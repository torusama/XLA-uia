"""Optional model inference and per-image comparison measurements."""
import base64
import threading
import time
from pathlib import Path

from .dataset import EXPECTED_ARCHIVE_SHA256, png_bytes


class ModelRegistry:
    """Load a ViT checkpoint and leave an explicit slot for EfficientViT."""

    def __init__(self, vit_checkpoint=None):
        self.models = {}
        self.errors = {"vit": "Chưa có checkpoint ViT.",
                       "efficientvit": "EfficientViT chưa được tích hợp."}
        self.lock = threading.Lock()
        self.device = None
        if vit_checkpoint:
            self._load_vit(Path(vit_checkpoint))

    def _load_vit(self, checkpoint_path):
        try:
            import torch

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
                raise ValueError("Checkpoint không thuộc phiên bản BTXRD này")
            architecture = checkpoint.get("settings", {}).get("architecture")
            if architecture == "segmenter_vit_mask_transformer":
                from xla.models.segmenter_vit import SegmenterViT

                model = SegmenterViT(pretrained=False)
            elif architecture is None:
                from xla.models.vit_segmentation import ViTSegmentation

                model = ViTSegmentation(pretrained=False)
            else:
                raise ValueError(f"Unsupported ViT architecture: {architecture}")
            model.load_state_dict(checkpoint["model"], strict=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(self.device).eval()
            self.models["vit"] = model
            self.errors.pop("vit", None)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self.errors["vit"] = f"Không thể nạp checkpoint ViT: {exc}"

    def statuses(self):
        return {name: {"ready": name in self.models, "message": self.errors.get(name, "Sẵn sàng")}
                for name in ("vit", "efficientvit")}

    def infer(self, name, image, ground_truth):
        if name not in self.models:
            return {"ready": False, "error": self.errors.get(name, "Model chưa sẵn sàng")}
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        model = self.models[name]
        tensor = TF.normalize(TF.to_tensor(image), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            logits = model(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency_ms = (time.perf_counter() - start) * 1000
        prediction = logits[0, 0].sigmoid().ge(0.5).cpu().numpy()
        truth = np.asarray(ground_truth) > 0
        tp = int((prediction & truth).sum())
        fp = int((prediction & ~truth).sum())
        fn = int((~prediction & truth).sum())
        dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        overlay = np.zeros((*prediction.shape, 4), dtype=np.uint8)
        overlay[prediction] = [255, 103, 78, 180]
        encoded = base64.b64encode(png_bytes(Image.fromarray(overlay, "RGBA"))).decode("ascii")
        return {"ready": True, "mask_url": f"data:image/png;base64,{encoded}",
                "dice": dice, "iou": iou, "latency_ms": latency_ms,
                "params_m": sum(p.numel() for p in model.parameters()) / 1_000_000,
                "gflops": None, "device": str(self.device)}
