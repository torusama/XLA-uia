"""Optional model inference and per-image comparison measurements."""
import base64
import importlib.machinery
import importlib.util
import threading
import time
from pathlib import Path

from .dataset import EXPECTED_ARCHIVE_SHA256, png_bytes


def _patch_efficientvit_package():
    """Keep the EfficientViT import path limited to segmentation needs."""
    try:
        spec = importlib.util.find_spec("efficientvit")
        if spec is None or not spec.submodule_search_locations:
            return
        root = Path(spec.submodule_search_locations[0])
        nn_init = root / "models" / "nn" / "__init__.py"
        if nn_init.exists():
            nn_init.write_text(
                "from .act import *\nfrom .norm import *\nfrom .ops import *\nfrom .triton_rms_norm import *\n",
                encoding="utf-8",
            )
        evit_init = root / "models" / "efficientvit" / "__init__.py"
        if evit_init.exists():
            evit_init.write_text("from .backbone import *\nfrom .seg import *\n", encoding="utf-8")
    except OSError:
        pass


def _ensure_triton_mock():
    """Allow efficientvit imports on environments without the triton package."""
    try:
        import triton  # noqa: F401
        import triton.language  # noqa: F401
        return
    except Exception:
        pass
    import sys
    import types

    def _make_module(name):
        module = types.ModuleType(name)
        module.__path__ = []
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = module
        return module

    _triton = _make_module("triton")
    _triton_language = _make_module("triton.language")
    _triton.backends = _make_module("triton.backends")
    _triton.backends.compiler = _make_module("triton.backends.compiler")
    _triton.compiler = _make_module("triton.compiler")
    _triton.compiler.compiler = _make_module("triton.compiler.compiler")
    _triton.language = _triton_language
    _triton.backends = _triton.backends
    for _name, _value in {
        "jit": lambda fn=None, **kwargs: (fn if fn is not None else (lambda x: x)),
        "cdiv": lambda a, b: (a + b - 1) // b,
        "next_power_of_2": lambda x: 1 << (x - 1).bit_length(),
    }.items():
        setattr(_triton, _name, _value)
    for _name, _value in {
        "constexpr": object,
        "dtype": object,
        "program_id": lambda *args, **kwargs: 0,
        "arange": lambda *args, **kwargs: None,
        "zeros": lambda *args, **kwargs: None,
        "float32": float,
        "load": lambda *args, **kwargs: None,
        "store": lambda *args, **kwargs: None,
        "where": lambda *args, **kwargs: None,
        "sum": lambda *args, **kwargs: None,
    }.items():
        setattr(_triton_language, _name, _value)


class ModelRegistry:
    """Load ViT and EfficientViT checkpoints for the demo."""

    def __init__(self, vit_checkpoint=None, efficientvit_checkpoint=None):
        self.models = {}
        self.errors = {"vit": "Chưa có checkpoint ViT.",
                       "efficientvit": "EfficientViT chưa được tích hợp."}
        self.lock = threading.Lock()
        self.device = None
        if vit_checkpoint:
            self._load_vit(Path(vit_checkpoint))
        if efficientvit_checkpoint:
            self._load_efficientvit(Path(efficientvit_checkpoint))

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

    def _load_efficientvit(self, checkpoint_path):
        try:
            _patch_efficientvit_package()
            _ensure_triton_mock()
            import torch
            import torch.nn as nn

            from efficientvit.seg_model_zoo import create_efficientvit_seg_model

            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            core = create_efficientvit_seg_model(name="efficientvit-seg-b1-ade20k", pretrained=False)
            for output_op in core.head.output_ops:
                last_layer = output_op.op_list[-1]
                in_channels = last_layer.conv.in_channels
                last_layer.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

            core.load_state_dict(state_dict, strict=True)

            class EVITSeg(torch.nn.Module):
                def __init__(self, core):
                    super().__init__()
                    self.core = core

                def forward(self, images):
                    logits = self.core(images)
                    return torch.nn.functional.interpolate(
                        logits, size=images.shape[-2:], mode="bilinear", align_corners=False
                    )

            model = EVITSeg(core)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(self.device).eval()
            self.models["efficientvit"] = model
            self.errors.pop("efficientvit", None)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self.errors["efficientvit"] = f"Không thể nạp checkpoint EfficientViT: {exc}"

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
        if name == "efficientvit":
            tensor = TF.normalize(TF.to_tensor(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
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