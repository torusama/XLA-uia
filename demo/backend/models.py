"""Optional model inference and per-image comparison measurements."""
import base64
import hashlib
import importlib
import importlib.util
import platform
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

from .dataset import EXPECTED_ARCHIVE_SHA256, ROOT, png_bytes
from .resources import ResourceMonitor

DEFAULT_VIT_CHECKPOINT = ROOT / "outputs/kaggle_checks/vit-detail/best_vit_detail.pt"
DEFAULT_EFFICIENTVIT_CHECKPOINT = ROOT / "outputs/kaggle_checks/efficientvit/best_evit_seg_btxrd.pth"
DETAIL_VIT_ARCHITECTURE = "DetailViT"


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_efficientvit_factory():
    """Import only segmentation components without modifying installed files."""
    if "efficientvit.seg_model_zoo" in sys.modules:
        return sys.modules["efficientvit.seg_model_zoo"].create_efficientvit_seg_model
    spec = importlib.util.find_spec("efficientvit")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Install the EfficientViT segmentation dependency")
    root = Path(spec.submodule_search_locations[0])
    importlib.import_module("efficientvit.models")

    def namespace(name, directory):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        module.__package__ = name
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        sys.modules[name] = module
        return module

    nn_package = namespace("efficientvit.models.nn", root / "models" / "nn")
    if importlib.util.find_spec("triton") is None:
        triton_module = types.ModuleType("efficientvit.models.nn.triton_rms_norm")

        class UnsupportedTritonRMSNorm:
            @staticmethod
            def apply(*args, **kwargs):
                raise RuntimeError("Triton RMSNorm is unavailable; this adapter only supports B1")

        triton_module.TritonRMSNorm2dFunc = UnsupportedTritonRMSNorm
        sys.modules[triton_module.__name__] = triton_module
    for name in ("act", "norm", "ops"):
        module = importlib.import_module(f"efficientvit.models.nn.{name}")
        for exported in module.__all__:
            setattr(nn_package, exported, getattr(module, exported))
    model_package = namespace("efficientvit.models.efficientvit", root / "models" / "efficientvit")
    for name in ("backbone", "seg"):
        module = importlib.import_module(f"efficientvit.models.efficientvit.{name}")
        for exported in module.__all__:
            setattr(model_package, exported, getattr(module, exported))
    return importlib.import_module("efficientvit.seg_model_zoo").create_efficientvit_seg_model


class ModelRegistry:
    """Load ViT and EfficientViT checkpoints for the demo."""

    def __init__(self, vit_checkpoint=None, efficientvit_checkpoint=None, vit_threshold=0.5):
        if not 0 < vit_threshold < 1:
            raise ValueError("vit_threshold must be between 0 and 1")
        self.vit_threshold = vit_threshold
        self.models = {}
        self.errors = {"vit": "Chưa có checkpoint ViT.",
                       "efficientvit": "EfficientViT chưa được tích hợp."}
        self.lock = threading.Lock()
        self.complexity_cache = {}
        self.model_metadata = {}
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
            settings = checkpoint.get("settings", {})
            architecture = checkpoint.get("architecture") or settings.get("architecture")
            if architecture == DETAIL_VIT_ARCHITECTURE:
                from xla.models.vit_detail import DetailViT

                model = DetailViT(pretrained=False)
            elif architecture == "segmenter_vit_mask_transformer":
                from xla.models.segmenter_vit import SegmenterViT

                mask_norm_eps = settings.get("mask_norm_eps", 1e-5)
                model = SegmenterViT(pretrained=False, mask_norm_eps=mask_norm_eps)
            elif architecture == "segmenter_official_mask_transformer":
                from xla.models.segmenter_official import OfficialSegmenter

                model = OfficialSegmenter(pretrained=False)
            elif architecture is None:
                from xla.models.vit_segmentation import ViTSegmentation

                model = ViTSegmentation(pretrained=False)
            else:
                raise ValueError(f"Unsupported ViT architecture: {architecture}")
            model.load_state_dict(checkpoint["model"], strict=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(self.device).eval()
            self.models["vit"] = model
            self.model_metadata["vit"] = {"checkpoint_sha256": checkpoint_digest(checkpoint_path),
                                           "architecture": architecture or "vit_conv_head"}
            self.errors.pop("vit", None)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self.errors["vit"] = f"Không thể nạp checkpoint ViT: {exc}"

    def _load_efficientvit(self, checkpoint_path):
        try:
            create_efficientvit_seg_model = _load_efficientvit_factory()
            import torch
            from torch import nn

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
            self.model_metadata["efficientvit"] = {
                "checkpoint_sha256": checkpoint_digest(checkpoint_path),
                "architecture": "efficientvit_seg_b1"}
            self.errors.pop("efficientvit", None)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self.errors["efficientvit"] = f"Không thể nạp checkpoint EfficientViT: {exc}"

    def statuses(self):
        return {name: {"ready": name in self.models, "message": self.errors.get(name, "Sẵn sàng")}
                for name in ("vit", "efficientvit")}

    def _profile_gflops(self, model, tensor):
        """Count supported forward operations once per model and input shape."""
        from torch.utils.flop_counter import FlopCounterMode

        key = (id(model), tuple(tensor.shape), str(tensor.dtype), str(tensor.device))
        if key not in self.complexity_cache:
            with FlopCounterMode(display=False) as counter:
                model(tensor)
            total = counter.get_total_flops()
            if total <= 0:
                raise RuntimeError("No supported floating-point operations were counted")
            self.complexity_cache[key] = total / 1_000_000_000
        return self.complexity_cache[key]

    def infer(self, name, image, ground_truth):
        if name not in self.models:
            return {"ready": False, "error": self.errors.get(name, "Model chưa sẵn sàng")}
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        model = self.models[name]
        architecture = self.model_metadata.get(name, {}).get("architecture")
        if name == "efficientvit" or architecture == DETAIL_VIT_ARCHITECTURE:
            tensor = TF.normalize(TF.to_tensor(image), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
            tensor = TF.normalize(TF.to_tensor(image), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        tensor = tensor.unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            try:
                gflops = self._profile_gflops(model, tensor)
                gflops_error = None
            except (ImportError, RuntimeError, NotImplementedError) as exc:
                gflops = None
                gflops_error = str(exc)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            gpu_baseline = None
            if self.device.type == "cuda":
                gpu_baseline = torch.cuda.memory_allocated(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            started_at = datetime.now(timezone.utc).isoformat()
            with ResourceMonitor() as monitor:
                start = time.perf_counter()
                logits = model(tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latency_ms = (time.perf_counter() - start) * 1000
            resources = monitor.metrics(latency_ms)
            gpu_peak = torch.cuda.max_memory_allocated(self.device) if gpu_baseline is not None else None
            resources.update(gpu_peak_mb=gpu_peak / 1024**2 if gpu_peak is not None else None,
                             gpu_delta_mb=(gpu_peak - gpu_baseline) / 1024**2
                             if gpu_peak is not None else None)
        probabilities = (logits[0].softmax(dim=0)[1] if logits.shape[1] == 2
                         else logits[0, 0].sigmoid())
        threshold = self.vit_threshold if name == "vit" else 0.5
        prediction = probabilities.ge(threshold).cpu().numpy()
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
                "inference_started_at": started_at, **resources,
                "dice": dice, "iou": iou, "tp": tp, "fp": fp, "fn": fn, "latency_ms": latency_ms,
                "params_m": sum(p.numel() for p in model.parameters()) / 1_000_000,
                "gflops": gflops, "gflops_error": gflops_error,
                "gflops_method": "PyTorch forward FLOPs; multiply-add=2; supported operators only",
                "measurement_config": {**self.model_metadata.get(name, {}),
                                       "dataset_sha256": EXPECTED_ARCHIVE_SHA256,
                                       "threshold": threshold, "input_shape": list(tensor.shape),
                                       "device": str(self.device), "dtype": str(tensor.dtype),
                                       "hardware": torch.cuda.get_device_name(self.device)
                                       if self.device.type == "cuda" else platform.processor(),
                                       "torch_version": str(torch.__version__),
                                       "torch_threads": torch.get_num_threads(),
                                       "measurement_version": 2},
                "device": str(self.device)}
