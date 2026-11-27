"""Offline checks for the official Mask Transformer integration."""
import torch
from torch import nn
from torch.nn import functional as F

from xla.models import segmenter_vit
from xla.vendor.segmenter.decoder import MaskTransformer


def test_two_class_mask_normalization_retains_gradient_with_larger_epsilon():
    def gradient(eps):
        scores = torch.tensor([[-0.1, 0.1]], requires_grad=True)
        output = F.layer_norm(scores, (2,), eps=eps)
        (output[0, 1] - output[0, 0]).backward()
        return scores.grad.norm()

    assert gradient(0.01) > 100 * gradient(1e-5)


def test_mask_transformer_binary_logits_and_gradients(monkeypatch):
    class FakeBackbone(nn.Module):
        embed_dim = 64
        num_prefix_tokens = 1

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, self.embed_dim)

        def reset_classifier(self, _):
            pass

        def forward_features(self, images):
            patches = F.avg_pool2d(images, 16).flatten(2).transpose(1, 2)
            patches = self.proj(patches)
            return torch.cat((patches[:, :1], patches), dim=1)

    monkeypatch.setattr(segmenter_vit, "build_vit", lambda **kwargs: FakeBackbone())
    model = segmenter_vit.SegmenterViT(pretrained=False).eval()
    assert isinstance(model.decoder, MaskTransformer)
    assert model.decoder.n_cls == 2 and len(model.decoder.blocks) == 2
    images = torch.randn(2, 3, 32, 32)
    tokens = model.backbone.forward_features(images)[:, 1:]
    scores = F.interpolate(model.decoder(tokens, (32, 32)), size=(32, 32),
                           mode="bilinear", align_corners=False)
    logits = model(images)
    assert logits.shape == (2, 1, 32, 32)
    assert torch.allclose(logits.sigmoid(), scores.softmax(dim=1)[:, 1:2], atol=1e-6)
    target = torch.zeros_like(logits)
    target[:, :, 8:24, 8:24] = 1
    F.binary_cross_entropy_with_logits(logits, target).backward()
    assert model.decoder.cls_emb.grad is not None
    assert torch.isfinite(model.decoder.cls_emb.grad).all()
    assert model.decoder.cls_emb.grad.abs().sum() > 0
    assert segmenter_vit.SegmenterViT(pretrained=False, mask_norm_eps=0.01).decoder.mask_norm.eps == 0.01


def test_segmenter_training_uses_own_architecture_and_saves_metadata(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path

    from test_vit_segmentation import make_archive

    archive = tmp_path / "dataset.zip"
    output = tmp_path / "output"
    make_archive(archive)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_segmenter.py"
    spec = importlib.util.spec_from_file_location("train_segmenter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "train", "--archive", str(archive),
                                     "--output", str(output), "--epochs", "1",
                                     "--batch-size", "1", "--device", "cpu", "--from-scratch"])
    module.main()
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert checkpoint["epoch"] == 1 and (output / "final.pt").exists()
    assert config["architecture"] == "segmenter_vit_mask_transformer"
    assert config["mask_norm_eps"] == 0.01
