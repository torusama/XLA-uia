"""Small-lesion loss and end-to-end A/B smoke checks."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn

from xla.engine.vit_segmentation import segmentation_loss


def test_extra_term_only_uses_small_lesions_and_backpropagates():
    logits = torch.full((3, 1, 20, 20), -1.0, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[0, :, :4, :4] = 1
    masks[1, :, :16, :16] = 1
    base = segmentation_loss(logits, masks, 10)
    enhanced = segmentation_loss(logits, masks, 10, small_dice_weight=0.5)
    extra_grad = torch.autograd.grad(enhanced - base, logits)[0]
    assert enhanced > base
    assert extra_grad[0, 0, 0, 0] < 0
    assert extra_grad[0, 0, -1, -1] > 0
    assert extra_grad[1:].abs().max() < 1e-8
    assert torch.isfinite(extra_grad).all()
    no_small = masks.clone(); no_small[0] = 0
    assert torch.equal(segmentation_loss(logits, no_small, 10),
                       segmentation_loss(logits, no_small, 10, small_dice_weight=0.5))


def test_small_dice_trial_saves_control_and_candidate(tmp_path, monkeypatch):
    from test_vit_segmentation import make_archive

    archive, checkpoint, out = tmp_path / "dataset.zip", tmp_path / "source.pt", tmp_path / "out"
    make_archive(archive)
    torch.save({"model": nn.Conv2d(3, 1, 1).state_dict(), "epoch": 1,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "settings": {"architecture": "segmenter_vit_mask_transformer", "mask_norm_eps": 0.01}}, checkpoint)
    script = Path(__file__).resolve().parents[1] / "scripts/improve_segmenter.py"
    spec = importlib.util.spec_from_file_location("small_dice_trial", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "--archive", str(archive), "--checkpoint", str(checkpoint),
                        "--output", str(out), "--weights", "10", "--small-dice-trial",
                        "--epochs", "1", "--batch-size", "1", "--device", "cpu"])
    module.main()
    report = json.loads((out / "comparison.json").read_text())
    assert report["split"] == "validation" and report["validation_view"] == "full_image"
    assert set(report["arms"]) == {"control", "small_dice"}
    for arm, expected in (("control", 0.0), ("small_dice", 0.5)):
        cfg = json.loads((out / arm / "config.json").read_text())
        assert cfg["small_dice_weight"] == expected
        assert (out / arm / "best.pt").exists()
        assert torch.load(out / arm / "final.pt", weights_only=True)["settings"]["small_dice_weight"] == expected
