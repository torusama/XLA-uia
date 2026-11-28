"""Offline checks for paired augmentation and controlled fine-tuning."""
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from xla.data.augmentation import PairedAugmentation


def test_paired_geometry_is_reproducible_and_keeps_binary_masks():
    mask = torch.zeros(1, 224, 224)
    mask[:, 60:160, 70:150] = 1
    image = mask.expand(3, -1, -1) * 2 - 1
    dataset = PairedAugmentation([(image, mask)])
    random.seed(42)
    result_image, result_mask = dataset[0]
    random.seed(42)
    repeated = dataset[0]
    assert torch.equal(result_mask, repeated[1])
    assert torch.equal(result_image, repeated[0])
    assert set(result_mask.unique().tolist()) <= {0, 1}
    assert ((result_image[:1] > 0) == result_mask.bool()).float().mean() > 0.99


@pytest.mark.parametrize("trial", ["weight", "loss", "crop"])
def test_comparison_preserves_source_and_saves_both_arms(tmp_path, monkeypatch, trial):
    from test_vit_segmentation import make_archive

    archive, checkpoint, output = tmp_path / "data.zip", tmp_path / "source.pt", tmp_path / "run"
    make_archive(archive)
    torch.save({"model": nn.Conv2d(3, 1, 1).state_dict(), "epoch": 44,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "settings": {"architecture": "segmenter_vit_mask_transformer", "mask_norm_eps": 0.01}},
               checkpoint)
    script = Path(__file__).resolve().parents[1] / "scripts/improve_segmenter.py"
    spec = importlib.util.spec_from_file_location("improve_segmenter", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "--archive", str(archive), "--checkpoint",
                                     str(checkpoint), "--output", str(output), "--epochs", "1",
                                     "--device", "cpu", "--batch-size", "1"])
    if trial == "loss":
        sys.argv.extend(["--weights", "10", "--dice-reductions", "batch", "image",
                         "--expected-source-sha256", module.sha256(checkpoint)])
    elif trial == "crop":
        sys.argv.extend(["--weights", "10", "--crop-trial", "--crop-probability", "1",
                         "--expected-source-sha256", module.sha256(checkpoint)])
        views = []
        original_run_epoch = module.run_epoch

        def tracked_epoch(model, loader, device, optimizer=None, *args, **kwargs):
            views.append((optimizer is not None, loader.dataset))
            return original_run_epoch(model, loader, device, optimizer, *args, **kwargs)

        monkeypatch.setattr(module, "run_epoch", tracked_epoch)
    original_digest = module.sha256(checkpoint)
    module.main()
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert result["split"] == "validation"
    expected = {"weight": {"weight_50", "weight_10"},
                "loss": {"dice_batch", "dice_image"},
                "crop": {"full_image", "lesion_crop"}}[trial]
    assert set(result["arms"]) == expected
    assert result["validation_winner"] in {"source", *expected}
    assert module.sha256(checkpoint) == original_digest
    for arm in result["arms"]:
        assert (output / arm / "best.pt").is_file()
        assert torch.load(output / arm / "final.pt", weights_only=True)["epoch"] == 1
        if trial == "loss":
            final = torch.load(output / arm / "final.pt", weights_only=True)
            assert final["settings"]["dice_reduction"] == arm.removeprefix("dice_")
            assert final["settings"]["positive_weight"] == 10
        elif trial == "crop":
            final = torch.load(output / arm / "final.pt", weights_only=True)
            assert final["settings"]["lesion_crop"]["probability"] == (arm == "lesion_crop")
            assert final["settings"]["dice_reduction"] == "batch"
            assert final["settings"]["validation_view"] == "full_image"
            history = json.loads((output / arm / "history.json").read_text())
            assert history[0]["crop_probability"] == (arm == "lesion_crop")
    if trial == "crop":
        from xla.data.hf_segmentation import PolygonArchive
        from xla.data.lesion_crop import LesionCropDataset

        assert result["validation_view"] == "full_image"
        assert all(isinstance(dataset, LesionCropDataset if training else PolygonArchive)
                   for training, dataset in views)


def test_image_dice_weights_images_equally_and_handles_empty_masks():
    from xla.engine.vit_segmentation import segmentation_loss

    logits = torch.full((3, 1, 16, 16), -1.0, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, :, :2, :2] = 1
    target[1, :, :12, :12] = 1
    image_loss = segmentation_loss(logits, target, 10, "image")
    individual = torch.stack([segmentation_loss(x[None], y[None], 10)
                              for x, y in zip(logits, target)]).mean()
    assert torch.allclose(image_loss, individual)
    assert not torch.isclose(image_loss, segmentation_loss(logits, target, 10, "batch"))
    image_loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 0] < 0 < logits.grad[2, 0, 0, 0]


def test_loss_rejects_unknown_reduction():
    from xla.engine.vit_segmentation import segmentation_loss

    with pytest.raises(ValueError, match="dice_reduction"):
        segmentation_loss(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 10, "unknown")
