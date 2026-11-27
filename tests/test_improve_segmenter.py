"""Offline checks for paired augmentation and controlled fine-tuning."""
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

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


def test_comparison_preserves_source_and_saves_both_arms(tmp_path, monkeypatch):
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
    module.main()
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert result["split"] == "validation"
    assert set(result["arms"]) == {"weight_50", "weight_10"}
    assert result["validation_winner"] in ("source", "weight_50", "weight_10")
    for arm in result["arms"]:
        assert (output / arm / "best.pt").is_file()
        assert torch.load(output / arm / "final.pt", weights_only=True)["epoch"] == 1
