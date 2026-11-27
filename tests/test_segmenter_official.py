"""Offline checks for the upstream Segmenter training pipeline."""
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from xla.engine.segmenter_official import RandomHorizontalFlip, run_epoch


def test_flip_keeps_image_and_mask_aligned():
    image = torch.zeros(3, 2, 3)
    image[:, 0, 0] = 1
    mask = image[:1].clone()
    data = RandomHorizontalFlip([(image, mask)], probability=1)
    result_image, result_mask = data[0]
    assert result_image[0, 0, 2] == result_mask[0, 0, 2] == 1
    assert result_image[0, 0, 0] == result_mask[0, 0, 0] == 0


def test_epoch_uses_unweighted_cross_entropy_and_argmax():
    model = nn.Conv2d(1, 2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[[-1.0]]], [[[1.0]]]]))
    images = torch.tensor([[[[-1.0, 1.0]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])
    scores = model(images)
    expected = F.cross_entropy(scores, target[:, 0].long()).item()
    stats = run_epoch(model, DataLoader(TensorDataset(images, target)), torch.device("cpu"))
    assert abs(stats["loss"] - expected) < 1e-7
    assert stats["dice"] == stats["iou"] == 1


def test_training_script_records_official_protocol_and_checkpoints(tmp_path, monkeypatch):
    from test_vit_segmentation import make_archive

    archive, output = tmp_path / "dataset.zip", tmp_path / "output"
    make_archive(archive)
    script = Path(__file__).resolve().parents[1] / "scripts/train_segmenter_official.py"
    spec = importlib.util.spec_from_file_location("train_segmenter_official", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OfficialSegmenter", lambda **kwargs: nn.Conv2d(3, 2, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "--archive", str(archive),
                                     "--output", str(output), "--epochs", "1",
                                     "--batch-size", "1", "--device", "cpu", "--from-scratch"])
    module.main()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["loss"] == "cross_entropy" and config["optimizer"] == "SGD"
    assert config["scheduler"] == "polynomial" and config["weight_decay"] == 0
    assert (output / "best.pt").is_file() and (output / "final.pt").is_file()
