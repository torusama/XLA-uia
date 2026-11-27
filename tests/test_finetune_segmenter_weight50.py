"""Offline checks for the weight-50 fine-tuning experiment."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch import nn


def test_weight50_trial_loads_checkpoint_and_records_configuration(tmp_path, monkeypatch):
    from test_vit_segmentation import make_archive

    archive, output = tmp_path / "data.zip", tmp_path / "output"
    make_archive(archive)
    old_model = nn.Conv2d(3, 1, 1)
    checkpoint = tmp_path / "best.pt"
    torch.save({"epoch": 87, "model": old_model.state_dict(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "val": {"dice": 0.5345},
                "settings": {"architecture": "segmenter_vit_mask_transformer",
                             "positive_weight": 100.0, "mask_norm_eps": 0.01}}, checkpoint)
    script = Path(__file__).resolve().parents[1] / "scripts/finetune_segmenter_weight50.py"
    spec = importlib.util.spec_from_file_location("finetune_weight50", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "--archive", str(archive), "--checkpoint",
                                     str(checkpoint), "--output", str(output), "--epochs", "1",
                                     "--batch-size", "1", "--device", "cpu"])
    module.main()
    settings = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert settings["positive_weight"] == 50 and settings["trial_start_epoch"] == 87
    assert settings["optimizer_state"] == "reinitialized"
    assert settings["baseline_val"]["dice"] == 0.5345
    for filename in ("best.pt", "final.pt", "resume.pt", "history.json"):
        assert (output / filename).is_file()
    result = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert result["epoch"] == 1 and result["settings"]["positive_weight"] == 50
