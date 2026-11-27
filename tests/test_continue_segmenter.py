"""Offline checks for continuation of a weights-only Segmenter run."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


def test_continue_keeps_epoch_number_and_prior_best(tmp_path, monkeypatch):
    from test_vit_segmentation import make_archive

    archive, output = tmp_path / "dataset.zip", tmp_path / "output"
    make_archive(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    history = [{"epoch": n, "train": {}, "val": {"dice": 1.0 if n == 46 else 0.0}}
               for n in range(1, 51)]
    prior_history = tmp_path / "history.json"
    prior_history.write_text(json.dumps(history), encoding="utf-8")
    old_model = nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        old_model.weight.fill_(0.123)
    settings = {"architecture": "segmenter_vit_mask_transformer",
                "positive_weight": 100.0, "mask_norm_eps": 0.01}
    final, best = tmp_path / "old_final.pt", tmp_path / "old_best.pt"
    torch.save({"model": old_model.state_dict(), "epoch": 50, "val": history[-1]["val"],
                "archive_sha256": digest, "settings": settings}, final)
    torch.save({"model": old_model.state_dict(), "epoch": 46, "val": history[45]["val"],
                "archive_sha256": digest, "settings": settings}, best)
    path = Path(__file__).resolve().parents[1] / "scripts/continue_segmenter.py"
    spec = importlib.util.spec_from_file_location("continue_segmenter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(path), "--archive", str(archive),
                                     "--checkpoint", str(final), "--prior-best", str(best),
                                     "--prior-history", str(prior_history), "--output", str(output),
                                     "--additional-epochs", "2", "--batch-size", "1",
                                     "--device", "cpu"])
    module.main()
    new = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    saved_best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    resume = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)
    combined = json.loads((output / "history.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in combined] == list(range(1, 53))
    assert new["epoch"] == 52 and saved_best["epoch"] == 46
    assert new["settings"]["optimizer_state"] == "reinitialized"
    assert resume["epoch"] == 52 and resume["optimizer"]["state"]
    assert not torch.equal(new["model"]["weight"], old_model.weight)


def test_continue_rejects_incomplete_history(tmp_path, monkeypatch):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"data")
    history = tmp_path / "history.json"
    history.write_text("[]", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"epoch": 50}, checkpoint)
    path = Path(__file__).resolve().parents[1] / "scripts/continue_segmenter.py"
    spec = importlib.util.spec_from_file_location("continue_segmenter_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "--archive", str(archive),
                                     "--checkpoint", str(checkpoint), "--prior-best", str(checkpoint),
                                     "--prior-history", str(history), "--output", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="complete history"):
        module.main()
