"""Offline tests for polygon masks, split leakage, and segmentation training."""
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("timm")
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from xla.data.hf_segmentation import make_splits
from xla.engine.vit_segmentation import run_epoch, segmentation_loss
from xla.models import vit_segmentation


def test_weighted_loss_pushes_sparse_foreground_up():
    logits = torch.full((2, 1, 10, 10), -2.0, requires_grad=True)
    masks = torch.zeros_like(logits)
    masks[0, 0, 0, 0] = 1
    loss = segmentation_loss(logits, masks, positive_weight=10.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 0, 0] < 0 < logits.grad[1, 0, 0, 0]


def test_weighted_loss_handles_empty_masks():
    logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
    loss = segmentation_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def make_archive(path):
    with ZipFile(path, "w") as archive:
        for split, name, color, normal in [
            ("train", "duplicate", 50, False), ("train", "unique", 100, False),
            ("val", "validation", 50, False), ("test", "normal", 150, True),
        ]:
            image = Image.new("RGB", (224, 224), color=(color, 0, 0))
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            category = "Normal" if normal else "Fractured"
            annotation = {"image_name": f"{name}.png", "image_width": 224,
                          "image_height": 224, "class_label": category,
                          "annotations": [] if normal else [{"polygons": [
                              [[10, 10], [20, 10], [20, 20], [10, 20]]]}]}
            archive.writestr(f"{split}/images/{category}/{name}.png", stream.getvalue())
            archive.writestr(f"{split}/annotations/{category}/{name}.json", json.dumps(annotation))


def test_polygon_mask_and_duplicate_exclusion(tmp_path):
    archive = tmp_path / "dataset.zip"
    make_archive(archive)
    train, val, test = make_splits(archive)
    assert (len(train), len(val), len(test), train.excluded) == (1, 1, 1, 1)
    image, mask = train[0]
    assert image.shape == (3, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert mask[0, 15, 15] == 1 and mask[0, 30, 30] == 0
    assert test[0][1].sum() == 0


def test_head_shape_and_training_step(monkeypatch):
    class FakeBackbone(nn.Module):
        embed_dim = 8
        num_prefix_tokens = 1
        def __init__(self):
            super().__init__()
            self.patch_embed = type("Patch", (), {"grid_size": (2, 2)})()
            self.projection = nn.Linear(3, 8)
        def reset_classifier(self, unused):
            pass
        def forward_features(self, images):
            pooled = images.mean(dim=(2, 3))
            return self.projection(pooled).unsqueeze(1).expand(-1, 5, -1)

    monkeypatch.setattr(vit_segmentation, "build_vit", lambda **kwargs: FakeBackbone())
    model = vit_segmentation.ViTSegmentation(pretrained=False)
    images = torch.randn(2, 3, 16, 16)
    masks = torch.zeros(2, 1, 16, 16)
    masks[:, :, 4:8, 4:8] = 1
    assert model(images).shape == masks.shape
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.head[-1].weight.detach().clone()
    metrics = run_epoch(model, DataLoader(list(zip(images, masks)), batch_size=2),
                        torch.device("cpu"), optimizer)
    assert 0 <= metrics["dice"] <= 1 and 0 <= metrics["iou"] <= 1
    assert not torch.equal(before, model.head[-1].weight)


def test_conflicting_labels_removed_from_every_split(tmp_path):
    archive = tmp_path / "dataset.zip"
    make_archive(archive)
    with ZipFile(archive, "a") as z:
        for split, name, color, normal in [("train", "wrong", 50, True),
                                           ("val", "clean", 200, False)]:
            stream = io.BytesIO()
            Image.new("RGB", (224, 224), color=(color, 0, 0)).save(stream, format="PNG")
            category = "Normal" if normal else "Fractured"
            annotation = {"image_name": f"{name}.png", "image_width": 224,
                          "image_height": 224, "class_label": category,
                          "annotations": [] if normal else [{"polygons": [
                              [[10, 10], [20, 10], [20, 20]]]}]}
            z.writestr(f"{split}/images/{category}/{name}.png", stream.getvalue())
            z.writestr(f"{split}/annotations/{category}/{name}.json", json.dumps(annotation))
    train, val, test = make_splits(archive)
    assert (len(train), len(val), len(test)) == (1, 1, 1)
    assert (train.excluded, val.excluded, test.excluded) == (2, 1, 0)
    assert not (train.hashes & val.hashes or train.hashes & test.hashes or val.hashes & test.hashes)


def test_training_cli_saves_best_and_final(tmp_path, monkeypatch):
    import importlib.util
    import sys

    archive = tmp_path / "dataset.zip"
    output = tmp_path / "output"
    make_archive(archive)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_vit_segmentation.py"
    spec = importlib.util.spec_from_file_location("train_vit_segmentation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ViTSegmentation", lambda **kwargs: nn.Conv2d(3, 1, 1))
    monkeypatch.setattr(sys, "argv", [str(script), "train", "--archive", str(archive),
                                     "--output", str(output), "--epochs", "2", "--batch-size", "1",
                                     "--device", "cpu", "--from-scratch"])
    module.main()
    final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert final["epoch"] == 2 and best["epoch"] in (1, 2)
    assert len(json.loads((output / "history.json").read_text(encoding="utf-8"))) == 2
