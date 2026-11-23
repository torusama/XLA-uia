"""Offline checks for the demo's test-set and comparison contract."""
import csv
import io
import json
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.request import urlopen
from zipfile import ZipFile

import pytest

pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image
from torch import nn

from demo.backend.dataset import EXPECTED_ARCHIVE_SHA256
from demo.backend.dataset import TestDataset as BTXRDTestDataset
from demo.backend.models import ModelRegistry
from demo.backend.server import DemoHandler


@pytest.fixture
def sample_dataset(tmp_path):
    image_path = "test/images/tumor/example.jpg"
    annotation_path = "test/annotations/tumor/example.json"
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "black").save(image, format="JPEG")
    annotation = {"image_width": 16, "image_height": 16, "annotations": [
        {"polygons": [[[2, 2], [8, 2], [8, 8], [2, 8]]]}]}
    archive = tmp_path / "sample.zip"
    with ZipFile(archive, "w") as z:
        z.writestr(image_path, image.getvalue())
        z.writestr(annotation_path, json.dumps(annotation))
    manifest = tmp_path / "test.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "annotation", "category", "binary_label"])
        writer.writeheader()
        writer.writerow({"image": image_path, "annotation": annotation_path,
                         "category": "tumor", "binary_label": "tumor"})
    return BTXRDTestDataset(archive, manifest, verify=False)


def test_dataset_rasterizes_ground_truth(sample_dataset):
    _, image, mask = sample_dataset.read(0)
    assert image.size == (16, 16)
    assert mask.getpixel((4, 4)) == 255
    assert mask.getpixel((0, 0)) == 0
    with pytest.raises(IndexError):
        sample_dataset.read(-1)


def test_inference_uses_actual_mask_and_parameters():
    registry = ModelRegistry()
    model = nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(1)
    registry.models["vit"] = model.eval()
    registry.device = torch.device("cpu")
    result = registry.infer("vit", Image.new("RGB", (16, 16)), Image.new("L", (16, 16), 255))
    assert result["dice"] == result["iou"] == 1
    assert result["latency_ms"] >= 0 and result["params_m"] > 0
    assert result["mask_url"].startswith("data:image/png;base64,")
    assert registry.infer("efficientvit", None, None)["ready"] is False


def test_checkpoint_load_checks_dataset_and_model(tmp_path, monkeypatch):
    from xla.models import vit_segmentation

    monkeypatch.setattr(vit_segmentation, "ViTSegmentation", lambda **kwargs: nn.Conv2d(3, 1, 1))
    checkpoint = tmp_path / "checkpoint.pt"
    state = nn.Conv2d(3, 1, 1).state_dict()
    torch.save({"model": state, "archive_sha256": EXPECTED_ARCHIVE_SHA256}, checkpoint)
    assert ModelRegistry(checkpoint).statuses()["vit"]["ready"]
    torch.save({"model": state, "archive_sha256": "wrong"}, checkpoint)
    assert not ModelRegistry(checkpoint).statuses()["vit"]["ready"]


def test_load_segmenter_checkpoint(tmp_path, monkeypatch):
    from xla.models import segmenter_vit

    monkeypatch.setattr(segmenter_vit, "SegmenterViT", lambda **kwargs: nn.Conv2d(3, 1, 1))
    checkpoint = tmp_path / "segmenter.pt"
    torch.save({"model": nn.Conv2d(3, 1, 1).state_dict(),
                "archive_sha256": EXPECTED_ARCHIVE_SHA256,
                "settings": {"architecture": "segmenter_vit_mask_transformer"}}, checkpoint)
    assert ModelRegistry(checkpoint).statuses()["vit"]["ready"]


def test_http_samples_and_compare(sample_dataset, tmp_path):
    class Handler(DemoHandler):
        dataset = sample_dataset
        registry = ModelRegistry()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{url}/api/samples") as response:
            data = json.load(response)
        assert data["samples"][0]["name"] == "example.jpg"
        with urlopen(f"{url}/api/ground-truth?index=0") as response:
            assert Image.open(io.BytesIO(response.read())).getpixel((4, 4)) == 255
        with urlopen(f"{url}/api/compare?index=0") as response:
            result = json.load(response)
        assert not result["models"]["vit"]["ready"]
        assert not result["models"]["efficientvit"]["ready"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
