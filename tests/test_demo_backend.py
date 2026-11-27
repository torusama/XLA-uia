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
    assert result["gflops"] == pytest.approx(16 * 16 * 3 * 2 / 1e9)
    assert result["gflops_error"] is None
    assert result["mask_url"].startswith("data:image/png;base64,")
    assert registry.infer("efficientvit", None, None)["ready"] is False


def test_flop_profile_is_cached_separately_from_prediction():
    class CountedConv(nn.Conv2d):
        calls = 0

        def forward(self, tensor):
            self.calls += 1
            return super().forward(tensor)

    registry = ModelRegistry()
    model = CountedConv(3, 1, 1).eval()
    registry.models["vit"] = model
    registry.device = torch.device("cpu")
    image, mask = Image.new("RGB", (16, 16)), Image.new("L", (16, 16))
    first = registry.infer("vit", image, mask)
    second = registry.infer("vit", image, mask)
    assert first["gflops"] == second["gflops"] > 0
    assert model.calls == 3


def test_vit_threshold_does_not_change_efficientvit_threshold():
    registry = ModelRegistry(vit_threshold=0.8)
    model = nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(0.405465)
    registry.models = {"vit": model.eval(), "efficientvit": model.eval()}
    registry.device = torch.device("cpu")
    image = Image.new("RGB", (16, 16))
    mask = Image.new("L", (16, 16), 255)
    assert registry.infer("vit", image, mask)["dice"] == 0
    assert registry.infer("efficientvit", image, mask)["dice"] == 1


def test_efficientvit_factory_keeps_installed_files_unchanged():
    import hashlib
    import importlib.util
    from pathlib import Path

    from demo.backend.models import _load_efficientvit_factory

    spec = importlib.util.find_spec("efficientvit")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("EfficientViT dependency is not installed")
    root = Path(spec.submodule_search_locations[0])
    files = [root / "models/nn/__init__.py", root / "models/efficientvit/__init__.py"]
    before = [hashlib.sha256(path.read_bytes()).digest() for path in files]
    factory = _load_efficientvit_factory()
    model = factory(name="efficientvit-seg-b1-ade20k", pretrained=False)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert before == [hashlib.sha256(path.read_bytes()).digest() for path in files]


def test_inference_respects_selected_threshold():
    registry = ModelRegistry(vit_threshold=0.8)
    model = nn.Conv2d(3, 1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(1)
    registry.models["vit"] = model.eval()
    registry.device = torch.device("cpu")
    result = registry.infer("vit", Image.new("RGB", (16, 16)), Image.new("L", (16, 16), 255))
    assert result["dice"] == result["iou"] == 0


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


def test_load_official_segmenter_checkpoint(tmp_path, monkeypatch):
    from xla.models import segmenter_official

    monkeypatch.setattr(segmenter_official, "OfficialSegmenter", lambda **kwargs: nn.Conv2d(3, 2, 1))
    checkpoint = tmp_path / "segmenter.pt"
    torch.save({"model": nn.Conv2d(3, 2, 1).state_dict(),
                "archive_sha256": EXPECTED_ARCHIVE_SHA256,
                "settings": {"architecture": "segmenter_official_mask_transformer"}}, checkpoint)
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
