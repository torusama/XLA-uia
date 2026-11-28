"""Read the vetted BTXRD test split and its polygon ground truth."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/raw/xla-seminar-uia/btxrd_final.zip"
MANIFEST = ROOT / "data/splits/btxrd/test.csv"
EXPECTED_ARCHIVE_SHA256 = "59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b"


class TestDataset:
    """Keep source images unchanged; rasterize masks on demand."""

    def __init__(self, archive=ARCHIVE, manifest=MANIFEST, verify=True):
        self.archive = Path(archive)
        if verify:
            digest = hashlib.sha256()
            with self.archive.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != EXPECTED_ARCHIVE_SHA256:
                raise ValueError("BTXRD archive checksum mismatch")
        with Path(manifest).open(encoding="utf-8", newline="") as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError("Test split is empty")
        with ZipFile(self.archive) as z:
            names = set(z.namelist())
            for row in self.rows:
                if (not row["image"].startswith("test/images/")
                        or not row["annotation"].startswith("test/annotations/")
                        or row["image"] not in names or row["annotation"] not in names):
                    raise ValueError("Invalid test split manifest")

    def item(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.rows):
            raise IndexError("Sample index out of range")
        return self.rows[index]

    def describe(self, index):
        row = self.item(index)
        return {"index": index, "total": len(self.rows), "name": Path(row["image"]).name,
                "category": row["category"], "label": row["binary_label"]}

    def read(self, index):
        return self._read(index)[:3]

    def _read(self, index, audit=False):
        row = self.item(index)
        with ZipFile(self.archive) as z:
            image_bytes = z.read(row["image"])
            annotation = json.loads(z.read(row["annotation"]))
        with Image.open(io.BytesIO(image_bytes)) as original:
            if audit and (original.size != (224, 224) or original.mode != "RGB"):
                raise ValueError("unsupported_image_format")
            image = original.convert("RGB")
        if image.size != (annotation["image_width"], annotation["image_height"]):
            raise ValueError("annotation_size_mismatch")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for item in annotation["annotations"]:
            for polygon in item.get("polygons", []):
                if len(polygon) < 3:
                    raise ValueError("invalid_polygon")
                if audit and any(len(point) != 2 or not (
                        0 <= point[0] < image.width and 0 <= point[1] < image.height)
                        for point in polygon):
                    raise ValueError("polygon_out_of_bounds")
                draw.polygon([tuple(point) for point in polygon], fill=255)
        return image_bytes, image, mask, annotation

    def comparison_selection(self):
        """Screen only image/annotation quality and byte duplicates before inference."""
        included, excluded = [], []
        seen = set()
        for index, row in enumerate(self.rows):
            sample = self.describe(index)
            sample.pop("total")
            try:
                image_bytes, image, mask, annotation = self._read(index, audit=True)
                digest = hashlib.sha256(image_bytes).hexdigest()
                sample["sha256"] = digest
                if row.get("sha256") != digest:
                    raise ValueError("image_checksum_mismatch")
                expected_label = "normal" if row["category"] == "Normal" else "tumor"
                if (annotation["class_label"] != row["category"] or sample["label"] != expected_label
                        or bool(mask.getbbox()) != (expected_label == "tumor")):
                    raise ValueError("label_mask_mismatch")
                if all(low == high for low, high in image.getextrema()):
                    raise ValueError("uniform_image")
                if digest in seen:
                    raise ValueError("duplicate_image_bytes")
            except ValueError as error:
                excluded.append(dict(sample, reason=str(error)))
            except (OSError, KeyError, TypeError, IndexError):
                excluded.append(dict(sample, reason="unreadable_image_or_annotation"))
            else:
                seen.add(digest)
                included.append(sample)
        return {"rule": "btxrd_quality_v1", "archive_sha256": EXPECTED_ARCHIVE_SHA256,
                "source_count": len(self.rows), "included_by_label": dict(Counter(
                    sample["label"] for sample in included)),
                "included": included, "excluded": excluded}


def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
