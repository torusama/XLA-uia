"""Read the pinned Hugging Face archive and rasterize polygon masks."""
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PolygonArchive(Dataset):
    """Binary union of all annotated polygons; normal samples have empty masks."""

    def __init__(self, archive: Path, split: str, excluded_hashes: set[str] | None = None):
        self.archive = Path(archive)
        self.samples = []
        self.excluded = 0
        self._zip = None
        self._zip_pid = None
        excluded_hashes = excluded_hashes or set()
        with ZipFile(self.archive) as z:
            names = set(z.namelist())
            annotations = sorted(n for n in names if n.startswith(f"{split}/annotations/")
                                 and n.endswith(".json"))
            for name in annotations:
                annotation = json.loads(z.read(name))
                category = Path(name).parent.name
                image_name = f"{split}/images/{category}/{annotation['image_name']}"
                if image_name not in names:
                    raise ValueError(f"Missing image for {name}: {image_name}")
                digest = hashlib.sha256(z.read(image_name)).hexdigest()
                if digest in excluded_hashes:
                    self.excluded += 1
                    continue
                self.samples.append((image_name, name, digest))
        if not self.samples:
            raise ValueError(f"No samples in {self.archive} split {split}")

    @property
    def hashes(self) -> set[str]:
        return {sample[2] for sample in self.samples}

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        state["_zip_pid"] = None
        return state

    def __getitem__(self, index):
        image_name, annotation_name, _ = self.samples[index]
        if self._zip is None or self._zip_pid != os.getpid():
            if self._zip is not None:
                self._zip.close()
            self._zip = ZipFile(self.archive)
            self._zip_pid = os.getpid()
        annotation = json.loads(self._zip.read(annotation_name))
        with Image.open(io.BytesIO(self._zip.read(image_name))) as source:
            image = source.convert("RGB")
        expected = (annotation["image_width"], annotation["image_height"])
        if image.size != expected or expected != (224, 224):
            raise ValueError(f"Image and annotation dimensions do not match 224x224: {image_name}")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        count = 0
        for item in annotation["annotations"]:
            for polygon in item.get("polygons", []):
                if len(polygon) < 3:
                    raise ValueError(f"Invalid polygon: {annotation_name}")
                draw.polygon([tuple(point) for point in polygon], fill=1)
                count += 1
        if not count and annotation["class_label"].lower() != "normal":
            raise ValueError(f"Non-normal sample has no polygons: {annotation_name}")
        pixels = TF.normalize(TF.to_tensor(image), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        target = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(1, 224, 224)
        return pixels, target.float()


def make_splits(archive: Path):
    """Remove conflicting labels and train duplicates of held-out images."""
    splits = [PolygonArchive(archive, name) for name in ("train", "val", "test")]
    categories = {}
    for split in splits:
        for image_name, _, digest in split.samples:
            categories.setdefault(digest, set()).add(Path(image_name).parent.name)
    conflicts = {digest for digest, labels in categories.items() if len(labels) > 1}
    for split in splits:
        original = len(split.samples)
        split.samples = [sample for sample in split.samples if sample[2] not in conflicts]
        split.excluded = original - len(split.samples)
    train, val, test = splits
    if val.hashes & test.hashes:
        raise ValueError("Validation and test contain identical images; resolve the split first")
    held_out = val.hashes | test.hashes
    original = len(train.samples)
    train.samples = [sample for sample in train.samples if sample[2] not in held_out]
    train.excluded += original - len(train.samples)
    if any(not split.samples for split in splits):
        raise ValueError("A split is empty after removing conflicting and duplicate images")
    return train, val, test
