"""Audit the BTXRD archive and export the splits used by ViT training."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from xla.data.hf_segmentation import PolygonArchive, make_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=Path("data/raw/xla-seminar-uia/btxrd_final.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/btxrd"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    expected_hash = "59ce94244fdf997f0c68e6a378690a4359e6c411f7fda7ac2f3994489637884b"
    if archive_hash != expected_hash:
        raise ValueError("Archive checksum does not match the pinned BTXRD revision")
    all_splits = {name: PolygonArchive(args.archive, name) for name in ("train", "val", "test")}
    clean_splits = dict(zip(("train", "val", "test"), make_splits(args.archive)))
    classes_by_hash = defaultdict(set)
    for dataset in all_splits.values():
        for image_name, _, digest in dataset.samples:
            classes_by_hash[digest].add(Path(image_name).parent.name)
    conflict_hashes = {key for key, values in classes_by_hash.items() if len(values) > 1}
    held_out = clean_splits["val"].hashes | clean_splits["test"].hashes
    excluded = []
    counts = {}
    channel_sum = [0, 0, 0]
    channel_square_sum = [0, 0, 0]
    pixel_count = 0
    with ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        image_files = {n for n in names if "/images/" in n and n.lower().endswith((".jpg", ".jpeg"))}
        referenced_images = set()
        for split_name, dataset in all_splits.items():
            selected = {sample[0] for sample in clean_splits[split_name].samples}
            counts[split_name] = Counter()
            for image_name, annotation_name, digest in dataset.samples:
                referenced_images.add(image_name)
                annotation = json.loads(archive.read(annotation_name))
                category = Path(image_name).parent.name
                if annotation["class_label"] != category:
                    raise ValueError(f"Class mismatch: {annotation_name}")
                if (annotation["image_width"], annotation["image_height"]) != (224, 224):
                    raise ValueError(f"Bad annotation size: {annotation_name}")
                polygons = [p for item in annotation["annotations"]
                            for p in item.get("polygons", [])]
                if (category == "Normal") != (not polygons):
                    raise ValueError(f"Unexpected polygon presence: {annotation_name}")
                for polygon in polygons:
                    if len(polygon) < 3 or any(len(point) != 2 or not
                            (0 <= point[0] < 224 and 0 <= point[1] < 224)
                            for point in polygon):
                        raise ValueError(f"Invalid polygon: {annotation_name}")
                with Image.open(io.BytesIO(archive.read(image_name))) as image:
                    if image.size != (224, 224) or image.mode != "RGB":
                        raise ValueError(f"Unexpected image format: {image_name}")
                    image.load()
                    if split_name == "train" and image_name in selected:
                        histogram = image.histogram()
                        pixel_count += image.width * image.height
                        for channel in range(3):
                            values = histogram[channel * 256:(channel + 1) * 256]
                            channel_sum[channel] += sum(i * n for i, n in enumerate(values))
                            channel_square_sum[channel] += sum(i * i * n for i, n in enumerate(values))
                if image_name in selected:
                    counts[split_name][category] += 1
                else:
                    reason = "conflicting_labels" if digest in conflict_hashes else "held_out_duplicate"
                    if reason == "held_out_duplicate" and (split_name != "train" or digest not in held_out):
                        raise ValueError(f"Unexpected exclusion: {image_name}")
                    excluded.append((split_name, image_name, annotation_name, digest, category, reason))
        if image_files != referenced_images:
            raise ValueError("Archive contains unreferenced images or missing image files")
    args.output.mkdir(parents=True, exist_ok=False)
    fields = ["image", "annotation", "sha256", "category", "binary_label"]
    for split_name, dataset in clean_splits.items():
        with (args.output / f"{split_name}.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for image, annotation, digest in dataset.samples:
                category = Path(image).parent.name
                writer.writerow([image, annotation, digest, category,
                                 "normal" if category == "Normal" else "tumor"])
    with (args.output / "excluded.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_split", "image", "annotation", "sha256", "category", "reason"])
        writer.writerows(excluded)
    mean = [s / pixel_count / 255 for s in channel_sum]
    std = [((channel_square_sum[i] / pixel_count / (255 * 255) - mean[i] ** 2) ** 0.5)
           for i in range(3)]
    report = {"archive": str(args.archive), "archive_sha256": archive_hash,
              "source_revision": "cf8b0569b18e43e7cecab412a06e60aeaf258b32",
              "split_policy": "source splits; remove conflicting labels and train duplicates of held-out images",
              "image_size": [224, 224], "image_mode": "RGB",
              "counts_by_category": {name: dict(sorted(c.items())) for name, c in counts.items()},
              "excluded_by_reason": dict(Counter(row[-1] for row in excluded)),
              "train_rgb_mean_0_1": mean, "train_rgb_std_0_1": std,
              "training_normalization_mean": [0.5, 0.5, 0.5],
              "training_normalization_std": [0.5, 0.5, 0.5]}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
