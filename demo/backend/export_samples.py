"""Export the vetted test images and masks for the private static demo."""
import argparse
import json
from pathlib import Path

from .dataset import TestDataset, png_bytes


def export_samples(destination, data):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    samples = []
    for index in range(len(data.rows)):
        image_bytes, _, mask = data.read(index)
        image_name = f"{index:04d}.jpg"
        mask_name = f"{index:04d}.png"
        (destination / image_name).write_bytes(image_bytes)
        (destination / mask_name).write_bytes(png_bytes(mask))
        sample = data.describe(index)
        sample.update(image_url=f"./samples/{image_name}",
                      ground_truth_url=f"./samples/{mask_name}")
        samples.append(sample)
    (destination / "index.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(samples)} images and ground truth masks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_samples(args.destination, TestDataset())


if __name__ == "__main__":
    main()
