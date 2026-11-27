"""Audit a Segmenter checkpoint on validation without evaluating the test set."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xla.data.hf_segmentation import make_splits
from xla.engine.segmentation_audit import ThresholdAudit
from xla.models.segmenter_vit import SegmenterViT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["settings"].get("architecture") != "segmenter_vit_mask_transformer":
        raise ValueError("Not a Segmenter Mask Transformer checkpoint")
    if checkpoint["archive_sha256"] != sha256(args.archive):
        raise ValueError("Checkpoint and archive do not match")
    _, val_set, _ = make_splits(args.archive)
    model = SegmenterViT(pretrained=False,
                         mask_norm_eps=checkpoint["settings"]["mask_norm_eps"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    audit = ThresholdAudit([n / 100 for n in range(10, 100, 5)])
    loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    with torch.inference_mode():
        for batch_index, (images, target) in enumerate(loader):
            probabilities = model(images.to(args.device)).sigmoid()
            audit.update(probabilities, target.to(args.device))
            if (batch_index + 1) % 25 == 0:
                print(f"Validation batches: {batch_index + 1}/{len(loader)}", flush=True)
    results = audit.results()
    report = {"split": "validation", "samples": len(val_set),
              "normal_images": audit.normal_images, "positive_images": audit.positive_images,
              "checkpoint_epoch": checkpoint["epoch"],
              "checkpoint_sha256": sha256(args.checkpoint),
              "archive_sha256": checkpoint["archive_sha256"],
              "thresholds": results, "best_threshold": max(results, key=lambda r: r["dice"])}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
