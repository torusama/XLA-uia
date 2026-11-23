"""Train or evaluate ViT with the Segmenter Mask Transformer decoder."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from xla.data.hf_segmentation import make_splits
from xla.engine.vit_segmentation import run_epoch
from xla.models.segmenter_vit import SegmenterViT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=4)
        command.add_argument("--workers", type=int, default=0)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train = commands.choices["train"]
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=0.0001)
    train.add_argument("--positive-weight", type=float, default=10.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--from-scratch", action="store_true")
    commands.choices["evaluate"].add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")
    if args.command == "train" and (args.epochs < 1 or args.learning_rate <= 0
                                    or args.positive_weight <= 0):
        parser.error("epochs, learning-rate and positive-weight must be positive")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    digest = sha256(args.archive)
    train_set, val_set, test_set = make_splits(args.archive)
    print(f"Samples: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}; "
          f"excluded: train={train_set.excluded}, val={val_set.excluded}, "
          f"test={test_set.excluded}", flush=True)
    if args.command == "evaluate":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["archive_sha256"] != digest:
            raise ValueError("Checkpoint was trained on a different archive")
        model = SegmenterViT(pretrained=False).to(device)
        model.load_state_dict(checkpoint["model"])
        loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)
        positive_weight = checkpoint.get("settings", {}).get("positive_weight")
        print(json.dumps(run_epoch(model, loader, device, positive_weight=positive_weight), indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SegmenterViT(pretrained=not args.from_scratch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    history = []
    best = -1.0
    settings = {"architecture": "segmenter_vit_mask_transformer",
                "decoder_source_commit": "20d1bfad354165ee45c3f65972a4d9c131f58d53",
                "archive": str(args.archive), "archive_sha256": digest, "seed": args.seed,
                "epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "positive_weight": args.positive_weight,
                "pretrained": not args.from_scratch,
                "excluded_train": train_set.excluded, "excluded_val": val_set.excluded,
                "excluded_test": test_set.excluded}
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Starting Segmenter training: {args.epochs} epochs, device={device}", flush=True)
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, args.positive_weight)
        validation = run_epoch(model, val_loader, device, positive_weight=args.positive_weight)
        record = {"epoch": epoch, "train": training, "val": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "val": validation,
                      "archive_sha256": digest, "settings": settings}
        if validation["dice"] > best:
            best = validation["dice"]
            torch.save(checkpoint, args.output / "best.pt")
        if epoch == args.epochs:
            torch.save(checkpoint, args.output / "final.pt")


if __name__ == "__main__":
    main()
