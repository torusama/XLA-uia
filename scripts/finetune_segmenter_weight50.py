"""Fine-tune an existing Segmenter checkpoint with a new foreground weight."""
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
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    old = checkpoint["settings"]
    if old.get("architecture") != "segmenter_vit_mask_transformer" or checkpoint["epoch"] != 87:
        raise ValueError("Expected Segmenter best checkpoint from epoch 87")
    digest = sha256(args.archive)
    if checkpoint["archive_sha256"] != digest:
        raise ValueError("Checkpoint and archive do not match")
    if old["positive_weight"] != 100.0:
        raise ValueError("Expected the previous run with positive weight 100")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train, val, test = make_splits(args.archive)
    print(f"Samples: train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    model = SegmenterViT(pretrained=False, mask_norm_eps=old["mask_norm_eps"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    baseline = checkpoint["val"]
    settings = dict(old)
    settings.update({"positive_weight": 50.0, "previous_positive_weight": 100.0,
                     "epochs": args.epochs, "trial_start_epoch": checkpoint["epoch"],
                     "trial_seed": args.seed, "batch_size": args.batch_size,
                     "learning_rate": 0.000002, "optimizer_state": "reinitialized",
                     "baseline_val": baseline, "source_checkpoint_sha256": sha256(args.checkpoint)})
    del checkpoint
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.01)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Starting weight-50 trial from epoch 87, baseline val Dice={baseline['dice']:.6f}; "
          "AdamW reinitialized", flush=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, args.device, optimizer, 50.0)
        validation = run_epoch(model, val_loader, args.device, positive_weight=50.0)
        record = {"trial_epoch": epoch, "train": training, "val": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        output = {"model": model.state_dict(), "epoch": epoch, "val": validation,
                  "archive_sha256": digest, "settings": settings}
        if validation["dice"] > best:
            best = validation["dice"]
            torch.save(output, args.output / "best.pt")
        if epoch % 5 == 0 or epoch == args.epochs:
            snapshot = dict(output, optimizer=optimizer.state_dict(),
                            torch_rng_state=torch.get_rng_state(),
                            cuda_rng_states=torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else [],
                            python_rng_state=random.getstate())
            torch.save(snapshot, args.output / "resume.pt")
        if epoch == args.epochs:
            torch.save(output, args.output / "final.pt")


if __name__ == "__main__":
    main()
