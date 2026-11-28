"""Audit a pinned Segmenter checkpoint on small train masks; no test-set selection."""
import argparse
import hashlib
import json
import random
import subprocess
import types
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from xla.data.hf_segmentation import make_splits
from xla.engine.vit_segmentation import segmentation_loss
from xla.models.segmenter_vit import SegmenterViT
from xla.vendor.segmenter.utils import init_weights
from xla.vendor.segmenter.vit import VisionTransformer

COMMIT = "20d1bfad354165ee45c3f65972a4d9c131f58d53"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_decoder(repo):
    blocks_source = subprocess.check_output(["git", "-C", str(repo), "show",
                                             f"{COMMIT}:segm/model/blocks.py"], text=True)
    blocks = types.ModuleType("upstream_blocks")
    exec(compile(blocks_source, "upstream_blocks.py", "exec"), blocks.__dict__)  # noqa: S102
    source = subprocess.check_output(["git", "-C", str(repo), "show",
                                      f"{COMMIT}:segm/model/decoder.py"], text=True)
    source = source.replace("from segm.model.blocks import Block, FeedForward", "")
    source = source.replace("from segm.model.utils import init_weights", "")
    module = types.ModuleType("upstream_decoder")
    module.__dict__.update(Block=blocks.Block, FeedForward=blocks.FeedForward, init_weights=init_weights)
    exec(compile(source, "upstream_decoder.py", "exec"), module.__dict__)  # noqa: S102
    return module.MaskTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1:
        parser.error("steps must be positive")
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(6)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    archive_hash = sha256(args.archive)
    if checkpoint.get("archive_sha256") != archive_hash:
        raise ValueError("Checkpoint and dataset mismatch")
    train, _, _ = make_splits(args.archive)
    selected = []
    for index in range(len(train)):
        image, mask = train[index]
        if 16 <= mask.sum().item() < 256:
            selected.append((index, image, mask))
            if len(selected) == 2:
                break
    if len(selected) != 2:
        raise ValueError("Need two small foreground train masks")
    images = torch.stack([x[1] for x in selected]).to(args.device)
    masks = torch.stack([x[2] for x in selected]).to(args.device)
    model = SegmenterViT(pretrained=False, mask_norm_eps=checkpoint["settings"]["mask_norm_eps"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    with torch.no_grad():
        features = model.backbone.forward_features(images)[:, model.backbone.num_prefix_tokens:, :]
    encoder = VisionTransformer(image_size=(224, 224), patch_size=16, n_layers=12,
                                d_model=768, d_ff=3072, n_heads=12, n_cls=1000,
                                dropout=0.0, drop_path_rate=0.1)
    missing, unexpected = encoder.load_state_dict(model.backbone.state_dict(), strict=False)
    if set(missing) != {"head.weight", "head.bias"} or unexpected:
        raise ValueError(f"Unexpected encoder state mismatch: {missing}, {unexpected}")
    encoder.to(args.device).eval()
    with torch.no_grad():
        original_features = encoder(images, return_features=True)[:, 1:]
    encoder_diff = (features - original_features).abs().max().item()
    with torch.no_grad():
        current_mask = model.decoder(features, (224, 224))
        original_mask = model.decoder(original_features, (224, 224))
        def probability(scores):
            scores = F.interpolate(scores, (224, 224), mode="bilinear", align_corners=False)
            return (scores[:, 1:2] - scores[:, 0:1]).sigmoid()
        current_prob = probability(current_mask)
        other_prob = probability(original_mask)
        encoder_probability_diff = (current_prob - other_prob).abs().max().item()
        changed_pixels = int(((current_prob >= .7) != (other_prob >= .7)).sum())
    source_norms = dict(model.backbone.named_modules())
    adjusted = 0
    for name, layer in encoder.named_modules():
        source_layer = source_norms.get(name)
        if isinstance(layer, torch.nn.LayerNorm) and isinstance(source_layer, torch.nn.LayerNorm):
            layer.eps = source_layer.eps
            adjusted += 1
    with torch.no_grad():
        adjusted_features = encoder(images, return_features=True)[:, 1:]
    adjusted_diff = (features - adjusted_features).abs().max().item()
    source_eps = model.backbone.norm.eps
    del encoder
    original = upstream_decoder(args.upstream)(n_cls=2, patch_size=16, d_encoder=768,
        n_layers=2, n_heads=12, d_model=768, d_ff=3072, drop_path_rate=0.0, dropout=0.1)
    original.load_state_dict(model.decoder.state_dict(), strict=True)
    original.mask_norm.eps = model.decoder.mask_norm.eps
    original.to(args.device).eval()
    with torch.no_grad():
        expected = original(features, (224, 224))
        actual = model.decoder(features, (224, 224))
        logits = F.interpolate(actual, size=(224, 224), mode="bilinear", align_corners=False)
        two_class = logits.softmax(dim=1)[:, 1:2]
        binary = (logits[:, 1:2] - logits[:, 0:1]).sigmoid()
    decoder_diff = (expected - actual).abs().max().item()
    adapter_diff = (two_class - binary).abs().max().item()
    print(f"Encoder diff={encoder_diff}; decoder diff={decoder_diff}; adapter diff={adapter_diff}", flush=True)
    if decoder_diff > 1e-6 or adapter_diff > 1e-6:
        raise AssertionError("Decoder or binary adapter mismatch")
    del original
    model.decoder.eval()  # deterministic decoder-only overfit with frozen backbone
    optimizer = torch.optim.AdamW(model.parameters() if args.full_model else model.decoder.parameters(),
                                  lr=2e-6 if args.full_model else 2e-5, weight_decay=0.01)
    history = []
    images_out = []
    for step in range(args.steps + 1):
        if args.full_model:
            features = model.backbone.forward_features(images)[:, model.backbone.num_prefix_tokens:, :]
        output = model.decoder(features, (224, 224))
        output = F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)
        logits = output[:, 1:2] - output[:, 0:1]
        loss = segmentation_loss(logits, masks, 10, "batch")
        pred = logits.sigmoid() >= 0.7
        per_image = []
        for prediction, truth in zip(pred, masks.bool()):
            tp = int((prediction & truth).sum()); fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            per_image.append({"tp": tp, "fp": fp, "fn": fn,
                              "dice": 2 * tp / (2 * tp + fp + fn)})
        if step in (0, args.steps):
            images_out.append(pred.detach().cpu())
        if step == args.steps:
            history.append({"step": step, "loss": float(loss.detach()), "images": per_image})
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {name: float(param.grad.norm()) for name, param in model.decoder.named_parameters()
                 if param.grad is not None}
        if not grads or not all(torch.isfinite(torch.tensor(v)) for v in grads.values()):
            raise ValueError("Missing or nonfinite decoder gradients")
        if step in (0, 1) or (step + 1) % 10 == 0:
            history.append({"step": step, "loss": float(loss.detach()), "images": per_image,
                            "gradient_norms": {k: grads[k] for k in
                                ("cls_emb", "proj_patch", "proj_classes", "mask_norm.weight")}})
            if args.full_model:
                history[-1]["encoder_gradient_norm"] = float(model.backbone.patch_embed.proj.weight.grad.norm())
            print(json.dumps(history[-1]), flush=True)
        optimizer.step()
    args.output.mkdir(parents=True)
    canvas = Image.new("RGB", (224 * 3, 250 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (_, image, mask) in enumerate(selected):
        rgb = (((image.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())
        for j, label in enumerate((mask.bool(), images_out[0][i], images_out[1][i])):
            panel = Image.fromarray(rgb).convert("RGBA")
            overlay = Image.new("RGBA", panel.size, (255, 55, 30, 0))
            overlay.putalpha(Image.fromarray(label[0].byte().numpy() * 150))
            canvas.paste(Image.alpha_composite(panel, overlay).convert("RGB"), (j * 224, i * 250 + 22))
            draw.text((j * 224 + 4, i * 250 + 3), ("GT", "Before", "After")[j], fill="black")
    canvas.save(args.output / "overfit.png")
    report = {"archive_sha256": archive_hash, "checkpoint_sha256": sha256(args.checkpoint),
              "upstream_commit": COMMIT, "upstream_decoder_max_abs_diff": decoder_diff,
              "vendored_upstream_encoder_max_abs_diff": encoder_diff,
              "encoder_norm_eps_timm": source_eps, "encoder_norm_eps_original": 1e-5,
              "encoder_matched_norm_layers": adjusted,
              "encoder_max_abs_diff_after_matching_norm_eps": adjusted_diff,
              "encoder_probability_max_abs_diff": encoder_probability_diff,
              "encoder_changed_pixels_at_0_7": changed_pixels,
              "binary_probability_max_abs_diff": adapter_diff, "device": args.device,
              "selection": "first two train masks with 16-255 pixels", "steps": args.steps,
              "frozen_backbone": not args.full_model, "decoder_eval_mode": True,
              "learning_rate": 2e-6 if args.full_model else 2e-5,
              "threshold": 0.7, "samples": [{"index": x[0], "name": train.samples[x[0]][0],
                                               "gt_pixels": int(x[2].sum())} for x in selected],
              "history": history}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
