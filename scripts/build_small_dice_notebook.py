"""Package small-lesion Dice A/B for a separate Kaggle run."""
import json

from build_lesion_crop_notebook import ROOT, build_notebook


def build():
    notebook = build_notebook()
    notebook["cells"][0]["source"] = [
        "# BTXRD ViT: small-lesion Dice A/B\n",
        "Same checkpoint, batch-4 paired augmentation, 20 epochs and batch Dice + weighted BCE. ",
        "Candidate adds 0.5 times mean soft Dice loss for augmented masks with 1-255 pixels; ",
        "control uses the original loss. Empty and larger masks do not enter the extra term.\n",
        "Validation uses full images; test is not evaluated. Compare foreground micro Dice, ",
        "small-lesion Dice and normal false positives; do not compare losses across arms.\n",
    ]
    cell = notebook["cells"][4]
    source = "".join(cell["source"])
    source = source.replace("vit_lesion_crop_ab", "vit_small_dice_ab")
    source = source.replace("'--crop-trial', '--crop-probability', '0.5'", "'--small-dice-trial'")
    source = source.replace("Both crop trials finished", "Both small Dice trials finished")
    compile(source, "small-dice-run", "exec")
    cell["source"] = source.splitlines(keepends=True)
    return notebook


def main():
    output = ROOT / "notebooks/kaggle_btxrd_vit_small_dice_ab"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"id": "vtan2405/xla-btxrd-vit-small-dice-ab-20-epochs",
                "title": "XLA BTXRD ViT Small Dice AB 20 Epochs", "code_file": "vit_small_dice_ab.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": ["vtan2405/xla-btxrd-vit-improvement-ab-20-epochs"],
                "model_sources": []}
    (output / "vit_small_dice_ab.ipynb").write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf8")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(f"Prepared {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
