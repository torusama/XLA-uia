"""Thin adapter for the original ImageNet-21k ViT-B/16 PyTorch port."""

MODEL_NAME = "vit_base_patch16_224.orig_in21k"


def build_vit(*, pretrained: bool = True, num_classes: int = 2):
    """Create timm ViT; pretrained backbone may be downloaded by timm.

    The checkpoint has no classification head, so timm initializes a new head.
    pretrained=False is intended for offline tests, not a pretrained experiment.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError('Install the ViT extra: python -m pip install -e ".[vit]"') from exc
    try:
        return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        mode = "pretrained weights" if pretrained else "model"
        raise RuntimeError(f"Could not load {mode} for {MODEL_NAME}: {exc}") from exc


def preprocessing_config(model) -> dict:
    """Return the model's actual timm preprocessing settings."""
    from timm.data import resolve_model_data_config

    return resolve_model_data_config(model)
