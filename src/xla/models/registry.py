"""Build supported model adapters; future models can use the same interface."""

from xla.models.vit import MODEL_NAME, build_vit


def build_model(config: dict):
    """Build a classification model from the model YAML mapping."""
    if config.get("task") != "classification" or config.get("name") != MODEL_NAME:
        raise ValueError(f"Unsupported model/task: {config.get('name')}/{config.get('task')}")
    if config.get("source") != "timm" or config.get("input_channels") != 3:
        raise ValueError("ViT requires source=timm and input_channels=3")
    if config.get("input_size") != [224, 224]:
        raise ValueError("This ViT configuration requires input_size [224, 224]")
    if not isinstance(config.get("pretrained"), bool):
        raise TypeError("pretrained must be true or false")
    return build_vit(pretrained=config["pretrained"], num_classes=config["num_classes"])
