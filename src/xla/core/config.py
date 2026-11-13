"""Load and validate Phase 0 YAML configuration files."""
from pathlib import Path

import yaml

from xla.core.paths import resolve_reference

REQUIRED = {
    "experiment": ("experiment_name", "dataset_config", "model_config", "seed", "device",
                   "output_dir", "batch_size", "num_workers", "learning_rate", "epochs", "metrics"),
    "dataset": ("name", "root", "task", "class_names", "metadata_file", "annotations_dir",
                "split", "input_size", "normalization"),
    "model": ("name", "task", "source", "pretrained", "num_classes", "input_channels",
              "input_size", "notes"),
}


def load_config(path: Path, kind: str) -> dict:
    """Read a mapping and check its required top-level fields."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")  # noqa: TRY004
    missing = [key for key in REQUIRED[kind] if key not in data]
    if missing:
        raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
    if kind == "dataset" and (not isinstance(data['split'], dict) or
                              not {'strategy', 'seed'} <= data['split'].keys()):
        raise ValueError(f"{path}: split requires strategy and seed")
    return data


def check_experiment(path: Path, root: Path) -> list[str]:
    """Validate the experiment and referenced configs without requiring datasets."""
    experiment = load_config(path, "experiment")
    notices = []
    for key, kind in (("dataset_config", "dataset"), ("model_config", "model")):
        value = experiment[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: {key} must be a nonempty path")
        target = resolve_reference(value, root)
        config = load_config(target, kind)
        if config.get("task") not in ("classification", "segmentation"):
            raise ValueError(f"{target}: task must be classification or segmentation")
        def find_todos(value, prefix="", source=target):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from find_todos(child, f"{prefix}.{key}" if prefix else key, source)
            elif value == "TODO_VERIFY":
                yield f"{source}: {prefix} = TODO_VERIFY"

        notices.extend(find_todos(config))
        if kind == "dataset":
            dataset_task = config["task"]
        elif config["task"] != dataset_task:
            raise ValueError("Dataset and model tasks do not match")
    return notices
