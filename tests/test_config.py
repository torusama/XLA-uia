from pathlib import Path

import pytest

from xla.core.config import check_experiment, load_config
from xla.core.paths import project_root


def test_example_configs():
    root = project_root()
    for task in ("classification", "segmentation"):
        notices = check_experiment(root / f"configs/experiments/fracatlas_{task}.yaml", root)
        assert any("TODO_VERIFY" in notice for notice in notices)


def test_invalid_yaml(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("key: [", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read config"):
        load_config(path, "experiment")


def test_missing_reference(tmp_path: Path):
    path = tmp_path / "experiment.yaml"
    path.write_text("""experiment_name: x
dataset_config: missing.yaml
model_config: model.yaml
seed: 42
device: auto
output_dir: outputs
batch_size: 1
num_workers: 0
learning_rate: 0.1
epochs: 1
metrics: []
""", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot read config"):
        check_experiment(path, tmp_path)
