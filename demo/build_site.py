"""Build static site assets from frontend and the vetted BTXRD test split."""
import shutil
from pathlib import Path

from demo.backend.dataset import TestDataset
from demo.backend.export_samples import export_samples


def main():
    root = Path(__file__).resolve().parent
    output = root / "dist"
    if output.exists():
        raise FileExistsError(f"Build directory already exists: {output}")
    dataset = TestDataset()
    shutil.copytree(root / "frontend", output)
    export_samples(output / "samples", dataset)


if __name__ == "__main__":
    main()
