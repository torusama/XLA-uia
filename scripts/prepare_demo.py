"""Download the pinned public BoneScope assets and verify each file."""
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import Request, urlopen

KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download"
DATASET = "vtan2405/bonescope-btxrd-splits"
CHECKPOINTS = "vtan2405/bonescope-btxrd-checkpoints"
DATASET_VERSION = 2
CHECKPOINTS_VERSION = 1


def ensure_file(dataset, version, remote_name, destination, expected_sha256, digest):
    if destination.is_file() and digest(destination) == expected_sha256:
        return

    query = urlencode({"datasetVersionNumber": version})
    url = f"{KAGGLE_DOWNLOAD}/{dataset}/{remote_name}?{query}"
    request = Request(url, headers={"User-Agent": "BoneScope/1.0"})
    print(f"Downloading {destination.name} from Kaggle...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        temp_file = Path(staging) / destination.name
        with urlopen(request, timeout=60) as response, temp_file.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if digest(temp_file) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch for {destination.name}")
        temp_file.replace(destination)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo.backend.dataset import ARCHIVE, EXPECTED_ARCHIVE_SHA256, MANIFEST
    from demo.backend.models import (
        DEFAULT_EFFICIENTVIT_CHECKPOINT,
        DEFAULT_VIT_CHECKPOINT,
        checkpoint_digest,
    )

    assets = [
        # Kaggle expands uploaded .zip files. The .bin retains the exact original archive.
        (DATASET, DATASET_VERSION, "btxrd_final.bin", ARCHIVE, EXPECTED_ARCHIVE_SHA256),
        (DATASET, DATASET_VERSION, MANIFEST.name, MANIFEST,
         "ec5fdbbd8460eb0d63043eba38eba39f3e2c859768f02bcd7890b559d00a0fd2"),
    ]
    if not os.environ.get("XLA_VIT_CHECKPOINT"):
        assets.append((CHECKPOINTS, CHECKPOINTS_VERSION, DEFAULT_VIT_CHECKPOINT.name, DEFAULT_VIT_CHECKPOINT,
                       "234c2f0a9be7f5b70462fcd91830f58ff696cedcc5166d5ec29ff28693309271"))
    if not os.environ.get("XLA_EFFICIENTVIT_CHECKPOINT"):
        assets.append((CHECKPOINTS, CHECKPOINTS_VERSION, DEFAULT_EFFICIENTVIT_CHECKPOINT.name,
                       DEFAULT_EFFICIENTVIT_CHECKPOINT,
                       "3fd14076d7c0935992556ea7a33f772ce02398c25efa431c3c11acc9ac78e5ae"))
    for dataset, version, filename, path, checksum in assets:
        try:
            ensure_file(dataset, version, filename, path, checksum, checkpoint_digest)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Could not prepare {path.name}: {error}") from error
    print("BTXRD demo assets verified.", flush=True)


if __name__ == "__main__":
    main()
