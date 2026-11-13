"""Phase 0 commands for environment and configuration checks."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

from xla.core.config import check_experiment
from xla.core.paths import project_root


def doctor() -> int:
    """Report required and optional packages without requiring a GPU."""
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Project root: {project_root()}")
    missing = False
    for package in ("PyYAML", "pytest", "ruff", "torch"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
            missing |= package == "PyYAML"
        print(f"{package}: {version}" + (" (required)" if package == "PyYAML" else " (optional)"))
    try:
        import torch
        print(f"CUDA: {'available' if torch.cuda.is_available() else 'not available'}")
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"CUDA: not available ({type(exc).__name__})")
    return int(missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ViT / EfficientViT research scaffold")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the environment")
    check = commands.add_parser("config-check", help="Validate experiment YAML")
    check.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    try:
        notices = check_experiment(args.config, project_root())
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    print("Config valid; dataset files are not required in Phase 0.")
    for notice in notices:
        print(notice)
    return 0
