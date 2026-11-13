"""Resolve configuration references relative to the repository."""
from pathlib import Path


def project_root() -> Path:
    """Locate the source checkout independently of the current directory."""
    return Path(__file__).resolve().parents[3]


def resolve_reference(value: str, root: Path) -> Path:
    """Resolve a project-relative config reference, or preserve an absolute path."""
    path = Path(value)
    return path if path.is_absolute() else root / path
