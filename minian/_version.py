"""Resolved package version: importlib.metadata first, then pyproject.toml."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_package_version() -> str:
    """Same semantics as installs from GitHub: matches pyproject.toml / wheel metadata."""
    try:
        return version("minian")
    except PackageNotFoundError:
        pass
    try:
        import tomllib

        root = Path(__file__).resolve().parent.parent
        pyproject = root / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return data["project"]["version"]
    except (OSError, KeyError, ValueError, TypeError):
        return "0.0.0"
