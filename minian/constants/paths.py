"""Dataset folder names and conventional path helpers."""

from __future__ import annotations

import os.path

MINIAN = "minian"
MINIAN_CONFIG_FILENAME = f"{MINIAN}_config.json"
MINIAN_CONFIG_EFFECTIVE_FILENAME = f"{MINIAN}_config_effective.json"
MINIAN_INTERMEDIATE = f"{MINIAN}_intermediate"


def minian_folder_under(parent: str) -> str:
    """Return ``os.path.join(parent, MINIAN)`` — default dataset folder under a session root."""
    return os.path.join(parent, MINIAN)


def get_minian_intermediate_path(parent: str | None = None) -> str:
    """Return the conventional scratch folder path (``MINIAN_INTERMEDIATE``).

    With ``parent=None``, returns an absolute ``<cwd>/{MINIAN_INTERMEDIATE}`` (same idea as
    :attr:`minian.config.PipelineConfig.intpath` defaults).

    With ``parent`` set, returns ``os.path.join(os.path.abspath(parent), MINIAN_INTERMEDIATE)``
    (e.g. pass the session / video data directory so scratch lives next to that folder).
    """
    if parent is None:
        return os.path.abspath(MINIAN_INTERMEDIATE)
    return os.path.join(os.path.abspath(parent), MINIAN_INTERMEDIATE)
