"""Loads config/config.yaml and resolves project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and turn every entry of ``paths`` into an absolute ``Path``.

    ``root`` overrides the directory the paths are resolved against (used by tests
    to write artefacts into a temporary folder).
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(root) if root else PROJECT_ROOT
    cfg["paths"] = {name: (base / rel).resolve() for name, rel in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg
