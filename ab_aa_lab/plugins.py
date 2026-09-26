"""Load a strategy (or any other plugin) from a Python file at runtime."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_plugin(path: str | Path) -> None:
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    module_name = f"ab_aa_lab_plugin_{file_path.stem}_{file_path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import plugin {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
