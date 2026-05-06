from __future__ import annotations

import sys
from pathlib import Path


def add_repo_path(root: str | Path, relative_path: str) -> Path:
    path = (Path(root) / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required repository path not found: {path}")
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path

