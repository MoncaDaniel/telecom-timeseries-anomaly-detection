# src/utils/paths.py
from pathlib import Path

def ensure_parent_dir(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
