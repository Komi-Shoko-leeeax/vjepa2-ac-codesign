
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


class PlanningTraceWriter:
    """Simple append-only JSONL + NPZ trace writer."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out_dir / "trace.jsonl"

    def write_record(self, record: Dict[str, Any]):
        serializable = {}
        for k, v in record.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = v.item()
            else:
                serializable[k] = v
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def save_arrays(self, name: str, **arrays):
        path = self.out_dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return path
