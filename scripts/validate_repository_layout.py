from __future__ import annotations

import sys
from pathlib import Path


repo = Path(__file__).resolve().parents[1]
required = [
    repo / "configs/eval_2_1/vitG-384",
    repo / "configs/eval_2_1/vitg-384",
    repo / "configs/train_2_1/vitG16",
    repo / "configs/train_2_1/vitg16",
    repo / "vjepa2_bench/model_adapter.py",
    repo / "vjepa2_bench/robosuite_adapter.py",
    repo / "notebooks/utils/mpc_utils.py",
]

missing = [str(p.relative_to(repo)) for p in required if not p.exists()]
if missing:
    print("ERROR: repository layout is incomplete or case-collided:")
    for item in missing:
        print(" -", item)
    sys.exit(2)

print("Repository layout check: OK")
print("Repository root:", repo)
print("Case-sensitive V-JEPA2/V-JEPA2.1 config directories are present.")
print("Next: run scripts/check_vjepa2_ac_adapter.py with an official checkpoint.")
