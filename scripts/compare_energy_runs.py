from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vjepa2_bench.metrics import compare_rankings  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, help="NPZ from FP reference run")
parser.add_argument("--test", required=True, help="NPZ from quantized/test run")
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--out", default="")
args = parser.parse_args()

ref_npz = np.load(args.reference)
test_npz = np.load(args.test)
for key in ("energies", "candidate_actions"):
    if key not in ref_npz or key not in test_npz:
        raise KeyError(f"Both NPZ files must contain '{key}'")

for key in ("candidate_actions", "current_frames", "goal_frames", "state7"):
    ref_value = np.asarray(ref_npz[key])
    test_value = np.asarray(test_npz[key])
    if ref_value.shape != test_value.shape or not np.array_equal(ref_value, test_value):
        raise ValueError(
            f"Reference and test runs differ in '{key}'. Rank metrics are only valid "
            "when current observation, goal, state and candidate pool are identical."
        )

ref_actions = np.asarray(ref_npz["candidate_actions"])
ref = np.asarray(ref_npz["energies"])
test = np.asarray(test_npz["energies"])
metrics = compare_rankings(ref, test, topk=args.topk)
metrics["candidate_pool_verified_identical"] = True
metrics["num_candidates"] = int(ref.size)

text = json.dumps(metrics, indent=2)
print(text)
if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
