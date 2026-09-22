from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", default=".")
parser.add_argument("--case", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
parser.add_argument("--checkpoint", default="", help="Optional local official V-JEPA2-AC .pt checkpoint")
parser.add_argument("--base-url", default="", help="Optional checkpoint base URL override")
parser.add_argument("--return-latents", action="store_true")
args = parser.parse_args()

repo = Path(args.repo_root).resolve()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from vjepa2_bench.model_adapter import VJEPA2ACAdapter  # noqa: E402

case = np.load(args.case)
required = ["current_frames", "goal_frames", "state7", "candidate_actions"]
missing = [k for k in required if k not in case]
if missing:
    raise KeyError(f"Case file missing keys: {missing}")

candidate_actions = np.asarray(case["candidate_actions"], dtype=np.float32)
if candidate_actions.ndim != 3 or candidate_actions.shape[-1] != 7:
    raise ValueError(
        f"candidate_actions must be [S,H,7], got {candidate_actions.shape}"
    )

adapter = VJEPA2ACAdapter(
    repo_root=repo,
    device=args.device,
    dtype=args.dtype,
    checkpoint_path=args.checkpoint or None,
    base_url=args.base_url or None,
)
res = adapter.candidate_costs(
    current_frames=case["current_frames"],
    goal_frames=case["goal_frames"],
    state7=case["state7"],
    candidate_actions=candidate_actions,
    return_latents=args.return_latents,
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "case": str(Path(args.case).resolve()),
    "dtype": args.dtype,
    "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else "official_url",
    "num_candidates": int(len(res.energies)),
    "horizon": int(candidate_actions.shape[1]),
    "elapsed_ms": res.elapsed_ms,
    "peak_cuda_mb": res.peak_cuda_mb,
    "best_idx": int(np.argmin(res.energies)),
    "best_energy": float(np.min(res.energies)),
    "mean_energy": float(np.mean(res.energies)),
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

np.savez_compressed(
    out.with_suffix(".npz"),
    energies=res.energies,
    candidate_actions=candidate_actions,
    current_frames=case["current_frames"],
    goal_frames=case["goal_frames"],
    state7=case["state7"],
    **({"final_latents": res.final_latents} if res.final_latents is not None else {}),
)

print(json.dumps(payload, indent=2))
