from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", default=".")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
parser.add_argument("--checkpoint", default="", help="Optional local official V-JEPA2-AC .pt checkpoint")
parser.add_argument("--base-url", default="", help="Optional checkpoint base URL override")
args = parser.parse_args()

repo = Path(args.repo_root).resolve()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from vjepa2_bench.model_factory import build_official_local  # noqa: E402

encoder, predictor = build_official_local(
    repo_root=repo,
    device=args.device,
    dtype=args.dtype,
    checkpoint_path=args.checkpoint or None,
    base_url=args.base_url or None,
)

print("OK: local V-JEPA2-AC architecture + checkpoint loaded")
print("repo:", repo)
print("encoder:", type(encoder).__name__)
print("predictor:", type(predictor).__name__)
print("predictor depth:", len(predictor.predictor_blocks))
print("action encoder in_features:", predictor.action_encoder.in_features)
print("state encoder in_features:", predictor.state_encoder.in_features)
print("predictor num_frames:", predictor.num_frames)
print("predictor grid:", predictor.grid_height, predictor.grid_width)
