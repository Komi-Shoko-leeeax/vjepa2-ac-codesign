from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(
    description=(
        "Create a reproducible offline candidate-action case. Baseline V-JEPA2-AC "
        "planning keeps rotational action deltas at zero."
    )
)
parser.add_argument("--current", required=True, help=".npy [T,H,W,3]")
parser.add_argument("--goal", required=True, help=".npy [T,H,W,3]")
parser.add_argument("--state7", required=True, help=".npy [7]")
parser.add_argument("--out", required=True)
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--horizon", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--translation-std", type=float, default=0.03)
parser.add_argument(
    "--rotation-std",
    type=float,
    default=0.0,
    help=(
        "Default is 0 to match the official V-JEPA2-AC CEM baseline. Nonzero values "
        "are for offline analysis only unless an explicit simulator action conversion "
        "has been validated."
    ),
)
parser.add_argument("--gripper-std", type=float, default=0.5)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
a = np.zeros((args.samples, args.horizon, 7), dtype=np.float32)
a[..., :3] = rng.normal(0, args.translation_std, size=a[..., :3].shape)
if args.rotation_std != 0.0:
    a[..., 3:6] = rng.normal(0, args.rotation_std, size=a[..., 3:6].shape)
a[..., 6] = np.clip(
    rng.normal(0, args.gripper_std, size=a[..., 6].shape), -0.75, 0.75
)

current = np.load(args.current)
goal = np.load(args.goal)
state7 = np.asarray(np.load(args.state7), dtype=np.float32)
if state7.shape != (7,):
    raise ValueError(f"state7 must have shape (7,), got {state7.shape}")

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out,
    current_frames=current,
    goal_frames=goal,
    state7=state7,
    candidate_actions=a,
    seed=np.asarray(args.seed, dtype=np.int64),
)
print(out.resolve())
