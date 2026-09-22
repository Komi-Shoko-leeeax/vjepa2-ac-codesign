from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(
    description=(
        "OFFLINE candidate-batch sweep. This measures fixed recorded candidate sets; "
        "it is not the final closed-loop CEM sample-sweep benchmark."
    )
)
parser.add_argument("--repo-root", default=".")
parser.add_argument(
    "--case-template",
    required=True,
    help="Template containing {samples}, e.g. benchmark_data/case_S{samples}.npz",
)
parser.add_argument("--out-dir", default="runs/offline_candidate_sweep")
parser.add_argument("--samples", nargs="+", type=int, default=[32, 64, 128, 256, 400, 800])
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
parser.add_argument("--checkpoint", default="")
parser.add_argument("--base-url", default="")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for samples in args.samples:
    case = args.case_template.format(samples=samples)
    out = out_dir / f"S{samples}.json"
    cmd = [
        sys.executable,
        "scripts/run_recorded_case.py",
        "--repo-root",
        args.repo_root,
        "--case",
        case,
        "--out",
        str(out),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    row = json.loads(out.read_text(encoding="utf-8"))
    row["samples"] = samples
    row["benchmark_kind"] = "offline_fixed_candidate_batch"
    rows.append(row)

fields = sorted({k for row in rows for k in row})
with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(out_dir / "summary.csv")
