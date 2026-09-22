# V-JEPA2-AC Benchmark Adapter

This repository is a fork of `facebookresearch/vjepa2` plus a benchmark / profiling extension for V-JEPA2-AC deployment research.

## Scope

The current stage deliberately does **not** change the V-JEPA2-AC algorithm. It preserves the official encoder, AC predictor, DROID state/action convention, pose update logic, and final-frame normalized latent L1 energy, while adding infrastructure for:

- reproducible recorded candidate cases;
- offline candidate-batch sweeps;
- later closed-loop CEM sample sweeps;
- FP16 / W8A16 / W8A8 comparison hooks;
- energy / rank metrics;
- robosuite Panda integration;
- later FPGA comparison on VEK280.

## Important: use a case-sensitive filesystem

The upstream repository contains directory names that differ only by case, for example:

```text
configs/eval_2_1/vitG-384/
configs/eval_2_1/vitg-384/

configs/train_2_1/vitG16/
configs/train_2_1/vitg16/
```

Do **not** maintain this repository in a normal case-insensitive Windows checkout. Use Linux / a server / WSL's Linux filesystem, for example:

```bash
cd ~
git clone <your-fork-url>
cd vjepa2-ac-codesign
```

Using VS Code Remote-WSL is fine. A `.gitattributes` file is included to keep text files on LF line endings, but it cannot fix Windows case-insensitive directory collisions.

## Install

Install official dependencies, then adapter dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-adapter.txt
```

## Checkpoint loading

The upstream snapshot in this fork leaves `src.hub.backbones.VJEPA_BASE_URL` pointed to a localhost testing endpoint. The adapter does **not** modify upstream model code.

Instead, `vjepa2_bench/model_factory.py` invokes the official model factory while using either:

1. a local official checkpoint (recommended for reproducible experiments), or
2. the public base URL already documented in upstream source.

Recommended smoke test with a local checkpoint:

```bash
python scripts/check_vjepa2_ac_adapter.py \
  --repo-root . \
  --checkpoint /path/to/vjepa2-ac-vitg.pt
```

Or allow the adapter to use the public model base URL:

```bash
python scripts/check_vjepa2_ac_adapter.py --repo-root .
```

You may override it with `VJEPA2_BASE_URL` or `--base-url`.

## Official semantics preserved

The adapter follows the official energy-landscape / DROID code:

```text
RGB frames
  -> app.vjepa_droid.transforms.make_transforms
  -> official V-JEPA2 encoder
  -> LayerNorm latent
  -> official AC predictor rollout
  -> notebooks.utils.mpc_utils.compute_new_pose
  -> final-frame latent
  -> mean absolute latent distance to goal
```

The state convention is:

```text
[x, y, z, euler_x, euler_y, euler_z, gripper_closedness]
```

where Euler angles are XYZ radians and gripper closedness must be normalized to `[0, 1]`.

## Robosuite bridge: what is and is not validated

`vjepa2_bench/robosuite_adapter.py` converts Panda EEF quaternion observations to **XYZ Euler angles**, matching the official DROID/V-JEPA2-AC convention.

It intentionally refuses to use raw Panda finger qpos as the V-JEPA2-AC gripper state. You must calibrate:

```text
fully open -> closedness 0
fully closed -> closedness 1
```

before `state7_from_obs()` is used.

The official baseline CEM currently samples translation + gripper and inserts zero rotational deltas. Therefore the first simulation benchmark should also use:

```text
[dx, dy, dz, 0, 0, 0, dgripper]
```

Do not directly copy robosuite OSC rotational commands into V-JEPA2-AC action[3:6]; their rotation semantics are not numerically identical.

Also validate translation scaling and gripper command execution before claiming closed-loop benchmark results.

## Recorded case

A recorded case contains:

```text
current_frames      [T,H,W,3]
goal_frames         [T,H,W,3]
state7              [7]
candidate_actions   [S,Horizon,7]
```

Generate one:

```bash
python scripts/generate_candidate_case.py \
  --current current.npy \
  --goal goal.npy \
  --state7 state7.npy \
  --samples 64 \
  --horizon 2 \
  --seed 0 \
  --out benchmark_data/case_S64.npz
```

Run it:

```bash
python scripts/run_recorded_case.py \
  --repo-root . \
  --checkpoint /path/to/vjepa2-ac-vitg.pt \
  --case benchmark_data/case_S64.npz \
  --out runs/case_S64.json
```

The output NPZ deliberately stores the candidate pool, current/goal frames and state so FP16 / W8A16 / W8A8 comparisons can verify that the inputs are identical.

## Offline sample sweep

```bash
python scripts/run_sample_sweep.py \
  --repo-root . \
  --checkpoint /path/to/vjepa2-ac-vitg.pt \
  --case-template 'benchmark_data/case_S{samples}.npz' \
  --samples 32 64 128 256 400 800
```

**Important:** this script is an *offline fixed-candidate batch sweep*. It measures predictor cost / memory for recorded candidate sets. It is **not yet** the final closed-loop CEM sample-sweep benchmark.

The later formal benchmark must run full CEM iterations and closed-loop task success for each sample count.

## Quantization comparison

The required precision variants remain:

```text
FP16
W8A16
W8A8
```

`configs/precision_factories.example.yaml` only defines the interface. W8A16 / W8A8 must be connected to the actual quantized implementation; fake dynamic quantization must not be reported as the hardware quantization result.

For ranking comparisons, both runs must use the exact same current frame, goal frame, state and candidate action tensor. `scripts/compare_energy_runs.py` verifies these inputs before computing metrics.

## Required metrics

- latent L1 / cosine (diagnostics);
- energy error (diagnostics);
- Spearman rank correlation;
- Kendall tau;
- Top-K overlap;
- selected-action / argmin consistency;
- pairwise ranking flip rate;
- latency;
- peak CUDA memory;
- later: task success, full CEM history and predictor-call count;
- later FPGA: latency, throughput, power/energy, DDR traffic and resource utilization.

See `INTEGRATION_CHECKLIST.md` before formal experiments.
