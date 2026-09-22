# V-JEPA2-AC benchmark integration checklist

## 0. Repository integrity

- [ ] Work in Linux / WSL Linux filesystem / server filesystem, not a normal Windows case-insensitive checkout.
- [ ] Both `configs/eval_2_1/vitG-384` and `configs/eval_2_1/vitg-384` exist.
- [ ] Both `configs/train_2_1/vitG16` and `configs/train_2_1/vitg16` exist.
- [ ] `git diff --check` reports no whitespace errors.
- [ ] Upstream tracked files are unchanged except for explicitly intended project changes.

## 1. Official model baseline

- [ ] `python scripts/check_vjepa2_ac_adapter.py --repo-root . --checkpoint <official.pt>` succeeds.
- [ ] Official `notebooks/energy_landscape_example.ipynb` runs unchanged.
- [ ] A fixed recorded case matches the official notebook's encoder latent, final predictor latent, L1 energy and best candidate within the selected dtype tolerance.
- [ ] Save the exact checkpoint path / hash and Git commit used for each experiment.

## 2. V-JEPA2-AC state / action semantics

- [ ] State is `[x,y,z,euler_xyz,gripper_closedness]`.
- [ ] Euler representation is XYZ radians, not rotation vector.
- [ ] Gripper state is calibrated to `[0,1]` closedness.
- [ ] Baseline candidate action rotation `[3:6]` is zero, matching the official CEM baseline.
- [ ] If nonzero rotation is introduced later, an explicit robosuite-OSC <-> DROID/V-JEPA2 conversion has been validated.

## 3. Robosuite bridge

- [ ] Use Panda and inspect the installed controller action split.
- [ ] Verify camera orientation and whether vertical flipping is required.
- [ ] Verify EEF quaternion ordering before converting to Euler.
- [ ] Record fully-open and fully-closed Panda gripper qpos and calibrate closedness.
- [ ] Validate translation action scaling between V-JEPA2-AC physical deltas and robosuite controller commands.
- [ ] Validate gripper execution semantics/sign.
- [ ] Do not report closed-loop benchmark numbers until these mappings are verified.

## 4. Reproducible recorded cases

For every formal comparison save:

- [ ] current RGB frames;
- [ ] goal RGB frames;
- [ ] `state7`;
- [ ] exact `candidate_actions`;
- [ ] random seed;
- [ ] checkpoint version/hash;
- [ ] Git commit;
- [ ] benchmark config.

## 5. Mandatory sample tests

Required sample counts:

- [ ] 32
- [ ] 64
- [ ] 128
- [ ] 256
- [ ] 400
- [ ] 800

Two distinct experiments must not be confused:

1. [ ] **Offline fixed-candidate batch sweep** for predictor latency / VRAM scaling.
2. [ ] **Formal closed-loop CEM sample sweep** for CEM convergence, task success and total planning cost.

## 6. Mandatory precision tests

- [ ] FP16
- [ ] W8A16
- [ ] W8A8

For FP16 vs quantized ranking comparisons:

- [ ] same checkpoint semantics;
- [ ] same current observation;
- [ ] same goal;
- [ ] same state;
- [ ] exact same candidate action tensor;
- [ ] same CEM settings and seeds.

## 7. Metrics

### Diagnostic
- [ ] latent L1
- [ ] latent cosine
- [ ] energy MAE / relative MAE

### Planning / decision
- [ ] Spearman
- [ ] Kendall tau
- [ ] Top-K / elite overlap
- [ ] selected-action consistency
- [ ] pairwise ranking flip rate

### Closed-loop / efficiency
- [ ] task success
- [ ] total planning latency
- [ ] predictor latency
- [ ] predictor calls
- [ ] peak VRAM
- [ ] CEM iteration history

## 8. FPGA comparison later

- [ ] same model/checkpoint version
- [ ] same sample/CEM settings
- [ ] same start-goal pairs and seeds
- [ ] latency / throughput
- [ ] power / energy
- [ ] DDR traffic
- [ ] LUT / FF / BRAM / URAM / DSP utilization
- [ ] timing closure
