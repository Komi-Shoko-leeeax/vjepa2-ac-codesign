# Benchmark adapter repair notes

This working tree was repaired from the uploaded fork before packaging.

## Fixed

1. Restored every upstream tracked file exactly to Git `HEAD` (`204698b`) before reapplying project additions.
2. Recreated case-sensitive upstream directories lost by the Windows checkout:
   - `configs/eval_2_1/vitG-384`
   - `configs/eval_2_1/vitg-384`
   - `configs/train_2_1/vitG16`
   - `configs/train_2_1/vitg16`
3. Added `.gitattributes` to keep text files on LF line endings.
4. Kept upstream model code untouched; checkpoint URL handling is isolated in `vjepa2_bench/model_factory.py`.
5. Corrected robosuite state rotation from rotation-vector to XYZ Euler radians.
6. Removed raw finger-qpos-as-gripper-state behavior; gripper closedness now requires open/closed calibration to `[0,1]`.
7. Candidate rotational deltas default to zero, matching the official V-JEPA2-AC CEM baseline.
8. Candidate gripper deltas are clipped to the official CEM baseline range `[-0.75, 0.75]`.
9. Quantization/ranking comparison now verifies identical current frames, goal frames, state and candidate actions.
10. `run_sample_sweep.py` is explicitly labelled as an offline fixed-candidate sweep, not a final closed-loop CEM benchmark.
11. Added optional local checkpoint loading to improve reproducibility on the future server.

## Intentionally not claimed as complete yet

- robosuite translation-command scaling;
- robosuite gripper execution command/sign;
- non-zero rotational action conversion between DROID/V-JEPA2-AC and OSC_POSE;
- closed-loop CEM benchmark runner;
- actual W8A16 / W8A8 model builders;
- stable-worldmodel closed-loop integration.

These should be completed only after the server environment and exact robosuite/controller version are available.
