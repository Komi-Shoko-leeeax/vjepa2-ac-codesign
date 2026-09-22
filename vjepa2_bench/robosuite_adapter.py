from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RoboSuiteFrame:
    rgb: np.ndarray
    state7: np.ndarray
    raw_obs: Dict[str, Any]


def make_panda_env(
    task: str = "Lift",
    camera_name: str = "agentview",
    camera_height: int = 256,
    camera_width: int = 256,
    control_freq: int = 20,
    horizon: int = 500,
    seed: Optional[int] = None,
):
    """
    Create a robosuite Panda environment.

    This is only the environment bridge.  Before benchmark collection, inspect
    the installed robosuite version's action split and calibrate the gripper.
    The V-JEPA2-AC baseline initially keeps rotational action deltas at zero.
    """
    import robosuite as suite
    from robosuite import load_composite_controller_config

    controller_config = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        task,
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=camera_name,
        camera_heights=camera_height,
        camera_widths=camera_width,
        control_freq=control_freq,
        horizon=horizon,
    )
    if seed is not None and hasattr(env, "seed"):
        env.seed(seed)
    return env


class RobosuitePandaAdapter:
    """
    Convert robosuite observations to the 7-D state convention expected by the
    official V-JEPA2-AC pose update helper::

        [x, y, z, euler_x, euler_y, euler_z, gripper_closedness]

    Rotation is XYZ Euler (radians), matching ``app/vjepa_droid/droid.py`` and
    ``notebooks/utils/mpc_utils.py``.  It is deliberately *not* a rotation
    vector.

    Gripper closedness must be calibrated to [0, 1].  The adapter refuses to
    silently use raw Panda finger qpos as V-JEPA2-AC closedness.
    """

    def __init__(
        self,
        env,
        camera_name: str = "agentview",
        image_flip_vertical: bool = True,
        gripper_open_qpos: Sequence[float] | float | None = None,
        gripper_closed_qpos: Sequence[float] | float | None = None,
        quat_order: str = "xyzw",
    ):
        self.env = env
        self.camera_name = camera_name
        self.image_flip_vertical = image_flip_vertical
        if quat_order not in {"xyzw", "wxyz"}:
            raise ValueError("quat_order must be 'xyzw' or 'wxyz'")
        self.quat_order = quat_order
        self._gripper_open_metric = self._gripper_metric(gripper_open_qpos)
        self._gripper_closed_metric = self._gripper_metric(gripper_closed_qpos)

    @staticmethod
    def _gripper_metric(value) -> float | None:
        """Return an aperture-like scalar robust to mirrored finger joint signs."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise ValueError("Gripper calibration qpos cannot be empty")
        return float(np.mean(np.abs(arr)))

    def _find_key(self, obs, candidates):
        for k in candidates:
            if k in obs:
                return obs[k]
        raise KeyError(
            "None of the expected observation keys were found: "
            + ", ".join(candidates)
        )

    def rgb_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        key = f"{self.camera_name}_image"
        if key not in obs:
            raise KeyError(f"Expected camera key '{key}', got {list(obs.keys())}")
        rgb = np.asarray(obs[key])
        if self.image_flip_vertical:
            rgb = np.flipud(rgb)
        return np.ascontiguousarray(rgb)

    def _gripper_qpos_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.asarray(
            self._find_key(
                obs,
                ["robot0_gripper_qpos", "robot0_right_gripper_qpos"],
            ),
            dtype=np.float32,
        ).reshape(-1)

    def calibrate_gripper(self, open_obs: Dict[str, Any], closed_obs: Dict[str, Any]) -> None:
        """Calibrate raw Panda qpos to V-JEPA2-AC closedness: open=0, closed=1."""
        self._gripper_open_metric = float(np.mean(np.abs(self._gripper_qpos_from_obs(open_obs))))
        self._gripper_closed_metric = float(np.mean(np.abs(self._gripper_qpos_from_obs(closed_obs))))
        if np.isclose(self._gripper_open_metric, self._gripper_closed_metric):
            raise ValueError("Open/closed gripper qpos calibration points are indistinguishable")

    @property
    def gripper_is_calibrated(self) -> bool:
        return self._gripper_open_metric is not None and self._gripper_closed_metric is not None

    def gripper_closedness_from_obs(self, obs: Dict[str, Any]) -> float:
        if not self.gripper_is_calibrated:
            raise RuntimeError(
                "Gripper closedness is not calibrated. Provide gripper_open_qpos and "
                "gripper_closed_qpos to RobosuitePandaAdapter, or call "
                "calibrate_gripper(open_obs, closed_obs). Raw finger qpos must not be "
                "used directly as the V-JEPA2-AC [0,1] closedness state."
            )
        current = float(np.mean(np.abs(self._gripper_qpos_from_obs(obs))))
        denom = self._gripper_closed_metric - self._gripper_open_metric
        closedness = (current - self._gripper_open_metric) / denom
        return float(np.clip(closedness, 0.0, 1.0))

    def state7_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        pos = np.asarray(
            self._find_key(obs, ["robot0_eef_pos", "robot0_right_eef_pos"]),
            dtype=np.float32,
        ).reshape(3)

        quat = np.asarray(
            self._find_key(obs, ["robot0_eef_quat", "robot0_right_eef_quat"]),
            dtype=np.float64,
        ).reshape(4)

        # scipy expects xyzw. Make quaternion ordering an explicit adapter setting
        # so the installed robosuite version can be verified rather than assumed.
        if self.quat_order == "wxyz":
            quat = quat[[1, 2, 3, 0]]
        euler_xyz = Rotation.from_quat(quat).as_euler("xyz", degrees=False).astype(np.float32)
        grip = np.array([self.gripper_closedness_from_obs(obs)], dtype=np.float32)
        return np.concatenate([pos, euler_xyz, grip], axis=0).astype(np.float32)

    def unpack(self, obs: Dict[str, Any]) -> RoboSuiteFrame:
        return RoboSuiteFrame(
            rgb=self.rgb_from_obs(obs),
            state7=self.state7_from_obs(obs),
            raw_obs=obs,
        )

    @staticmethod
    def assert_vjepa_action7(action: np.ndarray, require_zero_rotation: bool = True):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"Expected V-JEPA2-AC action shape (7,), got {action.shape}")
        if require_zero_rotation and not np.allclose(action[3:6], 0.0, atol=1e-7):
            raise ValueError(
                "Baseline bridge requires V-JEPA2-AC rotational deltas action[3:6] == 0. "
                "robosuite OSC_POSE rotation commands are not numerically identical to "
                "the DROID/V-JEPA2-AC Euler-delta convention. Add an explicit conversion "
                "before enabling rotational actions."
            )

    @staticmethod
    def validate_robosuite_action_shape(action: np.ndarray):
        action = np.asarray(action)
        if action.shape != (7,):
            raise ValueError(
                f"Expected a 7-D Panda controller action, got {action.shape}. "
                "Inspect the installed robosuite controller action split before benchmarking."
            )
