
from __future__ import annotations

import numpy as np

from .model_adapter import VJEPA2ACAdapter


class StableWorldModelCostBridge:
    """
    Optional bridge for a custom stable-worldmodel environment.

    Expected info dictionary keys:
        current_frames: [T,H,W,3]
        goal_frames:    [T,H,W,3]
        state7:         [7]

    `action_candidates` must already be V-JEPA2-AC-compatible [S,H,7].

    This class intentionally does not attempt to reinterpret PushT/TwoRoom 2-D
    actions as 7-D V-JEPA2-AC actions.
    """

    def __init__(self, adapter: VJEPA2ACAdapter):
        self.adapter = adapter

    def get_cost(self, info, action_candidates):
        result = self.adapter.candidate_costs(
            current_frames=np.asarray(info["current_frames"]),
            goal_frames=np.asarray(info["goal_frames"]),
            state7=np.asarray(info["state7"], dtype=np.float32),
            candidate_actions=np.asarray(action_candidates, dtype=np.float32),
        )
        return result.energies
