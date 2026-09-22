
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model_factory import build_official_local


@dataclass
class CostResult:
    energies: np.ndarray
    final_latents: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    peak_cuda_mb: float = 0.0


class VJEPA2ACAdapter:
    """
    Thin adapter around the official V-JEPA2-AC encoder/predictor.

    The implementation follows the official `energy_landscape_example.ipynb`:
      - 256 crop
      - each source frame duplicated temporally before the encoder
      - layer-normalized latent representations
      - autoregressive AC predictor rollout
      - mean absolute error on the final frame tokens as energy
    """

    def __init__(
        self,
        repo_root: str | Path,
        device: str = "cuda",
        dtype: str = "fp16",
        encoder=None,
        predictor=None,
        normalize_reps: bool = True,
        checkpoint_path: str | Path | None = None,
        base_url: str | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.normalize_reps = normalize_reps

        if encoder is None or predictor is None:
            encoder, predictor = build_official_local(
                self.repo_root,
                device=device,
                dtype=dtype,
                checkpoint_path=checkpoint_path,
                base_url=base_url,
            )
        self.encoder = encoder.eval()
        self.predictor = predictor.eval()

        # Imports from the local V-JEPA2 fork.
        import sys
        repo_str = str(self.repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from app.vjepa_droid.transforms import make_transforms

        self.transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=256,
        )

        self.tokens_per_frame = int((256 // self.encoder.patch_size) ** 2)

        # The official energy-landscape notebook's implementation lives under
        # ``notebooks/utils`` in this fork. Older forks may expose the same
        # module as ``utils.mpc_utils`` instead.
        try:
            from notebooks.utils.mpc_utils import compute_new_pose
        except Exception as exc:
            try:
                from utils.mpc_utils import compute_new_pose
            except Exception as legacy_exc:
                compute_new_pose = None
                self._pose_import_error = (
                    "notebooks.utils.mpc_utils: "
                    f"{exc}; utils.mpc_utils: {legacy_exc}"
                )
        self.compute_new_pose = compute_new_pose

    @property
    def model_dtype(self):
        return next(self.predictor.parameters()).dtype

    def _as_clip(self, frames: np.ndarray) -> torch.Tensor:
        """
        frames: [T,H,W,3] uint8/float numpy array.
        Returns transformed tensor [1,C,T,H,W].
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"frames must be [T,H,W,3], got {tuple(frames.shape)}"
            )
        clip = self.transform(frames).unsqueeze(0)
        return clip.to(self.device, dtype=self.model_dtype)

    @torch.no_grad()
    def encode_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        Reproduces forward_target() from the official energy-landscape notebook.
        """
        c = self._as_clip(frames)
        B, C, T, H, W = c.shape
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1)
        c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = self.encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def _ensure_pose_update(self):
        if self.compute_new_pose is None:
            raise RuntimeError(
                "Could not import the official compute_new_pose helper from the local "
                f"V-JEPA2 fork. Original import error: {self._pose_import_error}"
            )

    @torch.no_grad()
    def candidate_costs_from_latents(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        state7: np.ndarray | torch.Tensor,
        candidate_actions: np.ndarray | torch.Tensor,
        return_latents: bool = False,
    ) -> CostResult:
        """
        Evaluate candidate action sequences with the official AC predictor semantics.

        z_current: [1, tokens_per_frame, D] (or longer context ending in current frame)
        z_goal:    [1, tokens_per_frame, D] (goal final frame is used)
        state7:    [7] or [1,7]
        candidate_actions: [S,H,7]

        Energy = mean(abs(final_predicted_frame - goal_frame)).
        """
        self._ensure_pose_update()

        actions = torch.as_tensor(
            candidate_actions, device=self.device, dtype=self.model_dtype
        )
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError(
                f"candidate_actions must be [S,H,7], got {tuple(actions.shape)}"
            )
        S, H, _ = actions.shape

        state = torch.as_tensor(state7, device=self.device, dtype=self.model_dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (1, 7):
            raise ValueError(f"state7 must be [7] or [1,7], got {tuple(state.shape)}")

        z_hist = z_current.to(self.device, dtype=self.model_dtype).repeat(S, 1, 1)
        s_hist = state.unsqueeze(1).repeat(S, 1, 1)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time
            t0 = time.perf_counter()

        for t in range(H):
            a_hist = actions[:, : t + 1]
            pred = self.predictor(z_hist, a_hist, s_hist)
            z_next = pred[:, -self.tokens_per_frame :]
            if self.normalize_reps:
                z_next = F.layer_norm(z_next, (z_next.size(-1),))

            s_next = self.compute_new_pose(
                s_hist[:, -1:], actions[:, t : t + 1]
            )
            z_hist = torch.cat([z_hist, z_next], dim=1)
            s_hist = torch.cat([s_hist, s_next], dim=1)

        goal = z_goal[:, -self.tokens_per_frame :].to(
            self.device, dtype=self.model_dtype
        )
        final_pred = z_hist[:, -self.tokens_per_frame :]
        energies = torch.mean(torch.abs(final_pred - goal), dim=(1, 2))

        if self.device.type == "cuda":
            end.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = float(start.elapsed_time(end))
            peak_mb = float(torch.cuda.max_memory_allocated(self.device) / (1024**2))
        else:
            import time
            elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
            peak_mb = 0.0

        latents_np = (
            final_pred.detach().float().cpu().numpy() if return_latents else None
        )
        return CostResult(
            energies=energies.detach().float().cpu().numpy(),
            final_latents=latents_np,
            elapsed_ms=elapsed_ms,
            peak_cuda_mb=peak_mb,
        )

    @torch.no_grad()
    def candidate_costs(
        self,
        current_frames: np.ndarray,
        goal_frames: np.ndarray,
        state7: np.ndarray,
        candidate_actions: np.ndarray,
        return_latents: bool = False,
    ) -> CostResult:
        z_current_all = self.encode_frames(current_frames)
        z_goal_all = self.encode_frames(goal_frames)
        z_current = z_current_all[:, -self.tokens_per_frame :]
        z_goal = z_goal_all[:, -self.tokens_per_frame :]
        return self.candidate_costs_from_latents(
            z_current=z_current,
            z_goal=z_goal,
            state7=state7,
            candidate_actions=candidate_actions,
            return_latents=return_latents,
        )
