from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Tuple

import torch
import torch.nn as nn


ModelPair = Tuple[nn.Module, nn.Module]
DEFAULT_VJEPA2_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"


def _ensure_repo_importable(repo_root: str | Path) -> Path:
    repo = Path(repo_root).resolve()
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _load_ac_checkpoint(encoder: nn.Module, predictor: nn.Module, checkpoint_path: str | Path) -> None:
    """Load the official V-JEPA2-AC encoder/predictor state dictionaries."""
    from src.hub.backbones import _clean_backbone_key

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if "encoder" not in state_dict or "predictor" not in state_dict:
        raise KeyError(
            "Expected an official V-JEPA2-AC checkpoint containing 'encoder' and "
            "'predictor' entries."
        )

    encoder_state = _clean_backbone_key(dict(state_dict["encoder"]))
    predictor_state = _clean_backbone_key(dict(state_dict["predictor"]))
    encoder.load_state_dict(encoder_state, strict=False)
    predictor.load_state_dict(predictor_state, strict=True)


def _cast_model_pair(encoder: nn.Module, predictor: nn.Module, device: str, dtype: str) -> ModelPair:
    dev = torch.device(device)
    encoder = encoder.eval().to(dev)
    predictor = predictor.eval().to(dev)

    if dtype == "fp16":
        encoder = encoder.half()
        predictor = predictor.half()
    elif dtype == "bf16":
        encoder = encoder.to(dtype=torch.bfloat16)
        predictor = predictor.to(dtype=torch.bfloat16)
    elif dtype == "fp32":
        encoder = encoder.float()
        predictor = predictor.float()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return encoder, predictor


def build_official_local(
    repo_root: str | Path,
    device: str = "cuda",
    dtype: str = "fp16",
    checkpoint_path: str | Path | None = None,
    base_url: str | None = None,
) -> ModelPair:
    """
    Build V-JEPA2-AC from the local fork using the official model factory.

    Why this does not simply use ``torch.hub.load(..., pretrained=True)``:
    the upstream snapshot used by this project currently leaves its module-level
    VJEPA_BASE_URL pointed at a localhost testing server.  We do not modify the
    upstream source.  Instead this adapter either:

    1. instantiates the official architecture and loads a user-supplied local
       checkpoint, or
    2. temporarily supplies the public V-JEPA2 base URL while invoking the
       official factory.

    The model architecture and checkpoint loading semantics remain the official
    V-JEPA2-AC ones.
    """
    repo = _ensure_repo_importable(repo_root)
    from src.hub import backbones

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        encoder, predictor = backbones.vjepa2_ac_vit_giant(pretrained=False)
        _load_ac_checkpoint(encoder, predictor, ckpt)
    else:
        url = base_url or os.environ.get("VJEPA2_BASE_URL") or DEFAULT_VJEPA2_BASE_URL
        old_url = backbones.VJEPA_BASE_URL
        try:
            backbones.VJEPA_BASE_URL = url.rstrip("/")
            encoder, predictor = backbones.vjepa2_ac_vit_giant(pretrained=True)
        finally:
            backbones.VJEPA_BASE_URL = old_url

    return _cast_model_pair(encoder, predictor, device=device, dtype=dtype)


def resolve_factory(spec: str) -> Callable[..., ModelPair]:
    """
    Resolve ``module.submodule:function`` into a callable.

    A real W8A16 / W8A8 implementation should expose a function like::

        def build(repo_root, device, checkpoint_path=None):
            return encoder, predictor

    This intentionally avoids substituting fake/dynamic quantization for the
    quantization implementation that will eventually be evaluated on hardware.
    """
    if ":" not in spec:
        raise ValueError("Factory spec must be 'python.module:function'")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn
