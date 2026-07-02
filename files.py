"""Load and save audio-token embedding tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_embeddings(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ["audio_tokens", "embeddings", "tokens", "hidden_states"]:
                if key in obj:
                    obj = obj[key]
                    break
        if not torch.is_tensor(obj):
            raise TypeError(f"{path} does not contain a tensor or supported tensor dict.")
        return obj.float()

    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()

    raise ValueError(f"Unsupported input suffix: {suffix}. Use .pt, .pth, or .npy.")


def save_embeddings(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        torch.save(tensor.cpu(), path)
        return

    if suffix == ".npy":
        np.save(path, tensor.detach().cpu().numpy())
        return

    raise ValueError(f"Unsupported output suffix: {suffix}. Use .pt, .pth, or .npy.")
