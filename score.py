"""Residual scoring functions for ARP-family compression."""

from __future__ import annotations

import torch

from info import ScoreMode


def layer_norm_token(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    mean = x_float.mean(dim=-1, keepdim=True)
    std = x_float.std(dim=-1, keepdim=True, unbiased=False)
    return (x_float - mean) / (std + eps)


def residual_scores(
    segment: torch.Tensor,
    mean_token: torch.Tensor,
    score_mode: ScoreMode = "raw_l2",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute token-level residual scores inside one local window."""
    if score_mode == "raw_l2":
        return torch.norm((segment - mean_token).float(), dim=-1)

    if score_mode == "ln_l2":
        seg_norm = layer_norm_token(segment, eps)
        mean_norm = layer_norm_token(mean_token, eps)
        return torch.norm(seg_norm - mean_norm, dim=-1)

    if score_mode == "cosine":
        seg_float = segment.float()
        mean_float = mean_token.float().expand_as(seg_float)
        similarity = torch.nn.functional.cosine_similarity(seg_float, mean_float, dim=-1, eps=eps)
        return 1.0 - similarity

    if score_mode == "diag_maha":
        seg_float = segment.float()
        mean_float = mean_token.float()
        var = seg_float.var(dim=0, keepdim=True, unbiased=False)
        diff = seg_float - mean_float
        return torch.sqrt(((diff * diff) / (var + eps)).sum(dim=-1) + eps)

    raise ValueError(f"Unknown score_mode={score_mode}")
