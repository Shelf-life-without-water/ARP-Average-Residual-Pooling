"""Utility functions for token-budget control and local windows."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def validate_tokens(audio_tokens: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(audio_tokens):
        raise TypeError("audio_tokens must be a torch.Tensor with shape [T, D].")
    if audio_tokens.ndim != 2:
        raise ValueError(f"audio_tokens must have shape [T, D], got {tuple(audio_tokens.shape)}")
    if audio_tokens.shape[0] < 1:
        raise ValueError("audio_tokens must contain at least one token.")
    return audio_tokens


def target_length(
    num_tokens: int,
    keep_ratio: Optional[float] = None,
    target_len: Optional[int] = None,
) -> int:
    if target_len is None and keep_ratio is None:
        raise ValueError("Either keep_ratio or target_len must be provided.")
    if target_len is None:
        target_len = int(round(float(keep_ratio) * int(num_tokens)))
    return max(1, min(int(target_len), int(num_tokens)))


def fixed_windows(num_tokens: int, window_size: int) -> List[Tuple[int, int]]:
    window_size = max(1, int(window_size))
    return [(s, min(num_tokens, s + window_size)) for s in range(0, num_tokens, window_size)]


def avg_bins(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Uniform average pooling to exactly target_len tokens."""
    x = validate_tokens(x)
    T = int(x.shape[0])
    target_len = max(1, min(int(target_len), T))
    if target_len >= T:
        return x

    edges = torch.linspace(0, T, target_len + 1, device=x.device).round().long()
    outputs = []
    for i in range(target_len):
        s = int(edges[i].item())
        e = int(edges[i + 1].item())
        if e <= s:
            e = min(T, s + 1)
        outputs.append(x[s:e].mean(dim=0, keepdim=True))
    return torch.cat(outputs, dim=0)


def allocate_budget(lengths: torch.Tensor, total_budget: int) -> torch.Tensor:
    """Allocate an integer budget to regions according to region lengths."""
    if lengths.ndim != 1:
        raise ValueError("lengths must be a 1D tensor.")
    if int(lengths.numel()) == 0:
        raise ValueError("lengths must be non-empty.")

    total_budget = int(total_budget)
    if total_budget < int(lengths.numel()):
        raise ValueError("total_budget must be at least the number of regions.")

    raw = lengths.float() / lengths.float().sum() * total_budget
    quotas = torch.floor(raw).long().clamp_min(1)

    diff = int(total_budget - quotas.sum().item())
    if diff > 0:
        fractional = raw - torch.floor(raw)
        order = torch.argsort(fractional, descending=True)
        for idx in order[:diff]:
            quotas[int(idx.item())] += 1
    elif diff < 0:
        need = -diff
        order = torch.argsort(quotas, descending=True)
        for idx in order:
            j = int(idx.item())
            if need <= 0:
                break
            if quotas[j] > 1:
                quotas[j] -= 1
                need -= 1

    if int(quotas.sum().item()) != total_budget:
        raise RuntimeError("Budget allocation failed to match total_budget.")
    return quotas
