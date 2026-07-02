"""Average-Residual Pooling (ARP)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from info import CompressionInfo, ScoreMode
from score import residual_scores
from util import avg_bins, fixed_windows, target_length, validate_tokens


def arp_compress(
    audio_tokens: torch.Tensor,
    keep_ratio: Optional[float] = None,
    target_len: Optional[int] = None,
    window_size: int = 8,
    score_mode: ScoreMode = "raw_l2",
    eps: float = 1e-6,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CompressionInfo]]:
    """
    Average-Residual Pooling (ARP).

    The sequence is split into local windows. Each window keeps one mean token.
    The remaining budget is assigned to original tokens with high residual
    scores around the local mean.
    """
    x = validate_tokens(audio_tokens)
    T = int(x.shape[0])
    B = target_length(T, keep_ratio=keep_ratio, target_len=target_len)

    if B >= T:
        out = x
    elif B == 1:
        out = x.mean(dim=0, keepdim=True)
    else:
        windows = fixed_windows(T, window_size)

        if len(windows) >= B:
            out = avg_bins(x, B)
        else:
            mean_slots: List[Tuple[float, torch.Tensor]] = []
            residual_candidates: List[Tuple[float, float, torch.Tensor]] = []

            for s, e in windows:
                segment = x[s:e]
                mean_token = segment.mean(dim=0, keepdim=True)
                mean_slots.append((float(s) - 0.1, mean_token))

                scores = residual_scores(segment, mean_token, score_mode=score_mode, eps=eps)
                window_variation = scores.mean()

                for local_idx in range(segment.shape[0]):
                    global_idx = s + local_idx
                    rank_score = float((scores[local_idx] * (window_variation + eps)).item())
                    residual_candidates.append((rank_score, float(global_idx), segment[local_idx:local_idx + 1]))

            residual_budget = max(0, B - len(mean_slots))
            residual_candidates = sorted(residual_candidates, key=lambda item: item[0], reverse=True)
            residual_candidates = residual_candidates[:residual_budget]

            selected: List[Tuple[float, torch.Tensor]] = []
            selected.extend(mean_slots)
            selected.extend((pos, token) for _, pos, token in residual_candidates)
            selected = sorted(selected, key=lambda item: item[0])

            out = torch.cat([token for _, token in selected], dim=0)
            if out.shape[0] > B:
                out = out[:B]
            elif out.shape[0] < B:
                pad = x[-1:].expand(B - out.shape[0], -1)
                out = torch.cat([out, pad], dim=0)

    if return_info:
        info = CompressionInfo(
            method="arp",
            original_tokens=T,
            compressed_tokens=int(out.shape[0]),
            target_tokens=B,
            keep_ratio=keep_ratio,
            window_size=int(window_size),
            score_mode=score_mode,
        )
        return out, info
    return out
