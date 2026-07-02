"""Multiscale Average-Residual Pooling (MARP)."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from arp import arp_compress
from info import CompressionInfo, ScoreMode
from util import allocate_budget, target_length, validate_tokens


def marp_compress(
    audio_tokens: torch.Tensor,
    keep_ratio: Optional[float] = None,
    target_len: Optional[int] = None,
    window_size: int = 8,
    num_regions: int = 4,
    score_mode: ScoreMode = "raw_l2",
    eps: float = 1e-6,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CompressionInfo]]:
    """
    Multiscale Average-Residual Pooling (MARP).

    MARP selects the top M-1 adjacent token-change positions as region
    boundaries, allocates the target budget to regions by temporal length, and
    applies ARP inside each region.
    """
    x = validate_tokens(audio_tokens)
    T = int(x.shape[0])
    B = target_length(T, keep_ratio=keep_ratio, target_len=target_len)

    if B >= T:
        out = x
    elif B == 1:
        out = x.mean(dim=0, keepdim=True)
    else:
        M = max(1, min(int(num_regions), T, B))
        if M <= 1 or T <= 2:
            out = arp_compress(
                x,
                target_len=B,
                window_size=window_size,
                score_mode=score_mode,
                eps=eps,
                return_info=False,
            )
        else:
            change_scores = torch.norm((x[1:] - x[:-1]).float(), dim=-1)
            num_boundaries = min(M - 1, T - 1)
            boundaries = torch.topk(change_scores, k=num_boundaries, largest=True).indices + 1
            boundaries = torch.sort(boundaries.long()).values

            starts = torch.cat([
                torch.zeros(1, device=x.device, dtype=torch.long),
                boundaries,
            ])
            ends = torch.cat([
                boundaries,
                torch.tensor([T], device=x.device, dtype=torch.long),
            ])

            lengths = ends - starts
            quotas = allocate_budget(lengths.cpu(), B).to(x.device)

            outputs = []
            for start, end, quota in zip(starts.tolist(), ends.tolist(), quotas.tolist()):
                if end <= start:
                    end = min(T, start + 1)
                region = x[start:end]
                compressed_region = arp_compress(
                    region,
                    target_len=int(quota),
                    window_size=window_size,
                    score_mode=score_mode,
                    eps=eps,
                    return_info=False,
                )
                outputs.append(compressed_region)

            out = torch.cat(outputs, dim=0)
            if out.shape[0] > B:
                out = out[:B]
            elif out.shape[0] < B:
                pad = x[-1:].expand(B - out.shape[0], -1)
                out = torch.cat([out, pad], dim=0)

    if return_info:
        info = CompressionInfo(
            method="marp",
            original_tokens=T,
            compressed_tokens=int(out.shape[0]),
            target_tokens=B,
            keep_ratio=keep_ratio,
            window_size=int(window_size),
            num_regions=int(num_regions),
            score_mode=score_mode,
        )
        return out, info
    return out
