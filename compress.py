"""Unified ARP/MARP compression wrapper."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from arp import arp_compress
from info import CompressionInfo, MethodName, ScoreMode
from marp import marp_compress


def compress_tokens(
    audio_tokens: torch.Tensor,
    method: MethodName = "arp",
    keep_ratio: Optional[float] = None,
    target_len: Optional[int] = None,
    window_size: int = 8,
    num_regions: int = 4,
    score_mode: ScoreMode = "raw_l2",
    eps: float = 1e-6,
    return_info: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, CompressionInfo]]:
    if method == "arp":
        return arp_compress(
            audio_tokens,
            keep_ratio=keep_ratio,
            target_len=target_len,
            window_size=window_size,
            score_mode=score_mode,
            eps=eps,
            return_info=return_info,
        )
    if method == "marp":
        return marp_compress(
            audio_tokens,
            keep_ratio=keep_ratio,
            target_len=target_len,
            window_size=window_size,
            num_regions=num_regions,
            score_mode=score_mode,
            eps=eps,
            return_info=return_info,
        )
    raise ValueError(f"Unknown method={method}. Expected 'arp' or 'marp'.")
