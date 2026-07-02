"""Shared types for ARP and MARP compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ScoreMode = Literal["raw_l2", "ln_l2", "cosine", "diag_maha"]
MethodName = Literal["arp", "marp"]


@dataclass
class CompressionInfo:
    method: str
    original_tokens: int
    compressed_tokens: int
    target_tokens: int
    keep_ratio: Optional[float]
    window_size: int
    num_regions: Optional[int] = None
    score_mode: str = "raw_l2"
