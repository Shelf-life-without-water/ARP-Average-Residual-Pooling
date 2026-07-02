"""Command-line utility for ARP/MARP compression on saved embeddings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compress import compress_tokens
from files import load_embeddings, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress audio-token embeddings with ARP or MARP.")
    parser.add_argument("--input", type=Path, required=True, help="Input .pt/.pth/.npy tensor with shape [T, D].")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt/.pth/.npy tensor with shape [B, D].")
    parser.add_argument("--method", choices=["arp", "marp"], default="arp")
    parser.add_argument("--keep-ratio", type=float, default=None, help="Keep ratio rho, e.g., 0.75.")
    parser.add_argument("--target-len", type=int, default=None, help="Explicit target token count B.")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--num-regions", type=int, default=4, help="Only used by MARP.")
    parser.add_argument(
        "--score-mode",
        choices=["raw_l2", "ln_l2", "cosine", "diag_maha"],
        default="raw_l2",
        help="Residual scoring function.",
    )
    parser.add_argument("--info", type=Path, default=None, help="Optional JSON path for compression metadata.")
    args = parser.parse_args()

    audio_tokens = load_embeddings(args.input)
    compressed, info = compress_tokens(
        audio_tokens,
        method=args.method,
        keep_ratio=args.keep_ratio,
        target_len=args.target_len,
        window_size=args.window_size,
        num_regions=args.num_regions,
        score_mode=args.score_mode,
        return_info=True,
    )
    save_embeddings(args.output, compressed)

    if args.info is not None:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        args.info.write_text(json.dumps(asdict(info), indent=2), encoding="utf-8")

    print(f"input_shape={tuple(audio_tokens.shape)}")
    print(f"output_shape={tuple(compressed.shape)}")
    print(json.dumps(asdict(info), indent=2))


if __name__ == "__main__":
    main()
