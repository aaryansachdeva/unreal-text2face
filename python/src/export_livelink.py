"""
Convert a generated [T, 61] tensor into an Apple LiveLink Face CSV.

This is the format Unreal's LiveLink plugin and MetaHuman animation pipelines
consume natively. The output column layout matches what the iPhone Live Link
Face app produces:

    Timecode, BlendshapeCount, <61 float channels>

Usage:
    python src/export_livelink.py --input outputs/my_sample.pt
    python src/export_livelink.py --input outputs/my_sample.pt --target-fps 60
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import CHANNEL_NAMES, N_CHANNELS


def upsample_linear(x: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    """Linearly interpolate a [T, C] array from src_fps to dst_fps.
    Returns an [T', C] array where T' = T * dst_fps / src_fps (rounded).
    """
    if src_fps == dst_fps:
        return x
    T_src, C = x.shape
    T_dst = int(round(T_src * dst_fps / src_fps))
    # Sample the source at evenly spaced positions in [0, T_src - 1]
    src_idx = np.linspace(0.0, T_src - 1, T_dst)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.clip(lo + 1, 0, T_src - 1)
    w = (src_idx - lo).astype(np.float32)[:, None]     # [T_dst, 1]
    return (1 - w) * x[lo] + w * x[hi]


def frames_to_timecode(frame_index: int, fps: int) -> str:
    """Build a LiveLink-style HH:MM:SS:FF.subframe timecode starting from 00:00:00:00.
    fps is rounded to nearest int for the FF field."""
    total_frames = frame_index
    frames = total_frames % fps
    total_secs = total_frames // fps
    seconds = total_secs % 60
    total_mins = total_secs // 60
    minutes = total_mins % 60
    hours = total_mins // 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}.000"


def write_livelink_csv(motion: np.ndarray, out_path: Path, fps: int) -> None:
    """Write [T, 61] array as a LiveLink Face CSV."""
    assert motion.shape[1] == N_CHANNELS, f"expected 61 channels, got {motion.shape[1]}"
    T = motion.shape[0]
    header = ["Timecode", "BlendshapeCount"] + CHANNEL_NAMES
    lines = [",".join(header)]
    for i in range(T):
        tc = frames_to_timecode(i, fps)
        vals = ",".join(f"{v:.10f}" for v in motion[i])
        lines.append(f"{tc},{N_CHANNELS},{vals}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to a .pt file produced by sample.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (defaults to alongside input with .csv)")
    parser.add_argument("--target-fps", type=int, default=60,
                        help="Output frame rate. 60 matches LiveLink Face app default.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    payload = torch.load(in_path, weights_only=False)
    if isinstance(payload, dict) and "motion" in payload:
        motion = payload["motion"].numpy() if torch.is_tensor(payload["motion"]) else np.asarray(payload["motion"])
        src_fps = payload.get("fps", 30)
        prompt = payload.get("prompt", "")
    else:
        # Bare tensor
        motion = payload.numpy() if torch.is_tensor(payload) else np.asarray(payload)
        src_fps = 30
        prompt = ""

    if motion.ndim != 2 or motion.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected [T, 61] motion tensor, got shape {motion.shape}")

    print(f"Input:     {in_path}")
    print(f"  shape:   {motion.shape}")
    print(f"  src fps: {src_fps}")
    if prompt:
        print(f"  prompt:  {prompt}")

    if args.target_fps != src_fps:
        motion = upsample_linear(motion, src_fps, args.target_fps)
        print(f"  upsampled: {motion.shape} @ {args.target_fps} fps")

    out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")
    write_livelink_csv(motion, out_path, args.target_fps)
    print(f"\nWrote LiveLink CSV: {out_path}")
    print(f"  {motion.shape[0]} frames @ {args.target_fps} fps "
          f"= {motion.shape[0] / args.target_fps:.2f} seconds")


if __name__ == "__main__":
    main()
