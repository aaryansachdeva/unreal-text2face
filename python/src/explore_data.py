"""
Dataset exploration and statistics.

Walks the Express4D data directory, validates CSV format, reports what's usable,
and computes per-channel mean/std on the training split. Saves stats to
`stats/` for the dataset class to use during training.

Run this ONCE before training:
    python src/explore_data.py

Output:
    stats/mean.npy      — [61] per-channel mean over training set
    stats/std.npy       — [61] per-channel std over training set
    stats/metadata.json — frame counts, split sizes, sanity info
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# The 61 columns we keep (everything after Timecode + BlendshapeCount)
# This matches LiveLink Face app output and Express4D's convention.
CHANNEL_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight",
    "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawRight", "JawLeft", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight", "MouthLeft",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut",
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]
assert len(CHANNEL_NAMES) == 61


def load_clip_csv(csv_path: Path) -> np.ndarray:
    """Load a LiveLink CSV and return [T, 61] float32 array.

    LiveLink format: Timecode, BlendshapeCount, <61 values>
    We drop the first two columns.
    """
    df = pd.read_csv(csv_path)
    # Drop the first two columns regardless of exact header names
    data = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    if data.shape[1] != 61:
        raise ValueError(
            f"Expected 61 channels in {csv_path.name}, got {data.shape[1]}"
        )
    return data


def load_caption(txt_path: Path) -> str:
    """Load a text caption file. Format: 'caption#POS-tagged tokens'.
    We keep only the caption half (before '#')."""
    raw = txt_path.read_text(encoding="utf-8").strip()
    if "#" in raw:
        raw = raw.split("#")[0].strip()
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="ExpressData",
        help="Path to the ExpressData directory",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="stats",
        help="Where to write mean.npy / std.npy / metadata.json",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    stats_dir = Path(args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = data_root / "data"
    txt_dir = data_root / "texts"
    train_split = (data_root / "train.txt").read_text(encoding="utf-8").strip().splitlines()
    test_split = (data_root / "test.txt").read_text(encoding="utf-8").strip().splitlines()

    train_split = [s.strip() for s in train_split if s.strip()]
    test_split = [s.strip() for s in test_split if s.strip()]

    print(f"Data root:  {csv_dir}")
    print(f"Train clips in split: {len(train_split)}")
    print(f"Test clips in split:  {len(test_split)}")

    # Validate: every clip in the split should have both CSV and TXT
    print("\nValidating train split...")
    valid_train = []
    missing = []
    for clip_id in train_split:
        csv_path = csv_dir / f"{clip_id}.csv"
        txt_path = txt_dir / f"{clip_id}.txt"
        if not csv_path.exists():
            missing.append(f"CSV missing: {clip_id}")
            continue
        if not txt_path.exists():
            missing.append(f"TXT missing: {clip_id}")
            continue
        valid_train.append(clip_id)

    print(f"  Valid train clips: {len(valid_train)}")
    if missing:
        print(f"  Missing: {len(missing)} (showing first 5)")
        for m in missing[:5]:
            print(f"    {m}")

    # Same for test
    print("\nValidating test split...")
    valid_test = []
    for clip_id in test_split:
        csv_path = csv_dir / f"{clip_id}.csv"
        txt_path = txt_dir / f"{clip_id}.txt"
        if csv_path.exists() and txt_path.exists():
            valid_test.append(clip_id)
    print(f"  Valid test clips: {len(valid_test)}")

    # Walk training CSVs and accumulate stats + frame length distribution
    print("\nComputing per-channel mean/std over training set...")
    # Use two-pass Welford style to avoid memory blowup — but for 700 clips,
    # just concatenate; it's only ~10MB total.
    all_frames = []
    frame_lengths = []
    sample_captions = []

    for clip_id in tqdm(valid_train, desc="Loading train CSVs"):
        csv_path = csv_dir / f"{clip_id}.csv"
        txt_path = txt_dir / f"{clip_id}.txt"
        try:
            data = load_clip_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {clip_id}: {e}")
            continue
        if len(data) < 10:
            continue  # too short, skip
        all_frames.append(data)
        frame_lengths.append(len(data))

        if len(sample_captions) < 5:
            sample_captions.append((clip_id, load_caption(txt_path)))

    stacked = np.concatenate(all_frames, axis=0)  # [total_frames, 61]
    print(f"\nTotal training frames: {len(stacked)}")
    print(f"  ~ {len(stacked) / 60.0 / 60.0:.1f} minutes at 60 fps source")

    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    # Guard against zero-std channels (if any channel is always 0)
    std = np.where(std < 1e-6, 1.0, std)

    print("\nPer-channel stats (first 10 + last 9):")
    print(f"{'channel':<30} {'mean':>10} {'std':>10}")
    for i in list(range(10)) + list(range(52, 61)):
        print(f"  {CHANNEL_NAMES[i]:<28} {mean[i]:>10.5f} {std[i]:>10.5f}")

    # Frame length histogram
    frame_lengths = np.array(frame_lengths)
    print(f"\nFrame length distribution (source = 60 fps):")
    print(f"  min={frame_lengths.min()}  max={frame_lengths.max()}  "
          f"mean={frame_lengths.mean():.0f}  median={int(np.median(frame_lengths))}")
    for pct in [50, 75, 90, 95, 99]:
        v = int(np.percentile(frame_lengths, pct))
        print(f"  p{pct:<2d}: {v:>4d} frames  ({v / 60:.1f}s @ 60fps  "
              f"-> {v // 2} frames @ 30fps)")

    # Save
    np.save(stats_dir / "mean.npy", mean)
    np.save(stats_dir / "std.npy", std)

    metadata = {
        "channel_names": CHANNEL_NAMES,
        "n_channels": 61,
        "source_fps": 60,
        "valid_train_clips": len(valid_train),
        "valid_test_clips": len(valid_test),
        "total_train_frames": int(len(stacked)),
        "frame_length_min": int(frame_lengths.min()),
        "frame_length_max": int(frame_lengths.max()),
        "frame_length_mean": float(frame_lengths.mean()),
        "frame_length_p95": int(np.percentile(frame_lengths, 95)),
        "frame_length_p99": int(np.percentile(frame_lengths, 99)),
        "sample_captions": [
            {"clip_id": cid, "caption": cap} for cid, cap in sample_captions
        ],
    }
    with open(stats_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  {stats_dir / 'mean.npy'}")
    print(f"  {stats_dir / 'std.npy'}")
    print(f"  {stats_dir / 'metadata.json'}")
    print("\nSample captions:")
    for cid, cap in sample_captions:
        print(f"  [{cid}] {cap}")


if __name__ == "__main__":
    main()
