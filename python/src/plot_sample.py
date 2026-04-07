"""
Plot a generated .pt sample as time series curves for key channels.

Saves a PNG to outputs/<name>_plot.png. Useful for eyeballing whether the
model actually produces expected motion for a prompt before hauling it into
Unreal.

Usage:
    python src/plot_sample.py --input outputs/test3_yawn.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, save to file
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CHANNEL_NAMES

# Groups of channels we care about for quick eyeballing
GROUPS = {
    "Mouth (speech/expression)": [
        ("JawOpen", 17),
        ("MouthFunnel", 19),
        ("MouthPucker", 20),
        ("MouthSmileLeft", 23),
        ("MouthSmileRight", 24),
        ("MouthFrownLeft", 25),
        ("MouthFrownRight", 26),
    ],
    "Eyes": [
        ("EyeBlinkLeft", 0),
        ("EyeBlinkRight", 7),
        ("EyeSquintLeft", 5),
        ("EyeSquintRight", 12),
        ("EyeWideLeft", 6),
        ("EyeWideRight", 13),
    ],
    "Brows / Cheeks / Nose": [
        ("BrowDownLeft", 41),
        ("BrowDownRight", 42),
        ("BrowInnerUp", 43),
        ("BrowOuterUpLeft", 44),
        ("BrowOuterUpRight", 45),
        ("CheekPuff", 46),
        ("NoseSneerLeft", 49),
    ],
    "Head + Tongue": [
        ("HeadYaw", 52),
        ("HeadPitch", 53),
        ("HeadRoll", 54),
        ("TongueOut", 51),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    in_path = Path(args.input)
    payload = torch.load(in_path, weights_only=False)
    if isinstance(payload, dict) and "motion" in payload:
        motion = payload["motion"].numpy() if torch.is_tensor(payload["motion"]) else np.asarray(payload["motion"])
        prompt = payload.get("prompt", "")
        fps = payload.get("fps", 30)
    else:
        motion = payload.numpy() if torch.is_tensor(payload) else np.asarray(payload)
        prompt = ""
        fps = 30

    T = motion.shape[0]
    t = np.arange(T) / fps

    # Sanity check the expected channels
    for group_name, channels in GROUPS.items():
        for name, idx in channels:
            assert CHANNEL_NAMES[idx] == name, (
                f"Channel mismatch at {idx}: expected {name}, got {CHANNEL_NAMES[idx]}"
            )

    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"Generated animation — {prompt}", fontsize=12)

    for ax, (group_name, channels) in zip(axes, GROUPS.items()):
        for name, idx in channels:
            ax.plot(t, motion[:, idx], label=name, linewidth=1.4)
        ax.set_ylabel(group_name, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_plot.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
