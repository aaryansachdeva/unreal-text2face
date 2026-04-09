"""
Express4D PyTorch Dataset (v3).

Loads paired (text, face motion) training examples from the Express4D
LiveLink Face CSV format. Normalizes using precomputed stats, downsamples
from 60fps source to a target fps (default 30), and pads to a fixed window.

v3 augmentations (training split only; eval is always deterministic):
    - Horizontal mirror with text L/R swap:
        ~50% of training samples are flipped. Channel pairs (EyeBlinkLeft,
        EyeBlinkRight), (JawLeft, JawRight), etc. are swapped, and yaw/roll
        rotations change sign. Captions get their 'left'/'right' words
        swapped to keep labels consistent. This doubles the effective
        training set for free.
    - Variable sub-window cropping:
        With probability 0.3, crop a random-length window from the clip
        instead of using the full clip (or the fixed max_frames center
        crop). Exposes the model to many more (window, caption) pairs
        per clip.
    - Random word dropout on captions:
        Each non-L/R word is dropped with prob p (default 0.1). Forces
        the text encoder + cross-attention to attend to more than one
        way of expressing the same idea.

The raw CSVs have shape [T_60fps, 61]. After processing we return:
    motion: [max_frames, 61] float32 (normalized, zero-padded)
    mask:   [max_frames]     bool     (True = real frame, False = padding)
    text:   str                       (the caption, possibly augmented)
    length: int                       (number of real frames before padding)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Same constant as in explore_data.py
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
N_CHANNELS = 61
SOURCE_FPS = 60


# ---------------------------------------------------------------------------
# Horizontal mirror lookup table
# ---------------------------------------------------------------------------
# MIRROR_PERM[i] is the channel index that channel i maps TO under a
# horizontal left<->right flip of the face. MIRROR_SIGN[i] is -1 if that
# channel's value also negates (for yaw/roll rotations) and +1 otherwise.
#
# Pairs:
#   EyeX{Left} <-> EyeX{Right}           (14 channels, 7 pairs)
#   JawLeft    <-> JawRight
#   MouthLeft  <-> MouthRight
#   MouthXLeft <-> MouthXRight           (various mouth pairs)
#   BrowXLeft  <-> BrowXRight
#   CheekSquintLeft <-> CheekSquintRight
#   NoseSneerLeft   <-> NoseSneerRight
#   LeftEyeYaw  <-> RightEyeYaw    (+ sign flip)
#   LeftEyePitch<-> RightEyePitch
#   LeftEyeRoll <-> RightEyeRoll   (+ sign flip)
# Sign flips (without swap):
#   HeadYaw, HeadRoll

def _build_mirror_table() -> Tuple[np.ndarray, np.ndarray]:
    """Smart mirror (v8): swap Left/Right EXPRESSION channels only.
    Head rotation (52-54) and eye rotation (55-60) are LEFT UNTOUCHED
    so the model preserves natural head movement patterns."""
    name_to_idx = {n: i for i, n in enumerate(CHANNEL_NAMES)}
    perm = list(range(N_CHANNELS))
    sign = np.ones(N_CHANNELS, dtype=np.float32)

    def swap(a: str, b: str) -> None:
        i, j = name_to_idx[a], name_to_idx[b]
        perm[i], perm[j] = j, i

    # Swap Left/Right expression blendshapes (indices 0-51 only)
    # Skip LeftEye*/RightEye* rotation channels (55-60)
    rotation_names = {"LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
                      "RightEyeYaw", "RightEyePitch", "RightEyeRoll"}
    for name in list(name_to_idx.keys()):
        if name in rotation_names:
            continue  # don't touch eye rotations
        if "Left" in name:
            mirrored = name.replace("Left", "Right")
            if mirrored in name_to_idx and mirrored not in rotation_names:
                if name_to_idx[name] < name_to_idx[mirrored]:
                    swap(name, mirrored)

    # Non-suffix spatial pairs (expressions only)
    swap("JawLeft", "JawRight")
    swap("MouthLeft", "MouthRight")

    # NO sign flips on HeadYaw/HeadRoll — leave head rotation as-is
    # NO swaps or sign flips on eye rotations — leave as-is

    return np.asarray(perm, dtype=np.int64), sign


MIRROR_PERM, MIRROR_SIGN = _build_mirror_table()


def mirror_motion(x: np.ndarray) -> np.ndarray:
    """Smart mirror: swap Left/Right expression blendshapes but preserve
    head and eye rotation channels unchanged. This gives L/R symmetry
    for expressions without killing head movement.
    """
    return (x[:, MIRROR_PERM] * MIRROR_SIGN).astype(np.float32)


# ---------------------------------------------------------------------------
# Text augmentation
# ---------------------------------------------------------------------------

# Word-level left/right swap, case-preserving. Matches whole words only
# so we don't break "leftover", "righteous", etc.
_LR_RE = re.compile(
    r"\b(left|right|leftward|rightward|clockwise|counterclockwise)\b",
    flags=re.IGNORECASE,
)
_LR_SWAP = {
    "left": "right",
    "right": "left",
    "leftward": "rightward",
    "rightward": "leftward",
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
}


def mirror_text(s: str) -> str:
    """Swap left<->right vocabulary in a caption while preserving case.

    'looking to the Left' -> 'looking to the Right'
    'LEFTWARD glance'     -> 'RIGHTWARD glance'
    'leftover thoughts'   -> 'leftover thoughts'   (not a whole-word match)
    """
    def repl(m: "re.Match[str]") -> str:
        w = m.group(0)
        target = _LR_SWAP[w.lower()]
        if w.isupper():
            return target.upper()
        if w[0].isupper():
            return target.capitalize()
        return target
    return _LR_RE.sub(repl, s)


# Words we never drop so the model doesn't lose spatial cues. The rest
# of the prompt is fair game.
_PROTECTED_WORDS = {
    "left", "right", "leftward", "rightward",
    "up", "down", "upward", "downward",
    "in", "out", "inward", "outward",
    "clockwise", "counterclockwise",
}


def drop_words(s: str, p: float) -> str:
    """Randomly drop each non-protected whitespace word with probability p.

    Uses numpy's global RNG so it inherits the per-worker seed installed
    by train.py's worker_init_fn.
    """
    if p <= 0:
        return s
    words = s.split()
    if len(words) <= 2:
        return s
    kept: List[str] = []
    for w in words:
        if w.lower().strip(".,;:!?") in _PROTECTED_WORDS:
            kept.append(w)
        elif np.random.random() >= p:
            kept.append(w)
    if len(kept) < 2:
        return s  # don't return a nearly empty caption
    return " ".join(kept)


# ---------------------------------------------------------------------------
# CSV / caption loaders (unchanged from v1)
# ---------------------------------------------------------------------------

def _load_csv(csv_path: Path) -> np.ndarray:
    """Load a LiveLink CSV as [T, 61] float32. Drops Timecode + BlendshapeCount."""
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    if data.shape[1] != N_CHANNELS:
        raise ValueError(f"{csv_path.name}: expected 61 channels, got {data.shape[1]}")
    return data


def _load_caption(txt_path: Path) -> str:
    """Load a caption, dropping the POS-tagged half after '#'."""
    raw = txt_path.read_text(encoding="utf-8").strip()
    if "#" in raw:
        raw = raw.split("#")[0].strip()
    return raw


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Express4DDataset(Dataset):
    """
    Args:
        data_root:         path to ExpressData (containing data/, texts/, train.txt, test.txt)
        stats_dir:         path to where mean.npy and std.npy live
        split:             'train' or 'test'
        target_fps:        30 (downsample factor 2 from 60fps source)
        max_frames:        240 = 8 seconds at 30fps
        min_frames:        16  -- filter out anything shorter
        augment:           bool, enable augmentation (default True for train, False for test)
        mirror_prob:       probability of horizontal mirror per sample (default 0.5)
        word_drop_prob:    probability of dropping each non-protected word (default 0.1)
        subwindow_prob:    probability of variable sub-window crop (default 0.3)
        cfg_drop_prob:     probability of dropping the ENTIRE caption (replaced with "")
                           to train the unconditional path for classifier-free guidance.
                           Applied AFTER mirror+word_drop so it overrides both. (default 0.1)
    """

    def __init__(
        self,
        data_root: str | Path,
        stats_dir: str | Path,
        split: str = "train",
        target_fps: int = 30,
        max_frames: int = 240,
        min_frames: int = 16,
        augment: bool | None = None,
        mirror_prob: float = 0.5,
        word_drop_prob: float = 0.1,
        subwindow_prob: float = 0.3,
        cfg_drop_prob: float = 0.1,
    ):
        assert split in ("train", "test")
        assert SOURCE_FPS % target_fps == 0, (
            f"target_fps must divide {SOURCE_FPS}, got {target_fps}"
        )
        self.data_root = Path(data_root)
        self.split = split
        self.target_fps = target_fps
        self.downsample = SOURCE_FPS // target_fps
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.augment = augment if augment is not None else (split == "train")
        self.mirror_prob = mirror_prob if self.augment else 0.0
        self.word_drop_prob = word_drop_prob if self.augment else 0.0
        self.subwindow_prob = subwindow_prob if self.augment else 0.0
        self.cfg_drop_prob = cfg_drop_prob if self.augment else 0.0

        # Load stats
        self.mean = np.load(Path(stats_dir) / "mean.npy").astype(np.float32)  # [61]
        self.std = np.load(Path(stats_dir) / "std.npy").astype(np.float32)    # [61]
        assert self.mean.shape == (N_CHANNELS,)
        assert self.std.shape == (N_CHANNELS,)

        # Load split list
        split_file = self.data_root / f"{split}.txt"
        clip_ids = [
            s.strip()
            for s in split_file.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]

        # Filter to clips that have both CSV and TXT and meet the minimum length
        self.items: List[Tuple[str, Path, Path]] = []
        dropped_missing = 0
        dropped_short = 0
        for cid in clip_ids:
            csv_path = self.data_root / "data" / f"{cid}.csv"
            txt_path = self.data_root / "texts" / f"{cid}.txt"
            if not csv_path.exists() or not txt_path.exists():
                dropped_missing += 1
                continue
            try:
                n_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            except Exception:
                dropped_missing += 1
                continue
            n_target = n_rows // self.downsample
            if n_target < self.min_frames:
                dropped_short += 1
                continue
            self.items.append((cid, csv_path, txt_path))

        print(
            f"[Express4DDataset:{split}] "
            f"{len(self.items)} clips usable "
            f"(dropped {dropped_missing} missing, {dropped_short} too short) "
            f"augment={self.augment} mirror_p={self.mirror_prob} "
            f"word_drop_p={self.word_drop_prob} subwin_p={self.subwindow_prob} "
            f"cfg_drop_p={self.cfg_drop_prob}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        clip_id, csv_path, txt_path = self.items[idx]
        data = _load_csv(csv_path)            # [T_src, 61]
        data = data[:: self.downsample]        # [T_src/ds, 61]
        T = len(data)

        # --- Window selection ---
        # Pick a contiguous window of length L from the clip.
        # - augment mode:
        #     * with prob subwindow_prob, L is random in [min_frames+8, min(T, max_frames)]
        #     * else L = min(T, max_frames)
        #     * start is random in [0, T-L]
        # - eval mode:
        #     * L = min(T, max_frames), start is the center crop
        max_L = min(T, self.max_frames)
        if self.augment:
            use_subwindow = (
                np.random.random() < self.subwindow_prob
                and max_L > self.min_frames + 8
            )
            if use_subwindow:
                L = int(np.random.randint(self.min_frames + 8, max_L + 1))
            else:
                L = max_L
            start = int(np.random.randint(0, T - L + 1))
        else:
            L = max_L
            start = (T - L) // 2
        data = data[start : start + L]
        T = L

        # --- Caption ---
        caption = _load_caption(txt_path)

        # --- Augmentation (mirror + word dropout) ---
        if self.augment and self.mirror_prob > 0 and np.random.random() < self.mirror_prob:
            data = mirror_motion(data)
            caption = mirror_text(caption)
        if self.augment and self.word_drop_prob > 0:
            caption = drop_words(caption, self.word_drop_prob)

        # --- CFG: unconditional dropout (replace entire caption with "") ---
        if self.augment and self.cfg_drop_prob > 0 and np.random.random() < self.cfg_drop_prob:
            caption = ""

        # --- Normalize ---
        data = self.normalize(data)

        # --- Pad to max_frames ---
        padded = np.zeros((self.max_frames, N_CHANNELS), dtype=np.float32)
        padded[:T] = data
        mask = np.zeros(self.max_frames, dtype=bool)
        mask[:T] = True

        return {
            "motion": torch.from_numpy(padded),      # [max_frames, 61]
            "mask":   torch.from_numpy(mask),        # [max_frames]
            "text":   caption,                       # str
            "length": T,                             # int
            "clip_id": clip_id,
        }


def collate_fn(batch):
    """Stack the per-sample tensors and keep the list fields as lists."""
    return {
        "motion": torch.stack([b["motion"] for b in batch]),      # [B, T, 61]
        "mask":   torch.stack([b["mask"] for b in batch]),        # [B, T]
        "text":   [b["text"] for b in batch],                     # list[str], len B
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "clip_id": [b["clip_id"] for b in batch],
    }


# ---------------------------------------------------------------------------
# DataLoader worker init: give each worker its own numpy RNG seed.
# Without this, all workers would share the same numpy seed and produce
# identical augmentation sequences -- which defeats the point.
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
