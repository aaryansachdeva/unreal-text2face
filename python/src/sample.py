"""
Sample / inference from a trained TextToFace checkpoint.

Usage:
    python src/sample.py --prompt "A person looks to the side thoughtfully" --frames 120
    python src/sample.py --prompt "Grace is confused" --frames 180 --out-name grace_confused

Produces a .pt tensor at outputs/<name>.pt with shape [T, 61], in
denormalized ARKit blendshape space (values in the original LiveLink range).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import TextToFace


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[TextToFace, dict]:
    """Load a training checkpoint. Uses EMA weights if available (smoother output)."""
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = payload.get("args", {})

    # v8 architecture defaults — matches train.py
    model = TextToFace(
        max_frames=saved_args.get("max_frames", 240),
        latent_dim=saved_args.get("latent_dim", 768),
        n_layers=saved_args.get("n_layers", 6),
        n_heads=saved_args.get("n_heads", 12),
        ff_dim=saved_args.get("ff_dim", 2048),
        dropout=saved_args.get("dropout", 0.25),
    ).to(device)

    # Prefer EMA weights — smoother, what training validates against.
    # Fall back to raw weights for older checkpoints that didn't save EMA.
    if "ema_model" in payload:
        model.load_state_dict(payload["ema_model"])
    else:
        model.load_state_dict(payload["model"])
    model.eval()
    return model, saved_args


def generate(
    model: TextToFace,
    prompt: str,
    n_frames: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    guidance_scale: float = 1.0,
) -> np.ndarray:
    """Run inference and return denormalized [T, 61] float32.

    When guidance_scale > 1.0, uses classifier-free guidance:
        output = uncond + guidance_scale * (cond - uncond)
    This amplifies what the text *specifically* describes, producing
    more exaggerated expressions. Requires a model trained with
    cfg_drop_prob > 0.  guidance_scale=1.0 disables CFG (single pass).
    """
    with torch.no_grad():
        cond = model([prompt], n_frames=n_frames)  # [1, T, 61]
        if guidance_scale != 1.0:
            uncond = model([""], n_frames=n_frames)  # [1, T, 61]
            out_norm = uncond + guidance_scale * (cond - uncond)
        else:
            out_norm = cond
    out_norm = out_norm[0].cpu().numpy()                # [T, 61]
    # Denormalize back into raw ARKit space
    out = out_norm * std + mean
    # Clip expression blendshapes (first 52 = ARKit 52 including TongueOut) into sane range.
    # Rotations (indices 52-60) can go outside this and that's fine.
    expression_slice = slice(0, 52)
    out[:, expression_slice] = np.clip(out[:, expression_slice], 0.0, 1.0)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/best.pt")
    parser.add_argument("--stats-dir", type=str,
                        default="stats")
    parser.add_argument("--out-dir", type=str,
                        default="outputs")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text description of the desired face animation")
    parser.add_argument("--frames", type=int, default=120,
                        help="Number of frames to generate (30fps = 4 sec at 120)")
    parser.add_argument("--out-name", type=str, default=None,
                        help="Output filename (without extension). Auto-generated if omitted.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frame rate these outputs are in (passed along to LiveLink export)")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="Classifier-free guidance scale. 1.0=off, 1.5=moderate, 2.0=strong")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed (not used currently but reserved)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model, saved_args = load_checkpoint(ckpt_path, device)
    n_train_params = model.count_trainable()
    print(f"Model: {n_train_params / 1e6:.2f}M trainable params")

    mean = np.load(Path(args.stats_dir) / "mean.npy")  # [61]
    std = np.load(Path(args.stats_dir) / "std.npy")    # [61]

    max_frames = saved_args.get("max_frames", 240)
    if args.frames > max_frames:
        print(f"Warning: requested {args.frames} frames, model max is {max_frames}. Clamping.")
        args.frames = max_frames

    print(f"\nPrompt: {args.prompt}")
    print(f"Frames: {args.frames} ({args.frames / args.fps:.1f}s @ {args.fps}fps)")
    if args.guidance != 1.0:
        print(f"Guidance scale: {args.guidance}")

    out = generate(model, args.prompt, args.frames, mean, std, device,
                   guidance_scale=args.guidance)
    print(f"Generated shape: {out.shape}")
    print(f"Value ranges per region:")
    print(f"  expression bs [0..50]: min={out[:,:51].min():.3f} max={out[:,:51].max():.3f}")
    print(f"  TongueOut       [51]:  min={out[:,51].min():.3f}   max={out[:,51].max():.3f}")
    print(f"  head rotations  [52:55]: min={out[:,52:55].min():.3f} max={out[:,52:55].max():.3f}")
    print(f"  eye rotations   [55:61]: min={out[:,55:61].min():.3f} max={out[:,55:61].max():.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-name from first few words of prompt if no name given
    if args.out_name is None:
        slug = "_".join(args.prompt.lower().split()[:6])
        slug = "".join(c if c.isalnum() or c == "_" else "" for c in slug)
        args.out_name = slug[:48] or "sample"

    out_path = out_dir / f"{args.out_name}.pt"
    torch.save({
        "motion": torch.from_numpy(out),       # [T, 61]
        "prompt": args.prompt,
        "fps": args.fps,
        "n_frames": args.frames,
    }, out_path)
    print(f"\nSaved: {out_path}")
    print(f"To turn this into a LiveLink CSV for Unreal, run:")
    print(f'  python src/export_livelink.py --input "{out_path}"')


if __name__ == "__main__":
    main()
