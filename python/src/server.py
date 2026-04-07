"""
TextToFace sidecar server.

Holds the trained model resident and serves generated expression curve
blocks over HTTP for real-time use from Unreal Engine (or any other
client). The model's 52 ARKit blendshape outputs are translated to
MetaHuman 'ctrl_expressions_*' rig curve names before being returned,
so the Unreal side can write them directly into the face rig without
knowing about ARKit at all.

Start:
    python src/server.py --port 8765

Health check:
    curl http://127.0.0.1:8765/health

Generate:
    curl -X POST http://127.0.0.1:8765/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "A person looks surprised", "frames": 240, "fps": 60}'

Response shape:
    {
      "prompt":         "A person looks surprised",
      "duration":       4.0,
      "fps":            60,
      "n_frames":       240,
      "generation_ms":  48.7,
      "curves": {
        "ctrl_expressions_browraiseinl":  [0.02, 0.05, ...],
        "ctrl_expressions_jawopen":       [0.01, 0.12, ...],
        ...
      }
    }
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mh_mapping import MH_CURVE_NAMES, ARKIT_CURVE_NAMES, arkit_to_mh_curves

# All 61 ARKit channel names in model output order (same as dataset.py CHANNEL_NAMES)
CHANNEL_NAMES = list(ARKIT_CURVE_NAMES) + [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
]
from model import TextToFace


# =============================================================================
# Request / response schemas
# =============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Stage-direction text")
    frames: int = Field(240, ge=1, le=480, description="Number of output frames")
    fps: int = Field(60, ge=1, le=120, description="Output frame rate")
    guidance: float = Field(1.0, ge=0.0, le=5.0,
                            description="CFG scale. 1.0=off, 1.5=moderate, 2.0=strong exaggeration")


class GenerateResponse(BaseModel):
    prompt: str
    duration: float
    fps: int
    n_frames: int
    generation_ms: float
    curves: dict[str, list[float]]
    arkit_raw: dict[str, list[float]] = {}  # Raw ARKit channels for LiveLink


class HealthResponse(BaseModel):
    status: str
    device: Optional[str] = None
    max_frames: Optional[int] = None
    model_params_m: Optional[float] = None
    n_mh_curves: int


# =============================================================================
# Global model state - loaded once at startup
# =============================================================================

_model: Optional[TextToFace] = None
_mean: Optional[np.ndarray] = None
_std: Optional[np.ndarray] = None
_device: Optional[torch.device] = None
_max_frames: int = 240
_param_count_m: float = 0.0


def load_model(ckpt_path: str, stats_dir: str) -> None:
    global _model, _mean, _std, _device, _max_frames, _param_count_m

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server] device: {_device}")

    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    print(f"[server] loading checkpoint: {ckpt}")

    payload = torch.load(ckpt, map_location=_device, weights_only=False)
    saved_args = payload.get("args", {})
    _max_frames = int(saved_args.get("max_frames", 240))

    _model = TextToFace(
        max_frames=_max_frames,
        latent_dim=saved_args.get("latent_dim", 512),
        n_layers=saved_args.get("n_layers", 4),
        n_heads=saved_args.get("n_heads", 8),
        ff_dim=saved_args.get("ff_dim", 1024),
        dropout=saved_args.get("dropout", 0.25),
    ).to(_device)
    _model.load_state_dict(payload["model"])
    _model.eval()

    _mean = np.load(Path(stats_dir) / "mean.npy")
    _std = np.load(Path(stats_dir) / "std.npy")

    _param_count_m = sum(p.numel() for p in _model.parameters()) / 1e6
    print(f"[server] model ready: {_param_count_m:.2f}M params, max_frames={_max_frames}")

    # GPU warmup: first inference pays CUDA kernel JIT cost, do it now so
    # the first real request hits a hot model.
    with torch.no_grad():
        _ = _model(["warmup"], n_frames=8)
    print(f"[server] warmup complete, serving")


def generate_curves(prompt: str, n_frames: int, fps: int,
                    guidance: float = 1.0) -> dict:
    """Run inference, map ARKit -> MH curves, return a dict ready to serialize."""
    assert _model is not None and _mean is not None and _std is not None

    if n_frames > _max_frames:
        n_frames = _max_frames

    t0 = time.perf_counter()
    with torch.no_grad():
        cond = _model([prompt], n_frames=n_frames)              # [1, T, 61]
        if guidance != 1.0:
            uncond = _model([""], n_frames=n_frames)            # [1, T, 61]
            out_norm = uncond + guidance * (cond - uncond)
        else:
            out_norm = cond
    arkit = out_norm[0].cpu().numpy() * _std + _mean            # [T, 61]
    # Clip expression channels (0..51) into a sane range; head/eye rotation
    # channels (52..60) can go outside and that's fine.
    arkit[:, :52] = np.clip(arkit[:, :52], 0.0, 1.0)

    # --- Save raw ARKit values BEFORE amplification (for LiveLink path) ---
    # The LiveLink AnimBP (ABP_MH_LiveLink) applies its own scaling
    # (head rotation × 50, pitch × -50), so raw values must be clean.
    arkit_raw_clean = arkit.copy()

    # --- Post-processing: amplify weak channels (for legacy AnimNode path only) ---
    EXPR_GAIN = {
        0: 10.0,   # EyeBlinkLeft
        7: 10.0,   # EyeBlinkRight
        1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0,    # EyeLook L
        8: 3.0, 9: 3.0, 10: 3.0, 11: 3.0,  # EyeLook R
    }
    for ch_idx, gain in EXPR_GAIN.items():
        arkit[:, ch_idx] = np.clip(arkit[:, ch_idx] * gain, 0.0, 1.0)

    curves = arkit_to_mh_curves(arkit)

    # Head rotation curves for the AnimNode path (backward compat)
    curves["HeadYaw"]   = arkit[:, 52].tolist()
    curves["HeadPitch"] = arkit[:, 53].tolist()
    curves["HeadRoll"]  = arkit[:, 54].tolist()
    curves["HeadControlSwitch"] = [1.0] * n_frames

    # Raw ARKit channels for the LiveLink path — all 61 channels with
    # original Apple ARKit names, UNAMPLIFIED. The MetaHuman LiveLink
    # AnimBP handles its own scaling (head rot × 50 etc.).
    arkit_raw = {}
    for i, name in enumerate(CHANNEL_NAMES):
        arkit_raw[name] = arkit_raw_clean[:, i].tolist()

    gen_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "prompt": prompt,
        "duration": n_frames / fps,
        "fps": fps,
        "n_frames": n_frames,
        "generation_ms": round(gen_ms, 2),
        "curves": curves,
        "arkit_raw": arkit_raw,
    }


# =============================================================================
# FastAPI app
# =============================================================================

app = FastAPI(title="TextToFace sidecar", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ready" if _model is not None else "loading",
        device=str(_device) if _device is not None else None,
        max_frames=_max_frames if _model is not None else None,
        model_params_m=round(_param_count_m, 2) if _model is not None else None,
        n_mh_curves=len(MH_CURVE_NAMES),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        result = generate_curves(req.prompt, req.frames, req.fps, req.guidance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")
    print(
        f"[server] generated '{req.prompt[:40]}' "
        f"({result['n_frames']}f, {result['generation_ms']}ms)"
    )
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="checkpoints/best.pt")
    parser.add_argument("--stats-dir", default="stats")
    parser.add_argument("--host",      default="127.0.0.1")
    parser.add_argument("--port",      type=int, default=8765)
    args = parser.parse_args()

    load_model(args.ckpt, args.stats_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
