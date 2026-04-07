"""
Training loop for TextToFace (v4).

Masked L1 reconstruction + masked velocity loss + masked acceleration loss,
AdamW with cosine LR schedule + warmup, mixed precision, tensorboard logging,
best-checkpoint saving, EARLY STOPPING.

Typical invocation:
    python src/train.py

Key differences vs v1:
    - Max epochs 500 -> 120 (v1 hit its best val at epoch ~33 then
      regressed for 467 epochs)
    - Early stopping: halt if val_loss fails to improve for 25 epochs
    - Weight decay 1e-4 -> 1e-2 (standard transformer value)
    - Dropout 0.1 -> 0.2
    - Velocity loss weight 0.5 -> 1.0
    - Acceleration loss added with weight 0.25
    - DataLoader worker_init_fn to properly seed numpy per worker
      (otherwise all workers would draw identical augmentation)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Express4DDataset, collate_fn, worker_init_fn
from model import TextToFace


# ----------------------------------------------------------------------
# Per-channel loss weights — boost blinks, eye gaze, head rotation
# ----------------------------------------------------------------------
# Channel indices (from dataset.py CHANNEL_NAMES):
#   0: EyeBlinkLeft       7: EyeBlinkRight        (blinks)
#   1-6: EyeLook/Squint/WideLeft                   (eye expression L)
#   8-13: EyeLook/Squint/WideRight                 (eye expression R)
#   52-54: HeadYaw/Pitch/Roll                       (head rotation)
#   55-60: L/R EyeYaw/Pitch/Roll                    (eye gaze rotation)

def build_channel_weights(n_channels: int = 61) -> torch.Tensor:
    """Per-channel weight tensor. Higher = model pays more attention."""
    w = torch.ones(n_channels, dtype=torch.float32)

    # Blinks: very underrepresented, boost hard
    w[0] = 5.0   # EyeBlinkLeft
    w[7] = 5.0   # EyeBlinkRight

    # Eye expression channels (squint, wide, look directions)
    for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        w[i] = 3.0

    # Head rotation
    w[52] = 4.0  # HeadYaw
    w[53] = 4.0  # HeadPitch
    w[54] = 4.0  # HeadRoll

    # Eye gaze rotations
    for i in [55, 56, 57, 58, 59, 60]:
        w[i] = 3.0

    return w


# ----------------------------------------------------------------------
# Loss functions (with per-channel weights)
# ----------------------------------------------------------------------

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              channel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted L1 loss over masked positions.
    pred, target: [B, T, C]
    mask: [B, T] (True = real frame)
    channel_weights: [C] (optional, default uniform)
    """
    diff = (pred - target).abs()                  # [B, T, C]
    if channel_weights is not None:
        diff = diff * channel_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    mask_expanded = mask.unsqueeze(-1).float()    # [B, T, 1]
    loss = (diff * mask_expanded).sum() / (mask_expanded.sum() * pred.shape[-1] + 1e-8)
    return loss


def masked_velocity(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    channel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted L1 on frame-to-frame differences."""
    pred_vel = pred[:, 1:] - pred[:, :-1]
    tgt_vel = target[:, 1:] - target[:, :-1]
    diff = (pred_vel - tgt_vel).abs()
    if channel_weights is not None:
        diff = diff * channel_weights.unsqueeze(0).unsqueeze(0)

    valid = (mask[:, 1:] & mask[:, :-1]).float()
    valid = valid.unsqueeze(-1)

    loss = (diff * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)
    return loss


def masked_acceleration(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        channel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted L1 on second differences."""
    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    tgt_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    diff = (pred_acc - tgt_acc).abs()
    if channel_weights is not None:
        diff = diff * channel_weights.unsqueeze(0).unsqueeze(0)

    valid = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float()
    valid = valid.unsqueeze(-1)

    loss = (diff * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)
    return loss


# ----------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ----------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Reproducibility knobs (best-effort; full determinism is not a goal
    # here since we benefit from the stochasticity of augmentation).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data ---
    print("\nLoading datasets...")
    train_ds = Express4DDataset(
        data_root=args.data_root,
        stats_dir=args.stats_dir,
        split="train",
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        mirror_prob=args.mirror_prob,
        word_drop_prob=args.word_drop_prob,
        subwindow_prob=args.subwindow_prob,
        cfg_drop_prob=args.cfg_drop_prob,
    )
    val_ds = Express4DDataset(
        data_root=args.data_root,
        stats_dir=args.stats_dir,
        split="test",
        target_fps=args.target_fps,
        max_frames=args.max_frames,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    # --- Model ---
    print("\nBuilding model...")
    model = TextToFace(
        max_frames=args.max_frames,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    n_trainable = model.count_trainable()
    print(f"Trainable params: {n_trainable / 1e6:.2f}M")

    # --- Optimizer + schedule ---
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, max(50, total_steps // 20))  # 5% warmup, capped
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # --- Channel weights ---
    channel_weights = build_channel_weights().to(device)
    print(f"Channel weights: blink=5x, eye=3x, head=4x, eye_rot=3x, rest=1x")

    # --- Mixed precision ---
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Tensorboard ---
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Training loop ---
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    step = 0
    start_time = time.time()

    print(
        f"\nStarting training: up to {args.epochs} epochs, "
        f"{steps_per_epoch} steps/epoch, {total_steps} max steps"
    )
    print(f"Warmup steps: {warmup_steps}")
    print(f"Early stopping patience: {args.patience} epochs")
    print(f"Mixed precision: {use_amp}")
    print(
        f"Loss weights: recon=1.0  velocity={args.lambda_vel}  "
        f"acceleration={args.lambda_acc}"
    )

    for epoch in range(args.epochs):
        # ===== train =====
        model.train()
        # CLIP must stay in eval mode (no dropout / BN stats updates)
        model.clip_text.eval()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_vel = 0.0
        epoch_acc = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} [train]")
        for batch in pbar:
            motion = batch["motion"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            texts = batch["text"]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(texts, n_frames=args.max_frames)
                recon = masked_l1(pred, motion, mask)
                vel = masked_velocity(pred, motion, mask)
                acc = masked_acceleration(pred, motion, mask)
                loss = recon + args.lambda_vel * vel + args.lambda_acc * acc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_vel += vel.item()
            epoch_acc += acc.item()

            if step % args.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/recon", recon.item(), step)
                writer.add_scalar("train/vel", vel.item(), step)
                writer.add_scalar("train/acc", acc.item(), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon.item():.4f}",
                "vel": f"{vel.item():.4f}",
                "acc": f"{acc.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        n_train_batches = len(train_loader)
        epoch_loss /= n_train_batches
        epoch_recon /= n_train_batches
        epoch_vel /= n_train_batches
        epoch_acc /= n_train_batches

        # ===== validation =====
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_vel = 0.0
        val_acc = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"epoch {epoch+1}/{args.epochs} [val]")
            for batch in pbar:
                motion = batch["motion"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                texts = batch["text"]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(texts, n_frames=args.max_frames)
                    recon = masked_l1(pred, motion, mask, channel_weights)
                    vel = masked_velocity(pred, motion, mask, channel_weights)
                    acc = masked_acceleration(pred, motion, mask, channel_weights)
                    loss = recon + args.lambda_vel * vel + args.lambda_acc * acc

                val_loss += loss.item()
                val_recon += recon.item()
                val_vel += vel.item()
                val_acc += acc.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        n_val_batches = len(val_loader)
        val_loss /= n_val_batches
        val_recon /= n_val_batches
        val_vel /= n_val_batches
        val_acc /= n_val_batches

        elapsed = time.time() - start_time
        print(
            f"\nepoch {epoch+1}/{args.epochs}  "
            f"train_loss={epoch_loss:.4f} (recon={epoch_recon:.4f} "
            f"vel={epoch_vel:.4f} acc={epoch_acc:.4f})  "
            f"val_loss={val_loss:.4f} (recon={val_recon:.4f} "
            f"vel={val_vel:.4f} acc={val_acc:.4f})  "
            f"elapsed={elapsed/60:.1f}min"
        )

        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("val/recon", val_recon, step)
        writer.add_scalar("val/vel", val_vel, step)
        writer.add_scalar("val/acc", val_acc, step)

        # ===== checkpointing + early stopping =====
        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "val_recon": val_recon,
            "val_vel": val_vel,
            "val_acc": val_acc,
            "args": vars(args),
        }
        torch.save(ckpt_payload, Path(args.ckpt_dir) / "latest.pt")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(ckpt_payload, Path(args.ckpt_dir) / "best.pt")
            print(f"  ** new best val_loss={val_loss:.4f}, saved best.pt")
        else:
            patience_counter += 1
            print(
                f"  no improvement ({patience_counter}/{args.patience}) "
                f"since epoch {best_epoch+1} (best={best_val:.4f})"
            )
            if patience_counter >= args.patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1}. "
                    f"Best val_loss={best_val:.4f} at epoch {best_epoch+1}."
                )
                break

    writer.close()
    print(
        f"\nTraining complete. Best val_loss={best_val:.4f} at epoch {best_epoch+1}."
    )
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--data-root", type=str, default="ExpressData")
    parser.add_argument("--stats-dir", type=str, default="stats")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")

    # Data
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=240,
                        help="240 = 8 seconds at 30fps")
    parser.add_argument("--num-workers", type=int, default=2)

    # Augmentation (train split only)
    parser.add_argument("--mirror-prob", type=float, default=0.0,
                        help="Probability of horizontal mirror per training sample (0=off)")
    parser.add_argument("--word-drop-prob", type=float, default=0.1,
                        help="Probability of dropping each non-protected caption word")
    parser.add_argument("--subwindow-prob", type=float, default=0.3,
                        help="Probability of variable sub-window crop per training sample")
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1,
                        help="Probability of dropping entire caption for classifier-free guidance")

    # Model (v3-large: 6 layers, 768d, 12 heads, 2048 ff = ~48.5M trainable)
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max epochs. Training usually stops earlier via --patience.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-vel", type=float, default=1.0,
                        help="Weight on velocity (smoothness) loss")
    parser.add_argument("--lambda-acc", type=float, default=0.25,
                        help="Weight on acceleration (jitter) loss")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log-every", type=int, default=20,
                        help="Tensorboard log every N training steps")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
