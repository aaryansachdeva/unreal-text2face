"""
Training loop for TextToFace (v5).

v5 changes (over v4):
    - FIX: channel weights now applied to TRAINING loss (v4 bug: only applied
      to val, so the model never actually learned boosted blink/head channels)
    - ADD: EMA (exponential moving average) of model weights — smooths training
      noise, typically 1-3% better at inference. Borrowed from Express4D/MDM.
    - ADD: blended L1+L2 loss (MSE penalizes peak errors harder, pushes the
      model to commit to strong expressions instead of hedging)
    - Batch size 32 -> 64 (we have 10GB VRAM headroom — bigger batch = more
      stable gradients)
    - LR 2e-4 -> 1e-4 (bigger model + bigger batch benefits from lower LR,
      matches Express4D baseline)
    - Adam beta2 0.99 -> 0.999 (matches Express4D, smoother second moments)
    - Recalibrated channel weights (less extreme than v4's since they actually
      affect training now): blink=3x, eye=2x, head=3x, eye_rot=2x, rest=1x
"""
from __future__ import annotations

import argparse
import copy
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
# Per-channel loss weights
# ----------------------------------------------------------------------

def build_channel_weights(n_channels: int = 61) -> torch.Tensor:
    """Per-channel weight tensor applied to ALL loss terms (train + val)."""
    w = torch.ones(n_channels, dtype=torch.float32)

    # Blinks: sparse events the model tends to suppress
    w[0] = 3.0   # EyeBlinkLeft
    w[7] = 3.0   # EyeBlinkRight

    # Eye expression channels
    for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]:
        w[i] = 2.0

    # Head rotation — boosted for visible neck movement
    w[52] = 5.0  # HeadYaw
    w[53] = 5.0  # HeadPitch
    w[54] = 5.0  # HeadRoll

    # Eye gaze rotations
    for i in [55, 56, 57, 58, 59, 60]:
        w[i] = 2.0

    return w


# ----------------------------------------------------------------------
# Loss functions (blended L1 + L2, with per-channel weights)
# ----------------------------------------------------------------------

def masked_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                channel_weights: torch.Tensor, l2_ratio: float = 0.5) -> torch.Tensor:
    """Blended L1 + L2 reconstruction loss over masked positions.
    pred, target: [B, T, C]
    mask: [B, T] (True = real frame)
    channel_weights: [C]
    l2_ratio: blend ratio (0=pure L1, 1=pure L2, 0.5=equal blend)
    """
    diff = pred - target                              # [B, T, C]
    l1 = diff.abs()
    l2 = diff.square()
    blended = (1 - l2_ratio) * l1 + l2_ratio * l2    # [B, T, C]
    weighted = blended * channel_weights.unsqueeze(0).unsqueeze(0)
    mask_f = mask.unsqueeze(-1).float()               # [B, T, 1]
    return (weighted * mask_f).sum() / (mask_f.sum() * pred.shape[-1] + 1e-8)


def masked_velocity(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    channel_weights: torch.Tensor) -> torch.Tensor:
    """Weighted L1 on frame-to-frame differences."""
    pred_vel = pred[:, 1:] - pred[:, :-1]
    tgt_vel = target[:, 1:] - target[:, :-1]
    diff = (pred_vel - tgt_vel).abs()
    diff = diff * channel_weights.unsqueeze(0).unsqueeze(0)
    valid = (mask[:, 1:] & mask[:, :-1]).float().unsqueeze(-1)
    return (diff * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)


def masked_acceleration(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                        channel_weights: torch.Tensor) -> torch.Tensor:
    """Weighted L1 on second differences."""
    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    tgt_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    diff = (pred_acc - tgt_acc).abs()
    diff = diff * channel_weights.unsqueeze(0).unsqueeze(0)
    valid = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).float().unsqueeze(-1)
    return (diff * valid).sum() / (valid.sum() * pred.shape[-1] + 1e-8)


# ----------------------------------------------------------------------
# EMA (exponential moving average)
# ----------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update EMA parameters: ema = decay * ema + (1 - decay) * current."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


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

    # --- EMA ---
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    print(f"EMA enabled (decay={args.ema_decay})")

    # --- Optimizer + schedule ---
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        betas=(0.9, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(500, max(50, total_steps // 20))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # --- Channel weights (applied to BOTH train and val) ---
    channel_weights = build_channel_weights().to(device)
    cw_desc = "blink=3x, eye=2x, head=5x, eye_rot=2x, rest=1x"
    print(f"Channel weights: {cw_desc}")

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
    print(f"Loss: {1-args.l2_ratio:.0%} L1 + {args.l2_ratio:.0%} L2")
    print(
        f"Loss weights: recon=1.0  velocity={args.lambda_vel}  "
        f"acceleration={args.lambda_acc}"
    )

    for epoch in range(args.epochs):
        # ===== train =====
        model.train()
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
                recon = masked_loss(pred, motion, mask, channel_weights,
                                    l2_ratio=args.l2_ratio)
                vel = masked_velocity(pred, motion, mask, channel_weights)
                acc = masked_acceleration(pred, motion, mask, channel_weights)
                loss = recon + args.lambda_vel * vel + args.lambda_acc * acc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update
            update_ema(ema_model, model, args.ema_decay)

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

        # ===== validation (using EMA model) =====
        ema_model.eval()
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
                    pred = ema_model(texts, n_frames=args.max_frames)
                    recon = masked_loss(pred, motion, mask, channel_weights,
                                        l2_ratio=args.l2_ratio)
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
            "ema_model": ema_model.state_dict(),
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
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--num-workers", type=int, default=8)

    # Augmentation
    parser.add_argument("--mirror-prob", type=float, default=0.5,
                        help="Smart mirror: swaps L/R expressions only, preserves head rotation")
    parser.add_argument("--word-drop-prob", type=float, default=0.1)
    parser.add_argument("--subwindow-prob", type=float, default=0.3)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.1)

    # Model
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.25)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-vel", type=float, default=1.0)
    parser.add_argument("--lambda-acc", type=float, default=0.25)
    parser.add_argument("--l2-ratio", type=float, default=0.0,
                        help="Pure L1 (v8). 0=pure L1, 0.5=L1+L2 blend")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
