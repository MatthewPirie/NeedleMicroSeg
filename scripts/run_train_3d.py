# scripts/run_train_3d.py

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.data.dataset_3d import NeedleDataset3D
from src.data.extractors_3d import get_extractor
from src.data.augmentations_3d import build_train_transforms_3d, build_val_transforms_3d
from src.models.unet_3d import build_unet_3d
from src.train.losses import CompoundBCEDiceLoss
from src.train.trainer_3d import train_one_epoch_3d, validate_one_epoch_3d
from src.utils.normalization import get_normalizer
from src.utils.helper_functions import (
    _get_git_commit,
    _make_run_dir,
    _save_checkpoint,
    _set_seed,
)
from src.utils.visualization_3d import save_val_panels_3d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Core config
    parser.add_argument("--train_config", type=str, required=True)

    # Data paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask",
    )
    parser.add_argument(
        "--splits_file",
        type=str,
        default="/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask/optimum_patient_splits.json",
    )

    # Run management
    parser.add_argument("--runs_dir", type=str, default="runs_3d")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--resume_ckpt", type=str, default="")

    # Checkpointing controls
    parser.add_argument("--save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", action="store_false", dest="save_last")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")
    parser.add_argument("--save_every", type=int, default=0)

    # Model + training settings
    parser.add_argument("--model_variant", type=str, default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # AMP toggle
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------
    # Reproducibility + device
    # ------------------------
    _set_seed(int(args.seed))
    torch.set_num_threads(1)

    run_dir = _make_run_dir(args.runs_dir, args.run_name)
    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))
    metrics_path = run_dir / "metrics.jsonl"

    print(f"Run dir: {run_dir}", flush=True)

    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(_get_git_commit() + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pin_memory = torch.cuda.is_available() and int(args.num_workers) > 0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(torch.cuda.is_available() and bool(args.amp)),
    )

    # ------------------------
    # Load YAML config
    # ------------------------
    train_config_path = Path(args.train_config)
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config not found: {train_config_path}")

    with open(train_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    extractor_cfg = cfg.get("extractor", {})
    normalization_cfg = cfg.get("normalization", {})
    augmentations_cfg = cfg.get("augmentations", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})

    # ------------------------
    # Parse config values
    # ------------------------
    extractor_name = str(extractor_cfg.get("name", "temporal_window"))
    extractor_kwargs = dict(extractor_cfg.get("kwargs", {}))

    normalization_name = normalization_cfg.get("name", None)

    enabled_augs = list(augmentations_cfg.get("enabled", []))
    aug_kwargs = dict(augmentations_cfg.get("kwargs", {}))

    split_id = str(data_cfg.get("split_id", "kfold_cv_fold-0"))

    batch_size = int(train_cfg.get("batch_size", 8))
    epochs = int(train_cfg.get("epochs", 50))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 250))
    log_every = int(train_cfg.get("log_every", 50))

    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    momentum = float(train_cfg.get("momentum", 0.99))

    scheduler_name = str(train_cfg.get("lr_scheduler", "none")).lower()
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    poly_power = float(train_cfg.get("poly_power", 0.9))

    w_bce = float(loss_cfg.get("w_bce", 1.0))
    w_dice = float(loss_cfg.get("w_dice", 1.0))
    batch_dice = bool(loss_cfg.get("batch_dice", True))

    print(f"Train config: {train_config_path}", flush=True)
    print(f"Split ID: {split_id}", flush=True)
    print(f"Extractor: {extractor_name}", flush=True)

    # ------------------------
    # Build preprocessing
    # ------------------------
    extractor_fn = get_extractor(extractor_name)
    normalizer = get_normalizer(normalization_name)

    out_hw = tuple(extractor_kwargs.get("out_hw", (256, 256)))

    train_tf = build_train_transforms_3d(
        spatial_hw=out_hw,
        enabled_augs=enabled_augs,
        **aug_kwargs,
    )
    val_tf = build_val_transforms_3d()

    # ------------------------
    # Datasets + loaders
    # ------------------------
    train_ds = NeedleDataset3D(
        root=args.data_root,
        split="train",
        split_id=split_id,
        splits_file=args.splits_file,
        extractor_fn=extractor_fn,
        extractor_kwargs=extractor_kwargs,
        normalizer=normalizer,
        transform=train_tf,
        return_metadata=True,
    )

    val_ds = NeedleDataset3D(
        root=args.data_root,
        split="val",
        split_id=split_id,
        splits_file=args.splits_file,
        extractor_fn=extractor_fn,
        extractor_kwargs=extractor_kwargs,
        normalizer=normalizer,
        transform=val_tf,
        return_metadata=True,
    )

    dl_kwargs = {}
    if int(args.num_workers) > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=True,
        **dl_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
        **dl_kwargs,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)

    # ------------------------
    # Model + loss
    # ------------------------
    model, model_meta = build_unet_3d(
        in_channels=1,
        out_channels=1,
        variant=args.model_variant,
    )
    model = model.to(device)

    criterion = CompoundBCEDiceLoss(
        w_bce=w_bce,
        w_dice=w_dice,
        batch_dice=batch_dice,
    ).to(device)

    # ------------------------
    # Optimizer
    # ------------------------
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # ------------------------
    # LR scheduler
    # ------------------------
    total_steps = max(epochs * steps_per_epoch, 1)
    scheduler = None

    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )
    elif scheduler_name == "polynomial":
        def _poly(step: int) -> float:
            s = min(step, total_steps)
            return (1.0 - s / total_steps) ** poly_power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly)
    elif scheduler_name != "none":
        raise ValueError(f"Unknown lr_scheduler: {scheduler_name}")

    # ------------------------
    # Save config snapshot
    # ------------------------
    run_config = {
        "cli_args": vars(args),
        "train_config": cfg,
        "model": model_meta,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # ------------------------
    # Resume checkpoint
    # ------------------------
    start_epoch = 1
    best_val_dice = float("-inf")

    if args.resume_ckpt:
        ckpt = torch.load(Path(args.resume_ckpt), map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_dice = float(ckpt.get("extra", {}).get("best_val_dice", float("-inf")))

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_3d(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            log_every=log_every,
            pin_memory=pin_memory,
            scheduler=scheduler,
            scaler=scaler,
            amp=bool(args.amp),
        )
        t1 = time.time()

        val_metrics = validate_one_epoch_3d(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        t2 = time.time()

        writer.add_scalar("train/total_loss", train_metrics["train_total_loss"], epoch)
        writer.add_scalar("val/total_loss", val_metrics["val_total_loss"], epoch)
        writer.add_scalar("val/hard_dice_score", val_metrics["val_hard_dice_score"], epoch)
        writer.flush()

        print(
            f"[Epoch {epoch}] "
            f"time_sec={(t2 - t0):.1f} "
            f"train_loss={train_metrics['train_total_loss']:.4f} "
            f"val_loss={val_metrics['val_total_loss']:.4f} "
            f"val_dice={val_metrics['val_hard_dice_score']:.4f}",
            flush=True,
        )

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "time_sec": round(t2 - t0, 1),
                **train_metrics,
                **val_metrics,
            }) + "\n")

        if args.save_last:
            _save_checkpoint(
                run_dir / "checkpoint_last.pt",
                model,
                optimizer,
                epoch,
                extra={"best_val_dice": best_val_dice},
            )

        if args.save_best and val_metrics["val_hard_dice_score"] > best_val_dice:
            best_val_dice = val_metrics["val_hard_dice_score"]
            _save_checkpoint(
                run_dir / "checkpoint_best.pt",
                model,
                optimizer,
                epoch,
                extra={"best_val_dice": best_val_dice},
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                run_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                extra={"best_val_dice": best_val_dice},
            )

    # ------------------------
    # Post-training visualization
    # ------------------------
    best_ckpt = run_dir / "checkpoint_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded best checkpoint for validation panels.", flush=True)
    else:
        print("Best checkpoint not found, using current in-memory model for validation panels.", flush=True)

    save_val_panels_3d(
        model=model,
        val_ds=val_ds,
        device=device,
        save_dir=run_dir / "val_panels",
        n_samples=3,
        context_radius=2,
        prefix="final",
    )

    print("Training completed.", flush=True)
    writer.close()


if __name__ == "__main__":
    main()