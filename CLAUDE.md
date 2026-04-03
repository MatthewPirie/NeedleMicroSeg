# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeedleMicroSeg is a medical imaging project for segmenting needles in ultrasound cine sequences (video clips). It targets needle-fire frames — the specific frame where a needle is annotated — paired with binary segmentation masks stored in HDF5 format.

## Data

- **Dataset root:** `/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask`
- **Splits file used by training:** `<root>/optimum_patient_splits.json` (not `splits.json`)
- **Format:** Each sample is a pair of files in `<root>/data/`:
  - `.h5` containing `cine` (T, H, W) uint8, `needle_mask` (H, W) uint8, and `needle_mask_annotation_index` scalar
  - `.json` with metadata including `cine_id`
- **Splits:** Keys are split IDs (e.g. `"kfold_cv_fold-0"`), each with `"train"` and `"val"` lists of case IDs
- Case IDs are the first two dash-separated tokens of the filename (e.g. `case-001` from `case-001-frame-005.h5`)

## Code Structure

There are no `__init__.py` files outside of `src/data/extractors_2d/` and `src/data/extractors_3d/` — add repo root to `PYTHONPATH` or use `sys.path.insert` as in training scripts.

Both 2D and 3D pipelines share the same loss functions, normalizers, and helper utilities. The key architectural difference is that the 3D model takes `(B, 1, Z, H, W)` inputs, predicts `(B, 1, Z, H, W)` logits, and loss/metrics are computed only on the center slice `logits[:, :, center_index, :, :]` against the 2D mask `(B, 1, H, W)`.

### Data pipeline (`src/data/`)
- `dataset_2d.py` — `NeedleDataset2D`; loads single needle-fire frames as `float32`, applies normalizer then extractor, returns `{"image": (1,H,W), "mask": (1,H,W), ...}` as tensors
- `dataset_3d.py` — `NeedleDataset3D`; loads full cine then extracts a temporal window via extractor; internally expands mask to `(1,Z,H,W)` for joint spatial augmentation, then collapses back to `(1,H,W)` at the center index before returning
- `needle_fire_needle_mask_dataset.py` — legacy dataset returning uint8 numpy arrays; kept for EDA/notebooks
- `extractors_2d/` — registry: `full_frame_resize` (bilinear resize for image, nearest for mask)
- `extractors_3d/` — registry: `temporal_window` — extracts `z_window` frames centered on `annotation_index` with edge replication, then resizes spatially; returns `{"image": (Z,H,W), "mask": (H,W), "center_index": int}`
- `augmentations_2d.py` — `build_train_transforms_2d(spatial_hw, enabled_augs, **prob_overrides)`; `enabled_augs` from `{"flip","affine","elastic","noise","blur","shift","scale","contrast"}`
- `augmentations_3d.py` — `build_train_transforms_3d(spatial_hw, enabled_augs, **prob_overrides)`; spatial augs apply only to H/W axes (not Z); `enabled_augs` from `{"flip","translate","rotate_scale","noise","blur","shift","scale","contrast"}`

### Models (`src/models/`)
- `unet_2d.py` — `build_unet_2d(in_channels, out_channels, variant)` wraps MONAI `UNet`; variants `"small"` (5 levels) and `"base"` (8 levels)
- `unet_3d.py` — `build_unet_3d(in_channels, out_channels, variant)` uses MONAI `DynUNet` (falls back to `UNet` if unavailable); anisotropic kernels/strides — early layers use `(1,3,3)` kernels and `(1,2,2)` strides to preserve the Z dimension initially; variants `"small"` (5 levels) and `"base"` (7 levels)

### Training (`src/train/`)
- `losses.py` — `CompoundBCEDiceLoss` (BCE + soft Dice), `soft_dice_loss`, `hard_dice_score`
- `trainer_2d.py` — `train_one_epoch_2d` / `validate_one_epoch_2d`; fixed `steps_per_epoch` (nnU-Net style); AMP via `GradScaler`
- `trainer_3d.py` — `train_one_epoch_3d` / `validate_one_epoch_3d`; same fixed-steps pattern; center-slice extraction happens inside the trainer

### Utils (`src/utils/`)
- `normalization.py` — `get_normalizer(name)`; options: `"zscore_per_image"`, `"minmax_per_image"`, `None`/`"none"`
- `helper_functions.py` — `_set_seed`, `_make_run_dir`, `_save_checkpoint`, `_get_git_commit`
- `visualization.py` — `save_val_panel` for 2D; saves PNG grid of (image | GT mask | predicted mask | overlay)
- `visualization_3d.py` — `save_val_panels_3d(model, val_ds, device, save_dir, n_samples, context_radius, prefix)`; saves one PNG per sample showing context frames around the annotated slice

## Training Scripts

**2D:**
```bash
python scripts/run_train_2d.py \
    --train_config configs/train_2d/full_frame_resize.yaml \
    [--model_variant small|base] \
    [--run_name my_run] \
    [--resume_ckpt runs_2d/my_run/checkpoint_last.pt]
```

**3D:**
```bash
python scripts/run_train_3d.py \
    --train_config configs/train_3d/temporal_window_resized.yaml \
    [--model_variant small|base] \
    [--run_name my_run] \
    [--resume_ckpt runs_3d/my_run/checkpoint_last.pt]
```

Both output to `runs_2d/` or `runs_3d/` respectively: `checkpoint_last.pt`, `checkpoint_best.pt` (by val hard Dice), `metrics.jsonl`, `config.json`, `git_commit.txt`, `tb/` (TensorBoard).

**Inference (2D):**
```bash
python scripts/run_infer_2d.py \
    --ckpt_path runs_2d/<run>/checkpoint_best.pt \
    --split val --split_id kfold_cv_fold-0 \
    --sample_idx 0 --save_png out.png
```
Reads `config.json` next to the checkpoint to reconstruct extractor/normalizer settings.

## Config Format

Configs live in `configs/train_2d/` and `configs/train_3d/`. The 3D config adds `z_window` to the extractor kwargs and supports extra scheduler params (`min_lr`, `poly_power`, `momentum`):

```yaml
extractor:
  name: temporal_window        # 3D; use full_frame_resize for 2D
  kwargs:
    z_window: 10               # 3D only
    out_hw: [256, 256]

normalization:
  name: zscore_per_image

augmentations:
  enabled: []
  kwargs: {}

data:
  split_id: kfold_cv_fold-0

train:
  batch_size: 4
  epochs: 150
  lr: 0.0001
  weight_decay: 0.00001
  steps_per_epoch: 250
  optimizer: adam              # or sgd
  lr_scheduler: cosine         # or polynomial / none
  min_lr: 1e-6
  poly_power: 0.9
  momentum: 0.99               # sgd only

loss:
  w_bce: 1.0
  w_dice: 1.5
  batch_dice: true
```

## Environment

Runs on a SLURM cluster. No `requirements.txt` or `pyproject.toml`. Dependencies: `torch`, `h5py`, `numpy`, `monai`, `tensorboard`, `pyyaml`.
