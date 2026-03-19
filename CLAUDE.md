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

There are no `__init__.py` files outside of `src/data/extractors_2d/` — add repo root to `PYTHONPATH` or use `sys.path.insert` as in `scripts/run_train_2d.py`.

### Data pipeline (`src/data/`)
- `dataset_2d.py` — **primary dataset class** `NeedleDataset2D`; loads single needle-fire frames as `float32`, applies normalizer then extractor, returns `{"image": (1,H,W), "mask": (1,H,W), ...}` as tensors
- `needle_fire_needle_mask_dataset.py` — legacy dataset returning uint8 numpy arrays; kept for EDA/notebooks
- `extractors_2d/` — registry pattern for spatial extraction/resizing functions:
  - `__init__.py` provides `get_extractor(name)` and `available_extractors()`
  - `full_frame_resize.py` — `extract(image, mask, metadata, out_hw)` bilinear resize for image, nearest for mask
- `augmentations_2d.py` — `build_train_transforms_2d(spatial_hw, enabled_augs, **prob_overrides)` and `build_val_transforms_2d()` (returns `None`); `enabled_augs` is a list from `{"flip","affine","elastic","noise","blur","shift","scale","contrast"}`

### Models (`src/models/`)
- `unet_2d.py` — wraps MONAI `UNet`; `build_unet_2d(in_channels, out_channels, variant)` returns `(model, meta_dict)`; variants: `"small"` (5 levels) and `"base"` (8 levels), both use instance norm

### Training (`src/train/`)
- `losses.py` — `CompoundBCEDiceLoss` (BCE + soft Dice), `soft_dice_loss`, `hard_dice_score` (note: `loses.py` is a stale duplicate and can be deleted)
- `trainer_2d.py` — `train_one_epoch_2d` and `validate_one_epoch_2d`; uses fixed `steps_per_epoch` (nnU-Net style, not full-epoch); supports AMP via `GradScaler`

### Utils (`src/utils/`)
- `normalization.py` — `get_normalizer(name)` factory; options: `"zscore_per_image"`, `"minmax_per_image"`, `None`/`"none"`
- `helper_functions.py` — `_set_seed`, `_make_run_dir`, `_save_checkpoint`, `_get_git_commit`
- `visualization.py` — `save_val_panel(model, val_ds, device, save_path, n_samples=3)`; samples `n_samples` random frames, runs inference, saves a PNG grid of (image | GT mask | predicted mask | yellow overlay) with per-row Dice scores

## Training Script

```bash
cd /path/to/NeedleMicroSeg
python scripts/run_train_2d.py \
    --train_config configs/full_frame_resize.yaml \
    [--model_variant small|base] \
    [--run_name my_run] \
    [--resume_ckpt runs_2d/my_run/checkpoint_last.pt]
```

Outputs go to `runs_2d/<run_name>/`: `checkpoint_last.pt`, `checkpoint_best.pt` (by val Dice), `metrics.jsonl`, `config.json`, `git_commit.txt`, and `tb/` (TensorBoard logs).

## Config Format (`configs/full_frame_resize.yaml`)

```yaml
extractor:
  name: full_frame_resize      # key in extractors_2d registry
  kwargs:
    out_hw: [256, 256]

normalization:
  name: zscore_per_image       # or minmax_per_image / none

augmentations:
  enabled: []                  # list of augmentation names
  kwargs: {}

data:
  split_id: kfold_cv_fold-0

train:
  batch_size: 8
  epochs: 100
  lr: 0.0001
  weight_decay: 0.00001
  steps_per_epoch: 250         # optimizer steps per epoch
  optimizer: adam              # or sgd
  lr_scheduler: none           # or cosine / polynomial

loss:
  w_bce: 1.0
  w_dice: 1.0
  batch_dice: true
```

## Known Issues / Pending Work

- `src/train/loses.py` is a stale mis-spelled duplicate of `losses.py` and can be deleted.
- `scripts/slurm/` is empty; SLURM job scripts not yet written.

## Environment

Runs on a SLURM cluster. No `requirements.txt` or `pyproject.toml`. Dependencies: `torch`, `h5py`, `numpy`, `monai`, `tensorboard`, `pyyaml`.
