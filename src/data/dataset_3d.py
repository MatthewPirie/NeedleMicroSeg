# src/data/dataset_3d.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NeedleDataset3D(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        split_id: str | None = None,
        splits_file: str | Path | None = None,
        extractor_fn=None,
        extractor_kwargs: dict | None = None,
        normalizer=None,
        transform=None,
        return_metadata: bool = True,
    ):
        self.root = Path(root)
        self.data_dir = self.root / "data"

        self.extractor_fn = extractor_fn
        self.extractor_kwargs = extractor_kwargs or {}

        self.normalizer = normalizer
        self.transform = transform
        self.return_metadata = return_metadata

        self.split = split
        self.split_id = None

        case_ids = None

        # Load split definitions if train/val filtering is requested
        if split is not None:
            splits_file = (
                Path(splits_file)
                if splits_file is not None
                else (self.root / "optimum_patient_splits.json")
            )

            with open(splits_file) as f:
                splits_data = json.load(f)

            if split_id is None:
                split_id = next(iter(splits_data))

            self.split_id = split_id
            case_ids = set(splits_data[split_id][split])

        # Collect all h5 files
        all_h5 = sorted(self.data_dir.glob("*.h5"))

        # Filter files based on case ID in the split
        if case_ids is not None:
            filtered = []
            for p in all_h5:
                case_id = "-".join(p.stem.split("-")[:2])
                if case_id in case_ids:
                    filtered.append(p)
            all_h5 = filtered

        self.samples = all_h5

    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h5_path = self.samples[idx]
        json_path = h5_path.with_suffix(".json")

        # Load full cine, annotation index, and 2D mask
        with h5py.File(h5_path, "r") as f:
            annotation_index = int(f["needle_mask_annotation_index"][()])
            cine = f["cine"][:].astype(np.float32)         # (T, H, W)
            mask = f["needle_mask"][:].astype(np.float32)  # (H, W)

        # Metadata
        with open(json_path) as f:
            metadata = json.load(f)

        # Optional normalization
        if self.normalizer is not None:
            cine = self.normalizer(cine)

        # Extract temporal window
        if self.extractor_fn is not None:
            out = self.extractor_fn(
                cine=cine,
                mask=mask,
                annotation_index=annotation_index,
                metadata=metadata,
                **self.extractor_kwargs,
            )
            image = out["image"]   # (Z, H, W)
            mask = out["mask"]     # (H, W)
            center_index = out.get("center_index", image.shape[0] // 2)
        else:
            image = cine
            mask = mask
            center_index = annotation_index

        image = np.ascontiguousarray(image, dtype=np.float32)  # (Z, H, W)
        mask = np.ascontiguousarray(mask, dtype=np.float32)    # (H, W)

        Z = image.shape[0]

        # Expand mask across Z so MONAI spatial transforms can operate jointly
        mask_3d = np.repeat(mask[None, ...], Z, axis=0)  # (Z, H, W)

        sample = {
            "image": image[None, ...],      # (1, Z, H, W)
            "mask": mask_3d[None, ...],     # (1, Z, H, W)
        }

        if self.return_metadata:
            sample["metadata"] = metadata
            sample["cine_id"] = metadata.get("cine_id", h5_path.stem)
            sample["needle_mask_annotation_index"] = annotation_index
            sample["center_index"] = center_index
            sample["h5_path"] = str(h5_path)

        # Augmentations
        if self.transform is not None:
            sample = self.transform(sample)

        # Collapse mask back to the annotated slice only
        sample["mask"] = sample["mask"][:, center_index, :, :]  # (1, H, W)

        sample["image"] = torch.as_tensor(sample["image"]).float()  # (1, Z, H, W)
        sample["mask"] = torch.as_tensor(sample["mask"]).float()    # (1, H, W)

        return sample

