# src/data/dataset_2d.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NeedleDataset2D(Dataset):

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
            splits_file = Path(splits_file) if splits_file is not None else (self.root / "splits.json")

            with open(splits_file) as f:
                splits_data = json.load(f)

            if split_id is None:
                split_id = next(iter(splits_data))

            self.split_id = split_id
            case_ids = set(splits_data[split_id][split])

        # Collect all h5 files
        all_h5 = sorted(self.data_dir.glob("*.h5"))

        # Filter files based on patient ID in the split
        if case_ids is not None:
            filtered = []
            for p in all_h5:
                patient_id = "-".join(p.stem.split("-")[:2])
                if patient_id in case_ids:
                    filtered.append(p)
            all_h5 = filtered

        self.samples = all_h5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        h5_path = self.samples[idx]
        json_path = h5_path.with_suffix(".json")

        # Load annotated frame and mask
        with h5py.File(h5_path, "r") as f:
            annotation_index = int(f["needle_mask_annotation_index"][()])
            image = f["cine"][annotation_index].astype(np.float32)
            mask = f["needle_mask"][:].astype(np.float32)

        # Metadata (optional)
        with open(json_path) as f:
            metadata = json.load(f)

        # Optional normalization
        if self.normalizer is not None:
            image = self.normalizer(image)

        # Optional extractor (resize / patch logic later)
        if self.extractor_fn is not None:
            out = self.extractor_fn(
                image=image,
                mask=mask,
                metadata=metadata,
                **self.extractor_kwargs,
            )
            image = out["image"]
            mask = out["mask"]

        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)

        sample = {
            "image": image[None, ...],   # (1, H, W)
            "mask": mask[None, ...],     # (1, H, W)
        }

        if self.return_metadata:
            sample["metadata"] = metadata
            sample["cine_id"] = metadata.get("cine_id", h5_path.stem)
            sample["needle_mask_annotation_index"] = annotation_index
            sample["h5_path"] = str(h5_path)

        # Augmentations
        if self.transform is not None:
            sample = self.transform(sample)

        sample["image"] = torch.from_numpy(sample["image"]).float()
        sample["mask"] = torch.from_numpy(sample["mask"]).float()

        return sample