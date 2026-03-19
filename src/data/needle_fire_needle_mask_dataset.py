import json
from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


DATASET_ROOT = "/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask"

class NeedleFireNeedleMaskDataset(Dataset):
    """Dataset of annotated needle-fire frames with needle segmentation masks.

    Each item comes from an HDF5 file containing:
      - cine: (T, H, W) uint8 — full ultrasound cine clip
      - needle_mask: (H, W) uint8 — binary needle mask at the annotated frame
      - needle_mask_annotation_index: scalar — frame index of the annotation

    By default __getitem__ returns only the annotated needle-fire frame.
    Set return_full_cine=True to get the entire clip instead.
    """

    def __init__(
        self,
        root=None,
        split=None,
        split_id=None,
        splits_file=None,
        transform=None,
        return_full_cine=False,
    ):
        """
        Args:
            root: path to the dataset root (contains a ``data/`` subdirectory
                  with ``.h5`` / ``.json`` file pairs).  Defaults to the
                  canonical location on the cluster.
            split: e.g. ``"train"`` or ``"val"``.  Requires a splits file.
            split_id: key into the splits file (e.g. ``"kfold_cv_fold-0"``).
                  Defaults to the first key in the file.
            splits_file: path to a JSON splits file.  Defaults to
                  ``<root>/splits.json`` when ``split`` is given.
            transform: optional callable applied to each output dict.
            return_full_cine: if True, ``image`` in the output is the full
                (T, H, W) cine array; otherwise it is the single needle-fire
                frame (H, W).
        """
        root = Path(root or DATASET_ROOT)
        self.data_dir = root / "data"
        self.transform = transform
        self.return_full_cine = return_full_cine

        case_ids = None
        if split is not None:
            splits_file = splits_file or (root / "splits.json")
            with open(splits_file) as f:
                splits_data = json.load(f)
            if split_id is None:
                split_id = next(iter(splits_data))
            self.split_id = split_id
            case_ids = set(splits_data[split_id][split])

        all_h5 = sorted(self.data_dir.glob("*.h5"))
        if case_ids is not None:
            all_h5 = [p for p in all_h5 if "-".join(p.stem.split("-")[:2]) in case_ids]
        self.samples = all_h5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path = self.samples[idx]
        json_path = h5_path.with_suffix(".json")

        with h5py.File(h5_path, "r") as f:
            cine = f["cine"][:]                                    # (T, H, W) uint8
            needle_mask = f["needle_mask"][:]                      # (H, W) uint8
            annotation_index = int(f["needle_mask_annotation_index"][()])

        with open(json_path) as f:
            metadata = json.load(f)

        image = cine if self.return_full_cine else cine[annotation_index]

        out = {
            "image": image,
            "needle_mask": needle_mask,
            "needle_mask_annotation_index": annotation_index,
            "metadata": metadata,
            "cine_id": metadata["cine_id"],
        }

        if self.transform is not None:
            out = self.transform(out)

        return out

