from pathlib import Path
import json
import random
from collections import defaultdict

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_DIR = Path("/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask/data")
OUTPUT_FILE = Path("/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/dataset/multicenter_custom_split.json")

# --------------------------------------------------
# Settings
# --------------------------------------------------
VAL_FRAC = 0.2
SEED = 42
SPLIT_NAME = "multicenter_custom"

random.seed(SEED)

# --------------------------------------------------
# Collect unique patient IDs from .h5 files
# --------------------------------------------------
all_h5 = sorted(DATA_DIR.glob("*.h5"))

all_case_ids = set()
for p in all_h5:
    case_id = "-".join(p.stem.split("-")[:2])   # e.g. OL-004
    all_case_ids.add(case_id)

all_case_ids = sorted(all_case_ids)

# --------------------------------------------------
# Group case IDs by centre
# --------------------------------------------------
center_to_cases = defaultdict(list)

for case_id in all_case_ids:
    center = case_id.split("-")[0]   # e.g. OL
    center_to_cases[center].append(case_id)

# --------------------------------------------------
# Split within each centre
# --------------------------------------------------
train_cases = []
val_cases = []

for center, cases in sorted(center_to_cases.items()):
    cases = sorted(cases)
    random.shuffle(cases)

    n_total = len(cases)
    n_val = max(1, round(n_total * VAL_FRAC)) if n_total > 1 else 0

    val_center_cases = cases[:n_val]
    train_center_cases = cases[n_val:]

    train_cases.extend(train_center_cases)
    val_cases.extend(val_center_cases)

    print(f"{center}: total={n_total}, train={len(train_center_cases)}, val={len(val_center_cases)}")

# Final sort for cleanliness
train_cases = sorted(train_cases)
val_cases = sorted(val_cases)

# Safety check
overlap = set(train_cases) & set(val_cases)
assert len(overlap) == 0, f"Overlap found: {sorted(overlap)}"

# --------------------------------------------------
# Save JSON
# --------------------------------------------------
new_splits = {
    SPLIT_NAME: {
        "train": train_cases,
        "val": val_cases
    }
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(new_splits, f, indent=4)

print()
print(f"Saved to: {OUTPUT_FILE}")
print(f"Total train cases: {len(train_cases)}")
print(f"Total val cases:   {len(val_cases)}")
print(f"Total unique cases:{len(train_cases) + len(val_cases)}")