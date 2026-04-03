# scripts/run_infer_2d.py

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.dataset_2d import NeedleDataset2D
from src.data.extractors_2d import get_extractor
from src.models.unet_2d import build_unet_2d
from src.utils.normalization import get_normalizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=Path, 
                        default = "/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/runs_2d/full_frame_resize_test_2567085/checkpoint_best.pt")
    parser.add_argument("--dataset_root", type=str,
                        default="/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--split_id", type=str, default="leave-one-center-out_center-PU")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--model_variant", type=str, default=None, choices=["small", "base"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_png", type=str, 
                        default="/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/runs_2d/full_frame_resize_test_2567085/val_panels/inference_img.png")
    return parser.parse_args()


def load_run_config(ckpt_path: Path):
    run_dir = ckpt_path.parent
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.json next to checkpoint: {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    print()
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cfg = load_run_config(ckpt_path)

    train_cfg = run_cfg["train_config"]

    extractor_name = train_cfg["extractor"]["name"]
    extractor_kwargs = train_cfg["extractor"].get("kwargs", {})
    normalization_name = train_cfg["normalization"]["name"]
    split_id = args.split_id if args.split_id is not None else train_cfg["data"]["split_id"]

    model_variant = args.model_variant
    if model_variant is None:
        model_variant = run_cfg["model"]["model_variant"]
        
    normalizer = get_normalizer(normalization_name)
    extractor = get_extractor(extractor_name)

    ds = NeedleDataset2D(
        root=args.dataset_root,
        split=args.split,
        split_id=split_id,
        splits_file="/project/6106383/shared/2026-03-11_pwilson_needle-fire-needle-mask/optimum_patient_splits.json",
        normalizer=normalizer,
        extractor_fn=extractor,
        extractor_kwargs=extractor_kwargs,
        transform=None,
    )

    sample = ds[args.sample_idx]

    model, meta = build_unet_2d(
        in_channels=1,
        out_channels=1,
        variant=model_variant,
    )
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x = sample["image"].unsqueeze(0).to(device)   # (1,1,H,W)
        logits = model(x)
        probs = torch.sigmoid(logits)
        pred = (probs >= args.threshold).float()

    image = sample["image"].squeeze().cpu().numpy()
    gt = sample["mask"].squeeze().cpu().numpy()
    prob = probs.squeeze().cpu().numpy()
    pred_mask = pred.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT")
    axes[2].imshow(prob, cmap="gray")
    axes[2].set_title("Prob")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(pred_mask, cmap="autumn", alpha=0.4)
    axes[3].set_title("Pred overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if args.save_png:
        plt.savefig(args.save_png, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {args.save_png}")
    else:
        plt.show()

    print(f"split={args.split}")
    print(f"split_id={split_id}")
    print(f"sample_idx={args.sample_idx}")


if __name__ == "__main__":
    main()