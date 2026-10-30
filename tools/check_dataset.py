"""Check a flat image/mask dataset and train/val/test manifests."""
import argparse
import json
import sys

import numpy as np
from PIL import Image

from evaluate_masks import index_files, read_ids, read_mask, validate_options


def check(args):
    validate_options(args.num_classes, args.ignore_index)
    root = args.root
    images = index_files(root / "images", {".jpg", ".jpeg", ".png"})
    labels = index_files(root / "labels", {".png"})
    if not images:
        raise ValueError("No source images found")
    if set(images) != set(labels):
        raise ValueError("Image and label sample IDs differ")
    assigned = set()
    counts = {}
    for split in ("train", "val", "test"):
        ids = read_ids(root / "splits" / f"{split}.txt")
        overlap = assigned.intersection(ids)
        if overlap:
            raise ValueError(f"IDs occur in multiple splits: {sorted(overlap)}")
        assigned.update(ids)
        counts[split] = len(ids)
    if assigned != set(images):
        raise ValueError("Manifest membership must match all image/label IDs exactly")
    histogram = np.zeros(args.num_classes, dtype=np.int64)
    ignored = 0
    fully_ignored = []
    for sample, path in images.items():
        with Image.open(path) as image:
            image.load()
            size = image.size
        mask = read_mask(labels[sample], args.num_classes, args.ignore_index)
        if size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Image/label dimensions differ: {sample}")
        valid = mask != args.ignore_index
        ignored += int((~valid).sum())
        histogram += np.bincount(mask[valid].astype(np.int64), minlength=args.num_classes)
        if not valid.any():
            fully_ignored.append(sample)
    return dict(status="ok", samples=len(images), splits=counts,
                class_pixel_counts=histogram.tolist(), ignored_pixels=ignored,
                fully_ignored_samples=fully_ignored,
                note="Structural validation only; image-content duplication is not checked.")


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args), indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
