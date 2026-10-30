"""Evaluate paired class-ID PNG masks without model/framework dependencies."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def read_ids(path):
    ids = [s.strip() for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in ids):
        raise ValueError(f"Invalid sample ID in {path}; use stems without extensions or paths")
    return ids


def index_files(folder, extensions):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Missing directory: {folder}")
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_dir():
            raise ValueError(f"Nested directories are not supported: {path}")
        if path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file: {path}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", path.stem):
            raise ValueError(f"Invalid sample ID: {path.name}")
        if path.stem in result:
            raise ValueError(f"Duplicate filename stem: {path.stem}")
        result[path.stem] = path
    return result


def read_mask(path, num_classes, ignore_index=None):
    with Image.open(path) as image:
        arr = np.array(image)
    if arr.ndim != 2 or arr.dtype.kind not in "ui":
        raise ValueError(f"Expected a 2-D integer ID mask: {path}")
    valid = (arr >= 0) & (arr < num_classes)
    if ignore_index is not None:
        valid |= arr == ignore_index
    if not valid.all():
        raise ValueError(f"Invalid class IDs {np.unique(arr[~valid]).tolist()} in {path}")
    return arr


def validate_options(num_classes, ignore_index):
    if not 1 <= num_classes <= 256:
        raise ValueError("num-classes must be between 1 and 256")
    if 0 <= ignore_index < num_classes:
        raise ValueError("ignore-index must not overlap valid class IDs")


def evaluate(args):
    validate_options(args.num_classes, args.ignore_index)
    gt = index_files(args.gt, {".png"})
    pred = index_files(args.pred, {".png"})
    ids = read_ids(args.split) if args.split else sorted(gt)
    if not ids:
        raise ValueError("No samples selected")
    if not args.split and set(gt) != set(pred):
        raise ValueError("Ground-truth and prediction sample IDs differ")
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    ignored = 0
    for sample in ids:
        if sample not in gt or sample not in pred:
            raise ValueError(f"Missing ground truth or prediction: {sample}")
        target = read_mask(gt[sample], args.num_classes, args.ignore_index)
        output = read_mask(pred[sample], args.num_classes)
        if target.shape != output.shape:
            raise ValueError(f"Mask dimensions differ: {sample}")
        valid = target != args.ignore_index
        ignored += int((~valid).sum())
        encoded = target[valid].astype(np.int64) * args.num_classes + output[valid].astype(np.int64)
        confusion += np.bincount(encoded, minlength=args.num_classes ** 2).reshape(confusion.shape)
    count = int(confusion.sum())
    if count == 0:
        raise ValueError("Selected masks contain no valid ground-truth pixels")
    tp = confusion.diagonal().astype(float)
    rows = confusion.sum(axis=1)
    union = rows + confusion.sum(axis=0) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union != 0)
    recall = np.divide(tp, rows, out=np.full_like(tp, np.nan), where=rows != 0)
    percent = lambda values: [None if np.isnan(v) else float(v * 100) for v in values]
    return dict(samples=len(ids), valid_pixels=count, ignored_pixels=ignored,
                units="percent", class_ids=list(range(args.num_classes)),
                per_class_iou=percent(iou), per_class_recall=percent(recall),
                mIoU=float(np.nanmean(iou) * 100), mAcc=float(np.nanmean(recall) * 100),
                aAcc=float(tp.sum() / count * 100), confusion_matrix=confusion.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--ignore-index", type=int, default=255)
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args), indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
