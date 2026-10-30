# Dataset format

Input layout, class IDs and split-file format for the dataset checker and
mask evaluator.

```text
data/example/
├── images/
│   ├── sample001.jpg
│   └── sample002.png
├── labels/
│   ├── sample001.png
│   └── sample002.png
├── predictions/             # optional, for evaluate_masks.py
│   ├── sample001.png
│   └── sample002.png
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## IDs and pairing

- Directories are flat. Each sample ID is a filename stem without extension.
- IDs use only ASCII letters, digits, underscore and hyphen.
- Source images may use `.jpg`, `.jpeg` or `.png` (case-insensitive).
- Labels and predictions use `.png`. Image and mask stems must match uniquely.
- Ground truth has the same width and height as its source image.
- Labels must be two-dimensional integer class IDs. Grayscale (`L`), indexed
  palette (`P`) and integer PNG masks are accepted. For palette masks the
  stored index is the class ID; the displayed color is irrelevant.
- RGB/RGBA colored segmentation figures are not ID masks; decode them using
  the actual dataset's documented palette before evaluation.

## Default tactile-paving convention

| ID | Meaning |
|---|---|
| 0 | Background |
| 1 | Directional / strip tactile paving |
| 2 | Warning / dot tactile paving |
| 255 | Ignored ground-truth pixels |

The tools do not infer a class from an RGB color. They do not subtract one from
IDs or ignore background. Labels may contain only class IDs and the configured
ignore ID. Predictions may contain only class IDs; 255 is not an abstention
prediction. Fully ignored ground-truth samples are reported by the checker.

## Split manifests

Each text file is UTF-8 (a UTF-8 BOM is accepted), one sample ID per line:

```text
sample001
sample002
```

Blank lines are ignored; extensions, paths, comments and repeated IDs are not
accepted. The same ID cannot appear in multiple splits. Every image and label
must be assigned to exactly one of train/val/test. Empty split files are allowed
for structural checking but an evaluation manifest must select at least one ID.
This check does not prove the absence of duplicate image content under different IDs.

## Metric definitions

For accumulated confusion matrix C (rows: ground truth, columns: prediction):

- IoU_i = C_ii / (row_i + column_i - C_ii).
- Recall_i = C_ii / row_i; mAcc is the mean of defined class recalls.
- mIoU is the mean of defined class IoUs.
- aAcc = trace(C) / sum(C).

Zero-denominator class metrics are undefined (`null`), not zero. A class absent
from both masks is excluded from mIoU; an absent ground-truth class with false
positive predictions has IoU zero. If no valid pixels exist, evaluation fails.
Images are not averaged equally: valid pixels are accumulated across the dataset.

