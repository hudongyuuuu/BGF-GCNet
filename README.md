# BGF-GCNet

Dataset-format documentation and evaluation utilities for BGF-GCNet, a
boundary-guided fusion network for real-time tactile-paving segmentation.

The current release includes dataset validation and mask-based evaluation tools.
Model code, training configurations, pretrained weights and datasets are coming soon.

## Contents

```text
BGF-GCNet-release/
├── README.md
├── requirements.txt
├── requirements-tools.txt
├── THIRD_PARTY_NOTICES.md
├── docs/
│   └── dataset_format.md
├── tools/
│   ├── evaluate_masks.py
│   └── check_dataset.py
└── .gitignore
```

## Environment

The model was trained and evaluated in the following Linux environment.

| Component | Version |
|---|---|
| Python | 3.8.19 |
| PyTorch | 1.12.1+cu113 |
| torchvision | 0.13.1+cu113 |
| MMCV | 2.0.0 |
| MMEngine | 0.10.2 |
| MMSegmentation | 1.2.2 |
| NumPy | 1.24.4 |
| Pillow | 10.4.0 |
| OpenCV | 4.13.0.92 |
| SciPy | 1.10.1 |
| Matplotlib | 3.7.5 |
| CUDA (PyTorch / MMCV build) | 11.3 |
| cuDNN | 8.3.2 |

`requirements.txt` records the main research dependencies at their installed
versions. It uses the official PyTorch and OpenMMLab wheel sources for the
matching CUDA builds. Use Python 3.8.19 for this dependency set:

```bash
conda create -n bgf-gcnet python=3.8.19 -y
conda activate bgf-gcnet
python -m pip install -r requirements.txt
```

### Standalone tools

To run only the two mask utilities, install the smaller dependency file in a
separate Python 3.8 environment:

```bash
python -m pip install -r requirements-tools.txt
```

The standalone tools require only NumPy and Pillow and run on the CPU.

## Data checking

Prepare your own data according to [the format specification](docs/dataset_format.md).

```bash
python tools/check_dataset.py --root data/example
```

Checks image/label pairing, readable files, matching dimensions, integer label
IDs, split membership, duplicate manifest entries and cross-split overlap by ID.
Files in the dataset must match manifest membership exactly. Exit status is 0
on success and 1 on validation failure. This is a structural check: it does not
detect visually similar images or assess annotation quality.

## Mask evaluation

```bash
python tools/evaluate_masks.py --gt data/example/labels --pred data/example/predictions --split data/example/splits/test.txt
```

Predictions and ground truth must be single-channel class-ID PNGs with identical
dimensions and corresponding pixels. RGB visualizations and logits are not accepted.
Without `--split`, all ground-truth PNGs are evaluated and prediction membership
must match exactly. With `--split`, only listed pairs are evaluated.

Metrics are accumulated over all valid pixels before computing per-class IoU,
mIoU, mean class recall (mAcc) and overall pixel accuracy (aAcc). All classes,
including background, contribute when defined; undefined values are printed as
`null` and excluded from means. Results on stdout are JSON with percentages and
a ground-truth-row/prediction-column confusion matrix. Ground-truth ID 255 is
ignored; predictions must always use valid class IDs, including ignored areas.
Use `--num-classes` and `--ignore-index` for another ID convention.

The evaluator uses the supplied class-ID masks without resizing.
Model inference, prediction resizing and FPS measurement are separate steps.

## Dependencies

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream dependency links.
