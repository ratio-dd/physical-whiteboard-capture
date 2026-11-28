# LectureMath ink experiment

This directory contains a research-only comparison of local-background
logistic regression, a local-feature MLP, and a TinyFCN for segmenting
handwritten ink in corrected BGR lecture frames. The selected candidate for
this experiment is `local_background_mlp_v1` (11 input features, 24 hidden
units, 1 output). None of these models is imported by the whiteboard product
runtime, and this directory is not a release of LectureMath data.

## Data and rights

The official project page is [kdavila/lecturemath](https://github.com/kdavila/lecturemath/).
The [ACCESS2021 release README](https://github.com/kdavila/lecturemath/blob/master/ACCESS2021_release/README.md)
links to the official [LectureMath v1.1 annotation archive](https://www.dropbox.com/scl/fi/1s658riqvyfzhefttn8fi/IEEE_access_data_release.zip?rlkey=wcyk22jgtelzr9e2u5buwxbcu&dl=0)
and the [FCN-LectureNet model archive](https://www.dropbox.com/scl/fi/j4odjoft6sz6foe7c8qao/LectureNet_model_BIN_V34_final.dat?rlkey=903f7hronoc7ao360ykxbakwb&dl=0).

The archive README describes annotations and download contents, while the
repository also contains software and paper material. Those descriptions do
not by themselves settle every right attached to a video, frame, annotation,
or derived model. Obtain the data yourself, keep the upstream README and
notices, and use only material for which you have the necessary permission.
The archive is not treated here as either an unconditional redistribution
license or a blanket prohibition on every use. This repository does not copy
the archive, videos, frames, annotations, or its weights.

## Local manifest convention and construction

Keep an extracted archive and manifest outside Git (for example under the
local `data/` directory) and isolate each lecture in its own directory. The
selected trainer consumes one JSON manifest (not JSONL), with `pairs` and a
lecture-level `split`:

```json
{"mask":{"positive_values":[0]},"pairs":[{"lecture_id":"lecture-A","image":"LectureMath_lecture-A/keyframes/000123.png","mask":"LectureMath_lecture-A/binary/000123.png","split":"train"},{"lecture_id":"lecture-B","image":"LectureMath_lecture-B/keyframes/000456.png","mask":"LectureMath_lecture-B/binary/000456.png","split":"validation"}]}
```

The included `build_manifest.py` creates this shape from an extracted official
annotation archive without copying any data into Git. It expects each lecture
directory to contain matching `keyframes/` and `binary/` files:

```sh
.venv/bin/python experiments/lecturemath-ink/build_manifest.py \
  /path/to/extracted/IEEEAccess_Annotations \
  /path/to/extracted/manifest.json --validation-fraction .2 --seed 17
```

Place the output manifest beside the archive root's parent, as in the command
above, so its relative paths resolve. The split is deterministic at lecture
granularity; inspect the generated `validation_lectures` before training. For
a preregistered split, edit only the manifest's split fields, preserving the
lecture-level separation. In `pairs` form, `split` must be `train` or
`validation`; the current adapter uses `0` as ink. `mask.positive_values` is
required and is never guessed. A mask value listed in `positive_values` is ink;
this experiment commonly uses `0` for ink and non-zero for background.

`lecture_id` is the isolation key: do not mix frames or masks across lectures,
and keep a lecture in only one split. `image`, `mask`, and optional `ignore`
paths are relative to the manifest's directory. The trainer refuses missing
mask polarity and asserts that train and validation lecture IDs do not overlap.

Keep source URLs, retrieval date, checksums, and permission notes in the local
manifest. The trainer refuses missing mask polarity and asserts that train and
validation lecture IDs do not overlap.

## Model and reproducible entry points

`local_mlp.py` is the selected model's training entry point and
`local_mlp_model.py` is its frozen inference wrapper. From the repository root:

```sh
.venv/bin/python experiments/lecturemath-ink/local_mlp.py \
  data/datasets/lecturemath-ink/manifest.json \
  --out data/datasets/lecturemath-ink/local-mlp --seed 17
```

The output directory is ignored by the repository and contains `weights.npz`,
`report.json`, and `config.json`. `report.json` records the validation
operating point and metrics. The threshold is selected on validation by
allowing at most 1% false-positive rate among negative pixels, then taking the
highest recall operating point. The frozen threshold in `config.json` is the
only threshold accepted by the inference wrapper.

### Fixed validation comparator

The published comparison threshold for the current experiment is
`0.9955437183380127`. It was selected independently on the fixed 15-image
validation subset from the 104-pair official manifest, at original image
scale, using the same weak-ink definition and a fixed grid. This is not
necessarily the threshold in a newly trained `local_mlp.py` report: that
report selects its own threshold from whatever manifest it receives.

To reproduce the comparator, obtain the same official files and recreate the
same non-random manifest (including its exact lecture/frame selection), then
run the checkpoint through the public evaluator:

```sh
.venv/bin/python experiments/lecturemath-ink/select_threshold.py \
  /path/to/fixed-manifest.json \
  data/datasets/lecturemath-ink/local-mlp \
  --grid 4 --fpr .01 \
  --output data/datasets/lecturemath-ink/local-mlp/fixed-validation.json
```

The evaluator reads validation pairs only, preserves original pixels (the grid
controls deterministic iteration, not resizing or random crops), and reports
the selected threshold, precision, recall, FPR, and pixel counts. A manifest
generated with the ordinary random lecture split is useful for training smoke
tests, but does not reproduce the fixed comparator or any metric from it.

For a model-only smoke test, with no dataset dependency:

```python
from pathlib import Path
import cv2
from model import TinyFCN

model = TinyFCN.init(seed=17)
frame = cv2.imread("one-corrected-frame.png", cv2.IMREAD_COLOR)
probability, mask = model.predict(frame, threshold=0.5)
model.save(Path("model.npz"))
```

For MLP inference, load the run directory and use its frozen threshold:

```python
from local_mlp_model import load_checkpoint
model = load_checkpoint("data/datasets/lecturemath-ink/local-mlp")
probability, mask = model.predict(frame, known=known)
```

The MLP uses the 11-dimensional local-background feature map from
`local_logreg.py`, standardizes it from training pixels, and applies one ReLU
hidden layer of width 24 plus a sigmoid output. The logistic model and TinyFCN
remain comparison models; do not merge their checkpoints or metrics with the
selected MLP report. No LectureMath score is claimed here until a run's
`report.json` is inspected and its split, seed, checkpoint hash, threshold,
and source notices are recorded.

All claims about quality must be made on a lecture-held-out split and recorded
with the manifest, seed, checkpoint hash, threshold, and source notices. Do
not report a number until that experiment has actually been run.

## Vector verification

The model output is a pixel mask, not a vector drawing. The offline
`vector_pipeline.py` makes the auditable conversion explicit:

```python
from vector_pipeline import extract_hybrid
result = extract_hybrid(model, frame, known, threshold=chosen_threshold)
svg_paths = result.vector_paths
```

It requires known pixels and source-image support, deduplicates against the
classic extractor, and returns editable closed outlines tagged as
`model_candidate`. Compare the resulting SVG against the mask overlay at the
same frame size. Treat the SVG as measured geometry only: it does not prove
stroke order, semantic math structure, or hidden motion.

## Privacy boundary

This repository does not include the dataset or any user blind-test material.
Keep locally retained data and run outputs under the intended local data root,
verify their ignore status with `git check-ignore`, and stage public source
files explicitly rather than using a bulk staging command. Follow the
upstream archive notices and the permissions attached to the particular data
you obtained.
