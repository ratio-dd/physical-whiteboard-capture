"""Re-select a frozen threshold on a manifest's fixed validation set.

This evaluator reads only validation pairs from a caller-provided manifest and
does not alter a checkpoint. It is intentionally separate from local_mlp.py's
training report so a preregistered, fixed comparator can be reproduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from local_mlp_model import load_checkpoint


def _rows(spec: dict) -> list[dict]:
    if "pairs" in spec:
        return [r for r in spec["pairs"] if r.get("split") == "validation"]
    return list(spec.get("val", []))


def evaluate(model, root: Path, rows: list[dict], grid: int = 1) -> tuple[np.ndarray, np.ndarray]:
    probabilities, truth = [], []
    for row in rows:
        image = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(row)
        p = model.predict_proba(image)
        if p.shape != mask.shape:
            raise ValueError(f"shape mismatch for {row['image']}")
        # A fixed grid changes iteration order only; all pixels stay at source
        # resolution and no random crop or resize is introduced.
        for yy in np.array_split(np.arange(mask.shape[0]), max(1, grid)):
            for xx in np.array_split(np.arange(mask.shape[1]), max(1, grid)):
                if len(yy) and len(xx):
                    probabilities.append(p[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1].ravel())
                    truth.append((mask[yy[0]:yy[-1] + 1, xx[0]:xx[-1] + 1] == 0).ravel())
    if not probabilities:
        raise ValueError("manifest has no validation pairs")
    return np.concatenate(probabilities), np.concatenate(truth)


def choose_threshold(probability: np.ndarray, truth: np.ndarray, fpr: float = .01) -> dict:
    negative = np.sort(probability[~truth])[::-1]
    allowed = int(np.floor(fpr * len(negative)))
    threshold = float(np.nextafter(negative[allowed - 1], np.inf)) if allowed else float(np.nextafter(negative[0], np.inf))
    predicted = probability >= threshold
    tp = int(np.count_nonzero(predicted & truth)); fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth)); tn = int(np.count_nonzero(~predicted & ~truth))
    return {"threshold": threshold, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "false_positive_rate": fp / max(1, fp + tn),
            "positive_pixels": int(truth.sum()), "predicted_pixels": int(predicted.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("checkpoint", type=Path, help="local_mlp.py output directory")
    ap.add_argument("--grid", type=int, default=1, help="fixed row/column grid; 1 means one full-resolution tile")
    ap.add_argument("--fpr", type=float, default=.01)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.grid < 1 or not 0 < args.fpr < 1:
        ap.error("grid must be positive and fpr must be in (0, 1)")
    spec = json.loads(args.manifest.read_text())
    rows = _rows(spec)
    model = load_checkpoint(args.checkpoint)
    probability, truth = evaluate(model, args.manifest.parent, rows, args.grid)
    report = {"model_version": model.model_version, "validation_pairs": len(rows),
              "grid": args.grid, "selection_fpr": args.fpr,
              "selection": choose_threshold(probability, truth, args.fpr)}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
