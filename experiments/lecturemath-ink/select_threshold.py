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


def evaluate(model, root: Path, rows: list[dict], grid: int = 1, window: int = 128) -> tuple[np.ndarray, np.ndarray]:
    probabilities, truth = [], []
    for row in rows:
        image = cv2.imread(str(root / row["image"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(row)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"shape mismatch for {row['image']}")
        if image.shape[0] < window or image.shape[1] < window:
            raise ValueError(f"image is smaller than window for {row['image']}")
        # Tile centers sit at the centers of equal grid cells and are clamped
        # to the source bounds. This is the fixed comparator protocol, not a
        # resize or a random crop.
        for gy in range(grid):
            for gx in range(grid):
                cx = int(round((gx + .5) * image.shape[1] / grid))
                cy = int(round((gy + .5) * image.shape[0] / grid))
                x = max(0, min(image.shape[1] - window, cx - window // 2))
                y = max(0, min(image.shape[0] - window, cy - window // 2))
                tile = image[y:y + window, x:x + window]
                p = model.predict_proba(tile)
                probabilities.append(p.ravel())
                truth.append((mask[y:y + window, x:x + window] == 0).ravel())
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
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "false_positive_rate": fp / max(1, fp + tn),
            "positive_pixels": int(truth.sum()), "predicted_pixels": int(predicted.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("checkpoint", type=Path, help="local_mlp.py output directory")
    ap.add_argument("--grid", type=int, default=1, help="fixed row/column window grid; 4 means 16 windows per image")
    ap.add_argument("--window", type=int, default=128, help="original-resolution square window size")
    ap.add_argument("--fpr", type=float, default=.01)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.grid < 1 or args.window < 1 or not 0 < args.fpr < 1:
        ap.error("grid/window must be positive and fpr must be in (0, 1)")
    spec = json.loads(args.manifest.read_text())
    rows = _rows(spec)
    model = load_checkpoint(args.checkpoint)
    probability, truth = evaluate(model, args.manifest.parent, rows, args.grid, args.window)
    report = {"model_version": model.model_version, "validation_pairs": len(rows),
              "grid": args.grid, "window": args.window, "selection_fpr": args.fpr,
              "selection": choose_threshold(probability, truth, args.fpr)}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
