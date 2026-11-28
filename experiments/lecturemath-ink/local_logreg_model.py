"""Frozen-machine inference wrapper for the LectureMath local candidate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from local_logreg import features


class LocalLogRegModel:
    def __init__(self, weights: Path, config: Path):
        self.weights_path = Path(weights); self.config_path = Path(config)
        self.config = json.loads(self.config_path.read_text())
        self.model_version = self.config["model_version"]
        self.threshold = float(self.config["threshold"])
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("frozen threshold must be in (0,1)")
        with np.load(self.weights_path) as data:
            self.w = np.asarray(data["w"], dtype=np.float32)
            self.b = np.float32(np.asarray(data["b"], dtype=np.float32).reshape(-1)[0])
        if self.w.ndim != 1 or not np.isfinite(self.w).all() or not np.isfinite(self.b):
            raise ValueError("invalid checkpoint")

    def predict_proba(self, bgr, known=None):
        f = features(bgr)
        if known is not None:
            if known.shape != f.shape[:2]: raise ValueError("known shape differs")
            f = f.copy(); f[known == 0] = 0
        logits = f @ self.w + self.b
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)

    def predict(self, bgr, known=None, threshold=None):
        p = self.predict_proba(bgr, known)
        t = self.threshold if threshold is None else float(threshold)
        if t != self.threshold: raise ValueError("override of frozen threshold is forbidden")
        return p, (p >= np.float32(self.threshold)).astype(np.uint8) * 255


def load_checkpoint(run_dir: str | Path) -> LocalLogRegModel:
    root = Path(run_dir)
    return LocalLogRegModel(root / "weights.npz", root / "config.json")
