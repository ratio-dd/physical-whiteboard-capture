"""Stable research inference wrapper; threshold is deliberately mandatory."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from model import TinyFCN

def load_model(path: str | Path, threshold: float):
    if not 0 < threshold < 1: raise ValueError('threshold must be in (0,1)')
    return _Loaded(TinyFCN.load(path), float(threshold))

class _Loaded:
    def __init__(self, model, threshold): self.model=model; self.threshold=threshold
    def predict(self,bgr: np.ndarray, known: np.ndarray | None = None):
        return self.model.predict(bgr,known,self.threshold)
