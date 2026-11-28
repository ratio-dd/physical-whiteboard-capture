"""Conservative temporal rescue for weak-ink candidates.

This is an offline, opt-in prototype.  It remembers accepted *pixels* only;
it never caches a detector bounding box or invents pixels in unknown areas.
Rescue requires the pixel to remain known, a nearby source-colour/ridge cue,
and agreement with the accepted frame's local BGR value.  Call ``reset`` after
an erase or board change.
"""
from dataclasses import dataclass
import cv2
import numpy as np


def load_canonical_score(path, shape=None):
    """Load one frozen YOLO score map without changing its numeric values."""
    score = np.asarray(np.load(path), dtype=np.float32)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("canonical score must be a finite HxW array")
    if shape is not None and tuple(score.shape) != tuple(shape):
        raise ValueError(f"score shape {score.shape} differs from expected {shape}")
    return score


def canonical_raw_mask(score, known, threshold=0.7972984313964844):
    """Apply the frozen score threshold and known gate, with no source dilation."""
    score = np.asarray(score, dtype=np.float32); known = np.asarray(known)
    if score.shape != known.shape or known.dtype != np.uint8:
        raise ValueError("score/known shapes or dtype differ")
    return ((score >= np.float32(threshold)) & (known != 0)).astype(np.uint8) * 255


@dataclass(frozen=True)
class MemoryConfig:
    threshold: float = 0.7972984313964844
    colour_tolerance: float = 18.0
    radius: int = 1


class EvidenceMemory:
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config
        self._accepted = None
        self._reference = None

    def reset(self):
        self._accepted = None
        self._reference = None

    def update(self, bgr, known, score, source_support):
        """Return a uint8 mask; ``source_support`` is an independently derived mask.

        ``score`` is the canonical float32 YOLO score map.  On the first call,
        only score pixels at/above the fixed threshold are accepted.  Later,
        previously accepted pixels may survive a score drop only if source
        support remains nearby and the local BGR value has not changed beyond
        the configured tolerance.  All outputs are strictly gated by known.
        """
        bgr = np.asarray(bgr); known = np.asarray(known); score = np.asarray(score)
        support = np.asarray(source_support)
        h, w = known.shape
        if bgr.shape != (h, w, 3) or bgr.dtype != np.uint8:
            raise ValueError("bgr must be HxWx3 uint8")
        if known.dtype != np.uint8 or score.shape != (h, w) or support.shape != (h, w):
            raise ValueError("known, score and source_support shapes/dtypes are invalid")
        if self.config.radius < 0 or self.config.colour_tolerance < 0:
            raise ValueError("invalid memory configuration")
        yolo = (score >= np.float32(self.config.threshold)) & (known != 0)
        src = support.astype(bool)
        if self._accepted is None:
            self._accepted = yolo.copy()
            self._reference = bgr.copy()
        else:
            near = cv2.dilate(src.astype(np.uint8), np.ones((2*self.config.radius+1,)*2, np.uint8)) != 0
            diff = np.max(np.abs(bgr.astype(np.int16) - self._reference.astype(np.int16)), axis=2)
            stable = diff <= self.config.colour_tolerance
            rescue = self._accepted & (~yolo) & near & stable & (known != 0)
            self._accepted = (yolo | rescue) & (known != 0)
            # Refresh only accepted pixels currently supported by YOLO; a
            # rescued pixel's reference remains anchored to its original ink.
            self._reference[yolo] = bgr[yolo]
        return self._accepted.astype(np.uint8) * 255
