"""Offline faint-ink candidate -> vector evaluation pipeline.

This module is intentionally not imported by the service.  It makes the
candidate contract explicit for offline evaluation: the model returns a
probability map, candidates are restricted to observed pixels with source
image support, and geometry is produced by the existing extractor.  A mask is
never serialized as a vector and no pixels outside ``known`` are eligible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ink_geometry import extract_paths


class ProbabilityModel(Protocol):
    """The fixed faint-model prediction contract."""

    model_version: str

    def predict_proba(self, bgr: np.ndarray, known: np.ndarray | None = None) -> np.ndarray: ...


@dataclass(frozen=True)
class PipelineResult:
    baseline_paths: list[dict]
    vector_paths: list[dict]
    candidate_mask: np.ndarray
    model_version: str
    added_paths: int


def _source_support(bgr: np.ndarray) -> np.ndarray:
    """Conservative source evidence; no dilation or synthetic ink."""
    smooth = cv2.GaussianBlur(bgr, (3, 3), .7)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 5)
    contrast = np.abs(gray - local)
    edge = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    ycc = cv2.cvtColor(smooth, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    chroma = np.sqrt((ycc[..., 1] - 128) ** 2 + (ycc[..., 2] - 128) ** 2)
    return (contrast >= 1.5) | (edge >= 5.0) | (chroma >= 3.0)


def candidate_mask(model: ProbabilityModel, bgr: np.ndarray, known: np.ndarray,
                   threshold: float = .72) -> np.ndarray:
    """Return model candidates that have both known pixels and source support."""
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be an H×W uint8 BGR image")
    if known.dtype != np.uint8 or known.shape != bgr.shape[:2] or not np.all((known == 0) | (known == 255)):
        raise ValueError("known must be a binary H×W uint8 mask")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(bgr, known), dtype=np.float32)
    elif hasattr(model, "predict"):
        # LectureMath TinyFCN's fixed contract returns (probability, mask).
        probability = np.asarray(model.predict(bgr, known, threshold=threshold)[0], dtype=np.float32)
    else:
        raise TypeError("model must implement predict_proba or predict")
    if probability.ndim == 3 and probability.shape[-1] == 1:
        probability = probability[..., 0]
    if probability.shape != known.shape:
        raise ValueError("model probability map shape differs from board")
    return ((probability >= threshold) & (known == 255) & _source_support(bgr)).astype(np.uint8) * 255


def _baseline_occupancy(paths: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize existing geometry for conservative spatial de-duplication."""
    occupied = np.zeros(shape, np.uint8)
    for path in paths:
        rings = path.get("outline", [])
        for ring in rings:
            poly = np.rint(np.asarray(ring, np.float32)).astype(np.int32)
            if len(poly) >= 3:
                cv2.fillPoly(occupied, [poly], 255)
        branches = path.get("centerlines") or ([] if rings else [path])
        for branch in branches:
            poly = np.rint(np.asarray(branch.get("points", []), np.float32)).astype(np.int32)
            if len(poly) >= 2:
                width = max(1, int(round(float(path.get("width", 1)))))
                cv2.polylines(occupied, [poly], bool(branch.get("closed")), 255, width)
    return occupied


def _candidate_paths(bgr: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Vectorize model-only evidence without inventing pixels.

    The mask is already restricted to known/source-supported pixels.  Contours
    are deliberately retained as closed outlines: this is an editable SVG
    geometry primitive and makes the model contribution auditable separately
    from the classic extractor.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    paths = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < 2:
            continue
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        hierarchy = hierarchy[0]
        outer_indices = [i for i, h in enumerate(hierarchy) if h[3] < 0]
        if not outer_indices:
            continue
        rings = []
        primary = None
        for oi in outer_indices:
            contour = contours[oi].reshape(-1, 2).astype(float)
            if len(contour) < 3 or cv2.contourArea(contour.astype(np.float32)) < 1:
                continue
            contour[:, 0] += x; contour[:, 1] += y
            ring = contour.tolist() + [contour[0].tolist()]
            rings.append(ring)
            if primary is None: primary = ring
            child = hierarchy[oi][2]
            while child >= 0:
                hole = contours[child].reshape(-1, 2).astype(float)
                hole[:, 0] += x; hole[:, 1] += y
                if len(hole) >= 3:
                    rings.append(hole.tolist() + [hole[0].tolist()])
                child = hierarchy[child][0]
        if primary is None:
            continue
        pixels = component > 0
        b, g, r = np.mean(bgr[y:y + h, x:x + w][pixels], axis=0).astype(int)
        color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        paths.append({"points": primary, "closed": True, "centerlines": [],
                      "outline": rings, "layers": [{"outline": rings, "color": color}],
                      "color": color, "width": 1, "kind": "model_candidate",
                      "area": int(area), "evidence": "model_candidate"})
    return paths


def extract_hybrid(model: ProbabilityModel, bgr: np.ndarray, known: np.ndarray,
                   threshold: float = .72) -> PipelineResult:
    """Produce real geometry while retaining a classic comparison baseline.

    The classic extractor remains byte-for-byte the baseline. Model-only
    connected components are converted to closed editable outlines from the
    explicit candidate mask; no bundled/implicit model is consulted here.
    Callers must use ``vector_paths`` for SVG/InkML export, never the mask.
    """
    version = getattr(model, "model_version", "")
    # The packaged weak_ink_model predates the explicit metadata field; keep
    # its version auditable without changing the product model API.
    if not version and model.__class__.__name__ == "WeakInkV2Model":
        version = "weak_ink_v2"
    if not version:
        raise ValueError("model_version is required for auditable exports")
    candidates = candidate_mask(model, bgr, known, threshold)
    baseline = extract_paths(bgr, known, mode="classic")
    baseline_occupancy = _baseline_occupancy(baseline, known.shape)
    novel = candidates & (baseline_occupancy == 0).astype(np.uint8) * 255
    extra = _candidate_paths(bgr, novel)
    vectors = baseline + extra
    return PipelineResult(baseline, vectors, candidates, version, len(extra))
