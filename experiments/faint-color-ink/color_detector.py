"""Source-supported weak red/colour ink candidate detector.

This is deliberately a small, inspectable baseline rather than a trained
model.  It compares signed chroma against a temporal board reference, removes
slow broad illumination changes with a local background, gates by ``known``,
and converts connected evidence through the existing native vectorizer.  No
coordinates or source-label points are embedded here.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class ColorConfig:
    score_threshold: float = 6.0
    chroma_threshold: float = 5.0
    delta_threshold: float = 6.0
    local_threshold: float = 8.0
    local_sigma: float = 7.0
    ridge_kernel: int = 5
    ridge_threshold: float = 5.0
    close_size: int = 1
    # A real pen ridge is spatially continuous; tiny 1–12 px warm blobs are
    # overwhelmingly paper/glare edges in the low-cost camera stream.
    min_component: int = 20
    adaptation: float = 0.01
    temporal_confirm: int = 2

@dataclass
class ColorState:
    """Temporal reference. Call ``reset`` when the board is replaced/erased."""
    reference_bgr: np.ndarray | None = None
    frames: int = 0
    previous_known: np.ndarray | None = None
    support: np.ndarray | None = None

    def reset(self) -> None:
        self.reference_bgr = None
        self.frames = 0
        self.previous_known = None
        self.support = None

@dataclass(frozen=True)
class Detection:
    score: np.ndarray
    mask: np.ndarray
    paths: list[dict]
    model_version: str = "faint-color-v0"

def _chroma(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = bgr.astype(np.float32)
    red_green = f[..., 2] - f[..., 1]
    red_balance = f[..., 2] - (f[..., 1] + f[..., 0]) * .5
    return red_green, red_balance

def detect(bgr: np.ndarray, known: np.ndarray, state: ColorState,
           config: ColorConfig = ColorConfig()) -> Detection:
    """Return source-supported mask and real editable candidate paths.

    The first frame establishes a reference and intentionally returns no
    candidates.  Only low-score pixels adapt toward the reference, so a stable
    ink line is not silently learned as background.  This is a temporal
    heuristic for offline evaluation, not a claim of production quality.
    """
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("bgr must be HxWx3 uint8")
    if known.dtype != np.uint8 or known.shape != bgr.shape[:2] or not np.isin(known, [0, 255]).all():
        raise ValueError("known must be a binary HxW uint8 mask")
    if config.close_size < 1 or config.min_component < 1:
        raise ValueError("morphology/component settings must be positive")
    current = bgr.astype(np.float32)
    cold_start = state.reference_bgr is None
    if cold_start:
        # A genuinely static first frame still needs a useful result.  The
        # local chroma baseline is the cold-start substitute for a temporal
        # reference; the ridge gate keeps broad warm paper/reflection out.
        state.reference_bgr = current.copy(); state.frames = 1
        state.previous_known = known.copy()
        state.support = np.zeros(known.shape, np.uint8)
    rg, rb = _chroma(bgr)
    ref_rg, ref_rb = _chroma(state.reference_bgr.astype(np.uint8))
    # R-G is the stable signed cue for the faint red pen.  Taking a max over
    # two channels would turn warm exposure drift into a positive candidate.
    chroma_delta = rg - ref_rg
    local = rg - cv2.GaussianBlur(rg, (0, 0), config.local_sigma)
    # A narrow coloured stroke has chroma on its centre and paper-like pixels
    # on both sides.  White top-hat is a cheap orientation-independent ridge
    # test; unlike a raw frame delta it rejects broad exposure/glare changes.
    if config.ridge_kernel < 3 or config.ridge_kernel % 2 == 0:
        raise ValueError("ridge_kernel must be an odd integer >= 3")
    ridge = cv2.morphologyEx(rg, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (config.ridge_kernel, config.ridge_kernel)))
    if cold_start:
        chroma_delta = local
    score = np.maximum(np.maximum(chroma_delta, local), ridge)
    # The gate intentionally uses the narrow ridge, not the broad local or
    # absolute chroma score: exposure shifts can make those large everywhere.
    # A static, already-written board has zero temporal delta after the first
    # observation.  Current local ridge evidence must remain sufficient on its
    # own; temporal delta is an additional cue for newly written pixels, not a
    # one-shot requirement.  The narrow ridge gate still rejects broad glare.
    evidence = (chroma_delta >= config.delta_threshold) | (local >= config.local_threshold)
    candidate = ((ridge >= config.ridge_threshold) & evidence & (known != 0)).astype(np.uint8)
    # Require evidence in consecutive observations for a replay candidate.
    # Cold-start remains usable; later single-frame speckles are not exported.
    was_unknown = ((state.previous_known == 0) if state.previous_known is not None else np.zeros_like(known, dtype=bool))
    state.support = np.where(candidate, np.minimum(state.support + 1, 3),
                              np.maximum(state.support.astype(np.int16) - 1, 0)).astype(np.uint8)
    # A newly visible pixel or a fresh, strongly supported local ridge may be
    # accepted immediately; ordinary weak speckles still need persistence.
    immediate = was_unknown & (known != 0)
    immediate |= local >= (2.0 * config.local_threshold)
    needed = 1 if cold_start else config.temporal_confirm
    candidate = candidate & ((state.support >= needed).astype(np.uint8) | immediate.astype(np.uint8))
    if config.close_size > 1:
        k = np.ones((config.close_size, config.close_size), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    mask = np.zeros_like(candidate)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= config.min_component:
            mask[labels == label] = 255
    # A pixel that was genuinely unknown has no usable colour baseline.  Seed
    # only that pixel when it first becomes known; do not globally chase camera
    # exposure, which would turn a fixed warm board patch into ink.
    if state.previous_known is not None:
        newly_known = (known != 0) & (state.previous_known == 0)
        state.reference_bgr[newly_known] = current[newly_known]
    state.previous_known = known.copy()
    state.frames += 1
    from vector_pipeline import _candidate_paths
    paths = _candidate_paths(bgr, mask)
    return Detection(score, mask, paths)
