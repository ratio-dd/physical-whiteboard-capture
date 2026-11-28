"""Stable one-step API for the frozen v3-memory development candidate."""
from __future__ import annotations
from dataclasses import dataclass
import cv2, numpy as np
from color_detector import ColorConfig, ColorState, detect
from evidence_memory import EvidenceMemory, MemoryConfig
from vector_pipeline import _baseline_occupancy, _candidate_paths
from ink_geometry import extract_paths

@dataclass(frozen=True)
class PipelineConfig:
    color: ColorConfig = ColorConfig()
    memory: MemoryConfig = MemoryConfig()
    model_version: str = 'faint-color-memory-overlay-v3'
    min_length_px: float = 20.0

@dataclass(frozen=True)
class StepResult:
    raw_color_mask: np.ndarray
    final_mask: np.ndarray
    classic_paths: list[dict]
    vector_paths: list[dict]
    additions: list[dict]
    source_layer_count: int
    model_version: str

class FaintColorPipeline:
    """Initialize once per replay/session, then call ``step`` in seq order.

    Passing no score is the reproducible cold-start signed-colour path. Passing
    the frozen original-size YOLO score enables EvidenceMemory; it is still
    gated by current source-colour pixels and known, never by a remembered box.
    """
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        self.color_state = ColorState()
        self.memory = EvidenceMemory(config.memory)
        self._last_bgr = None
        self._last_known = None
        self._last_score = None
        self._last_result = None

    def step(self, bgr: np.ndarray, known: np.ndarray,
             frozen_yolo_score: np.ndarray | None = None) -> StepResult:
        # Replaying an identical observation must be idempotent.  In
        # particular, do not advance temporal-confirm counters merely because
        # a UI/API asked for the same frame twice.
        if (self._last_result is not None and np.array_equal(bgr, self._last_bgr)
                and np.array_equal(known, self._last_known)
                and ((frozen_yolo_score is None and self._last_score is None) or
                     (frozen_yolo_score is not None and np.array_equal(frozen_yolo_score, self._last_score)))):
            return self._last_result
        detection = detect(bgr, known, self.color_state, self.config.color)
        source_mask = detection.mask
        if frozen_yolo_score is not None:
            remembered = self.memory.update(bgr, known, frozen_yolo_score, source_mask)
            near = cv2.dilate((source_mask != 0).astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
            source_mask = np.where(near, np.maximum(source_mask, remembered), 0).astype(np.uint8)
        classic = extract_paths(bgr, known, mode='classic')
        occupied = _baseline_occupancy(classic, known.shape)
        # Keep classic dictionaries byte/renderer-compatible.  Adding layers
        # to a classic path makes the app hide its original centerline; use an
        # independent clipped overlay object instead.
        overlap = _candidate_paths(bgr, source_mask & (occupied != 0).astype(np.uint8) * 255)
        for item in overlap:
            item['kind'] = 'source_color_overlay'; item['evidence'] = 'source_color_overlay'
        raw = _candidate_paths(bgr, source_mask & (occupied == 0).astype(np.uint8) * 255)
        additions=[]
        for item in raw:
            pts=np.asarray(item.get('points',[]),float)
            if len(pts) and max(float(np.ptp(pts[:,0])),float(np.ptp(pts[:,1]))) >= self.config.min_length_px:
                additions.append(item)
        overlays = overlap
        result = StepResult(detection.mask, source_mask, classic, classic + overlays + additions,
                          overlays + additions, len(overlays),
                          self.config.model_version)
        self._last_bgr = np.array(bgr, copy=True); self._last_known = np.array(known, copy=True)
        self._last_score = None if frozen_yolo_score is None else np.array(frozen_yolo_score, copy=True)
        self._last_result = result
        return result
