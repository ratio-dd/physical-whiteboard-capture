"""Offline YOLO-instance mask -> editable whiteboard geometry adapter.

YOLO boxes and polygon perimeter changes are never treated as ink. Only the
instance pixel mask, restored into board coordinates, is passed to the frozen
vector pipeline. Unknown pixels are always zero and holes remain contour
hierarchy/even-odd geometry.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import cv2
import numpy as np

_PIPELINE_PATH = Path(__file__).parents[1] / "lecturemath-ink" / "vector_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("lecturemath_vector_pipeline_yolo", _PIPELINE_PATH)
_PIPELINE = importlib.util.module_from_spec(_SPEC)
import sys
sys.modules[_SPEC.name] = _PIPELINE
_SPEC.loader.exec_module(_PIPELINE)


class _MaskModel:
    model_version = "yolo_instance_mask_adapter_v1"
    def __init__(self, probability): self.probability = probability.astype(np.float32)
    def predict_proba(self, bgr, known=None):
        if self.probability.shape != bgr.shape[:2]: raise ValueError("restored mask shape differs")
        return self.probability


def restore_instance_masks(instances, board_shape, *, confidence_threshold=.5, pixel_threshold=.5):
    """Restore tile-local instance masks into HxW probability and confidence maps.

    Each instance must contain ``mask`` (float/binary local HxW), ``x``, ``y``
    and ``confidence``. Overlaps use max probability; confidence is max per
    pixel, so the same instance thresholds govern segmentation and export.
    """
    if not 0 < confidence_threshold <= 1 or not 0 < pixel_threshold <= 1:
        raise ValueError("thresholds must be in (0,1]")
    h, w = board_shape; probability=np.zeros((h,w),np.float32); confidence=np.zeros((h,w),np.float32)
    for item in instances:
        mask=np.asarray(item["mask"],np.float32); y,x=int(item["y"]),int(item["x"]); c=float(item["confidence"])
        if mask.ndim != 2 or not np.isfinite(mask).all() or not 0 <= c <= 1: raise ValueError("invalid instance")
        if c < confidence_threshold: continue
        bottom, right=max(0,y)+mask.shape[0], max(0,x)+mask.shape[1]
        sy0, sx0=max(0,-y), max(0,-x); sy1,sx1=min(mask.shape[0],h-y),min(mask.shape[1],w-x)
        if sy0>=sy1 or sx0>=sx1: continue
        tile=mask[sy0:sy1,sx0:sx1]; yy0,xx0=max(0,y),max(0,x); yy1,xx1=yy0+tile.shape[0],xx0+tile.shape[1]
        probability[yy0:yy1,xx0:xx1]=np.maximum(probability[yy0:yy1,xx0:xx1],tile)
        confidence[yy0:yy1,xx0:xx1]=np.maximum(confidence[yy0:yy1,xx0:xx1],np.where(tile>=pixel_threshold,c,0))
    return probability, confidence


def extract_yolo_hybrid(model_board, known, instances, *, confidence_threshold=.5, pixel_threshold=.5):
    """Return frozen classic baseline plus novel YOLO mask geometry."""
    if model_board.dtype != np.uint8 or model_board.ndim != 3: raise ValueError("model_board must be uint8 BGR")
    known=np.asarray(known,np.uint8)
    probability,confidence=restore_instance_masks(instances,known.shape,confidence_threshold=confidence_threshold,pixel_threshold=pixel_threshold)
    candidate=((probability>=pixel_threshold)&(confidence>=confidence_threshold)&(known==255)).astype(np.uint8)*255
    result=_PIPELINE.extract_hybrid(_MaskModel(candidate/255),model_board,known,threshold=.5)
    result.candidate_mask[:] = candidate
    return result
