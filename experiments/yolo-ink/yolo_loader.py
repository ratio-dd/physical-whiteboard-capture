"""Evaluator adapters using the production tiled_inference contract.

Both factories return the same full-frame probability map: max(binary native
instance mask * box confidence). The only difference is whether Ultralytics
is called once on the frame or on fixed overlapping tiles.
"""
from tiled_inference import TiledYOLOInkModel

CONF_FLOOR = 0.05
MASK_THRESHOLD = 0.5
TILE = 640
OVERLAP = 96
IMGSZ = 640

class _EvaluatorModel:
    def __init__(self, weights, mode):
        self.inner = TiledYOLOInkModel(weights, conf=CONF_FLOOR,
                                       mask_threshold=MASK_THRESHOLD,
                                       tile=TILE, overlap=OVERLAP, imgsz=IMGSZ)
        self.mode = mode
        self.model_version = self.inner.model_version + '-' + mode

    def predict(self, bgr):
        probability, _ = self.inner.predict(bgr, mode=self.mode)
        return probability

def load_full(weights):
    return _EvaluatorModel(weights, 'full')

def load_tiled(weights):
    return _EvaluatorModel(weights, 'tiled')

def load(weights):
    """Compatibility alias: production default is the tiled path."""
    return load_tiled(weights)

def _load_v2(weights, mode):
    global IMGSZ
    old = IMGSZ
    IMGSZ = 512
    try:
        return _EvaluatorModel(weights, mode)
    finally:
        IMGSZ = old

def load_full_v2(weights):
    return _load_v2(weights, 'full')

def load_tiled_v2(weights):
    return _load_v2(weights, 'tiled')
