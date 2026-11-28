"""Adapter for the frozen-feature semantic probe in the public evaluator."""
from pathlib import Path
from semantic_probe import load_model as load_semantic

RUN_DIR = Path(__file__).parents[2] / 'data/datasets/lecturemath-ink/yolo-semantic'
BACKBONE = Path(__file__).parents[2] / 'data/datasets/weak-ink-ml/yolo11s-seg.pt'

def load(_weights=None):
    model = load_semantic(str(BACKBONE), str(RUN_DIR))
    class Adapter:
        model_version = model.model_version
        def predict(self, bgr):
            probability, _ = model.predict(bgr)
            return probability
    return Adapter()
