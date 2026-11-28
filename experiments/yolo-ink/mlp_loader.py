"""Whole-frame adapter for the frozen public local MLP."""
from local_mlp_model import load_checkpoint

def load(run_dir):
    model = load_checkpoint(run_dir)
    class Adapter:
        model_version = model.model_version + '-whole-frame'
        def predict(self, bgr):
            return model.predict_proba(bgr)
    return Adapter()
