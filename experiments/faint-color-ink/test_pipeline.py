import sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(Path(__file__).parents[1]/'lecturemath-ink'))
from pipeline import FaintColorPipeline

def test_cold_start_step_returns_native_geometry():
    b=np.full((80,120,3),(235,245,247),np.uint8);cv2.line(b,(10,40),(100,40),(180,205,245),2);k=np.full((80,120),255,np.uint8)
    r=FaintColorPipeline().step(b,k);assert r.raw_color_mask.shape==k.shape;assert isinstance(r.vector_paths,list)

def test_exact_repeated_observation_is_idempotent():
    b=np.full((80,120,3),(235,245,247),np.uint8);cv2.line(b,(10,40),(100,40),(180,205,245),2);k=np.full((80,120),255,np.uint8);p=FaintColorPipeline();a=p.step(b,k);z=p.step(b,k)
    assert np.array_equal(a.final_mask,z.final_mask)
    assert len(a.vector_paths)==len(z.vector_paths)
