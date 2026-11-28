import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('yolo_tiled',Path(__file__).parents[1]/'experiments/yolo-ink/tiled_inference.py'); mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

class FakeModel(mod.TiledYOLOInkModel):
    def __init__(self): self.tile=8; self.overlap=2; self.mask_threshold=.5
    def _patch(self,image):
        h,w=image.shape[:2]; out=np.zeros((h,w),np.float32); out[:,max(0,w//2-1):min(w,w//2+1)]=.8; return out

class TiledInferenceTests(unittest.TestCase):
    def test_tail_tiles_and_known_preserve_original_shape(self):
        m=FakeModel(); b=np.zeros((13,19,3),np.uint8); k=np.full((13,19),255,np.uint8);k[:2,:]=0
        p,q=m.predict(b,k,mode='tiled'); self.assertEqual(p.shape,(13,19));self.assertEqual(q.dtype,np.uint8);self.assertEqual(int(q[:2].max()),0)
    def test_full_and_tiled_are_both_explicit_paths(self):
        m=FakeModel();b=np.zeros((13,19,3),np.uint8);a,_=m.predict(b,mode='full');c,_=m.predict(b,mode='tiled');self.assertEqual(a.shape,c.shape)
    def test_threshold_override_and_bad_mode_rejected(self):
        m=FakeModel();b=np.zeros((4,5,3),np.uint8)
        with self.assertRaises(ValueError):m.predict(b,threshold=.4)
        with self.assertRaises(ValueError):m.predict(b,mode='bad')

if __name__=='__main__':unittest.main()
