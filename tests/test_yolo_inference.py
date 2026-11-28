import importlib.util, sys, unittest
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('yolo_inference',Path(__file__).parents[1]/'experiments/yolo-ink/inference.py')
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)

class Box:
    def __init__(self,c): self.conf=np.asarray(c,np.float32)
class Masks:
    def __init__(self,m): self.data=np.asarray(m,np.float32)
class Result:
    def __init__(self,m,c): self.masks=Masks(m); self.boxes=Box(c)

class YoloInferenceTests(unittest.TestCase):
    def test_native_non_square_masks_and_box_scores(self):
        m=np.zeros((2,17,31),np.float32); m[0,2:5,4:8]=1; m[1,10:13,20:24]=1
        p=mod.combine_native_result(Result(m,[.8,.3]),(17,31))
        self.assertEqual(p.shape,(17,31)); self.assertEqual(p.dtype,np.float32)
        self.assertAlmostEqual(float(p[3,5]),.8); self.assertAlmostEqual(float(p[11,21]),.3)
    def test_empty_detection_is_zero(self):
        class Empty: masks=None
        p=mod.combine_native_result(Empty(),(19,37)); self.assertEqual(int(p.max()),0)
    def test_non_native_shape_rejected_not_resized(self):
        m=np.zeros((1,8,8),np.float32)
        with self.assertRaises(ValueError): mod.combine_native_result(Result(m,[.9]),(17,31))

if __name__=='__main__': unittest.main()
