import importlib.util, sys, unittest
from pathlib import Path
import cv2, numpy as np

spec=importlib.util.spec_from_file_location('yolo_vector_adapter',Path(__file__).parents[1]/'experiments/yolo-ink/vector_adapter.py')
mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)

class YoloAdapterTests(unittest.TestCase):
    def setUp(self):
        self.board=np.full((100,140,3),245,np.uint8); self.known=np.full((100,140),255,np.uint8)
    def test_zero_equals_classic_and_unknown_is_blocked(self):
        z=mod.extract_yolo_hybrid(self.board,self.known,[])
        c=mod._PIPELINE.extract_paths(self.board,self.known,mode='classic')
        self.assertEqual(z.vector_paths,c); self.assertEqual(int(z.candidate_mask.max()),0)
    def test_box_hole_is_not_filled_and_is_export_geometry(self):
        cv2.rectangle(self.board,(30,1),(80,46),(160,160,160),3)
        m=np.zeros((50,55),np.float32); cv2.rectangle(m,(0,0),(50,45),1,3); cv2.rectangle(m,(5,5),(45,40),0, -1)
        p=mod._PIPELINE._candidate_paths(self.board,m.astype(np.uint8)*255)[0]
        self.assertGreaterEqual(len(p['outline']),2)
        rendered=np.zeros((50,55),np.uint8)
        for ring in p['outline']:
            layer=np.zeros_like(rendered); cv2.fillPoly(layer,[np.rint(np.asarray(ring)).astype(np.int32)],255); rendered=cv2.bitwise_xor(rendered,layer)
        self.assertEqual(int(rendered[25,25]),0)
    def test_thin_line_chunk_restore_and_confidence_gate(self):
        cv2.line(self.board,(10,20),(120,20),(175,175,175),2)
        left=np.zeros((8,60),np.float32); right=np.zeros((8,70),np.float32); left[2:6,:]=1; right[2:6,:]=1
        inst=[{'x':10,'y':16,'mask':left,'confidence':.9},{'x':65,'y':16,'mask':right,'confidence':.9},{'x':0,'y':0,'mask':np.ones((3,3),np.float32),'confidence':.2}]
        r=mod.extract_yolo_hybrid(self.board,self.known,inst,confidence_threshold=.8,pixel_threshold=.5)
        self.assertGreater(r.added_paths,0); self.assertEqual(int(r.candidate_mask[:3,:3].max()),0)
    def test_crossing_and_coordinates_are_board_absolute(self):
        cv2.line(self.board,(40,10),(40,80),(170,170,170),2); cv2.line(self.board,(20,45),(100,45),(170,170,170),2)
        m=np.zeros((20,30),np.float32); m[:]=1
        r=mod.extract_yolo_hybrid(self.board,self.known,[{'x':35,'y':40,'mask':m,'confidence':.95}],confidence_threshold=.8,pixel_threshold=.5)
        p=[p for p in r.vector_paths if p.get('kind')=='model_candidate'][-1]; pts=np.asarray(p['points']); self.assertGreaterEqual(pts[:,0].min(),35); self.assertGreaterEqual(pts[:,1].min(),40)

if __name__=='__main__': unittest.main()
