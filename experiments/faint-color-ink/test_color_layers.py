import sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from color_layers import apply_source_color_layers

def test_only_overlapped_gray_stroke_gets_source_layer():
    b=np.full((80,120,3),(235,245,247),np.uint8); cv2.line(b,(10,30),(100,30),(180,180,180),3); cv2.line(b,(10,60),(100,60),(180,180,180),3)
    p={'outline':[[[8,28],[102,28],[102,33],[8,33],[8,28]]],'layers':[],'color':'#888888','width':3}; m=np.zeros((80,120),np.uint8);cv2.line(m,(10,30),(100,30),255,2)
    out=apply_source_color_layers([p],b,m);assert out[0].get('source_color_layers')==1

def test_nearby_gray_stroke_is_not_recolored():
    b=np.full((80,120,3),(235,245,247),np.uint8);p={'outline':[[[8,58],[102,58],[102,63],[8,63],[8,58]]],'layers':[],'color':'#888888'};m=np.zeros((80,120),np.uint8);cv2.line(m,(10,30),(100,30),255,2)
    assert 'source_color_layers' not in apply_source_color_layers([p],b,m)[0]
