"""YOLO-seg research inference: native full-frame score map plus mask.

This entrypoint intentionally predicts one original frame at a time. It does
not pretend to support tiled/letterboxed inputs: callers must provide a result
with ``retina_masks=True`` whose mask shape equals the input board shape.
"""
import cv2,numpy as np
try:
 from ultralytics import YOLO
except ImportError: YOLO=None

def _array(x):
 if hasattr(x,'detach'): x=x.detach().cpu().numpy()
 return np.asarray(x)

def combine_native_result(result, shape):
 """Return max(mask * instance box confidence), preserving original shape."""
 h,w=shape; out=np.zeros((h,w),np.float32)
 masks=getattr(getattr(result,'masks',None),'data',None)
 if masks is None: return out
 masks=_array(masks)
 if masks.ndim!=3 or masks.shape[1:]!=(h,w):
  raise ValueError(f"retina_masks shape {getattr(masks,'shape',None)} != original {(h,w)}")
 boxes=getattr(result,'boxes',None); conf=getattr(boxes,'conf',None) if boxes is not None else None
 scores=np.ones(masks.shape[0],np.float32) if conf is None else _array(conf).reshape(-1).astype(np.float32)
 if len(scores)!=masks.shape[0] or not np.all(np.isfinite(scores)): raise ValueError('instance confidence mismatch')
 for mask,score in zip(masks,scores): out=np.maximum(out,mask.astype(np.float32)*float(score))
 return out

class YOLOInkModel:
 def __init__(self,weights,conf=0.05,mask_threshold=0.5):
  if YOLO is None: raise RuntimeError('ultralytics is required for YOLO inference')
  self.model=YOLO(str(weights));self.conf=float(conf);self.mask_threshold=float(mask_threshold);self.model_version='yolo11s-seg-lecturemath-v1'
 def predict(self,bgr,known=None,threshold=None):
  t=self.mask_threshold if threshold is None else float(threshold); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); r=self.model.predict(rgb,conf=self.conf,verbose=False,device='cpu',retina_masks=True)[0]; p=combine_native_result(r,bgr.shape[:2])
  if known is not None:p=np.where(known>0,p,0).astype('f4')
  return p,(p>=t).astype('uint8')*255
def load_model(weights,conf=.05,mask_threshold=.5): return YOLOInkModel(weights,conf,mask_threshold)
