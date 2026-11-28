"""Full-frame and overlap-tiled YOLO-seg inference in original board space."""
from __future__ import annotations
from pathlib import Path
import cv2, numpy as np


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class TiledYOLOInkModel:
    model_version = "yolo11s-seg-lecturemath-v1-tiled"
    def __init__(self, weights, conf=.05, mask_threshold=.5, tile=640, overlap=96, imgsz=640, iou=.7, max_det=300):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required") from exc
        if tile <= 0 or not 0 <= overlap < tile: raise ValueError("invalid tile overlap")
        self.model=YOLO(str(weights)); self.conf=float(conf); self.mask_threshold=float(mask_threshold)
        self.tile=int(tile); self.overlap=int(overlap); self.imgsz=int(imgsz); self.iou=float(iou); self.max_det=int(max_det)

    def _patch(self, image):
        result=self.model.predict(image, conf=self.conf, imgsz=self.imgsz, iou=self.iou, max_det=self.max_det, verbose=False, device="cpu", retina_masks=True)[0]
        h,w=image.shape[:2]; out=np.zeros((h,w),np.float32)
        masks=getattr(getattr(result,"masks",None),"data",None)
        if masks is None: return out
        masks=_np(masks)
        if masks.ndim != 3 or masks.shape[1:] != (h,w): raise ValueError("retina mask is not patch-native")
        boxes=getattr(result,"boxes",None); conf=getattr(boxes,"conf",None) if boxes is not None else None
        scores=np.ones(masks.shape[0],np.float32) if conf is None else _np(conf).reshape(-1).astype(np.float32)
        if len(scores)!=len(masks): raise ValueError("mask/confidence count mismatch")
        for mask,score in zip(masks,scores): out=np.maximum(out,(mask.astype(np.float32)>0)*float(score))
        return out

    def _gates(self, probability, known):
        if known is not None:
            if known.shape != probability.shape: raise ValueError("known shape differs")
            probability=np.where(known>0,probability,0).astype(np.float32)
        return probability, (probability>=np.float32(self.mask_threshold)).astype(np.uint8)*255

    def predict_full(self, bgr, known=None):
        probability=self._patch(bgr); return self._gates(probability,known)

    def predict_tiled(self, bgr, known=None):
        h,w=bgr.shape[:2]; step=self.tile-self.overlap; probability=np.zeros((h,w),np.float32)
        ys=list(range(0,max(1,h-self.tile+1),step)); xs=list(range(0,max(1,w-self.tile+1),step))
        if not ys or ys[-1] != max(0,h-self.tile): ys.append(max(0,h-self.tile))
        if not xs or xs[-1] != max(0,w-self.tile): xs.append(max(0,w-self.tile))
        for y in ys:
            for x in xs:
                patch=bgr[y:min(h,y+self.tile),x:min(w,x+self.tile)]
                q=self._patch(patch); probability[y:y+q.shape[0],x:x+q.shape[1]]=np.maximum(probability[y:y+q.shape[0],x:x+q.shape[1]],q)
        return self._gates(probability,known)

    def predict(self,bgr,known=None,threshold=None,mode="full"):
        if threshold is not None and float(threshold)!=self.mask_threshold: raise ValueError("threshold override is forbidden")
        if mode=="full": return self.predict_full(bgr,known)
        if mode=="tiled": return self.predict_tiled(bgr,known)
        raise ValueError("mode must be full or tiled")


def load_model(weights, conf=.05, mask_threshold=.5, tile=640, overlap=96, imgsz=640, iou=.7, max_det=300):
    return TiledYOLOInkModel(weights,conf,mask_threshold,tile,overlap,imgsz,iou,max_det)
