"""Add source-colour layers without duplicating editable geometry."""
from __future__ import annotations
import cv2
import numpy as np

def _raster(path: dict, shape: tuple[int,int]) -> np.ndarray:
    out=np.zeros(shape,np.uint8)
    rings=path.get('outline') or []
    if rings:
        for i,ring in enumerate(rings):
            q=np.rint(np.asarray(ring,np.float32)).astype(np.int32)
            if len(q)>=3: cv2.fillPoly(out,[q],255 if i==0 else 0)
    else:
        for branch in path.get('centerlines') or []:
            q=np.rint(np.asarray(branch.get('points',[]),np.float32)).astype(np.int32)
            if len(q)>=2: cv2.polylines(out,[q],False,255,max(1,int(round(path.get('width',1)))))
    return out

def apply_source_color_layers(paths: list[dict], bgr: np.ndarray, candidate: np.ndarray,
                              min_overlap: int = 2) -> list[dict]:
    """Return copies with red/source layers only where mask overlaps classic ink.

    The classic outline/centerline is untouched. Each added layer is a clipped
    connected contour, so a neighbouring gray stroke is not recoloured merely
    because it is nearby. Candidate components outside classic are deliberately
    not handled here; callers pass those to ``_candidate_paths``.
    """
    if candidate.dtype != np.uint8 or candidate.shape != bgr.shape[:2]: raise ValueError('candidate shape/type')
    out=[]
    for path in paths:
        overlap=_raster(path,candidate.shape)
        hit=((overlap!=0)&(candidate!=0)).astype(np.uint8)*255
        if int((hit!=0).sum()) < min_overlap: out.append(path); continue
        n,lab,stats,_=cv2.connectedComponentsWithStats(hit,8); layers=[]
        for i in range(1,n):
            if stats[i,cv2.CC_STAT_AREA] < min_overlap: continue
            part=(lab==i).astype(np.uint8)*255; contours,h=cv2.findContours(part,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            rings=[]
            for c in contours:
                q=c.reshape(-1,2).astype(float).tolist()
                if len(q)>=3: rings.append(q+[q[0]])
            if not rings: continue
            ys,xs=np.where(part!=0); pix=bgr[ys,xs].mean(axis=0); color=f'#{int(pix[2]):02x}{int(pix[1]):02x}{int(pix[0]):02x}'
            layers.append({'outline':rings,'color':color,'source':'signed-color'})
        if layers:
            p=dict(path); p['layers']=list(path.get('layers') or [])+layers; p['source_color_layers']=len(layers); out.append(p)
        else: out.append(path)
    return out
