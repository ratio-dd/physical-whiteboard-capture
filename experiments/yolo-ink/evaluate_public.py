"""Public LectureMath evaluator for future YOLO-style ink models.

The predictor always receives the complete original-size BGR frame. Evaluation
is restricted to the frozen native-coordinate 4x4 grid of 128x128 windows; it
never resizes a score window into a full image. A predictor may return either:

  * ``{"masks": [N,H,W], "scores": [N]}`` (instance masks + confidence), or
  * ``{"probability": [H,W]}`` / an HxW array (soft pixel map).

Ultralytics-like objects with ``result.masks.data`` and ``result.boxes.conf``
are also accepted. `--mock` is a protocol smoke test only, never a YOLO score.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
MANIFEST=ROOT/'data/datasets/lecturemath/manifest_multi.json'; TILE=128; GRID=4; SEED=20261128

def sha256(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()

def rows_for(split):
    spec=json.loads(MANIFEST.read_text()); return [r for r in spec['pairs'] if r.get('split')==split]

def windows(im):
    h,w=im.shape[:2]
    for gy in range(GRID):
        for gx in range(GRID):
            cx=int(round((gx+.5)*w/GRID)); cy=int(round((gy+.5)*h/GRID))
            x=max(0,min(w-TILE,cx-TILE//2)); y=max(0,min(h-TILE,cy-TILE//2))
            yield gy,gx,x,y

def local_deficit(im):
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    return (paper.astype(np.float32)-gray.astype(np.float32))/255.0

def _array(x):
    return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

def normalize_prediction(raw, shape):
    """Return ('soft', HxW) or ('instances', masks, scores), preserving semantics."""
    h,w=shape
    if isinstance(raw,(list,tuple)) and len(raw)==1: raw=raw[0]
    if isinstance(raw,dict):
        if 'probability' in raw or 'prob' in raw or 'score_map' in raw:
            val=raw['probability'] if 'probability' in raw else raw['prob'] if 'prob' in raw else raw['score_map']
            p=_array(val).astype(np.float32)
            return 'soft',_map(p,shape)
        if 'masks' in raw:
            masks=_array(raw['masks']); scores=_array(raw.get('scores',raw.get('conf',[]))).reshape(-1).astype(np.float32)
            return _instances(masks,scores,shape)
    if hasattr(raw,'masks') and hasattr(raw,'boxes'):
        masks=_array(raw.masks.data); scores=_array(raw.boxes.conf).reshape(-1).astype(np.float32)
        return _instances(masks,scores,shape)
    p=_array(raw)
    if p.ndim==2: return 'soft',_map(p,shape)
    if p.ndim==3 and p.shape[0]==1: return 'soft',_map(p[0],shape)
    raise TypeError('predict must return HxW soft map, {probability}, {masks,scores}, or Ultralytics-like result')

def _map(p,shape):
    if p.shape!=shape:
        raise ValueError(f'soft map shape {p.shape} differs from original frame {shape}; predict full frame')
    if not np.all(np.isfinite(p)): raise ValueError('soft map contains non-finite scores')
    p=p.astype(np.float32)
    # Contract requires probabilities, not logits: reject obvious out-of-range maps.
    if p.min() < -1e-6 or p.max() > 1+1e-6: raise ValueError('soft map must be sigmoid probability in [0,1], not logits')
    return np.clip(p,0,1)

def _instances(masks,scores,shape):
    masks=np.asarray(masks); h,w=shape
    if masks.ndim!=3 or masks.shape[0]!=len(scores): raise ValueError('instance masks/scores length mismatch')
    if masks.shape[1:]!=shape: raise ValueError('instance masks must be full original-frame shape')
    if not np.all(np.isfinite(masks)) or not np.all(np.isfinite(scores)): raise ValueError('non-finite instance output')
    if scores.size and (scores.min()<0 or scores.max()>1): raise ValueError('instance scores must be confidence probabilities [0,1]')
    return 'instances',masks.astype(np.float32),scores.astype(np.float32)

def mask_at(pred, threshold, shape):
    kind=pred[0]
    if kind=='soft': return pred[1]>=threshold
    _,m,s=pred; return np.any((m>0.5) & (s[:,None,None]>=threshold),axis=0)

def metric(scores, truth, faded, kind):
    s=np.concatenate(scores); y=np.concatenate(truth); f=np.concatenate(faded); neg=s[~y]; budget=int(np.floor(.01*len(neg)))
    vals=np.sort(neg)[::-1]; lower=float(vals[budget-1]) if budget else float(vals[0]); t=float(np.nextafter(np.float32(lower),np.float32(np.inf)))
    p=s>=t; return counts(p,y,f,t,lower_bound=lower,negative_ties=int((neg==lower).sum()))

def counts(p,y,f,t,*,lower_bound=None,negative_ties=None):
    tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
    return {'threshold':float(t),'threshold_lower_bound':float(lower_bound if lower_bound is not None else t),'negative_ties_at_lower_bound':int(negative_ties if negative_ties is not None else 0),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'fpr':fp/max(1,fp+tn),'weak_recall':int((p&f).sum())/max(1,int(f.sum())),'positive_pixels':int(y.sum()),'negative_pixels':int((~y).sum()),'faded_positive_pixels':int(f.sum())}

class MockPredictor:
    mock=True
    def predict(self,im):
        d=local_deficit(im); return {'probability':np.clip(d*4,0,1).astype(np.float32)}

def load_predictor(spec, weights=None):
    if spec=='mock': return MockPredictor()
    mod,fn=spec.split(':',1); factory=getattr(importlib.import_module(mod),fn); return factory(weights) if weights is not None else factory()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split',choices=['validation','test'],default='validation'); ap.add_argument('--loader',help='module:function factory returning .predict(full_bgr)'); ap.add_argument('--weights'); ap.add_argument('--mock',action='store_true'); ap.add_argument('--limit',type=int,default=0,help='optional smoke limit; default evaluates the complete split'); ap.add_argument('--fixed-threshold',type=float,help='also report metrics at this pre-frozen score threshold'); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    pred=load_predictor('mock' if a.mock else a.loader,a.weights if a.weights else None)
    items=rows_for(a.split); items=items[:a.limit] if a.limit else items; scores=[]; truth=[]; faded=[]; timing=[]; metadata=[]; detail_all=[]; label_all=[]; prediction_stats=[]
    for row in items:
        im=cv2.imread(str(MANIFEST.parent/row['image']),1); gt=cv2.imread(str(MANIFEST.parent/row['mask']),0)==0
        if im is None: raise FileNotFoundError(row)
        d=local_deficit(im); detail_all.append(d.reshape(-1)); label_all.append(gt.reshape(-1)); t0=time.perf_counter(); raw=pred.predict(im); elapsed=(time.perf_counter()-t0)*1000; timing.append(elapsed); norm=normalize_prediction(raw,im.shape[:2])
        if norm[0]=='soft':
            support=int(np.count_nonzero(norm[1]>=.5)); detections=None; conf_range=None
        else:
            support=int(np.count_nonzero(np.any(norm[1]>.5,axis=0))); detections=int(len(norm[2])); conf_range=[float(norm[2].min()),float(norm[2].max())] if len(norm[2]) else None
        prediction_stats.append({'lecture_id':row['lecture_id'],'frame_id':row['frame_id'],'kind':norm[0],'detections':detections,'nonempty_support_pixels':support,'confidence_range':conf_range,'elapsed_ms':elapsed})
        for gy,gx,x,y in windows(im):
            sl=np.s_[y:y+TILE,x:x+TILE]; s=norm[1][sl] if norm[0]=='soft' else np.max(np.where(norm[1][1][:,None,None]>=0,norm[1][1],0),axis=0)[sl] if False else None
            if norm[0]=='soft': score=s.reshape(-1)
            else:
                mm,ss=norm[1],norm[2]; score=np.max(np.where(mm[:,sl[0],sl[1]]>0.5,ss[:,None,None],0),axis=0).reshape(-1)
            yy=gt[sl].reshape(-1); dd=d[sl].reshape(-1); scores.append(score);truth.append(yy);faded.append(yy&(dd<=0))
            metadata.append({'lecture_id':row['lecture_id'],'frame_id':row['frame_id'],'grid':[gy,gx],'xy':[x,y],'source_wh':[im.shape[1],im.shape[0]],'image_sha256':sha256(MANIFEST.parent/row['image']),'mask_sha256':sha256(MANIFEST.parent/row['mask'])})
    detail=np.concatenate(detail_all); labels=np.concatenate(label_all); cutoff=float(np.median(detail[labels])); faded=[y&(d<=cutoff) for y,d in zip(truth,[np.zeros_like(y,dtype=np.float32) for y in truth])]
    # Rebuild faded masks using the same pooled cutoff on each collected window.
    # `faded` above is replaced by a direct source-detail pass for clarity.
    faded=[]
    for row in items:
        im=cv2.imread(str(MANIFEST.parent/row['image']),1); d=local_deficit(im); gt=cv2.imread(str(MANIFEST.parent/row['mask']),0)==0
        for _,_,x,y in windows(im): faded.append((gt[y:y+TILE,x:x+TILE].reshape(-1)&(d[y:y+TILE,x:x+TILE].reshape(-1)<=cutoff)))
    # Expand instance predictions to score vectors above; threshold is exact global FPR cutoff.
    ss=np.concatenate(scores); yy=np.concatenate(truth); ff=np.concatenate(faded)
    fixed_metrics = counts(ss>=np.float32(a.fixed_threshold),yy,ff,float(a.fixed_threshold)) if a.fixed_threshold is not None else None
    # A fixed test threshold is an audit mode, never a selection mode. Keep the
    # primary metrics authoritative so downstream readers cannot accidentally
    # consume a test-quantile result.
    if a.split == 'test' and fixed_metrics is not None:
        primary_metrics = fixed_metrics
        selection_skipped = True
    else:
        primary_metrics = metric(scores,truth,faded,'')
        selection_skipped = False
    report={'protocol':{'manifest':str(MANIFEST),'split':a.split,'pairs':len(items),'limit':a.limit or None,'grid':[GRID,GRID],'tile':[TILE,TILE],'original_frame_inference':True,'seed':SEED,'positive':'mask==0','faded_cutoff':cutoff,'patch_count':len(scores),'windows':metadata,'mock':bool(a.mock),'fixed_threshold':a.fixed_threshold,'threshold_selection_skipped':selection_skipped},'model':{'loader':a.loader,'weights':a.weights,'kind':'mock' if a.mock else type(pred).__name__,'full_frame_ms':{'median':float(np.median(timing)),'mean':float(np.mean(timing)),'p90':float(np.percentile(timing,90))},'prediction_stats':prediction_stats,'metrics':primary_metrics}}
    if a.fixed_threshold is not None and not (a.split == 'test'):
        report['model']['fixed_threshold_metrics']=fixed_metrics
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
