"""Frozen inference wrapper for local_background_mlp_v1."""
from pathlib import Path
import json,numpy as np
from local_logreg import features

class LocalMLPModel:
 def __init__(self,run_dir):
  r=Path(run_dir); self.config=json.loads((r/'config.json').read_text()); self.threshold=float(self.config['threshold']);
  self.model_version=self.config.get('model_version','lecturemath_local_mlp_v1')
  if not 0<self.threshold<1: raise ValueError('invalid frozen threshold')
  d=np.load(r/'weights.npz'); self.w1=d['w1'];self.b1=d['b1'];self.w2=d['w2'];self.b2=float(d['b2'][0]);self.mu=d['mu'];self.sd=d['sd']
 def predict_proba(self,bgr,known=None):
  f=features(bgr); shape=f.shape[:2]; z=(f.reshape(-1,f.shape[-1])-self.mu)/self.sd; p=1/(1+np.exp(-np.clip(np.maximum(z@self.w1+self.b1,0)@self.w2+self.b2,-30,30))); p=p.reshape(shape).astype('f4')
  if known is not None:p=np.where(known>0,p,0).astype('f4')
  return p
 def predict(self,bgr,known=None,threshold=None):
  p=self.predict_proba(bgr,known); t=self.threshold if threshold is None else float(threshold)
  if not 0<t<1: raise ValueError('threshold must be in (0,1)')
  return p,(p>=t).astype('uint8')*255

def load_checkpoint(run_dir): return LocalMLPModel(run_dir)
