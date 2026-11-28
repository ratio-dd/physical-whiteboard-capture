"""Throttled, atomic Ultralytics progress JSON for the Windows panel."""
from __future__ import annotations
import json, os, tempfile, time
from pathlib import Path
import torch

class ProgressWriter:
    def __init__(self, path: Path, interval: float = 1.0):
        self.path=Path(path); self.interval=float(interval); self.last=0.0; self.started=time.monotonic()
    def _write(self, payload, path=None):
        path = Path(path or self.path); path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.progress-',suffix='.json',dir=str(path.parent),text=True)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f: json.dump(payload,f,ensure_ascii=False); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
    def update(self, trainer, force=False):
        now=time.monotonic()
        if not force and now-self.last < self.interval: return
        self.last=now; epoch=int(getattr(trainer,'epoch',-1))+1; total=int(getattr(trainer,'epochs',0)); batch=int(getattr(trainer,'batch_i',-1))+1; nb=int(getattr(trainer,'nb',0)); elapsed=max(.001,now-self.started); done=max(0,(epoch-1)*nb+batch); total_steps=max(1,total*nb); eta=max(0,elapsed*(total_steps-done)/max(1,done)); actual=Path(getattr(trainer,'save_dir',self.path))/ 'progress.json'
        losses={}
        vals=getattr(trainer,'tloss',None)
        if vals is not None:
            try:
                vals=vals.detach().float().cpu().tolist() if torch.is_tensor(vals) else list(vals)
                names=getattr(trainer,'loss_names',[]) or []
                losses={str(names[i]) if i<len(names) else f'loss_{i}':float(v) for i,v in enumerate(vals)}
            except Exception: pass
        self._write({'schema':'windows-cuda-progress-v1','state':'training','epoch':epoch,'epochs':total,'batch':batch,'batches':nb,'elapsed_s':elapsed,'eta_s':eta,'losses':losses,'updated_unix':time.time()},actual)
    def on_train_batch_end(self, trainer): self.update(trainer)
    def on_fit_epoch_end(self, trainer): self.update(trainer,True)
    def on_train_end(self, trainer):
        self.update(trainer,True)
        data={'schema':'windows-cuda-progress-v1','state':'completed','epoch':int(getattr(trainer,'epochs',0)),'epochs':int(getattr(trainer,'epochs',0)),'batch':int(getattr(trainer,'nb',0)),'batches':int(getattr(trainer,'nb',0)),'elapsed_s':max(.001,time.monotonic()-self.started),'eta_s':0.0,'losses':{},'updated_unix':time.time()}; self._write(data,Path(getattr(trainer,'save_dir',self.path))/'progress.json')
