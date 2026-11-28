"""Generic read-only ink JSON exporter for the v3-memory candidate.

It accepts an explicit History database/session and emits the app's
add/update/remove event shape. No private session, data path, or checkpoint is
embedded; callers provide those as arguments.
"""
from __future__ import annotations
import argparse, hashlib, json, sqlite3, tempfile
from pathlib import Path
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[2]
# Public code depends only on the repository and the shared vector pipeline;
# private evaluation helpers under ignored data/ are intentionally not added
# to import resolution.
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/lecturemath-ink')]
from history import History
from pipeline import FaintColorPipeline

def _center(path):
    q=np.asarray(path.get('points',[]),float); return q.mean(0) if len(q) else np.zeros(2)
def _digest(path): return hashlib.sha256(json.dumps(path,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build(db: Path, session: str, out: Path, memory_dir: Path|None=None, through: int|None=None, weights: Path|None=None):
    src=sqlite3.connect(f'file:{db.resolve()}?mode=ro',uri=True); tmp=Path(tempfile.mktemp(prefix='ink-export-',suffix='.sqlite3')); dst=sqlite3.connect(tmp);src.backup(dst);dst.close();src.close();h=History(tmp)
    row=h._db.execute('select seq from sessions where id=?',(session,)).fetchone()
    if not row: raise ValueError(f'unknown session {session}')
    n=min(int(row[0]),through or int(row[0])); pipe=FaintColorPipeline(); previous={}; next_id=1; events=[]; last_shape=None
    for seq in range(1,n+1):
        _,b,k=h.state(session,seq); score=np.load(memory_dir/f'seq-{seq:03d}.npy') if memory_dir else None; step=pipe.step(b,k,score); current={}; used=set()
        for path in step.vector_paths:
            q=_center(path); best=None; dist=1e9
            for ident,old in previous.items():
                if ident in used: continue
                d=float(np.linalg.norm(q-_center(old))); same=(ident.startswith('color-') == (path.get('kind')=='model_candidate'))
                if same and d<dist: best,dist=ident,d
            prefix='color-' if path.get('kind')=='model_candidate' else 'stroke-'
            ident=best if best is not None and dist<=24 else f'{prefix}{next_id:04d}'
            if ident not in previous: next_id+=1
            enriched={**path,'id':ident,'first_seq':previous.get(ident,{}).get('first_seq',seq),'direction':'unknown','direction_source':'unknown'};current[ident]=enriched;used.add(ident)
        changes=[]
        for ident,path in current.items(): changes.append({'op':'update' if ident in previous else 'add','id':ident,'stroke':path})
        for ident in previous:
            if ident not in current: changes.append({'op':'remove','id':ident})
        ts=h._db.execute('select observed_at,confirmed_at from events where session_id=? and seq=?',(session,seq)).fetchone() or (0,0)
        events.append({'seq':seq,'observed_at':int(ts[0]),'confirmed_at':int(ts[1]),'changes':changes});previous=current;last_shape=b.shape
    weight_hash=None
    if weights and weights.exists():
        hsh=hashlib.sha256();
        with weights.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): hsh.update(chunk)
        weight_hash=hsh.hexdigest()
    payload={'version':1,'session_id':session,'session_version':n,'through_seq':n,'width':last_shape[1],'height':last_shape[0],'recognition_mode':'source-color-experiment','model_version':pipe.config.model_version,'recognition_threshold':pipe.config.memory.threshold,'source_config':pipe.config.color.__dict__,'memory_config':pipe.config.memory.__dict__,'weights_sha256':weight_hash,'events':events}
    out.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf8');return payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--db',type=Path,required=True);p.add_argument('--session',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--memory-dir',type=Path);p.add_argument('--weights',type=Path);p.add_argument('--through',type=int);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);d=build(a.db,a.session,a.out,a.memory_dir,a.through,a.weights);print(json.dumps({'out':str(a.out),'events':len(d['events']),'through_seq':d['through_seq']}))
if __name__=='__main__':main()
