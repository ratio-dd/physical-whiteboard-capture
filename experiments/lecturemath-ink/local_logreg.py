"""One-shot LectureMath local-background logistic candidate (research only).

Uses only the manifest's train/validation pairs and lecture-level split. The
feature map is deliberately interpretable: local background deficit, BGR/YCrCb
colour differences, gradients, and multi-scale directional support. It never
reads the held-out test lectures or user captures.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2, numpy as np


def features(bgr):
    z=cv2.GaussianBlur(bgr,(3,3),.7); g=cv2.cvtColor(z,cv2.COLOR_BGR2GRAY).astype('f4')/255
    paper=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8)); deficit=np.clip(paper-g,0,1)
    local=cv2.GaussianBlur(g,(0,0),5); contrast=np.abs(g-local)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    edge=np.sqrt(gx*gx+gy*gy); ycc=cv2.cvtColor(z,cv2.COLOR_BGR2YCrCb).astype('f4')
    cr=np.abs(ycc[...,1]-128)/128; cb=np.abs(ycc[...,2]-128)/128
    dirs=[]
    signal=np.clip(deficit+.45*np.sqrt(cr*cr+cb*cb),0,1)
    for n in (5,9):
        dirs += [cv2.morphologyEx(signal,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(n,1)))-signal,
                 cv2.morphologyEx(signal,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(1,n)))-signal]
    return np.stack([g,deficit,contrast,edge,cr,cb,*dirs,np.ones_like(g)],-1).astype('f4')


def rows(root,spec,split):
    return [r for r in spec['pairs'] if r.get('split')==split]


def load(root,row):
    im=cv2.imread(str(root/row['image']),1); m=cv2.imread(str(root/row['mask']),0)
    if im is None or m is None: raise FileNotFoundError(row)
    return im,(m==0)


def collect(root,items,seed,per_image=6000):
    rng=np.random.default_rng(seed); xx=[]; yy=[]
    for row in items:
        im,gt=load(root,row); f=features(im).reshape(-1,features(im).shape[-1]); y=gt.reshape(-1)
        pos=np.flatnonzero(y); neg=np.flatnonzero(~y); n=min(per_image//2,len(pos),len(neg))
        if n==0: continue
        take=np.concatenate([rng.choice(pos,n,False),rng.choice(neg,n,False)]); rng.shuffle(take)
        xx.append(f[take]); yy.append(y[take].astype('f4'))
    return np.concatenate(xx),np.concatenate(yy)


def fit(x,y,seed=17,epochs=35,lr=.18):
    rng=np.random.default_rng(seed); w=np.zeros(x.shape[1],np.float32); b=0.;
    for _ in range(epochs):
        order=rng.permutation(len(y))
        for batch in np.array_split(order, max(1,len(order)//8192)):
            p=1/(1+np.exp(-np.clip(x[batch]@w+b,-30,30))); d=p-y[batch]
            w-=lr*(x[batch].T@d/len(batch)+.0005*w); b-=lr*float(d.mean())
    return w,b


def metrics(p,y,f):
    neg=~y; order=np.argsort(p[neg])[::-1]; negatives=p[neg][order]; allowed=int(np.floor(.01*len(negatives)))
    threshold=float(np.nextafter(negatives[allowed-1], np.inf)) if allowed else float(np.nextafter(negatives[0], np.inf))
    pred=p>=threshold; tp=int((pred&y).sum()); fp=int((pred&neg).sum()); tn=int((~pred&neg).sum()); fn=int((~pred&y).sum())
    faded=y&(f[:,1]<=np.median(f[y,1]))
    return {'threshold':threshold,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'false_positive_rate':fp/max(1,fp+tn),'faded_recall':float((pred&faded).sum()/max(1,faded.sum())),'positive_pixels':int(y.sum()),'predicted_pixels':int(pred.sum())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest',type=Path); ap.add_argument('--out',type=Path,default=Path('data/datasets/lecturemath-ink/local-logreg')); ap.add_argument('--seed',type=int,default=17); a=ap.parse_args()
    spec=json.loads(a.manifest.read_text()); root=a.manifest.parent; tr=rows(root,spec,'train'); va=rows(root,spec,'validation')
    assert len(tr)==75 and len(va)==15
    assert {r['lecture_id'] for r in tr}.isdisjoint({r['lecture_id'] for r in va})
    t=time.perf_counter(); x,y=collect(root,tr,a.seed); w,b=fit(x,y,a.seed)
    probs=[]; truth=[]; feats=[]
    for row in va:
        im,gt=load(root,row); f=features(im); probs.append((f.reshape(-1,f.shape[-1])@w+b)); truth.append(gt.reshape(-1)); feats.append(f.reshape(-1,f.shape[-1]))
    p=np.concatenate(probs); q=np.concatenate(truth); ff=np.concatenate(feats); report={'architecture':'local_background_logreg_v1','seed':a.seed,'train_pairs':len(tr),'validation_pairs':len(va),'train_lectures':sorted({r['lecture_id'] for r in tr}),'validation_lectures':sorted({r['lecture_id'] for r in va}),'features':12,'seconds':round(time.perf_counter()-t,3),'validation':metrics(1/(1+np.exp(-np.clip(p,-30,30))),q,ff)}
    report['features']=int(x.shape[1]); a.out.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.out/'weights.npz',w=w,b=np.array([b],np.float32)); (a.out/'report.json').write_text(json.dumps(report,indent=2))
    (a.out/'config.json').write_text(json.dumps({'model_version':'lecturemath_local_logreg_v1','threshold':report['validation']['threshold'],'source_report':'report.json'},indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
