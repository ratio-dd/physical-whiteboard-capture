"""Train/evaluate the context-aware V2 weak-ink candidate model."""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parents[2]))
from weak_ink_model import V2_FEATURES, WeakInkV2Model, _features_v2

OUT = Path(__file__).resolve().parent

def scene(seed: int, size: int = 80):
    rng = np.random.default_rng(seed)
    hue = int(seed * 17 % 15)
    base = np.full((size, size, 3), (238 + hue // 3, 239 + hue // 4, 232), np.uint8)
    gx = np.linspace(-12, 12, size, dtype=np.float32)[None, :, None]
    gy = np.linspace(-9, 9, size, dtype=np.float32)[:, None, None]
    base = np.clip(base.astype(np.float32) + gx + gy, 0, 255).astype(np.uint8)
    truth = np.zeros((size, size), np.uint8)
    mode = seed % 10
    if mode in (0, 1, 2, 3, 4, 5, 6):
        p0=(6 + seed % 9, 10 + (seed * 3) % 55); p1=(size - 7, 20 + (seed * 5) % 45)
        colors=[(55,55,55),(125,135,155),(90,105,175),(170,170,170),(90,110,205),(185,185,190),(135,145,180)]
        color=colors[mode]; width=1 if mode in (1,3,5) else 2
        cv2.line(base,p0,p1,color,width,cv2.LINE_AA); cv2.line(truth,p0,p1,255,width,cv2.LINE_AA)
    elif mode == 7:  # dirt, visually similar but isolated and not line-supported
        for _ in range(8):
            x=int(rng.integers(5,size-5)); y=int(rng.integers(5,size-5)); base[y:y+2,x:x+2]=(198,198,198)
    elif mode == 8:  # saturated glare plateau
        base[18:26, 5:size-5] = 255
    else:  # board rim, negative
        base[2:4,:]=(215,215,215); base[:,:3]=(215,215,215)
    # varied JPEG-like blur, noise and occasional local glare
    if seed % 3: base=cv2.GaussianBlur(base,(3,3),.6)
    base=np.clip(base.astype(np.float32)+rng.normal(0,1.7,base.shape),0,255).astype(np.uint8)
    return base,truth

def make_set(start,count):
    xs=[];ys=[]
    for seed in range(start,start+count):
        im,t=scene(seed);xs.append(_features_v2(im).reshape(-1,V2_FEATURES));ys.append((t.reshape(-1)>0).astype(np.float32))
    return np.concatenate(xs),np.concatenate(ys)

def fit(x,y,epochs=260,width=24):
    rng=np.random.default_rng(702);w1=rng.normal(0,.08,(V2_FEATURES,width)).astype(np.float32);b1=np.zeros(width,np.float32);w2=rng.normal(0,.08,(width,1)).astype(np.float32);b2=np.array([-2.8],np.float32)
    pos=max(1,int(y.sum()));neg=max(1,len(y)-pos);weight=np.where(y>0,.5/pos,.5/neg).astype(np.float32)
    for _ in range(epochs):
        h=np.tanh(x@w1+b1);p=1/(1+np.exp(-np.clip((h@w2+b2)[:,0],-30,30)));dz=((p-y)*weight)[:,None];dw2=h.T@dz;dh=(dz@w2.T)*(1-h*h);w2-=.9*dw2;b2-=.9*dz.sum(0);w1-=.9*(x.T@dh);b1-=.9*dh.sum(0)
    return {'w1':w1,'b1':b1,'w2':w2,'b2':b2}

def score(model,start,count,t=.72):
    rows=[]
    for seed in range(start,start+count):
        im,truth=scene(seed);p=model.predict_proba(im);pred=p>=t;gt=truth>0;tp=int(np.count_nonzero(pred&gt));fp=int(np.count_nonzero(pred&~gt));fn=int(np.count_nonzero(~pred&gt));rows.append((seed%10,tp,fp,fn,int(gt.sum()),int((~gt).sum())))
    a=np.asarray(rows);out={}
    for kind in range(10):
        q=a[a[:,0]==kind]
        if len(q):
            _,tp,fp,fn,pos,neg=q.sum(0);out[str(kind)]={'tp':int(tp),'fp':int(fp),'fn':int(fn),'positive_pixels':int(pos),'negative_pixels':int(neg),'precision':float(tp/max(1,tp+fp)),'recall':float(tp/max(1,tp+fn))}
    return out

def main():
    x,y=make_set(0,240);w=fit(x,y);m=WeakInkV2Model(w)
    parser=argparse.ArgumentParser(); parser.add_argument('--write-model',action='store_true',help='explicitly replace models/weak_ink_v2.npz'); args=parser.parse_args()
    model_path=OUT/'artifacts'/'weak_ink_v2.npz'; model_path.parent.mkdir(exist_ok=True)
    if args.write_model: model_path=OUT.parents[1]/'models'/'weak_ink_v2.npz'; model_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(model_path,**w)
    result={'version':'v2','seed':702,'train_scenes':240,'test_scenes':100,'threshold':.72,'unseen':score(m,1000,100),'features':V2_FEATURES,'model_path':'models/weak_ink_v2.npz','note':'Synthetic-only evidence; scores are candidate scores, not calibrated probabilities.'}
    (OUT/'results_v2.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__': main()
