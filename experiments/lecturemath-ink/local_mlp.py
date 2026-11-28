"""Last neural candidate: one hidden layer over the proven local features."""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent)); from local_logreg import rows,collect,load,features,metrics

def fit(x,y,seed=17,epochs=45,lr=.12,width=24):
    r=np.random.default_rng(seed); mu=x.mean(0); sd=x.std(0)+1e-4; z=(x-mu)/sd
    n=len(y); pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0); keep=np.r_[pos,neg];
    w1=r.normal(0,.08,(z.shape[1],width)).astype('f4'); b1=np.zeros(width,'f4'); w2=r.normal(0,.08,width).astype('f4'); b2=np.float32(0)
    for _ in range(epochs):
        order=r.permutation(keep)
        for batch in np.array_split(order,max(1,len(order)//8192)):
            h=np.maximum(z[batch]@w1+b1,0); p=1/(1+np.exp(-np.clip(h@w2+b2,-30,30))); d=(p-y[batch]); d1=(d[:,None]*w2)*(h>0)
            w2-=lr*(h.T@d/len(batch)+.0005*w2); b2-=lr*float(d.mean()); w1-=lr*(z[batch].T@d1/len(batch)+.0005*w1); b1-=lr*d1.mean(0)
    return w1,b1,w2,b2,mu,sd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest',type=Path);ap.add_argument('--out',type=Path,default=Path('data/datasets/lecturemath-ink/local-mlp'));ap.add_argument('--seed',type=int,default=17);a=ap.parse_args(); spec=json.loads(a.manifest.read_text());root=a.manifest.parent;tr=rows(root,spec,'train');va=rows(root,spec,'validation');assert {r['lecture_id'] for r in tr}.isdisjoint({r['lecture_id'] for r in va})
    t=time.perf_counter();x,y=collect(root,tr,a.seed);w1,b1,w2,b2,mu,sd=fit(x,y,a.seed);ps=[];ys=[];fs=[]
    for row in va:
        im,gt=load(root,row); f=features(im); z=(f.reshape(-1,f.shape[-1])-mu)/sd; h=np.maximum(z@w1+b1,0); ps.append(1/(1+np.exp(-np.clip(h@w2+b2,-30,30))));ys.append(gt.ravel());fs.append(f.reshape(-1,f.shape[-1]))
    p=np.concatenate(ps);q=np.concatenate(ys);ff=np.concatenate(fs);rep={'architecture':'local_background_mlp_v1','features':int(x.shape[1]),'hidden':24,'seed':a.seed,'train_pairs':len(tr),'validation_pairs':len(va),'seconds':round(time.perf_counter()-t,3),'validation':metrics(p,q,ff)};a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'weights.npz',w1=w1,b1=b1,w2=w2,b2=np.array([b2]),mu=mu,sd=sd);(a.out/'report.json').write_text(json.dumps(rep,indent=2));(a.out/'config.json').write_text(json.dumps({'model_version':'lecturemath_local_mlp_v1','threshold':rep['validation']['threshold'],'features':rep['features']},indent=2));print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
