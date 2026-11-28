"""Train the isolated LectureMath tiny FCN once a manifest is available.

Manifest JSON format: {"train":[{"image":"...","mask":"..."}],
"val":[...]} with paths relative to manifest. Outputs default to ignored
``data/datasets/lecturemath-ink/`` and never touch product weights.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from model import TinyFCN

def load_pair(root,row,polarity):
    im=cv2.imread(str(root/row['image']),cv2.IMREAD_COLOR); mask=cv2.imread(str(root/row['mask']),cv2.IMREAD_GRAYSCALE)
    if im is None or mask is None: raise FileNotFoundError(row)
    positive=polarity.get('positive_values')
    if not positive: raise ValueError('manifest must declare positive_values; refusing to guess mask polarity')
    target=np.isin(mask,positive).astype(np.float32)
    ignore=None
    if row.get('ignore'):
        ig=cv2.imread(str(root/row['ignore']),cv2.IMREAD_GRAYSCALE)
        if ig is None: raise FileNotFoundError(row['ignore'])
        ignore=ig>0
    return im,target[...,None],ignore

def crop(im,y,ignore,rng,size=128):
    h,w=im.shape[:2]
    if h<size or w<size: im=cv2.copyMakeBorder(im,0,max(0,size-h),0,max(0,size-w),cv2.BORDER_CONSTANT,value=255); y=cv2.copyMakeBorder(y,0,max(0,size-h),0,max(0,size-w),cv2.BORDER_CONSTANT); h,w=im.shape[:2]
    for _ in range(12):
        yy=int(rng.integers(0,h-size+1)); xx=int(rng.integers(0,w-size+1)); q=y[yy:yy+size,xx:xx+size]
        if q.mean()>0.001 or _==11: return im[yy:yy+size,xx:xx+size],q,ignore[yy:yy+size,xx:xx+size] if ignore is not None else None

def augment(im,y,ignore,rng):
    """Photometric augmentation; saturated glare is marked unknown, not ink."""
    z=im.astype(np.float32); z=z*rng.uniform(.82,1.12)+rng.uniform(-12,12)
    gains=rng.uniform(.94,1.06,3); z*=gains[None,None,:]
    if rng.random()<.35: z=cv2.GaussianBlur(z,(3,3),rng.uniform(.2,.8))
    z+=rng.normal(0,1.8,z.shape)
    ig=np.zeros(y.shape[:2],bool) if ignore is None else ignore.copy()
    if rng.random()<.30:
        h,w=ig.shape; gh=int(rng.integers(4,max(5,h//8))); gw=int(rng.integers(12,max(13,w//2))); yy=int(rng.integers(0,max(1,h-gh))); xx=int(rng.integers(0,max(1,w-gw)))
        z[yy:yy+gh,xx:xx+gw]=255; ig[yy:yy+gh,xx:xx+gw]=True
    return np.clip(z,0,255).astype(np.uint8),y,ig

def evaluate(model,root,rows,polarity):
    vals=[]; base=[]; weak=[]
    for row in rows:
        im,y,ig=load_pair(root,row,polarity); h,w=im.shape[:2]; ys=np.linspace(0,max(0,h-128),4).astype(int); xs=np.linspace(0,max(0,w-128),4).astype(int)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)/255; paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)); detail=np.clip(paper-gray,0,1)
        for yy in ys:
            for xx in xs:
                tile=im[yy:yy+128,xx:xx+128]; p,_=model.predict(tile); valid=np.ones(p.shape,bool) if ig is None else ~ig[yy:yy+128,xx:xx+128]; gt=y[yy:yy+128,xx:xx+128,0]>0
                vals.append((p[valid],gt[valid])); base.append(detail[yy:yy+128,xx:xx+128][valid])
    p=np.concatenate([a for a,b in vals]); g=np.concatenate([b for a,b in vals]); out=[]
    for t in np.linspace(.1,.95,18):
        q=p>=t; tp=np.count_nonzero(q&g); fp=np.count_nonzero(q&~g); fn=np.count_nonzero(~q&g); tn=np.count_nonzero(~q&~g)
        out.append({'threshold':float(t),'precision':float(tp/max(1,tp+fp)),'recall':float(tp/max(1,tp+fn)),'false_positive_rate':float(fp/max(1,fp+tn)),'positive_pixels':int(g.sum()),'predicted_pixels':int(q.sum())})
    eligible=[r for r in out if r['false_positive_rate']<=.01 and r['positive_pixels']>0]; chosen=max(eligible,key=lambda r:r['recall']) if eligible else min(out,key=lambda r:r['false_positive_rate'])
    detail=np.concatenate(base); positive_detail=detail[g]; cutoff=float(np.percentile(positive_detail,50)) if len(positive_detail) else 0.0; faded=(g)&(detail<=cutoff); pred=p>=chosen['threshold']; fp=np.count_nonzero(pred&~g); tn=np.count_nonzero(~pred&~g)
    baseline=(detail>=.04); btp=np.count_nonzero(baseline&g); bfp=np.count_nonzero(baseline&~g)
    return {'operating_points':out,'chosen_low_fpr':chosen,'faded_definition':'positive pixels with local deficit <= validation positive median','faded_cutoff':cutoff,'faded_recall':float(np.mean(pred[faded])) if faded.any() else None,'background_fpr':float(fp/max(1,fp+tn)),'contrast_baseline_detail_ge_0.04':{'precision':float(btp/max(1,btp+bfp)),'recall':float(btp/max(1,int(g.sum()))),'false_positive_rate':float(bfp/max(1,bfp+tn))},'non_background_validation':bool(g.any())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('manifest',type=Path); ap.add_argument('--out',type=Path,default=Path('data/datasets/lecturemath-ink')); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--seed',type=int,default=17); args=ap.parse_args()
    spec=json.loads(args.manifest.read_text()); root=args.manifest.parent
    if 'pairs' in spec:
        spec={"train":[r for r in spec['pairs'] if r.get('split')=='train'],"val":[r for r in spec['pairs'] if r.get('split')=='validation'],"mask":{"positive_values":[0]}}
    rng=np.random.default_rng(args.seed); model=TinyFCN.init(args.seed); log=[]; polarity=spec.get('mask',{})
    train_ids={r.get('lecture_id',r.get('id',r['image'])) for r in spec['train']}; val_ids={r.get('lecture_id',r.get('id',r['image'])) for r in spec.get('val',[])}
    assert not train_ids & val_ids, 'lecture-level train/val leakage'
    best=float('inf'); best_w=None; stale=0
    for epoch in range(args.epochs):
        order=rng.permutation(len(spec['train'])); losses=[]; t=time.perf_counter()
        for i in order:
            raw=load_pair(root,spec['train'][int(i)],polarity)
            for _ in range(3):
                im,y,ig=crop(*raw,rng); im,y,ig=augment(im,y,ig,rng); x=__import__('model').featurize(im)[None]
                if ig is not None: y=np.where(ig[...,None],0,y)
                loss,gr=model.loss_grad(x,y[None]);
                for w,g in zip(model.w,gr): w-=0.03*g
                losses.append(loss)
        vl=[]
        for row in spec.get('val',[]):
            im,y,ig=load_pair(root,row,polarity); p,_=model.predict(im); valid=np.ones(y.shape[:2],bool) if ig is None else ~ig; vl.append(float(-(y[...,0][valid]*np.log(p[valid]+1e-6)+(1-y[...,0][valid])*np.log(1-p[valid]+1e-6)).mean()))
        v=float(np.mean(vl)) if vl else float(np.mean(losses));
        if v < best-1e-4: best=v; best_w=[a.copy() for a in model.w]; stale=0
        else: stale+=1
        rec={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_loss':float(np.mean(vl)) if vl else None,'seconds':round(time.perf_counter()-t,3),'best':stale==0}; log.append(rec); print(json.dumps(rec))
        if stale>=6: break
    if best_w is not None: model.w=best_w
    args.out.mkdir(parents=True,exist_ok=True); model.save(args.out/'lecturemath_tinyfcn.npz'); report={'seed':args.seed,'epochs':args.epochs,'log':log,'validation':evaluate(model,root,spec.get('val',[]),polarity) if spec.get('val') else None}; (args.out/'training.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()
