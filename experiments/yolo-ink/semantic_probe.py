"""YOLO11s frozen high-resolution feature + semantic pixel head probe.

This is separate from instance-segmentation YOLO. The backbone/neck is frozen;
only a 1x1 semantic head is trained on the public LectureMath train split.
No user data or held-out test lectures are read.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np

IMG=320; FEAT=40

def letterbox(im, fill=255):
    h,w=im.shape[:2]; scale=min(IMG/w,IMG/h); nw,nh=round(w*scale),round(h*scale)
    x=cv2.resize(im,(nw,nh),interpolation=cv2.INTER_AREA); out=np.full((IMG,IMG,3),fill,np.uint8); x0=(IMG-nw)//2;y0=(IMG-nh)//2;out[y0:y0+nh,x0:x0+nw]=x;return out,(x0,y0,nw,nh)

def letterbox_mask(mask, geom):
    x0,y0,nw,nh=geom; out=np.zeros((IMG,IMG),np.float32); x=cv2.resize(mask.astype(np.float32),(nw,nh),interpolation=cv2.INTER_NEAREST); out[y0:y0+nh,x0:x0+nw]=x;return out

def load_pair(root,row):
    im=cv2.imread(str(root/row['image']),1); m=cv2.imread(str(root/row['mask']),0)
    if im is None or m is None: raise FileNotFoundError(row)
    z,g=letterbox(im); y=letterbox_mask(m==0,g); return z,y

class FrozenFeature:
    def __init__(self,weights):
        import torch
        from ultralytics import YOLO
        self.torch=torch; self.net=YOLO(str(weights)).model.eval(); self.out=None
        for p in self.net.parameters(): p.requires_grad=False
        self.hook=self.net.model[16].register_forward_hook(lambda _m,_i,o:setattr(self,'out',o))
    def __call__(self,bgr):
        x=self.torch.from_numpy(bgr.transpose(2,0,1)).float().div(255).unsqueeze(0)
        with self.torch.no_grad(): self.net(x)
        return self.out.detach()
    def close(self): self.hook.remove()

class YOLOSemanticModel:
    """Unified inference factory for the frozen-feature semantic checkpoint."""
    model_version='yolo11s-layer16-semantic-head-v1'
    def __init__(self, weights, head_path, threshold):
        import torch
        self.torch=torch; self.frozen=FrozenFeature(weights); self.head=torch.nn.Conv2d(128,1,1); self.head.load_state_dict(torch.load(head_path,map_location='cpu')['head']); self.head.eval(); self.threshold=float(threshold)
    def predict(self,bgr,known=None,threshold=None):
        if threshold is not None and float(threshold)!=self.threshold: raise ValueError('threshold override is forbidden')
        im,g=letterbox(bgr); f=self.frozen(im);
        with self.torch.no_grad(): p=self.torch.sigmoid(self.torch.nn.functional.interpolate(self.head(f),size=(IMG,IMG),mode='bilinear',align_corners=False))[0,0].numpy()
        x0,y0,nw,nh=g; full=np.zeros(bgr.shape[:2],np.float32); full=cv2.resize(p[y0:y0+nh,x0:x0+nw],(bgr.shape[1],bgr.shape[0]),interpolation=cv2.INTER_LINEAR)
        if known is not None: full=np.where(known>0,full,0).astype(np.float32)
        return full,(full>=np.float32(self.threshold)).astype(np.uint8)*255
    def close(self): self.frozen.close()

def operating(prob,truth,detail):
    y=truth.astype(bool); neg=~y; vals=np.sort(prob[neg])[::-1]; allowed=int(np.floor(.01*len(vals))); t=float(np.nextafter(vals[allowed-1],np.inf)) if allowed else float(np.nextafter(vals[0],np.inf)); pred=prob>=t; tp=int((pred&y).sum());fp=int((pred&neg).sum());tn=int((~pred&neg).sum()); faded=y&(detail<=np.median(detail[y])); return {'threshold':t,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+int((~pred&y).sum())),'faded_recall':float((pred&faded).sum()/max(1,faded.sum())),'fpr':fp/max(1,fp+tn),'positive_pixels':int(y.sum())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('manifest',type=Path);ap.add_argument('weights',type=Path);ap.add_argument('--out',type=Path,default=Path('data/datasets/lecturemath-ink/yolo-semantic'));ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--seed',type=int,default=17);a=ap.parse_args()
    import torch
    torch.manual_seed(a.seed); spec=json.loads(a.manifest.read_text());root=a.manifest.parent;tr=[r for r in spec['pairs'] if r['split']=='train'];va=[r for r in spec['pairs'] if r['split']=='validation']; assert len(tr)==75 and len(va)==15
    assert {r['lecture_id'] for r in tr}.isdisjoint({r['lecture_id'] for r in va})
    frozen=FrozenFeature(a.weights); cache=[]; start=time.perf_counter()
    for row in tr:
        im,y=load_pair(root,row); f=frozen(im); cache.append((f,y));
    head=torch.nn.Conv2d(128,1,1); opt=torch.optim.Adam(head.parameters(),lr=.02,weight_decay=1e-4); pos=sum(int(y.sum()) for _,y in cache); total=sum(y.size for _,y in cache); lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(total-pos)/max(1,pos)],dtype=torch.float32))
    for _ in range(a.epochs):
        for f,y in cache:
            yy=torch.from_numpy(cv2.resize(y,(FEAT,FEAT),interpolation=cv2.INTER_AREA)).float()[None,None]; logits=head(f); loss=lossfn(logits,yy);opt.zero_grad();loss.backward();opt.step()
    probs=[];truth=[];details=[]
    with torch.no_grad():
      for row in va:
        im,y=load_pair(root,row); f=frozen(im); p=torch.sigmoid(torch.nn.functional.interpolate(head(f),size=(IMG,IMG),mode='bilinear',align_corners=False))[0,0].numpy(); probs.append(p.reshape(-1));truth.append(y.reshape(-1)); gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype('f4')/255;paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8));details.append(np.clip(paper-gray,0,1).reshape(-1))
    p=np.concatenate(probs);y=np.concatenate(truth).astype(bool);d=np.concatenate(details); result={'model':'YOLO11s-layer16-frozen-plus-1x1-semantic-head','layer16_channels':128,'input':IMG,'feature':FEAT,'train_pairs':75,'validation_pairs':15,'epochs':a.epochs,'seconds':round(time.perf_counter()-start,3),'validation':operating(p,y,d)}
    # zero-feature ablation: only the trained bias is retained.
    z=np.full_like(p,float(head.bias.detach().numpy()[0])); result['feature_zero_ablation']=operating(1/(1+np.exp(-z)),y,d)
    a.out.mkdir(parents=True,exist_ok=True);torch.save({'head':head.state_dict(),'layer':16,'channels':128},a.out/'head.pt');(a.out/'report.json').write_text(json.dumps(result,indent=2));(a.out/'config.json').write_text(json.dumps({'model_version':YOLOSemanticModel.model_version,'threshold':result['validation']['threshold'],'weights':str(a.weights),'head':'head.pt'},indent=2));frozen.close();print(json.dumps(result,indent=2))

def load_model(weights, run_dir):
    cfg=json.loads((Path(run_dir)/'config.json').read_text()); return YOLOSemanticModel(weights,Path(run_dir)/cfg['head'],cfg['threshold'])
if __name__=='__main__':main()
