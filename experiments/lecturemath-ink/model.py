"""Tiny NumPy spatial-convolution ink segmenter (research-only).

Input is corrected BGR uint8 plus optional known mask. The model is deliberately
small and dependency-light: three same-padded 3x3 convolutions with ReLU and a
sigmoid output. It is not imported by the product runtime.
"""
from __future__ import annotations
import cv2
import numpy as np

CHANNELS = 4  # B,G,R, local grayscale deficit

def featurize(bgr: np.ndarray, known: np.ndarray | None = None) -> np.ndarray:
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected HxWx3 uint8 BGR")
    x = bgr.astype(np.float32) / 255.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    paper = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    deficit = np.clip(paper - g, 0, 1)
    out = np.concatenate([x, deficit[..., None]], axis=2)
    if known is not None:
        if known.shape != g.shape: raise ValueError("known shape differs")
        out *= (known > 0)[..., None]
    return out

def _pad(x, p=1):
    # Constant padding keeps the backward crop exact. Reflect padding requires
    # accumulating mirrored border gradients and silently biased early layers.
    return np.pad(x, ((0,0),(p,p),(p,p),(0,0)), mode='constant')

def _conv(x, w, b):
    # NHWC, same 3x3 convolution using explicit spatial shifts.
    z=np.zeros((x.shape[0],x.shape[1],x.shape[2],w.shape[3]),np.float32)
    q=_pad(x)
    for dy in range(3):
        for dx in range(3): z += q[:,dy:dy+x.shape[1],dx:dx+x.shape[2],:] @ w[dy,dx]
    return z+b

def _conv_grad(x,w,dz):
    q=_pad(x); dw=np.zeros_like(w); db=dz.sum((0,1,2)); dx=np.zeros_like(q)
    for dy in range(3):
        for dx0 in range(3):
            patch=q[:,dy:dy+x.shape[1],dx0:dx0+x.shape[2],:]
            dw[dy,dx0]=np.einsum('nhwc,nhwo->co',patch,dz)
            dx[:,dy:dy+x.shape[1],dx0:dx0+x.shape[2]] += dz @ w[dy,dx0].T
    return dx[:,1:-1,1:-1],dw,db

class TinyFCN:
    def __init__(self, weights): self.w=[np.asarray(a,np.float32) for a in weights]
    @classmethod
    def init(cls, seed=17):
        r=np.random.default_rng(seed)
        return cls([r.normal(0,.08,(3,3,CHANNELS,8)).astype('f4'),np.zeros(8,'f4'),r.normal(0,.08,(3,3,8,8)).astype('f4'),np.zeros(8,'f4'),r.normal(0,.08,(3,3,8,1)).astype('f4'),np.zeros(1,'f4')])
    def forward(self,x,training=False):
        w1,b1,w2,b2,w3,b3=self.w; z1=_conv(x,w1,b1); a1=np.maximum(z1,0); z2=_conv(a1,w2,b2); a2=np.maximum(z2,0); z3=_conv(a2,w3,b3); p=1/(1+np.exp(-np.clip(z3,-30,30))); return p,(z1,a1,z2,a2,z3)
    def loss_grad(self,x,y):
        p,c=self.forward(x,True);z1,a1,z2,a2,z3=c; n=np.prod(y.shape)
        # Balanced BCE + soft Dice keeps sparse/faint ink from being optimized
        # away as background. Positive weight is fixed before validation.
        posw=10.0; den=np.sum(p)+np.sum(y)+1e-5; num=2*np.sum(p*y)+1e-5
        dz3=((p-y)*(1+(posw-1)*y)/n) - (0.10*(2*y*den-num)/(den*den))
        da2,dw3,db3=_conv_grad(a2,self.w[4],dz3); dz2=da2*(z2>0); da1,dw2,db2=_conv_grad(a1,self.w[2],dz2); dz1=da1*(z1>0); _,dw1,db1=_conv_grad(x,self.w[0],dz1)
        bce=-(posw*y*np.log(p+1e-6)+(1-y)*np.log(1-p+1e-6)).sum()/n; loss=float(bce+0.10*(1-num/den)); return loss,[dw1,db1,dw2,db2,dw3,db3]
    def predict(self,bgr,known=None,threshold=.5):
        p,_=self.forward(featurize(bgr,known)[None]); p=p[0,:,:,0]
        if known is not None:
            if known.shape != p.shape: raise ValueError("known shape differs")
            p=np.where(known>0,p,0).astype(np.float32)
        return p,(p>=threshold).astype(np.uint8)*255
    def save(self,path): np.savez_compressed(path,**{f'w{i}':v for i,v in enumerate(self.w)})
    @classmethod
    def load(cls,path):
        with np.load(path) as d:return cls([d[f'w{i}'] for i in range(6)])
