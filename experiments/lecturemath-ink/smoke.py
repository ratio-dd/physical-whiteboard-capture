"""Random-input forward/backprop/serialization smoke; no dataset access."""
from pathlib import Path
import tempfile, numpy as np
from model import TinyFCN, featurize

rng=np.random.default_rng(123); bgr=rng.integers(0,256,(2,32,32,3),dtype=np.uint8); y=rng.integers(0,2,(2,32,32,1)).astype('f4'); m=TinyFCN.init(7); x=np.stack([featurize(a) for a in bgr]); loss,gr=m.loss_grad(x,y)
assert np.isfinite(loss) and all(g.shape==w.shape for g,w in zip(gr,m.w))
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'m.npz'; m.save(p); q=TinyFCN.load(p); a,_=m.forward(x); z,_=q.forward(x); assert np.allclose(a,z)
print({'ok':True,'x':x.shape,'loss':loss,'params':sum(w.size for w in m.w)})
