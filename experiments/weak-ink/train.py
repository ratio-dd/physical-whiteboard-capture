"""Train and evaluate the tiny weak-ink model on procedurally generated boards."""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from weak_ink_model import FEATURES, WeakInkModel, _features

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(812)


def scene(seed: int, size: int = 64):
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), rng.integers(225, 250), np.uint8)
    gradient = np.linspace(0, rng.integers(-22, 23), size, dtype=np.float32)[:, None, None]
    base = np.clip(base.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
    truth = np.zeros((size, size), np.uint8)
    kind = seed % 8
    if kind == 0:
        p0, p1 = (4, 10 + seed % 30), (size - 5, 20 + seed % 30)
        color = (40, 40, 40)
    elif kind == 1:
        p0, p1 = (8, size - 10), (size - 10, 9)
        color = (80, 105, 185)  # faint red in BGR
    elif kind == 2:
        p0, p1 = (10, 30), (size - 10, 30)
        color = (120, 125, 135)
    elif kind == 3:
        p0, p1 = (6, 8), (size - 8, 8)
        color = (205, 205, 205)  # thin dirt-like mark, negative
    elif kind == 4:
        p0, p1 = (8, 14), (size - 9, 45)
        color = (180, 190, 210)  # faint blue
    elif kind == 5:
        p0, p1 = (12, 32), (size - 12, 32)
        color = (145, 145, 145)  # thin grey
    elif kind == 6:
        p0, p1 = (5, 52), (size - 5, 18)
        color = (95, 120, 190)  # faint red, diagonal
    else:
        p0, p1 = (4, 10), (size - 5, 10)
        color = (255, 255, 255)  # pure bright glare, negative
    if kind not in (3, 7):
        cv2.line(base, p0, p1, color, 1 + seed % 3, cv2.LINE_AA)
        cv2.line(truth, p0, p1, 255, 1 + seed % 3, cv2.LINE_AA)
    elif kind == 3:
        cv2.line(base, p0, p1, color, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(base, (6, 8), (size - 7, 15), color, -1)
    # JPEG-like blur and sensor noise.
    base = cv2.GaussianBlur(base, (3, 3), 0.5)
    noise = rng.normal(0, 1.8, base.shape).astype(np.float32)
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return base, truth


def make_set(start, count):
    xs, ys = [], []
    for seed in range(start, start + count):
        image, truth = scene(seed)
        xs.append(_features(image).reshape(-1, FEATURES))
        ys.append((truth.reshape(-1) > 0).astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def fit(x, y, epochs=260, width=12):
    rng = np.random.default_rng(91)
    w1 = rng.normal(0, .12, (FEATURES, width)).astype(np.float32)
    b1 = np.zeros(width, np.float32)
    w2 = rng.normal(0, .12, (width, 1)).astype(np.float32)
    b2 = np.array([-3.0], np.float32)
    pos = max(1, int(y.sum())); neg = max(1, len(y) - pos)
    weight = np.where(y > 0, .5 / pos, .5 / neg).astype(np.float32)
    for _ in range(epochs):
        h = np.tanh(x @ w1 + b1); z = h @ w2 + b2
        p = 1 / (1 + np.exp(-np.clip(z[:, 0], -30, 30)))
        dz = ((p - y) * weight)[:, None]
        dw2 = h.T @ dz; db2 = dz.sum(0)
        dh = (dz @ w2.T) * (1 - h * h)
        w2 -= .8 * dw2; b2 -= .8 * db2; w1 -= .8 * (x.T @ dh); b1 -= .8 * dh.sum(0)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def metrics(model, start, count, threshold=.72):
    rows=[]
    for seed in range(start, start + count):
        im, truth = scene(seed); prob=model.predict_proba(im); pred=prob>=threshold; t=truth>0
        tp=np.count_nonzero(pred&t); fp=np.count_nonzero(pred&~t); fn=np.count_nonzero(~pred&t)
        # A conventional local-contrast baseline. It has no learned weights.
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        base=(paper.astype(np.int16)-gray.astype(np.int16))>=10
        btp=np.count_nonzero(base&t); bfp=np.count_nonzero(base&~t); bfn=np.count_nonzero(~base&t)
        rows.append((seed%8,tp,fp,fn,btp,bfp,bfn,int(t.sum()),int((~t).sum())))
    arr=np.asarray([r[1:] for r in rows]); tp,fp,fn,btp,bfp,bfn,pos,neg=arr.sum(0)
    per_scene={}
    for kind in range(8):
        part=arr[[r[0]==kind for r in rows]]
        if len(part)==0: continue
        qtp,qfp,qfn,qbtp,qbfp,qbfn,qpos,qneg=part.sum(0)
        per_scene[str(kind)]={"positive_pixels":int(qpos),"model":{"tp":int(qtp),"fp":int(qfp),"fn":int(qfn),"precision":float(qtp/max(1,qtp+qfp)),"recall":float(qtp/max(1,qtp+qfn))},"contrast_baseline":{"tp":int(qbtp),"fp":int(qbfp),"fn":int(qbfn),"precision":float(qbtp/max(1,qbtp+qbfp)),"recall":float(qbtp/max(1,qbtp+qbfn))}}
    return {"positive_pixels":int(pos),"negative_pixels":int(neg),"model":{"tp":int(tp),"fp":int(fp),"fn":int(fn),"precision":float(tp/max(1,tp+fp)),"recall":float(tp/max(1,tp+fn))},"contrast_baseline":{"tp":int(btp),"fp":int(bfp),"fn":int(bfn),"precision":float(btp/max(1,btp+bfp)),"recall":float(btp/max(1,btp+bfn))},"scenes":per_scene}


def main():
    x,y=make_set(0,160); weights=fit(x,y); model=WeakInkModel(weights)
    train=metrics(model,0,40); test=metrics(model,200,80)
    # Baseline uses a stricter local contrast criterion, without ML.
    parser=argparse.ArgumentParser(); parser.add_argument('--write-model', action='store_true', help='explicitly replace models/weak_ink_v1.npz')
    args=parser.parse_args()
    model_dir = OUT / "artifacts"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "weak_ink_v1.npz"
    if args.write_model:
        model_path = OUT.parents[1] / "models" / "weak_ink_v1.npz"
        model_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(model_path, **weights)
    result={"seed":812,"train_scenes":160,"test_scenes":80,"features":FEATURES,
            "threshold":.72,"train":train,"unseen_test":test,
            "model_path":"models/weak_ink_v1.npz",
            "note":"Synthetic-only evidence. Real whiteboard evaluation is blind and does not tune this model. Scores are candidate scores, not calibrated probabilities."}
    (OUT / "results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
