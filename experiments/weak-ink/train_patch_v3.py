"""A bounded V3 experiment: classify hard traditional candidates by local patch.

This is research-only. It never reads private captures and does not replace the
V1/V2 candidate APIs. Background families and line shapes are held out by seed
range to avoid claiming in-distribution generalisation as OOD performance.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
PATCH = 9


def scene(seed: int, size: int = 72):
    rng = np.random.default_rng(seed)
    family = seed % 6
    if family < 3:
        tone = np.array([238, 239, 232], np.float32) + rng.uniform(-5, 5, 3)
        image = np.broadcast_to(tone, (size, size, 3)).copy()
    elif family == 3:
        tone = np.array([228, 231, 220], np.float32)
        image = np.broadcast_to(tone, (size, size, 3)).copy()
    elif family == 4:
        tone = np.array([244, 244, 242], np.float32)
        image = np.broadcast_to(tone, (size, size, 3)).copy()
    else:
        tone = np.array([235, 236, 228], np.float32)
        image = np.broadcast_to(tone, (size, size, 3)).copy()
    image += np.linspace(-10, 10, size, dtype=np.float32)[:, None, None]
    truth = np.zeros((size, size), np.uint8)
    shape = seed % 5
    # Continuous shape family. Shapes 3/4 are held out by the evaluator.
    if shape == 0:
        p0 = (8, int(rng.integers(12, 55))); p1 = (size - 8, int(rng.integers(16, 58)))
        cv2.line(image, p0, p1, (145, 155, 180), int(rng.integers(1, 3)), cv2.LINE_AA)
        cv2.line(truth, p0, p1, 255, 1, cv2.LINE_AA)
    elif shape == 1:
        pts = np.array([[8, 50], [25, 18], [45, 47], [64, 14]], np.int32)
        cv2.polylines(image, [pts], False, (160, 160, 165), 1, cv2.LINE_AA)
        cv2.polylines(truth, [pts], False, 255, 1, cv2.LINE_AA)
    elif shape == 2:
        pts = np.array([[10, 38], [22, 25], [38, 24], [52, 35], [64, 52]], np.int32)
        cv2.polylines(image, [pts], False, (125, 140, 175), 1, cv2.LINE_AA)
        cv2.polylines(truth, [pts], False, 255, 1, cv2.LINE_AA)
    elif shape == 3:
        cv2.ellipse(image, (36, 36), (24, 13), 25, 0, 300, (150, 155, 165), 1, cv2.LINE_AA)
        cv2.ellipse(truth, (36, 36), (24, 13), 25, 0, 300, 255, 1, cv2.LINE_AA)
    else:
        pts = np.array([[8, 12], [30, 34], [16, 58], [60, 14], [44, 60]], np.int32)
        cv2.polylines(image, [pts], False, (140, 145, 170), 1, cv2.LINE_AA)
        cv2.polylines(truth, [pts], False, 255, 1, cv2.LINE_AA)
    # Hard negatives are matched to the same local contrast budget.
    if seed % 2 == 0:
        cv2.rectangle(image, (4, 5), (size - 5, 8), (216, 216, 211), -1)  # rim/glare
    if seed % 3 == 0:
        for _ in range(18):
            x, y = rng.integers(4, size - 4, 2)
            image[y - 1:y + 2, x - 1:x + 2] -= rng.uniform(5, 18)
    image = np.clip(image + rng.normal(0, 2.0, image.shape), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.3, .9))
    return image, truth


def candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    score = (paper.astype(np.int16) - gray.astype(np.int16))
    return score >= 4


def patches(image, points):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    pad = PATCH // 2
    padded = np.pad(gray, pad, mode='reflect')
    return np.stack([padded[y:y + PATCH, x:x + PATCH].reshape(-1)
                     for y, x in points], axis=0)


def predict_full_frame(image, weights, threshold=.5, known=None):
    """Run the research model on a complete BGR frame (outside candidates=0)."""
    w1,b1,w2,b2,mu,sd=weights
    cand=candidates(image)
    if known is not None:
        cand &= np.asarray(known, dtype=np.uint8) > 0
    yy,xx=np.where(cand)
    out=np.zeros(cand.shape, np.float32)
    if len(yy):
        x=(patches(image,list(zip(yy,xx)))-mu)/sd
        p=1/(1+np.exp(-np.clip((np.tanh(x@w1+b1)@w2+b2)[:,0],-30,30)))
        out[yy,xx]=p
    return out, out >= threshold


def full_frame_metrics(pred, truth):
    gt=np.asarray(truth)>0; pp=np.asarray(pred)>0
    tp=int(np.count_nonzero(pp & gt)); fp=int(np.count_nonzero(pp & ~gt))
    fn=int(np.count_nonzero(~pp & gt)); positives=int(gt.sum()); negatives=int((~gt).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'positive_pixels':positives,'negative_pixels':negatives,
            'precision':float(tp/max(1,tp+fp)),'recall':float(tp/max(1,tp+fn)),
            'false_positive_rate':float(fp/max(1,negatives))}


def dataset(seeds):
    xs, ys = [], []
    for seed in seeds:
        image, truth = scene(int(seed)); cand = candidates(image)
        yy, xx = np.where(cand)
        if len(yy) == 0: continue
        # Keep all positive candidates and a bounded sample of negatives.
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(yy))[:min(500, len(yy))]
        pts = list(zip(yy[order], xx[order]))
        xs.append(patches(image, pts)); ys.append((truth[yy[order], xx[order]] > 0).astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys)


def fit(x, y, width=10, epochs=180):
    rng = np.random.default_rng(44)
    # Keep the study bounded and balanced. Pixel prevalence is intentionally
    # not used as a deployment probability estimate.
    positives=np.flatnonzero(y>0); negatives=np.flatnonzero(y==0)
    take=min(len(positives), len(negatives), 12000)
    keep=np.r_[rng.choice(positives,take,replace=False), rng.choice(negatives,take,replace=False)]
    x=x[keep]; y=y[keep]
    mu=x.mean(0); sd=x.std(0)+1e-4; x=(x-mu)/sd
    w1 = rng.normal(0, .05, (x.shape[1], width)); b1=np.zeros(width); w2=rng.normal(0,.05,(width,1)); b2=np.zeros(1)
    for _ in range(epochs):
        h=np.tanh(x@w1+b1); p=1/(1+np.exp(-np.clip((h@w2+b2)[:,0],-30,30))); dz=((p-y)/len(y))[:,None]; dh=(dz@w2.T)*(1-h*h); w2-=1.2*(h.T@dz); b2-=1.2*dz.sum(0); w1-=1.2*(x.T@dh); b1-=1.2*dh.sum(0)
    return w1,b1,w2,b2,mu,sd


def eval_model(weights, seeds, threshold=.5):
    w1,b1,w2,b2,mu,sd=weights; rows=[]
    # Sweep the integer detail score so the comparison can use the closest
    # achievable false-positive budget, rather than cherry-picking one cutoff.
    baseline_rows = {int(t): [] for t in range(0, 42)}
    for seed in seeds:
        image,truth=scene(int(seed));cand=candidates(image); yy,xx=np.where(cand)
        if len(yy)==0: continue
        px=(patches(image,list(zip(yy,xx)))-mu)/sd
        p=1/(1+np.exp(-np.clip((np.tanh(px@w1+b1)@w2+b2)[:,0],-30,30))); gt=truth[yy,xx]>0
        pred=p>=threshold
        rows.append((int(np.count_nonzero(pred & gt)),int(np.count_nonzero(pred & ~gt)),int(np.count_nonzero(~pred & gt)),int(gt.sum()),int((~gt).sum()),int(np.count_nonzero(cand))))
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        paper=cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        detail=(paper.astype(np.int16)-gray.astype(np.int16))[yy,xx]
        for t in baseline_rows:
            bp=detail >= t
            baseline_rows[t].append((int(np.count_nonzero(bp & gt)),int(np.count_nonzero(bp & ~gt)),int(np.count_nonzero(~bp & gt))))
    a=np.asarray(rows); tp,fp,fn,pos,neg,candidates_total=a.sum(0)
    model = {'tp':int(tp),'fp':int(fp),'fn':int(fn),'positive_pixels':int(pos),'negative_candidates':int(neg),'candidate_pixels':int(candidates_total),'precision':float(tp/max(1,tp+fp)),'recall':float(tp/max(1,tp+fn))}
    baseline = {}
    for t, vals in baseline_rows.items():
        b=np.asarray(vals).sum(0); bt,bf,bfn=b
        baseline[str(t)]={'tp':int(bt),'fp':int(bf),'fn':int(bfn),'precision':float(bt/max(1,bt+bf)),'recall':float(bt/max(1,bt+bfn))}
    same_t=min(baseline, key=lambda t: (abs(baseline[t]['fp']-int(fp)), -baseline[t]['precision']))
    return {'model':model,'traditional_detail_thresholds':baseline,
            'traditional_same_fp_budget':{'detail_threshold':int(same_t),**baseline[same_t],
                                          'target_model_fp':int(fp)}}


def main():
    train_seeds=[s for s in range(0,300) if s%5<3]
    test_seeds=[s for s in range(1000,1400) if s%5>=3]
    x,y=dataset(train_seeds); weights=fit(x,y)
    partial=eval_model(weights,test_seeds)
    full=[]; base=[]
    for seed in test_seeds:
        image,truth=scene(int(seed)); _,pred=predict_full_frame(image,weights)
        full.append(full_frame_metrics(pred,truth))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        base.append(full_frame_metrics((paper.astype(np.int16)-gray.astype(np.int16))>=4,truth))
    def aggregate(rows):
        keys=('tp','fp','fn','positive_pixels','negative_pixels'); sums={k:sum(r[k] for r in rows) for k in keys}
        sums['precision']=sums['tp']/max(1,sums['tp']+sums['fp']); sums['recall']=sums['tp']/max(1,sums['tp']+sums['fn']); sums['false_positive_rate']=sums['fp']/max(1,sums['negative_pixels']); return sums
    result={'version':'v3-patch','train_scenes':len(train_seeds),'test_scenes':len(test_seeds),'test_shape_families':'ellipse and self-crossing polyline held out','traditional_candidate_threshold':4,'evaluation':partial,'full_frame_evaluation':{'patch_model':aggregate(full),'traditional_detail_4':aggregate(base)},'note':'Synthetic-only. Full-frame metrics count every truth pixel; candidates outside the traditional mask are predicted zero. No private pixels used.'}
    parser=argparse.ArgumentParser(); parser.add_argument('--write-output',action='store_true',help='explicitly replace the checked-in V3 result and weight'); args=parser.parse_args()
    output_dir=OUT if args.write_output else OUT/'artifacts'; output_dir.mkdir(exist_ok=True)
    (output_dir/'results_v3.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); np.savez_compressed(output_dir/'weak_ink_patch_v3.npz',w1=weights[0],b1=weights[1],w2=weights[2],b2=weights[3],mu=weights[4],sd=weights[5]); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
