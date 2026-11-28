"""Evaluate a candidate binary-mask directory against temporal-stress-v1.

Prediction convention: one grayscale PNG per scenario/frame at
<scenario>/<frame>.png; nonzero pixels mean predicted ink. Unknown occlusion
pixels are excluded from visible recall/FP, then reported separately.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np

def score(pred,truth,unknown):
    valid=unknown==0; p=pred>0; t=truth>0; tp=np.count_nonzero(p&t&valid); fp=np.count_nonzero(p&~t&valid); fn=np.count_nonzero(~p&t&valid)
    return {'precision':float(tp/max(1,tp+fp)),'recall':float(tp/max(1,tp+fn)),'fp_pixels':int(fp),'fn_pixels':int(fn),'pred_pixels':int(p.sum()),'unknown_pred_rate':float(np.mean(p[unknown>0])) if unknown.any() else 0.0}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pred_root',type=Path); ap.add_argument('--fixture',type=Path,default=Path('data/datasets/temporal-stress-v1')); ap.add_argument('--out',type=Path,default=None); args=ap.parse_args(); man=json.loads((args.fixture/'manifest.json').read_text()); rows=[]
    for s in man['scenarios']:
      for f in range(man['frames_per_scenario']):
        pred=cv2.imread(str(args.pred_root/s['name']/f'{f:03d}.png'),0); truth=cv2.imread(str(args.fixture/s['physical_truth']/f'{f:03d}.png'),0); vis=cv2.imread(str(args.fixture/s['visible_truth']/f'{f:03d}.png'),0); unk=cv2.imread(str(args.fixture/s['unknown_mask']/f'{f:03d}.png'),0)
        if pred is None: raise SystemExit(f'missing prediction {s["name"]}/{f:03d}.png')
        r=score(pred,vis,unk); erased=(30<=f<=39 and not s['erase_phase']); r.update({'scenario':s['name'],'frame':f,'phase':'initial' if f<10 else 'occluded' if f<20 else 'revisible' if f<30 else 'erased','erase_residual_rate':float(np.mean((pred>0)&(truth==0))) if f>=30 else None}); rows.append(r)
    report={'fixture':str(args.fixture.resolve()),'prediction_root':str(args.pred_root.resolve()),'thresholds':{'visible_recall_min':0.8,'blank_fp_rate_max':0.01,'erase_residual_rate_max':0.01,'unknown_pixels':'excluded from visible score; reported as unknown_pred_rate'},'rows':rows,'note':'This evaluator does not decide whether a hidden line should be hallucinated. Occlusion is unknown; post-erase ink is a false positive.'}
    out=args.out or args.pred_root/'temporal_stress_report.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps({'frames':len(rows),'out':str(out)}))
if __name__=='__main__': main()
