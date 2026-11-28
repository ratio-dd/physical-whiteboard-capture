"""Prepare LectureMath official binary masks as YOLO polygon labels.

Only the public LectureMath manifest is read. A component is an ink instance
and mask value 0 is positive. Tiles retain empty board/glare blocks so a model
sees negatives. The output is deliberately ignored local data.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def starts(n: int, tile: int, stride: int):
    if n <= tile: return [0]
    out = list(range(0, n - tile + 1, stride))
    if out[-1] != n - tile: out.append(n - tile)
    return out

def poly_for_component(comp: np.ndarray, x0: int, y0: int, w: int, h: int):
    contours, _ = cv2.findContours((comp > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) >= 3:
            q = cv2.approxPolyDP(c, 0.35, True).reshape(-1, 2).astype(np.float32)
        else:
            x, y = c.reshape(-1, 2)[0]
            q = np.array([[x-0.5,y-0.5],[x+0.5,y-0.5],[x+0.5,y+0.5],[x-0.5,y+0.5]], np.float32)
        q[:, 0] = np.clip(q[:, 0], 0, w-1); q[:, 1] = np.clip(q[:, 1], 0, h-1)
        if len(q) >= 3:
            out.append(q)
    return out

def labels_from_mask(mask: np.ndarray, max_instances: int = 256):
    ink = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    polys = []
    h, w = mask.shape
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 1: continue
        comp = (lab == i).astype(np.uint8)
        for q in poly_for_component(comp, 0, 0, w, h):
            q[:, 0] /= w; q[:, 1] /= h
            polys.append((area, q))
    polys.sort(key=lambda z: z[0], reverse=True)
    return [q for _, q in polys[:max_instances]]

def write_label(path: Path, polys):
    lines = []
    for q in polys:
        vals = ' '.join(f'{float(v):.6f}' for xy in q for v in xy)
        lines.append('0 ' + vals)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''))

def rasterize(label: Path, h: int, w: int):
    out = np.zeros((h, w), np.uint8)
    for line in label.read_text().splitlines():
        a = [float(x) for x in line.split()[1:]]
        p = np.array(a, np.float32).reshape(-1, 2)
        p[:, 0] *= w; p[:, 1] *= h
        cv2.fillPoly(out, [np.rint(p).astype(np.int32)], 1)
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', type=Path, default=ROOT/'data/datasets/lecturemath/manifest_multi.json'); ap.add_argument('--out', type=Path, default=ROOT/'data/datasets/lecturemath-yolo'); ap.add_argument('--tile', type=int, default=640); ap.add_argument('--overlap', type=int, default=96); ap.add_argument('--max-negative-per-pair', type=int, default=2); ap.add_argument('--audit', action='store_true'); args = ap.parse_args()
    spec = json.loads(args.manifest.read_text()); root = args.manifest.parent; stride = args.tile - args.overlap
    if args.out.exists(): shutil.rmtree(args.out)
    records=[]; audit=[]
    for row in spec['pairs']:
        split = row['split']; im = cv2.imread(str(root/row['image']), cv2.IMREAD_COLOR); mask = cv2.imread(str(root/row['mask']), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None or im.shape[:2] != mask.shape: raise RuntimeError(row)
        H,W=mask.shape; neg=0
        for y0 in starts(H,args.tile,stride):
            for x0 in starts(W,args.tile,stride):
                th=min(args.tile,H-y0); tw=min(args.tile,W-x0); mt=mask[y0:y0+th,x0:x0+tw]; polys=labels_from_mask(mt)
                if not polys:
                    if neg >= args.max_negative_per_pair: continue
                    neg += 1
                stem=f"{row['lecture_id']}_{row['frame_id']}_x{x0}_y{y0}"
                ip=args.out/'images'/split/(stem+'.jpg'); lp=args.out/'labels'/split/(stem+'.txt'); ip.parent.mkdir(parents=True,exist_ok=True); lp.parent.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(ip),im[y0:y0+th,x0:x0+tw], [cv2.IMWRITE_JPEG_QUALITY,95]); write_label(lp,polys)
                rec={'split':split,'lecture_id':row['lecture_id'],'frame_id':row['frame_id'],'image':str(ip.relative_to(args.out)),'label':str(lp.relative_to(args.out)),'tile_xywh':[x0,y0,tw,th],'instances':len(polys),'source_image':row['image'],'source_mask':row['mask']}; records.append(rec)
                if args.audit and polys:
                    rr=rasterize(lp,th,tw); gt=(mt==0).astype(np.uint8); inter=np.count_nonzero(rr&gt); union=np.count_nonzero(rr|gt); audit.append({'image':rec['image'],'gt_ink':int(gt.sum()),'poly_ink':int(rr.sum()),'intersection':int(inter),'iou':float(inter/max(1,union)),'recall':float(inter/max(1,int(gt.sum()))),'precision':float(inter/max(1,int(rr.sum())))})
    args.out.mkdir(parents=True,exist_ok=True); (args.out/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: ink\n')
    (args.out/'tile_manifest.json').write_text(json.dumps({'source_manifest':str(args.manifest),'tile':args.tile,'overlap':args.overlap,'positive_mask_value':0,'class_names':['ink'],'records':records},indent=2))
    if args.audit: (args.out/'polygon_raster_audit.json').write_text(json.dumps({'count':len(audit),'rows':audit,'mean_iou':float(np.mean([x['iou'] for x in audit])) if audit else None,'mean_recall':float(np.mean([x['recall'] for x in audit])) if audit else None,'mean_precision':float(np.mean([x['precision'] for x in audit])) if audit else None,'note':'YOLO polygons cannot encode holes explicitly; connected-component external contours are used.'},indent=2))
    print(json.dumps({'tiles':len(records),'positive_tiles':sum(r['instances']>0 for r in records),'negative_tiles':sum(r['instances']==0 for r in records),'audit':len(audit)}))
if __name__=='__main__': main()
