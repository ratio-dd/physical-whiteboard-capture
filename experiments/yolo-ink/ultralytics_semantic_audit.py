"""Ultralytics-style polygon2mask/downsample audit without user data.

Ultralytics is intentionally not imported here: this checks the documented
polygon label semantics at mask_ratio=4 (the common training setting) and
records the limitation explicitly. It is not a claim of running a model.
"""
import json
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[2]

def poly_mask(label,h,w):
    out=np.zeros((h,w),np.uint8)
    for line in label.read_text().splitlines():
        a=np.asarray([float(x) for x in line.split()[1:]],np.float32).reshape(-1,2)
        p=np.rint(a*np.array([w,h],np.float32)).astype(np.int32)
        cv2.fillPoly(out,[p],1)
    return out

def main():
    root=ROOT/'data/datasets/lecturemath-yolo'; spec=json.loads((root/'tile_manifest.json').read_text()); ratio=4; rows=[]; thin=[]; by={}
    for r in spec['records']:
        lab=root/r['label']; im=cv2.imread(str(root/r['image']),cv2.IMREAD_COLOR); h,w=im.shape[:2]
        gt=cv2.imread(str(ROOT/'data/datasets/lecturemath'/r['source_mask']),cv2.IMREAD_GRAYSCALE); x,y,tw,th=r['tile_xywh']; gt=gt[y:y+th,x:x+tw]; gt=(gt==0).astype(np.uint8)
        pred=poly_mask(lab,h,w) if r['instances'] else np.zeros((h,w),np.uint8)
        sh,sw=max(1,(h+ratio-1)//ratio),max(1,(w+ratio-1)//ratio)
        # Approximate polygon2mask at the loader's mask_ratio resolution.
        gd=cv2.resize(gt,(sw,sh),interpolation=cv2.INTER_NEAREST); pd=cv2.resize(pred,(sw,sh),interpolation=cv2.INTER_NEAREST)
        inter=int(np.count_nonzero(gd&pd)); rows.append({'split':r['split'],'image':r['image'],'instances':r['instances'],'full_recall':float(np.count_nonzero(gt&pred)/max(1,int(gt.sum()))),'ratio4_recall':float(inter/max(1,int(gd.sum()))),'ratio4_gt_pixels':int(gd.sum()),'ratio4_polygon_pixels':int(pd.sum())})
        n,l,st,_=cv2.connectedComponentsWithStats(gt,8)
        for i in range(1,n):
            area=int(st[i,cv2.CC_STAT_AREA])
            if area in (1,2):
                q=(l==i).astype(np.uint8); qd=cv2.resize(q,(sw,sh),interpolation=cv2.INTER_NEAREST); thin.append({'area':area,'survives_ratio4':bool(qd.any())})
        by.setdefault(r['split'],[]).append(r['instances'])
    negatives=[r for r in spec['records'] if r['instances']==0]
    pos=[x for x in rows if x['instances']>0]
    out={'loader_semantics':'Ultralytics-style normalized polygons rasterized then downsampled at mask_ratio=4; no Ultralytics package was imported in this environment','mask_ratio':4,'overlap_mask':'Not applicable to non-overlapping connected-component instances; no overlap was introduced. Holes remain a known polygon limitation.','tiles_by_split':{k:{'tiles':len(v),'positive':sum(x>0 for x in v),'negative':sum(x==0 for x in v),'mean_instances_all_tiles':float(np.mean(v)),'mean_instances_positive_tiles':float(np.mean([x for x in v if x>0])) if any(x>0 for x in v) else 0} for k,v in by.items()},'total_instances_labels':int(sum(sum(v) for v in by.values())),'full_resolution_mean_recall_all_tiles':float(np.mean([x['full_recall'] for x in rows])),'full_resolution_mean_recall_positive_tiles':float(np.mean([x['full_recall'] for x in pos])),'ratio4_mean_recall_all_tiles':float(np.mean([x['ratio4_recall'] for x in rows])),'ratio4_mean_recall_positive_tiles':float(np.mean([x['ratio4_recall'] for x in pos])),'ratio4_min_recall_positive_tiles':float(min(x['ratio4_recall'] for x in pos)),'ratio4_mean_gt_pixels':float(np.mean([x['ratio4_gt_pixels'] for x in pos])),'thin_components_area_1_or_2':{'count':len(thin),'survives_ratio4':sum(x['survives_ratio4'] for x in thin),'retention':float(sum(x['survives_ratio4'] for x in thin)/max(1,len(thin))),'note':'A source component can disappear after downsampling if it lies below the loader grid; this is measured, not silently discarded.'},'negative_tile_semantics':{'count':len(negatives),'interpretation':'Official binary mask has no zero-valued ink pixels in these tiles. They are blank/negative according to annotation; source evidence is insufficient to call them confirmed glare rather than genuinely blank board or unmarked content.','paths':[r['image'] for r in negatives]}}
    (root/'ultralytics_semantic_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
