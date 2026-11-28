"""Run the official Ultralytics 8.4.220 polygon2mask semantics on our labels."""
import json
from pathlib import Path
import cv2, numpy as np
from ultralytics.data.utils import polygon2mask

ROOT=Path(__file__).resolve().parents[2]

def read_polys(p,h,w):
    out=[]
    for line in p.read_text().splitlines():
        a=np.asarray([float(x) for x in line.split()[1:]],np.float32).reshape(-1,2)
        out.append((a*np.array([w,h],np.float32)).reshape(-1))
    return out

def u_mask(shape, polys, ratio):
    out=np.zeros((shape[0]//ratio,shape[1]//ratio),np.uint8)
    for p in polys:
        out=np.maximum(out,polygon2mask(shape,[p],1,ratio))
    return out

def main():
    root=ROOT/'data/datasets/lecturemath-yolo'; spec=json.loads((root/'tile_manifest.json').read_text()); allrows=[]; thin={2:[],4:[]}; micro={2:[0,0],4:[0,0]}
    for r in spec['records']:
        im=cv2.imread(str(root/r['image'])); h,w=im.shape[:2]; x,y,tw,th=r['tile_xywh']; src=cv2.imread(str(ROOT/'data/datasets/lecturemath'/r['source_mask']),0); gt=(src[y:y+th,x:x+tw]==0).astype(np.uint8)
        polys=read_polys(root/r['label'],w,h)
        for ratio in (2,4):
            pd=u_mask((h,w),polys,ratio) if polys else np.zeros((h//ratio,w//ratio),np.uint8); gd=cv2.resize(gt,(w//ratio,h//ratio))>0; inter=np.count_nonzero(pd.astype(bool)&gd); allrows.append({'split':r['split'],'ratio':ratio,'recall':inter/max(1,int(gd.sum())),'precision':inter/max(1,int(pd.sum())),'gt':int(gd.sum())})
            n,lab,st,_=cv2.connectedComponentsWithStats(gt,8); dist=cv2.distanceTransform(gt,cv2.DIST_L2,3)
            for i in range(1,n):
                area=int(st[i,cv2.CC_STAT_AREA]); bw=int(st[i,cv2.CC_STAT_WIDTH]); bh=int(st[i,cv2.CC_STAT_HEIGHT]); major=max(bw,bh); vals=dist[lab==i]; width=float(2*np.median(vals)) if len(vals) else 0
                if area in (1,2): micro[ratio][0]+=1; micro[ratio][1]+=int((cv2.resize((lab==i).astype(np.uint8),(w//ratio,h//ratio))>0).any())
                if area>=20 and major>=20 and width<=3.2:
                    c=(lab==i).astype(np.uint8); cd=cv2.resize(c,(w//ratio,h//ratio))>0; hit=inter2=np.count_nonzero(cd & (gd)); phit=np.count_nonzero(cd & (pd>0)); thin[ratio].append({'area':area,'width_est':width,'major':major,'gt_down_pixels':int(cd.sum()),'poly_recall':float(phit/max(1,int(cd.sum()))),'gt_survives':bool(cd.any())})
    out={'ultralytics_version':'8.4.220','function':'ultralytics.data.utils.polygon2mask','ratios':{},'thin_definition':'source connected component area>=20, major bbox dimension>=20, distance-transform median width estimate<=3.2px; this separates long thin strokes from 1-2px micro-dots','micro_components':{str(k):{'count':v[0],'survives_downsample':v[1],'retention':v[1]/max(1,v[0])} for k,v in micro.items()}}
    for ratio in (2,4):
        q=[x for x in allrows if x['ratio']==ratio]; t=thin[ratio]; out['ratios'][str(ratio)]={'all_tile_mean_recall':float(np.mean([x['recall'] for x in q])),'all_tile_mean_precision':float(np.mean([x['precision'] for x in q])),'positive_tile_mean_recall':float(np.mean([x['recall'] for x in q if x['gt']])),'thin_components':len(t),'thin_mean_poly_recall':float(np.mean([x['poly_recall'] for x in t])) if t else None,'thin_min_poly_recall':float(min(x['poly_recall'] for x in t)) if t else None,'thin_survival':sum(x['gt_survives'] for x in t)/max(1,len(t))}
    (root/'official_ultralytics_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
