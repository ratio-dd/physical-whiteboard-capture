"""Prepare YOLO-seg labels from the official LectureMath manifest only."""
from pathlib import Path
import json,cv2,os

def main(manifest,out):
 d=json.loads(Path(manifest).read_text()); root=Path(manifest).parent; out=Path(out); out.mkdir(parents=True,exist_ok=True)
 for split in ('train','validation','test'):
  s='val' if split=='validation' else split; (out/'images'/s).mkdir(parents=True,exist_ok=True); (out/'labels'/s).mkdir(parents=True,exist_ok=True)
  for row in [r for r in d['pairs'] if r['split']==split]:
   src=root/row['image']; dst=out/'images'/s/src.name
   if not dst.exists(): os.symlink(src.resolve(),dst)
   im=cv2.imread(str(root/row['mask']),0); h,w=im.shape; contours,_=cv2.findContours((im==0).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); lines=[]
   for c in contours:
    if cv2.contourArea(c)<2: continue
    p=c.reshape(-1,2); lines.append('0 '+' '.join(f'{x/w:.6f} {y/h:.6f}' for x,y in p))
   (out/'labels'/s/(src.stem+'.txt')).write_text('\n'.join(lines)+'\n')
 (out/'data.yaml').write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ink\n")
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('out');a=p.parse_args();main(a.manifest,a.out)
