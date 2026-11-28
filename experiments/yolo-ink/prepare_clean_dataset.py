"""Make a cache-free clean YOLO view from the explicit tile_manifest records."""
import json, os, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'data/datasets/lecturemath-yolo'; OUT=ROOT/'data/datasets/lecturemath-yolo-clean'
def main():
    if OUT.exists(): shutil.rmtree(OUT)
    spec=json.loads((SRC/'tile_manifest.json').read_text()); counts={}
    for split in ('train','validation','test'):
        rows=[r for r in spec['records'] if r['split']==split]; counts[split]=len(rows)
        (OUT/'images'/split).mkdir(parents=True); (OUT/'labels'/split).mkdir(parents=True)
        paths=[]
        for r in rows:
            si=SRC/r['image']; sl=SRC/r['label']; di=OUT/'images'/split/si.name; dl=OUT/'labels'/split/sl.name
            os.link(si,di); os.link(sl,dl); paths.append(str(di.resolve()))
        (OUT/f'{split}.txt').write_text('\n'.join(paths)+'\n')
    (OUT/'dataset.yaml').write_text(f"path: {OUT.resolve()}\ntrain: train.txt\nval: validation.txt\ntest: test.txt\nnames:\n  0: ink\n")
    (OUT/'provenance.json').write_text(json.dumps({'source_manifest':str((SRC/'tile_manifest.json').resolve()),'counts':counts,'selection':'tile_manifest records only; no symlink original keyframes; hardlinked image and matching label pairs'},indent=2))
    print(json.dumps(counts))
if __name__=='__main__': main()
