"""Generate a frozen synthetic temporal whiteboard stress set.

This is an evaluation fixture only; it does not invoke or modify any capture
or vectorization candidate. Physical ink truth, visible truth, and unknown
occlusion masks are stored separately so an evaluator cannot score an occluded
region as a false negative or reward ink hallucinated after erase.
"""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import cv2, numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/datasets/temporal-stress-v1'
SEED=20261128
H,W=240,320

def line_mask(spec):
    m=np.zeros((H,W),np.uint8)
    cv2.line(m,tuple(spec['p0']),tuple(spec['p1']),255,int(spec['thickness']),cv2.LINE_AA)
    return m

def board(seed):
    rng=np.random.default_rng(seed); base=np.full((H,W,3),238,np.float32)
    small=rng.normal(0,1.8,(30,40)).astype(np.float32); noise=cv2.resize(small,(W,H),interpolation=cv2.INTER_CUBIC)
    base+=noise[...,None]
    # fixed faint grime/board texture, not ink truth
    cv2.ellipse(base,(250,75),(54,18),-12,0,360,(3,3,3),-1)
    return np.clip(base,0,255).astype(np.uint8)

def render_background(img, frame, mode):
    x=35 if mode=='static_glare' else (15+((frame*9)%280)) if mode=='moving_glare' else 0
    if mode in ('static_glare','moving_glare'):
        cv2.rectangle(img,(x,0),(min(W,x+38),H),(255,255,255),-1)
        cv2.GaussianBlur(img,(0,0),3,dst=img)
    if mode=='exposure': img[:]=np.clip(img.astype(np.float32)*(0.72 if frame<20 else 1.18),0,255).astype(np.uint8)
    return img

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    scenarios=[
      ('gray_occlude_erase',{'color':(112,112,112),'thickness':1,'p0':(35,78),'p1':(285,92),'glare':None,'occlude':True,'erase':True}),
      ('red_thick_occlude_erase',{'color':(145,95,120),'thickness':3,'p0':(35,145),'p1':(278,118),'glare':None,'occlude':True,'erase':True}),
      ('faint_diagonal_static_glare',{'color':(158,158,158),'thickness':2,'p0':(30,195),'p1':(290,170),'glare':'static_glare','occlude':False,'erase':False}),
      ('line_through_moving_glare',{'color':(130,108,115),'thickness':2,'p0':(18,38),'p1':(300,218),'glare':'moving_glare','occlude':False,'erase':False}),
      ('global_exposure_shift',{'color':(135,135,135),'thickness':1,'p0':(45,112),'p1':(280,112),'glare':'exposure','occlude':False,'erase':False}),
      ('blank_board_negative',{'color':None,'thickness':0,'p0':(0,0),'p1':(0,0),'glare':'moving_glare','occlude':False,'erase':False}),
      ('two_lines_partial_occlusion',{'color':(150,150,150),'thickness':1,'p0':(25,55),'p1':(295,55),'glare':None,'occlude':True,'erase':False,'second':{'color':(125,90,110),'thickness':2,'p0':(25,185),'p1':(295,205)}}),
      ('static_board_texture_no_ink',{'color':None,'thickness':0,'p0':(0,0),'p1':(0,0),'glare':'static_glare','occlude':False,'erase':False}),
    ]
    manifest={'name':'temporal-stress-v1','seed':SEED,'canvas':[W,H],'frames_per_scenario':40,'phase':{'visible_0_9':'ink visible and should be detected','occluded_10_19':'physical ink remains but occluder is unknown','visible_20_29':'ink visible again','erased_30_39':'physical ink removed; detection is false positive'},'scenarios':[]}
    for si,(name,s) in enumerate(scenarios):
      for kind in ('frames','truth_physical','truth_visible','unknown'):(OUT/name/kind).mkdir(parents=True,exist_ok=True)
      base=board(SEED+si); physical=np.zeros((H,W),np.uint8)
      if s['color'] is not None: physical=np.maximum(physical,line_mask(s))
      if s.get('second'): physical=np.maximum(physical,line_mask(s['second']))
      for f in range(40):
        img=render_background(base.copy(),f,s['glare'] or '')
        visible=physical.copy(); unknown=np.zeros((H,W),np.uint8)
        if s['color'] is not None and not (s['erase'] and f>=30):
          for spec in [s]+([s['second']] if s.get('second') else []): cv2.line(img,tuple(spec['p0']),tuple(spec['p1']),tuple(spec['color']),int(spec['thickness']),cv2.LINE_AA)
        if s['occlude'] and 10<=f<=19:
          cv2.rectangle(img,(132,25),(192,215),(130,105,88),-1); cv2.rectangle(unknown,(132,25),(192,215),255,-1); visible[unknown>0]=0
        if s['erase'] and f>=30: visible[:]=0; physical_for_truth=np.zeros_like(physical)
        else: physical_for_truth=physical
        # JPEG-like degradation is deterministic and only affects input.
        ok,enc=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,48]); img=cv2.imdecode(enc,cv2.IMREAD_COLOR)
        cv2.imwrite(str(OUT/name/'frames'/f'{f:03d}.png'),img); cv2.imwrite(str(OUT/name/'truth_physical'/f'{f:03d}.png'),physical_for_truth); cv2.imwrite(str(OUT/name/'truth_visible'/f'{f:03d}.png'),visible); cv2.imwrite(str(OUT/name/'unknown'/f'{f:03d}.png'),unknown)
      manifest['scenarios'].append({'name':name,'line_specs':s,'frame_dir':f'{name}/frames','physical_truth':f'{name}/truth_physical','visible_truth':f'{name}/truth_visible','unknown_mask':f'{name}/unknown','negative':s['color'] is None,'erase_phase':bool(s['erase']),'occlusion_phase':bool(s['occlude'])})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));
    digest=hashlib.sha256()
    for p in sorted(OUT.rglob('*.png')): digest.update(p.relative_to(OUT).as_posix().encode()); digest.update(p.read_bytes())
    (OUT/'SHA256').write_text(digest.hexdigest()+'  (tree digest over PNG paths+bytes)\n')
    print(json.dumps({'scenarios':len(scenarios),'frames':len(scenarios)*40,'tree_sha256':digest.hexdigest()}))
if __name__=='__main__': main()
