"""Official Ultralytics YOLO segmentation training entry point."""
from pathlib import Path
from ultralytics import YOLO

def main(data, out, weights='yolo11s-seg.pt', epochs=3, imgsz=320, name=None, device='cpu'):
 model=YOLO(str(weights))
 model.train(data=str(data),model=str(weights),epochs=epochs,imgsz=imgsz,batch=4,device=device,workers=0,project=str(out),name=name or f'lecturemath_yolo11s_seg_{device}{imgsz}',pretrained=True,plots=False,patience=epochs,mask_ratio=2,max_det=500,cache=False,mosaic=0.0,mixup=0.0,deterministic=(device == 'cpu'))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--out',type=Path,default=Path('data/datasets/lecturemath-yolo-runs'));p.add_argument('--weights',default='yolo11s-seg.pt');p.add_argument('--epochs',type=int,default=3);p.add_argument('--imgsz',type=int,default=320);p.add_argument('--device',default='cpu');p.add_argument('--name');a=p.parse_args();main(a.data,a.out,a.weights,a.epochs,a.imgsz,a.name,a.device)
