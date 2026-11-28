"""Strict Windows/CUDA YOLO segmentation training entry point.

This module intentionally refuses CPU fallback.  It is portable: every input
path is supplied by the caller and no Mac/project absolute path is assumed.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from ultralytics import YOLO
from progress_callback import ProgressWriter

def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA reports no devices")
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    run_dir = out / args.name; run_dir.mkdir(parents=True, exist_ok=True)
    config = {"schema":"windows-cuda-yolo-v1","device":"cuda:0","cuda_device_count":torch.cuda.device_count(),"torch":torch.__version__,"data":str(args.data.resolve()),"weights":str(args.weights.resolve()),"imgsz":args.imgsz,"batch":args.batch,"epochs":args.epochs,"seed":args.seed,"name":args.name,"resume":bool(args.resume),"started_unix":time.time()}
    (run_dir / "windows-cuda-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"status":"CUDA ready","device":torch.cuda.get_device_name(0),"torch":torch.__version__,"config":str(run_dir/'windows-cuda-config.json')}, ensure_ascii=False), flush=True)
    model = YOLO(str(args.weights))
    progress = ProgressWriter(run_dir / "progress.json")
    model.add_callback("on_train_batch_end", progress.on_train_batch_end)
    model.add_callback("on_fit_epoch_end", progress.on_fit_epoch_end)
    model.add_callback("on_train_end", progress.on_train_end)
    model.train(data=str(args.data), model=str(args.weights), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=0, workers=0,
                project=str(out), name=args.name, seed=args.seed,
                pretrained=True, plots=False, patience=args.epochs,
                mask_ratio=2, max_det=500, cache=False, mosaic=0.0,
                mixup=0.0, deterministic=False, resume=args.resume,
                verbose=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("data", type=Path); p.add_argument("--out", type=Path, required=True); p.add_argument("--weights", type=Path, required=True); p.add_argument("--name", required=True); p.add_argument("--epochs", type=int, default=4); p.add_argument("--imgsz", type=int, default=512); p.add_argument("--batch", type=int, default=4); p.add_argument("--seed", type=int, default=0); p.add_argument("--resume", action="store_true"); main(p.parse_args())
