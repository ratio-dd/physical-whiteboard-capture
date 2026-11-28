"""Strict Metal-only YOLO segmentation finetune entry point.

This intentionally fails before training when MPS is unavailable.  It must be
run by the runtime owner in the visible tmux session; it never falls back to
CPU and does not touch the frozen CPU checkpoints.
"""
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO


def main(data: Path, out: Path, weights: Path, epochs: int, imgsz: int,
         batch: int, name: str) -> None:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing CPU fallback")
    model = YOLO(str(weights))
    # Force model materialization on Metal before Ultralytics starts its loop.
    model.model.to("mps")
    device_types = {str(p.device.type) for p in model.model.parameters()}
    if device_types != {"mps"}:
        raise RuntimeError(f"model parameters are not exclusively on MPS: {device_types}")
    print(f"MPS ready: torch={torch.__version__}, parameters={sum(p.numel() for p in model.model.parameters()):,}, device={device_types}", flush=True)
    model.train(
        data=str(data), model=str(weights), epochs=epochs, imgsz=imgsz,
        batch=batch, device="mps", workers=0, project=str(out), name=name,
        pretrained=True, plots=False, patience=epochs, mask_ratio=2,
        max_det=500, cache=False, mosaic=0.0, mixup=0.0,
        deterministic=False, verbose=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--name", default="lecturemath_yolo11s_seg_mps512_ft")
    a = p.parse_args()
    main(a.data, a.out, a.weights, a.epochs, a.imgsz, a.batch, a.name)
