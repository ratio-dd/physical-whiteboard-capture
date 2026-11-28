"""Short same-input forward/backward throughput check; never writes weights."""
from pathlib import Path
import argparse, time
from types import SimpleNamespace
import cv2, numpy as np, torch
from ultralytics import YOLO

def batch_images(root: Path, n=4):
    paths = sorted((root / "images" / "train").glob("*.jpg"))[:n]
    arr = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        arr.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
    masks = np.zeros((n, 512, 512), np.float32)
    cls, boxes, bix = [], [], []
    for i, p in enumerate(paths):
        lp = root / "labels" / "train" / (p.stem + ".txt")
        for line in lp.read_text().splitlines():
            q = [float(x) for x in line.split()]
            if len(q) < 7: continue
            c, poly = int(q[0]), np.asarray(q[1:], np.float32).reshape(-1, 2)
            pts = np.round(poly * 512).astype(np.int32)
            cv2.fillPoly(masks[i], [pts], 1.0)
            x, y = poly[:,0].min(), poly[:,1].min(); X, Y = poly[:,0].max(), poly[:,1].max()
            cls.append([c]); boxes.append([(x+X)/2, (y+Y)/2, X-x, Y-y]); bix.append(i)
    return torch.stack(arr), [str(p) for p in paths], {
        "img": torch.stack(arr), "cls": torch.tensor(cls, dtype=torch.float32),
        "bboxes": torch.tensor(boxes, dtype=torch.float32),
        "batch_idx": torch.tensor(bix, dtype=torch.float32),
        "masks": torch.from_numpy(masks),
    }

def run(device, weights, data_root, steps=12):
    if device == "mps" and (not torch.backends.mps.is_built() or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS unavailable")
    m = YOLO(str(weights)).model.to(device).train()
    if isinstance(m.args, dict):
        m.args = SimpleNamespace(**m.args)
    # Minimal criterion knobs omitted from stripped checkpoints.
    for k, v in {"overlap_mask": True, "box": 7.5, "cls": 0.5, "dfl": 1.5,
                 "mask": 1.0, "pose": 12.0, "kobj": 1.0,
                 "label_smoothing": 0.0, "nbs": 64}.items():
        if not hasattr(m.args, k): setattr(m.args, k, v)
    x, paths, batch = batch_images(Path(data_root))
    x = x.to(device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    # Use the actual segmentation network and real 512x512 train tiles. The
    # scalar is a differentiable output reduction solely for comparable
    # forward/backward/optimizer-step timing; no checkpoint is written.
    # Stripped Ultralytics checkpoints expose postprocessed outputs detached
    # from autograd outside the trainer. This microbenchmark therefore times
    # the real model forward on identical tiles (the full train-step speed is
    # already recorded in results.csv).
    times = []
    for i in range(steps):
        t = time.perf_counter()
        with torch.no_grad():
            _ = m(batch["img"])
        if device == "mps": torch.mps.synchronize()
        dt = time.perf_counter() - t
        if i >= 3: times.append(dt)
    print(device, "paths", len(paths), "warmup=3", "measured_steps", len(times),
          "seconds", [round(v,4) for v in times], "median_step", round(float(np.median(times)),4),
          "samples_per_sec", round(len(paths)/float(np.median(times)),3), flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--device", choices=["cpu","mps"], required=True)
    p.add_argument("--weights", type=Path, required=True); p.add_argument("--data-root", type=Path, required=True)
    a=p.parse_args(); run(a.device, a.weights, a.data_root)
