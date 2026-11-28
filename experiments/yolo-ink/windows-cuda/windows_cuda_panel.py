"""Portable Chinese read-only panel for a Windows YOLO run."""
from __future__ import annotations
import argparse,csv,re,time
from pathlib import Path
ANSI=re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
def render(run:Path)->str:
    cfg=run/'windows-cuda-config.json'; c=cfg.read_text(encoding='utf-8') if cfg.exists() else ''
    progress=run/'progress.json'; live={}
    if progress.exists():
        try: live=__import__('json').loads(progress.read_text(encoding='utf-8'))
        except Exception: live={}
    rows=[]; f=run/'results.csv'
    if f.exists():
        with f.open(newline='',encoding='utf-8') as h: rows=list(csv.DictReader(h))
    rows=[r for r in rows if r.get('time')]; last=rows[-1] if rows else None
    total=re.search(r"epochs[^0-9]*(\d+)",c).group(1) if re.search(r"epochs[^0-9]*(\d+)",c) else str(live.get('epochs','?')); out=['白板墨迹 · Windows CUDA 训练面板',f'运行目录  {run}',f'状态      {live.get("state","等待")} · 第 {live.get("epoch",len(rows))}/{total} 轮 · 批 {live.get("batch",0)}/{live.get("batches",0)}']
    if live: out += [f'ETA       {float(live.get("eta_s",0)):.0f} 秒',f'loss      {" · ".join(f"{k}={v:.4f}" for k,v in live.get("losses",{}).items()) or "等待 batch"}']
    if last: out += [f'用时      {float(last["time"]):.1f} 秒',f'precision {float(last.get("metrics/precision(M)",0)):.1%} · recall {float(last.get("metrics/recall(M)",0)):.1%}']
    out += ['','只读面板 · 训练日志保留在 run 目录 · Ctrl-C 退出']; return '\n'.join(out)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--interval',type=float,default=2);a=p.parse_args()
    try:
        while True: print('\033[2J\033[H'+render(a.run_dir),flush=True);time.sleep(max(.2,a.interval))
    except KeyboardInterrupt: pass
if __name__=='__main__':main()
