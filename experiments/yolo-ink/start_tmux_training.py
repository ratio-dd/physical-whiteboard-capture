"""Create a replayable two-pane training session (training + Chinese monitor).

The command refuses to reuse an existing session unless ``--attach`` is given;
run directories are explicit and therefore never silently overwritten.
"""
from __future__ import annotations
import argparse, datetime, shlex, shutil, subprocess
from pathlib import Path

def q(x): return shlex.quote(str(x))
def main():
    p=argparse.ArgumentParser(); p.add_argument('--session',default='whiteboard-yolo-mps'); p.add_argument('--run-root',type=Path); p.add_argument('--run-name'); p.add_argument('--monitor-run',type=Path,help='只在已有会话新增中文面板，不启动训练'); p.add_argument('--monitor-log',type=Path); p.add_argument('--data',type=Path); p.add_argument('--weights',type=Path); p.add_argument('--python',default='.venv-yolo/bin/python'); p.add_argument('--device',choices=['mps','cpu'],default='mps'); p.add_argument('--epochs',type=int,default=4); p.add_argument('--imgsz',type=int,default=512); p.add_argument('--batch',type=int,default=4); p.add_argument('--attach',action='store_true'); p.add_argument('--dry-run',action='store_true'); a=p.parse_args()
    if shutil.which('tmux') is None: raise SystemExit('tmux is required')
    exists=subprocess.run(['tmux','has-session','-t',a.session],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    if a.monitor_run:
        if not exists: raise SystemExit(f'session {a.session!r} does not exist; create it before adding a monitor')
        log=a.monitor_log or (a.monitor_run/'train.log'); monitor=f"exec {q(a.python)} {q(Path(__file__).resolve().parent/'dashboard_zh.py')} --run-dir {q(a.monitor_run)} --log {q(log)}"
        if a.dry_run: print(f'session={a.session}\n[中文面板] {monitor}'); return
        subprocess.run(['tmux','new-window','-t',a.session,'-n','中文面板',monitor],check=True)
        print(f'已在 {a.session} 新增“中文面板”窗口；原训练窗口未改动。'); return
    if not a.run_root or not a.data or not a.weights: raise SystemExit('--run-root、--data、--weights 为训练必填参数')
    if exists and not a.attach: raise SystemExit(f'session {a.session!r} already exists; use --attach or choose --session')
    name=a.run_name or ('yolo-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    run=a.run_root/name; log=run/'train.log'; run.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent; trainer=root/('train_mps.py' if a.device=='mps' else 'train.py'); panel=root/'dashboard_zh.py'
    if a.device=='mps': train=f"exec {q(a.python)} {q(trainer)} {q(a.data)} --out {q(a.run_root)} --weights {q(a.weights)} --epochs {a.epochs} --imgsz {a.imgsz} --batch {a.batch} --name {q(name)} 2>&1 | tee -a {q(log)}"
    else: train=f"exec {q(a.python)} {q(trainer)} {q(a.data)} --out {q(a.run_root)} --weights {q(a.weights)} --epochs {a.epochs} --imgsz {a.imgsz} --device cpu --name {q(name)} 2>&1 | tee -a {q(log)}"
    monitor=f"exec {q(a.python)} {q(panel)} --run-dir {q(run)} --log {q(log)}"
    if a.dry_run:
        print(f'session={a.session}\nrun={run}\n[训练] {train}\n[中文面板] {monitor}'); return
    if exists: subprocess.run(['tmux','attach-session','-t',a.session],check=True); return
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-n',name,train],check=True)
    subprocess.run(['tmux','split-window','-h','-t',a.session,monitor],check=True)
    subprocess.run(['tmux','select-pane','-t',f'{a.session}:0.0','-T','训练 · MPS' if a.device=='mps' else '训练 · CPU'],check=True)
    subprocess.run(['tmux','select-pane','-t',f'{a.session}:0.1','-T','中文面板'],check=True)
    print(f'已创建 tmux 会话 {a.session}，训练 pane 与中文面板 pane 均已启动。\ntmux attach -t {a.session}')
if __name__=='__main__': main()
