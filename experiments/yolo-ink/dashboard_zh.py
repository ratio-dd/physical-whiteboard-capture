"""中文、只读的 YOLO 训练面板。

它只读取 ``run-dir`` 的 args.yaml/results.csv 和显式传入的日志，不依赖
项目中的数据集、录像或机器绝对路径；适合放在 tmux 的第二个 pane。
"""
from __future__ import annotations
import argparse, csv, re, time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BATCH = re.compile(r"^\s*(\d+)/(\d+).*?(\d+)/(\d+)\s+(?:([\d.]+)it/s|([\d.]+)s/it)")

def _log(path: Path) -> str:
    if not path.exists(): return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 65536))
        return ANSI.sub("", f.read().decode(errors="replace"))

def _duration(seconds: float) -> str:
    total = max(0, round(seconds)); m, s = divmod(total, 60)
    return f"{m} 分 {s:02d} 秒" if m else f"{s} 秒"

def render(run: Path, log: Path) -> str:
    args = (run / "args.yaml").read_text(errors="replace") if (run / "args.yaml").exists() else ""
    def value(key: str, default: str = ""):
        m = re.search(rf"^{re.escape(key)}:\s*(.+)$", args, re.M)
        return m.group(1).strip() if m else default
    rows = []
    result = run / "results.csv"
    if result.exists():
        with result.open(newline="") as f: rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("time")]
    total = int(value("epochs", "0") or 0)
    text = _log(log)
    batch = next((m for line in reversed(text.splitlines()) if (m := BATCH.match(line))), None)
    out = ["白板墨迹 · MPS 训练面板", f"运行目录  {run}", f"设备      {value('device', '等待配置')} · 图像 {value('imgsz', '?')} · 轮数 {total or '?'}", ""]
    if total and len(rows) >= total: out.append(f"状态      已完成 {len(rows)}/{total} 轮 · 用时 {_duration(float(rows[-1]['time']))}")
    elif batch:
        cur, planned, step, steps = map(int, batch.groups()[:4]); rate = float(batch.group(5) or 1 / float(batch.group(6)))
        fill = round(20 * step / max(1, steps)); out += [f"状态      第 {cur}/{planned} 轮  {'█'*fill}{'░'*(20-fill)} {step}/{steps} 批", f"速度      {rate:.1f} 批/秒 · 本轮约 {_duration((steps-step)/max(rate,1e-9))}"]
    else: out.append(f"状态      已记录 {len(rows)} 轮，等待训练日志")
    if rows:
        last = rows[-1]; out += ["", f"最近一轮  recall {float(last.get('metrics/recall(M)', 0)):.1%} · precision {float(last.get('metrics/precision(M)', 0)):.1%}"]
    out.append("\n只读监控 · Ctrl-C 退出")
    return "\n".join(out)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--log", type=Path, required=True); p.add_argument("--interval", type=float, default=2); p.add_argument("--once", action="store_true"); a = p.parse_args()
    try:
        while True:
            print("\033[2J\033[H" + render(a.run_dir, a.log), flush=True)
            if a.once: return
            time.sleep(max(.2, a.interval))
    except KeyboardInterrupt: return
if __name__ == "__main__": main()
