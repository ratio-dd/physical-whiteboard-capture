#!/usr/bin/env python3
"""Deterministic synthetic occluded-stroke experiment.

The complete static curve is an input. Only progress along that curve is
predicted inside the hidden interval. No real capture or private dataset is
read by this experiment.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).parent
RNG = np.random.default_rng(194)
N, H0, H1 = 40, 12, 27
HIDDEN = np.arange(H0 + 1, H1)
PROFILES = ("constant", "ease", "pause", "burst")

def speed(name, t):
    if name == "constant": v = np.ones_like(t)
    elif name == "ease": v = .35 + 1.3 * (1 - (2 * t - 1) ** 2)
    elif name == "pause": v = np.where((t > .43) & (t < .60), 0., 1.)
    else: v = np.where(t < .38, 1.8, .45)
    p = np.cumsum(v)
    return p / p[-1]

def make_sample(profile):
    t = np.linspace(0, 1, N)
    progress = speed(profile, t)
    path = np.c_[t, .12 * np.sin(np.pi * t)]
    visible_velocity_before = np.diff(progress[:H0 + 1])[-3:]
    visible_velocity_after = np.diff(progress[H1:H1 + 4])[:3]
    features = np.r_[progress[H0], progress[H1], visible_velocity_before,
                     visible_velocity_after, (H1 - H0) / (N - 1)]
    return path, progress, features.astype(np.float32)

class MLP:
    def __init__(self, n_in, n_out):
        self.w1 = RNG.normal(0, .08, (n_in, 24)).astype(np.float32)
        self.b1 = np.zeros(24, np.float32)
        self.w2 = RNG.normal(0, .08, (24, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
    def forward(self, x):
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2, h
    def fit(self, x, y):
        for _ in range(1200):
            pred, h = self.forward(x); grad = 2 * (pred - y) / len(x)
            old = self.w2.copy(); self.w2 -= .006 * h.T @ grad; self.b2 -= .006 * grad.sum(0)
            dh = (grad @ old.T) * (1 - h * h); self.w1 -= .006 * x.T @ dh; self.b1 -= .006 * dh.sum(0)
    def predict(self, x): return self.forward(x)[0]

def main():
    train = [make_sample(RNG.choice(PROFILES)) for _ in range(900)]
    x = np.array([r[2] for r in train]); y = np.array([r[1][HIDDEN] for r in train])
    x_mu, x_sd = x.mean(0), x.std(0) + 1e-5; y_mu, y_sd = y.mean(0), y.std(0) + 1e-5
    models = []
    for _ in range(5):
        model = MLP(x.shape[1], len(HIDDEN)); model.fit((x-x_mu)/x_sd, (y-y_mu)/y_sd); models.append(model)
    rows = []
    for profile in PROFILES:
        path, progress, feat = make_sample(profile); truth = progress[HIDDEN]
        baseline = np.interp(HIDDEN, [H0, H1], [progress[H0], progress[H1]])
        guesses = np.array([(m.predict(((feat-x_mu)/x_sd)[None])[0] * y_sd + y_mu) for m in models])
        guess = guesses.mean(0); scale = np.linalg.norm(path[-1]-path[0]) * 1000
        rows.append({"profile": profile, "baseline_mae_px": float(np.abs(baseline-truth).mean()*scale), "mlp_mae_px": float(np.abs(guess-truth).mean()*scale), "uncertainty_px": float(np.abs(guesses-guess).mean()*scale), "path": path.tolist(), "truth": truth.tolist(), "baseline": baseline.tolist(), "prediction": guess.tolist()})
    a = np.linspace(0, 1, N); b = a.copy(); b[17:23] = b[16]; b[23:28] = np.linspace(b[16], b[27], 5)
    result = {"hidden_indices": HIDDEN.tolist(), "rows": rows, "ambiguity": {"visible_equal": bool(np.array_equal(a[:H0+1], b[:H0+1]) and np.array_equal(a[H1:], b[H1:])), "conclusion": "identical visible evidence can hide a pause or continuous motion"}}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (ROOT / "index.html").write_text(make_html(rows[1]), encoding="utf-8")
    print(json.dumps({r["profile"]: {k: r[k] for k in ("baseline_mae_px", "mlp_mae_px", "uncertainty_px")} for r in rows}, indent=2))

def make_html(row):
    payload = json.dumps(row)
    return f'''<!doctype html><meta charset="utf-8"><title>Occluded stroke experiment</title><style>body{{font:16px system-ui;max-width:850px;margin:2rem auto}}svg{{border:1px solid #ccd}}polyline{{fill:none;stroke-width:4;stroke-linecap:round}}.path{{stroke:#bbb;stroke-width:2}}.visible{{stroke:#111}}.guess{{stroke:#1769c2;stroke-dasharray:8 5}}button{{padding:.5rem 1rem}}</style><h1>Occluded stroke experiment</h1><p>Black is visible evidence, blue dashed is a clearly labelled inferred stroke. The hidden truth is only an offline evaluation reference.</p><button id="play">Play inferred stroke</button> <button id="reset">Reset</button><span id="status"></span><svg id="s" width="760" height="360"></svg><p id="m"></p><script>const d={payload},s=document.querySelector('#s'),st=document.querySelector('#status'),pts=d.path.map(p=>[60+p[0]*640,270-p[1]*180]);let i=0,t;function line(a,c){{const e=document.createElementNS('http://www.w3.org/2000/svg','polyline');e.className.baseVal=c;e.setAttribute('points',a.map(p=>p.join(',')).join(' '));s.append(e)}}function draw(){{s.innerHTML='';line(pts,'path');line(pts.slice(0,13),'visible');line(pts.slice(27),'visible');if(i)line(pts.slice(12,12+i),'guess');st.textContent=i?' inferred '+i+'/16':' two visible sides';if(i<16)t=setTimeout(()=>{{i++;draw()}},140)}}play.onclick=()=>{{clearTimeout(t);i=0;draw()}};reset.onclick=()=>{{clearTimeout(t);i=0;draw()}};draw();m.textContent='Baseline MAE '+d.baseline_mae_px.toFixed(2)+' px, MLP MAE '+d.mlp_mae_px.toFixed(2)+' px, uncertainty '+d.uncertainty_px.toFixed(2)+' px';</script>'''

if __name__ == "__main__": main()
