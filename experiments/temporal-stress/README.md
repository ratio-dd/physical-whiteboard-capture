# Temporal stress set v1

Run `physical-whiteboard/.venv/bin/python experiments/temporal-stress/generate.py` to create the ignored fixture `data/datasets/temporal-stress-v1/`.

It has 8 scenarios × 40 frames, deterministic seed `20261128`, 320×240 inputs, JPEG-quality-48 round-tripping, fixed paper texture, faint gray/red strokes, 1–3px widths, static and moving bright bands, exposure shifts, blank-board negatives, occlusion, reappearance, and true erase.

Each frame has separate `truth_physical`, `truth_visible`, and `unknown` masks. During occlusion, physical truth remains ink but visible truth is zero and the region is unknown; after erase, physical and visible truth are zero. An evaluator must exclude unknown pixels from recall/FP scoring, score erase residuals separately, and report latency as first visible frame detected after phase change. The 10 negative-looking cases are intentionally not labeled as glare truth unless the scenario name says so.

This is a frozen evaluation fixture only. It does not invoke candidate capture/vector algorithms and contains no user data.
