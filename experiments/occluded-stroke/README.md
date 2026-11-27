# Occluded stroke experiment

This is a small, public, synthetic-only experiment. It does not read the
private `data/` directory, the 194-event capture, or the production event log.

Run it from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python experiments/occluded-stroke/reconstruct.py
open experiments/occluded-stroke/index.html
```

The complete static curve is given as input after the person leaves. The
predictor estimates only progress along that curve during hidden samples. The
strict error set is samples 13–26. The red/black geometric baseline uses the
correct sample-time interpolation between visible endpoints, so its constant
speed error is approximately 0 px.

The NumPy MLP is trained from scratch on four synthetic speed templates
constant, ease, pause, and burst. It is a research comparison, not a trained
whiteboard model. Results are regenerated into `results.json` with seed 194.
Five initializations provide a disagreement estimate. A production system
should abstain when uncertainty is high and display any reconstruction as
inferred, separate from confirmed pixels.

The included ambiguity counterexample has identical visible samples but two
different hidden motions, one with a pause. No model can decide between them
without additional evidence such as saved video, more visible timestamps, or a
learned prior. The animation makes this distinction explicit. The black hidden
truth in the evaluation is never provided to the predictor.
