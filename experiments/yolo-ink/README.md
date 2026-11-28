# LectureMath YOLO ink experiment

This directory contains offline, research-only tooling for an Ultralytics
YOLO11 instance-segmentation experiment. It is not the product runtime and it
does not contain a checkpoint, training run, LectureMath image, mask, or
private capture.

## Licensing and inputs

The training data is the official LectureMath v1.1 annotation archive linked
by the [upstream ACCESS2021 README](https://github.com/kdavila/lecturemath/blob/master/ACCESS2021_release/README.md).
The archive's own license/readme notices should be retained: the local data
manifest records a CC BY-NC-SA 4.0 license file alongside a README claim of CC
BY 4.0 and a copyright note for RGB images. Treat that discrepancy as
unresolved and use only data for which you have obtained the required
permission. This repository does not redistribute the archive or derived
training data.

The model software is [Ultralytics](https://github.com/ultralytics/ultralytics),
whose repository is AGPL-3.0; review the [official license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
and [licensing guidance](https://docs.ultralytics.com/help/FAQ/#what-license-does-ultralytics-use)
before any deployment. The `yolo11s-seg.pt` starting checkpoint is an official
Ultralytics asset available from the [YOLO11 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0).
Download it yourself into an ignored local path and record its SHA-256 in your
private run manifest; do not commit the `.pt` file. An enterprise license may
be required for uses outside the AGPL-3.0 terms.

## Isolated environment

Keep Ultralytics and its large dependencies separate from the product `.venv`:

```sh
python3 -m venv .venv-yolo
.venv-yolo/bin/python -m pip install --upgrade pip
.venv-yolo/bin/python -m pip install 'ultralytics==8.4.220'
```

The exact version used for the polygon audits is recorded in the audit output;
the public source does not assume that Ultralytics is installed in the product
environment. No script downloads a model automatically during product
startup.

## Clean manifest and dataset preparation

Start from a local copy of the public LectureMath manifest. It must contain
lecture-disjoint `train`, `validation`, and `test` pairs, with `image`, `mask`,
`lecture_id`, and the declared `mask==0` ink polarity. Verify that every image
and mask hash matches the manifest before using it. Do not use user camera
captures, the physical-whiteboard session database, or a mixed lecture split.

Prepare an ignored local YOLO view with explicit paths:

```sh
.venv-yolo/bin/python experiments/yolo-ink/prepare_dataset.py \
  --manifest /path/to/manifest_multi.json \
  --out /path/to/local/lecturemath-yolo --audit
```

The converter uses 640-pixel tiles with 96-pixel overlap, connected-component
external contours, normalized YOLO polygons, and a small number of official
empty-mask negatives. `prepare_clean_dataset.py` creates a cache-free view from
the explicit tile manifest in its default ignored local data location. All outputs belong outside Git. The polygon audit
is a representation check, not a model-quality score; YOLO polygons cannot
represent holes exactly and tiny components can be lost at coarse mask ratios.

## Training and evaluation

### 可回看的 tmux 训练会话（Mac MPS）

公开版提供一个不依赖机器绝对路径的两 pane 启动器：左侧严格使用
`train_mps.py` 的 `device=mps`，右侧每 2 秒只读刷新中文进度面板。运行名
默认带时间戳；同名 tmux 会话会被拒绝，不会覆盖正在运行的训练，日志保存在
该 run 目录的 `train.log` 中，因此训练结束后仍可 `tmux attach` 回看。

```sh
python experiments/yolo-ink/start_tmux_training.py \
  --session whiteboard-yolo-mps-2026-11-28 \
  --run-root /path/to/local/yolo-runs \
  --data /path/to/local/dataset.yaml \
  --weights /path/to/yolo11s-seg.pt \
  --device mps --epochs 4 --imgsz 512
```

可先用 `--dry-run` 检查实际 pane 命令。CPU 对照使用 `--device cpu`；它会切换
到普通 `train.py`，不会伪装成 MPS。若已有训练会话只想增加一个中文监控窗口，
不启动新训练：

```sh
python experiments/yolo-ink/start_tmux_training.py \
  --session whiteboard-yolo-mps-2026-11-28 \
  --monitor-run /path/to/local/yolo-runs/existing-run
```

需要回看时执行 `tmux attach -t whiteboard-yolo-mps-2026-11-28`；该面板不读取
用户录像或产品数据库。

`benchmark_step.py` 只做相同 512 tile 的前向微基准，不写权重；它不能替代
完整训练速度评估。完整训练还包含反向、优化器和验证，实际耗时可能约为该微基准
的 4 倍，应以 `results.csv` 和训练日志为准。

Train only after checking the data permissions and the generated
`dataset.yaml`:

```sh
.venv-yolo/bin/python experiments/yolo-ink/train.py \
  /path/to/local/lecturemath-yolo-clean/dataset.yaml \
  --out /path/to/local/lecturemath-yolo-runs \
  --weights /path/to/yolo11s-seg.pt --epochs 3 --imgsz 320
```

The script uses CPU, no dataloader cache, `mask_ratio=2`, and writes the run
under the explicitly supplied ignored output directory. It is a training
entrypoint, not a claim that this model is suitable for live capture. The
final confidence/mask threshold is intentionally not frozen in this README;
it must be selected from the agreed validation protocol after the trained run
has been independently evaluated.

The public evaluator accepts a loader returning a full original-frame score
map or instance masks and evaluates the fixed 4×4 grid of 128×128 windows:

```sh
PYTHONPATH=experiments/yolo-ink \
.venv-yolo/bin/python experiments/yolo-ink/evaluate_public.py \
  --loader inference:load_model --weights /path/to/run/best.pt \
  --split validation --out /path/to/local/yolo-validation.json
```

It does not resize score windows, does not inspect test data when evaluating
validation, and reports the selected operating point plus timing and mask
support. `--mock` is only a protocol smoke test and must never be reported as
a YOLO result. Do not publish a metric until the manifest, checkpoint hash,
Ultralytics version, threshold, and data permissions are recorded together.

## Offline package and whiteboard preview

The vector adapter consumes precomputed instance masks and returns editable
geometry without changing the classic baseline. A local package for the
whiteboard viewer is a version-1 ink timeline JSON with these additional fields:

```json
{
  "version": 1,
  "session_id": "local-session-id",
  "session_version": 42,
  "through_seq": 42,
  "recognition_mode": "yolo-experiment",
  "model_version": "local-yolo-version",
  "weights_sha256": "checkpoint-sha256",
  "recognition_threshold": 0.7616142630577087,
  "width": 320,
  "height": 240,
  "events": []
}
```

Keep packages at `DIR/<session-id>/<session-head>.json`, outside Git, and start
the service explicitly:

```sh
./run.sh --replay-packages /path/to/local/replay-packages
```

The electronic-ink page shows “YOLO 分割（实验）” only when the package's
`session_version` matches the currently saved session head. It then supports
the existing sequence slider, original-board overlay, SVG export, and JSON
export. Without a matching package the option stays hidden; no checkpoint is
downloaded and live camera sampling remains classic. A package can therefore
be inspected without allowing an unvalidated model to mutate the recorder.

To generate a package for any later saved session, use the public command-line
builder once per session. It reads the existing SQLite history, runs the
explicit local tiled loader over each stored state, reuses the product ink
timeline and vector adapter, and atomically replaces only that session's
package file:

```sh
PYTHONPATH=experiments/yolo-ink \
.venv-yolo/bin/python experiments/yolo-ink/build_replay.py SESSION_ID \
  --data-dir /path/to/local/data \
  --output /path/to/local/replay-packages \
  --loader yolo_loader:load_tiled_v2 \
  --weights /path/to/local/yolo-v2-512-frozen.pt \
  --config /path/to/local/yolo-v2-512-frozen-config.json
```

The frozen v2 profile must declare `mode=tiled`, `imgsz=512`, `conf_floor=0.05`,
`iou=0.7`, `tile=640`, `overlap=96`, `max_det=300`, `mask_threshold=0.5`, the
checkpoint SHA-256, and validation threshold
`0.7616142630577087`. The known frozen checkpoint SHA begins
`b8ce752d3c2c…` and must be checked against the local file; the full value
belongs in the local config, not in a public weight file. The loader produces
the score map `max(binary native instance mask × instance confidence)`. The
builder applies the frozen validation threshold exactly once when passing the
score map to geometry; it does not reapply a default `0.72` gate.

The weights path must already exist; the builder refuses missing paths and
never invokes an automatic download. It does not open a camera, inspect
blind-label data, or require a special session. A synthetic fixture proves that
a score of `0.70` is excluded while `0.81` survives a `0.7616142630577087`
threshold, and covers package schema plus atomic write behavior; a real model
run still needs independent quality approval.

## Reproducibility boundary

Publicly reproducible steps are dataset preparation, model invocation,
validation protocol, vector adaptation, and package consumption. The official
checkpoint, trained `best.pt`, `runs/`, generated images/masks, session IDs,
camera frames, private evaluation reports, and final threshold are local
artifacts. This README deliberately contains no machine-absolute path, user
capture identifier, screenshot, or final model-quality claim.
