# 淡彩笔迹（实验）

这是一个本机、只读的实验回放模式，不改变默认 classic 识别，也不会自动下载
模型。`pipeline.py` 先用当前原板和 `known` 产生 source-colour evidence，再用
同一产品 `ink_geometry` 生成 SVG/事件几何；`ink_package.py` 将结果写成服务可读的
`source-color-experiment` replay package。

普通新会话可以用冷启动路径生成本机包：

```sh
PYTHONPATH=experiments/faint-color-ink:experiments/lecturemath-ink \
.venv/bin/python experiments/faint-color-ink/ink_package.py \
  --db /path/to/local/whiteboard.sqlite3 \
  --session SESSION_ID \
  --out /path/to/local/replay-packages/SESSION_ID/HEAD.source-color-experiment.json
```

如果另有经过授权、保存在本机的逐帧 score map，可显式传入
`--memory-dir /path/to/local/score-maps`；它必须按 `seq-001.npy` 这样的原尺度
文件命名。`--weights` 只记录本机权重的 SHA-256，不会上传或自动加载权重。
数据、score map、权重、数据库和生成包均应留在本机 ignored 目录；公开仓库只保留
此处的通用源码和合成测试。

将生成的 JSON 放入服务配置的 replay-packages 目录后，只有 session head 与包中
`session_version`/`through_seq` 完全匹配时，页面才显示“淡彩笔迹（实验）”。
它与 CPU/Metal YOLO 包使用不同的文件名和 mode，不覆盖既有包；没有匹配包时页面
不会显示该选项。
