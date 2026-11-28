# Physical Whiteboard MVP

这是“物理白板视觉同步系统”的本机 MVP。它用普通摄像头持续观察真实白板，在浏览器里显示透视校正后的数字白板，并保存可回放的状态变化。

## 环境准备

需要 macOS、Python 3.11 或更新版本，以及允许浏览器访问摄像头的权限。项目使用独立的 `.venv`。

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

若系统没有 `python3.13`，可以使用 `python3.11`。安装完成后统一使用 `.venv/bin/python` 或 `./run.sh`。

## 入口与边界

项目有四条分开的路径：实时采集（`./run.sh` → `server.py` → `vision.py`/`history.py`）、经典电子墨迹（已保存原板 → SVG/JSON）、淡墨/YOLO 回放（只读 replay package），以及 `experiments/` 下的离线训练/评估。只有实时采集是产品默认；训练权重不会被服务自动下载或加载，回放结果也不会改变实时确认层。

Windows CUDA 的私有训练包位于 ignored `data/datasets/windows-yolo11s-train700-package.zip`；解包后从包根目录运行 `train_windows_cuda.ps1`。它生成运行时 YAML、打开进度面板，并在 CUDA 不可用时拒绝 CPU fallback。默认 f54 权重 SHA-256 为 `f54bc7fe257515fce44775d87d0cba7443cc1f75cc4b1bb78e2998cc92aeaaaa`。LectureMath/Ultralytics 的数据和许可边界见 `experiments/yolo-ink/README.md`，不随公共仓库发布。

DocRes 只作为独立文档二值化对照，不是白板实例分割器；本机结果和来源保留在 ignored `data/datasets/weak-ink-ml/docres-official-cpu-v1/`，不能用 OCR 框代替笔画 GT。

## 启动

```sh
./run.sh
```

上面的命令均假定当前目录是仓库根目录，也就是包含 `run.sh` 的目录。

默认只监听本机 `127.0.0.1:8765`。打开 <http://127.0.0.1:8765/>，在浏览器提示时允许摄像头权限。也可以指定端口和数据目录：

```sh
./run.sh --port 8765 --data-dir ./data
```

## 第一次真实测试

1. 把 Mac 放在白板正前方，保证白板四角都能看到，尽量减少窗户或灯具反光。
2. 打开页面，选择实际摄像头，点击“开启摄像头”。浏览器地址必须是 `http://127.0.0.1:8765/` 或 `http://localhost:8765/`，否则浏览器通常不会授予摄像头权限。
3. 在视频预览上按左上、右上、右下、左下的顺序点选四个白板角点，再开始采集。摄像头移动后需要重新校准。
4. 写一小段黑色和彩色笔迹，停手等待数字板更新，再局部擦除。观察数字板、未知区域和时间线是否同步变化。
5. 走到白板前遮挡一部分，移动后再离开。被遮挡且尚未确认的区域应保持为暂缓/未知状态，不应把人体写进数字板。
6. 点击停止，打开历史会话并拖动时间线，确认每个事件都能重建对应白板状态。

浏览器会显示权限拒绝、没有可用设备、摄像头中断和服务断开等状态。可在 macOS“系统设置 → 隐私与安全性 → 摄像头”中为使用的浏览器打开权限，然后刷新页面重试。服务接收工作台发送的 JPEG 帧，并在本地数据目录保存白板状态、事件和快照，不会把原始摄像头视频作为视频文件保存。请把本地 `data/` 当作私人实验数据，不要把真实白板图片、数据库或摄像头帧提交到公开仓库。

## 当前能力

- 四点透视校正和局部光照处理，保留彩色笔迹并增强低对比度墨迹。
- 工作台向摄像头请求 1280×720 和 30 fps，按设备实际交付的新帧采样。界面分别显示最近 3 秒的实际采样帧率和成功处理帧率。服务较慢时，浏览器只保留最新待发画面，每次只发送一个 JPEG。区域至少稳定 3 个处理帧且持续 300 ms 才会确认更新。
- 分块稳定确认和遮挡暂缓。尚未可靠观察到的区域会显示为未知，不会强行补全。固定亮带反光会被标记为暂缓，并在界面提示调整光线。
- 通过 SQLite 保存稀疏变化补丁与定期快照，支持重新打开后重建每个状态。
- 历史页支持拖动时间线、上一步/下一步和按事件播放。回放按 32 个事件分段在本地缓存，当前段显示时预取下一段，拖动使用短暂防抖。
- 白板区域如果一直被遮挡，或内容在摄像头视野外被写入再擦除，系统无法恢复其真实历史，会保持未知或暂缓状态。

## 电子墨迹

完成一次采集后，从顶部“电子墨迹”进入工作台，选择一个会话即可按确认序列查看向量路径。拖动“确认序列”会在已经加载的时序数据上重放 `add`、`update` 和 `remove`，路径可以在 SVG 编辑器中单独选中和编辑。点击路径后可以查看其 ID、颜色、宽度、首次出现序号、确认时间、可观察点数以及方向依据。方向没有可靠依据时会明确显示为不确定；服务返回 `identity_ambiguous: true` 时，详情会标记存在多个合理前身。

电子墨迹对象按识别结果保存 `outline`、`centerlines` 和分层 `layers`。`outline` 使用偶奇规则保留 SVG 孔洞，`centerlines` 保留可分析的中心线对象数组，分层颜色与识别到的线宽会随对象保存。工作台的“墨迹”“原板”“叠合”按钮用于查看纯矢量、同一确认序号的原板，或在原板上对照矢量落点。

“识别方式”默认使用“标准 · OpenCV”，也可切换“淡墨增强（实验）”。这两个选项用于当前选中的已保存会话，切换会在当前会话和序号上重新派生，不会重新采集摄像头。首次生成可能需要数秒到几十秒，旧结果会保留到新结果成功返回，并废弃已经过时的请求。重算只能使用历史中已经保存的像素，早期采集时已经丢弃的像素无法找回。对于今后实时写入的淡线，系统会在数次一致采样并达到稳定条件后再确认；淡墨增强仍是弱线候选实验，反光处可能多识别，效果仍需用更多场景验证，不把它描述为已经解决所有淡字问题。服务返回的 `model_version` 会显示在详情状态并写入导出元数据。

如果本机存在预计算的 YOLO 实验回放包，可用 `--replay-packages DIR`
启动服务（包目录和包内权重均保持在本机 `data/`，不会自动下载模型）。包必须按
`DIR/<session-id>/<session-head>.json` 放置，并使用与电子墨迹相同的版本 1
时序字段，额外声明 `recognition_mode: "yolo-experiment"`、
`model_version`、`weights_sha256`、`session_version` 和 `through_seq`。只有当
`session_version` 与当前已保存会话 head 一致时，电子墨迹页才显示“YOLO 分割（实验）”；
没有匹配包时不会显示该选项，也不会改变实时采集。该模式是只读预计算消费路径，支持查看、
原板叠合和 SVG/JSON 导出；包中的模型质量、反光误检和空结果必须单独验收，不能据此把模型
升级为默认识别方式。

“导出 SVG”只导出当前查看序号的纯 SVG 路径，并在 SVG metadata 中记录实际识别方式。“导出墨迹 JSON”导出完整的版本 1 时序对象，包括会话尺寸、`through_seq`、全部 `events` 和 `recognition_mode`，文件名形如 `ink-<session-id>-seq-<view-seq>.json`。导出 JSON 中的 `seq` 和变化顺序表示系统确认可见的顺序，不代表真人下笔的真实先后，也不提供图形语义或笔画方向证据。系统识别的是可观察的路径几何，不能据此推断文字、公式或图形含义。

电子墨迹派生支持服务返回 `202 pending`。界面会保留“正在读取电子墨迹”的状态并继续轮询，每个 HTTP 请求有 20 秒超时；切换会话或离开页面会取消旧请求。派生仍未完成时显示可重试的状态，不会显示虚构的完成百分比。原始照片、摄像头视频和本地 SQLite 样本属于实验数据，不随 SVG 或 JSON 导出，也不应提交到公共仓库。

SVG 和 JSON 都只表达摄像头确认到的几何与颜色。白板过曝、固定反光或遮挡导致的区域没有可用信息时，导出不会补出真实线条或语义。

## 可选实时电子墨迹 runtime（实验入口）

默认采集服务不会加载深度模型。需要明确调用 `live_ink.runtime.OrderedInkRuntime` 才会开启
两级流水线：MPS FP16 的整帧官方 FCN-LectureNet 推理，随后由单一有序 CPU worker
调用现有电子墨迹 pipeline 和 SVG serializer。默认使用模型的 `text_mask` 第二输出作为
原板颜色/known 支持后的候选层；这不是作者最终黑墨二值分支，也不是“官方二值墨迹”直接输出。
队列默认长度为 2，`block` 背压保证
不丢状态；实时预览若明确接受丢帧，可选择 `drop_oldest`，丢帧数量会出现在
`RuntimeStats.dropped`，不会静默补写。

runtime 不包含模型权重，也不把 ignored 数据路径写死。构造
`LectureNetMPSBackend(model_root, weights)` 时分别传入本地的官方 AccessMath
源码目录和已核验的权重文件；backend 会拒绝在 MPS 不可用时回退 CPU。官方
LectureNet 源码和权重需按其原始发布渠道取得，并保留其 MIT 归属信息。当前已核验
的 V34 权重 SHA-256 为 `d3721339db6cc2d2de2a29816bfcaca38f2d4c4bc8a1eb75a7402a194604b144`，
但权重不随本仓库发布。

调用方必须注入每 session 独立的 `pipeline_factory` 和
`svg_serializer(result, width, height)`，并按严格递增的 `seq` 提交已经透视校正的
BGR/known 帧。`start(session_id)`、`submit(...)`、`get_result(...)`、`stop(drain=True)`
构成生命周期；停止后可用新的 session id 重启，旧结果不会混入新 session。运行时
发生 backend、顺序或序列化异常会进入 `failed` 状态并暴露在 `RuntimeStats.error`。

工作台通过 `GET /api/sessions/<session-id>/live-ink?after_seq=<seq>` 获取最新
SVG；这是当前公开的实时接口。服务仍接受旧的 `ink-live` 路径以兼容旧客户端，
但新调用方和文档应使用 `live-ink`。

该入口已接入服务端/UI，但仍是明确 opt-in 的实时实验：先运行
`tools/setup-lecturenet` 获取固定版本代码和 V34 权重，再用
`LECTURENET_LIVE=1 ./run.sh` 启动；页面的“LectureNet · 原板校验（实时实验）”选项
才会开启模型候选。默认 `./run.sh` 仍是 classic-only，不加载深度模型。先运行
`tests/test_runtime.py` 验证顺序、ID 隔离和重启行为；真实摄像头效果仍待真人验收。

## 验证边界

### 真实验证边界

真实低对比评估只代表冻结协议中人工复核的局部线段，不是整板或文字识别率。不同方法的覆盖数只能在同一原板、同一末帧、同一 72 点分母和同一容差下比较；不得把不同报告的局部点数拼成新的召回率。当前公开文档不发布私有采集图片、SQLite、会话 ID、录像或内部分数；下一轮真人验证仍需人工复核。

实时延迟也必须按来源解释：传统 720p 服务的本地剖析约为 30–40 ms 级单帧处理，保存事件还会增加开销；浏览器展示帧率、服务处理帧率和事件确认时间不是同一个指标。固定反光/遮挡会进入 unknown 或 held，不能把缺失观测解释成擦除。下一轮 Mac 摄像头真人验证应重新校准四角，固定光源与相机，记录原板快照、确认序列和 observed/confirmed 时间，并分别查看“原板/叠合/导出 SVG”；首次录淡红线应同时保留未增强原图和反光负例，待人工复核后再谈召回。

运行测试时使用项目自己的环境：

```sh
.venv/bin/python -m unittest discover -s tests -v
```

测试可以使用内存构造的 JPEG 或合成帧来检查状态和回放逻辑。合成帧通过不了真实摄像头、反光、遮挡和浏览器权限验收，真实摄像头验收必须按上面的步骤手动完成。

## 目录约定

- `server.py` 提供本地 HTTP API 和静态工作台。
- `vision.py` 负责四点透视、增强、变化稳定性和遮挡过滤。
- `history.py` 负责 SQLite 状态、快照、增量事件和回放。
- `web/` 是浏览器工作台。
- `tests/` 是不访问真实摄像头的受控测试。
- `data/` 是运行时数据，已被 `.gitignore` 忽略。

离线路径的入口文档：`experiments/faint-color-ink/README.md`（淡墨回放）、`experiments/yolo-ink/README.md`（LectureMath/YOLO 训练与评估）、`experiments/yolo-ink/windows-cuda/README.md`（Windows CUDA runner）和 `experiments/temporal-stress/README.md`（仅合成压力集）。过时 checkpoint、旧回放和真实采集应归档到本地 ignored 数据，不在公共 release 中改名后继续当作当前默认版本；删除/归档 release 资产需由 release 管理者单独执行，本 README 不宣称已发布任何私有权重或图片。

