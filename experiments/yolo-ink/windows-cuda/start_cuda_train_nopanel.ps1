$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$runner = Join-Path $root "tools\windows-cuda\run_windows_cuda.ps1"
$python = Join-Path $root ".venv-yolo\Scripts\python.exe"
$runtimeYaml = Join-Path $root "dataset.runtime.yaml"
@"
path: '$root'
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test
names:
  0: ink
"@ | Set-Content -LiteralPath $runtimeYaml -Encoding UTF8
& $runner -Data $runtimeYaml `
  -Weights (Join-Path $root "weights\f54_mps_initial.pt") `
  -Out (Join-Path $root "runs") -Name "yolo11s_seg_train700_cuda" `
  -Python $python -Epochs 6 -Imgsz 512 -Batch 4 -NoPanel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
