param(
  [Parameter(Mandatory=$true)][string]$Data,
  [Parameter(Mandatory=$true)][string]$Weights,
  [Parameter(Mandatory=$true)][string]$Out,
  [Parameter(Mandatory=$true)][string]$Name,
  [string]$Python = ".venv-yolo\\Scripts\\python.exe",
  [int]$Epochs = 4,
  [int]$Imgsz = 512,
  [int]$Batch = 4,
  [int]$Seed = 0,
  [switch]$NoPanel,
  [switch]$Resume
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
$cuda = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($cuda.Trim() -ne "1") { throw "CUDA unavailable; refusing CPU fallback" }
$script = Join-Path $PSScriptRoot "windows_cuda_train.py"
$panel = Join-Path $PSScriptRoot "windows_cuda_panel.py"
$runDir = Join-Path ([System.IO.Path]::GetFullPath($Out)) $Name
if (-not $NoPanel) {
  $py = (Resolve-Path -LiteralPath $Python).Path
  $panelCommand = "& '" + $py.Replace("'", "''") + "' '" + $panel.Replace("'", "''") + "' --run-dir '" + $runDir.Replace("'", "''") + "'"
  Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $panelCommand) -WorkingDirectory (Get-Location)
}
$args = @($script,$Data,"--out",$Out,"--weights",$Weights,"--name",$Name,"--epochs",$Epochs,"--imgsz",$Imgsz,"--batch",$Batch,"--seed",$Seed)
if ($Resume) { $args += "--resume" }
& $Python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
