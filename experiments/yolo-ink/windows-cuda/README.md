# Windows CUDA migration bundle

This bundle is prepared on Mac but has not been CUDA-tested here. It never
falls back to CPU: the PowerShell wrapper and Python entry point both require
`torch.cuda.is_available()`.

## Local Windows setup

Copy or unpack this public experiment bundle into a local Windows working
directory. Keep the launcher and training panel in separate visible PowerShell
windows. No remote host, SSH identity, or personal machine path is required.

From the clean unpack directory in PowerShell:

```powershell
py -3.11 -m venv .venv-yolo
.\.venv-yolo\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision ultralytics==8.4.220
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Run visibly with explicit paths; this does not use Mac logs or absolute paths:

The default initialization for the 700-image Windows run is the portable
original `f54` checkpoint copied to the Windows machine and verified there by
SHA-256. The completed Mac-best checkpoint is an explicit optional warm-start,
not the default and not a `--resume` operation: pass its copied path as
`-Weights` only if you intentionally want that comparison.

```powershell
.\experiments\yolo-ink\windows-cuda\run_windows_cuda.ps1 `
  -Data C:\wb\lecturemath-yolo-clean\dataset.yaml `
  -Weights C:\wb\checkpoints\new-model.pt `
  -Out C:\wb\runs\windows-cuda `
  -Name lecturemath_yolo11s_seg_cuda512_v4
```

The PowerShell launcher opens a second visible PowerShell panel by default.
The training callback atomically refreshes `progress.json` about once per
second with epoch, batch, loss and ETA; use `-NoPanel` only when opening the
panel manually.

In a second visible PowerShell window, monitor the same run:

```powershell
python .\experiments\yolo-ink\windows-cuda\windows_cuda_panel.py `
  --run-dir C:\wb\runs\windows-cuda\lecturemath_yolo11s_seg_cuda512_v4
```

The run records a machine-readable config and Ultralytics keeps `results.csv`,
weights, and the raw log in the explicit run directory. Use `-Resume` only
after checking that the named run's `weights\last.pt` is the intended one.
This bundle contains no user capture, checkpoint, or Mac training log.
