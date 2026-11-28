@echo off
setlocal
cd /d "%~dp0..\.."
if not exist .venv-yolo\Scripts\python.exe python -m venv .venv-yolo
echo [%date% %time%] Installing CUDA PyTorch > install_cuda.log
.venv-yolo\Scripts\python.exe -m pip install --upgrade pip >> install_cuda.log 2>&1
.venv-yolo\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 >> install_cuda.log 2>&1
.venv-yolo\Scripts\python.exe -m pip install ultralytics==8.4.220 >> install_cuda.log 2>&1
echo [%date% %time%] Installation exit=%errorlevel% >> install_cuda.log
