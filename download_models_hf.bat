@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   HeartMuLa Model Downloader
echo   (Using Hugging Face)
echo ========================================
echo.

REM Check if huggingface_hub is installed
.\python_embeded\python.exe -c "from huggingface_hub import snapshot_download" 2>nul
if errorlevel 1 (
    echo Installing huggingface_hub...
    .\python_embeded\python.exe -m pip install huggingface_hub --quiet
)

echo.
echo [1/2] Downloading HeartMuLa-RL-oss-3B-20260123...
echo.
.\python_embeded\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='HeartMuLa/HeartMuLa-RL-oss-3B-20260123', local_dir='./ckpt/HeartMuLa-oss-3B')"

if errorlevel 1 (
    echo [ERROR] Failed to download HeartMuLa model.
    pause
    exit /b 1
)

echo.
echo [2/2] Downloading HeartCodec-oss-20260123...
echo.
.\python_embeded\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='HeartMuLa/HeartCodec-oss-20260123', local_dir='./ckpt/HeartCodec-oss')"

if errorlevel 1 (
    echo [ERROR] Failed to download HeartCodec model.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Download Complete!
echo ========================================
echo.
echo Models saved to:
echo   - ckpt\HeartMuLa-oss-3B
echo   - ckpt\HeartCodec-oss
echo.
pause
