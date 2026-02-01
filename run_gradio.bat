@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   HeartLib Gradio Interface Launcher
echo ========================================
echo.

REM Check if python_embeded exists
if not exist "python_embeded\python.exe" (
    echo [ERROR] python_embeded\python.exe not found!
    echo Please ensure the python_embeded folder exists.
    pause
    exit /b 1
)

REM Check if models are downloaded
echo Checking model files...

if not exist "ckpt\HeartMuLa-oss-3B" (
    echo.
    echo [ERROR] Model not found: ckpt\HeartMuLa-oss-3B
    echo.
    echo Please download the models first using one of the following:
    echo   - download_models_hf.bat        ^(Hugging Face^)
    echo   - download_models_modelscope.bat ^(ModelScope^)
    echo.
    pause
    exit /b 1
)

if not exist "ckpt\HeartCodec-oss" (
    echo.
    echo [ERROR] Model not found: ckpt\HeartCodec-oss
    echo.
    echo Please download the models first using one of the following:
    echo   - download_models_hf.bat        ^(Hugging Face^)
    echo   - download_models_modelscope.bat ^(ModelScope^)
    echo.
    pause
    exit /b 1
)

echo Models found.

REM Check if gradio is installed
echo Checking Gradio installation...
.\python_embeded\python.exe -c "import gradio" 2>nul
if errorlevel 1 (
    echo [WARNING] Gradio not found. Installing...
    .\python_embeded\python.exe -m pip install gradio --quiet
)

REM Add local ffmpeg to PATH if exists
if exist "ffmpeg\bin\ffmpeg.exe" (
    echo FFmpeg found in local directory.
    set "PATH=%~dp0ffmpeg\bin;%PATH%"
) else (
    echo [NOTE] Local ffmpeg not found. Using system ffmpeg if available.
)

echo.
echo Starting Gradio interface...
echo.

REM Launch the Gradio app with unbuffered output
.\python_embeded\python.exe -u gradio_app.py %*

echo.
echo ========================================
echo   Gradio interface has stopped.
echo ========================================
pause
