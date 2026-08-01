@echo off
chcp 65001 >nul 2>&1
title DeepSeekChat - 本地 DeepSeek 对话框架

echo ==================================================
echo   DeepSeekChat - 本地 DeepSeek 对话框架
echo ==================================================
echo.

REM 检查 Python
set PYTHON=E:\Python311\python.exe
if not exist "%PYTHON%" (
    set PYTHON=python
)

REM 检查模型是否存在 (扫描 models 目录下任意 .gguf 文件)
set HAS_MODEL=0
if exist "models\*.gguf" set HAS_MODEL=1
if "%HAS_MODEL%"=="0" (
    echo [警告] 未找到模型文件!
    echo 正在运行模型下载脚本...
    echo.
    "%PYTHON%" download_model.py
    echo.
)

echo 正在启动 DeepSeekChat 服务...
echo 访问地址: http://127.0.0.1:7860
echo 按 Ctrl+C 停止服务
echo.

"%PYTHON%" app.py

pause
