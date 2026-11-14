@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=src"

echo ============================================
echo   MMGraph-Agent API check
echo ============================================
echo.

set /p DEEPSEEK_API_KEY=DeepSeek API Key [Enter to skip]: 
set /p DASHSCOPE_API_KEY=Qwen/DashScope API Key [Enter to skip]: 

if defined DEEPSEEK_API_KEY (
  if not defined MMGRAPH_EXTRACTOR_BASE_URL set "MMGRAPH_EXTRACTOR_BASE_URL=https://api.deepseek.com/v1/chat/completions"
  if not defined MMGRAPH_EXTRACTOR_MODEL set "MMGRAPH_EXTRACTOR_MODEL=deepseek-chat"
)
if defined DASHSCOPE_API_KEY (
  if not defined MMGRAPH_CAPTIONER_BASE_URL set "MMGRAPH_CAPTIONER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  if not defined MMGRAPH_CAPTIONER_MODEL set "MMGRAPH_CAPTIONER_MODEL=qwen3-vl-plus"
  if not defined MMGRAPH_CAPTIONER_ENABLED set "MMGRAPH_CAPTIONER_ENABLED=true"
)

set "PYEXE=python"
if defined MMGRAPH_PYTHON set "PYEXE=%MMGRAPH_PYTHON%"

echo.
"%PYEXE%" --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found: %PYEXE%
  echo Set MMGRAPH_PYTHON to a Python with numpy, or double-click run_check_api_local.bat.
  pause
  exit /b 1
)

"%PYEXE%" -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This Python has no numpy: %PYEXE%
  echo Options:
  echo   1^) double-click run_check_api_local.bat  ^(uses the bundled Python, no install^)
  echo   2^) double-click install_deps.bat          ^(needs internet^)
  pause
  exit /b 1
)

"%PYEXE%" scripts\check_api.py %*
echo.
echo Done. Press any key to close.
pause >nul
