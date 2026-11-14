@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=src"

echo ============================================
echo   MMGraph-Agent build + visualize
echo   (LLM extraction on the Chinese demo corpus)
echo ============================================
echo.

set /p DEEPSEEK_API_KEY=DeepSeek API Key (required): 
if not defined DEEPSEEK_API_KEY (
  echo [ERROR] DeepSeek key is required for LLM extraction.
  pause
  exit /b 1
)
set /p DASHSCOPE_API_KEY=Qwen/DashScope API Key [Enter to skip image captioning]: 

set "MMGRAPH_EXTRACTOR_BASE_URL=https://api.deepseek.com/v1/chat/completions"
set "MMGRAPH_EXTRACTOR_MODEL=deepseek-chat"
if defined DASHSCOPE_API_KEY (
  set "MMGRAPH_CAPTIONER_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
  set "MMGRAPH_CAPTIONER_MODEL=qwen3-vl-plus"
  set "MMGRAPH_CAPTIONER_ENABLED=true"
)

set "PYEXE=python"
if defined MMGRAPH_PYTHON set "PYEXE=%MMGRAPH_PYTHON%"

"%PYEXE%" -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python with numpy not found: %PYEXE%
  echo Use run_build_zh_local.bat instead.
  pause
  exit /b 1
)

echo.
echo [1/2] building hypergraph with LLM extraction ...
"%PYEXE%" scripts\build_hypergraph.py --config configs\default.toml --data-root data\smoke_zh --out out\mmkg_zh_api.json
if errorlevel 1 (
  echo [ERROR] build failed.
  pause
  exit /b 1
)

echo.
echo [2/2] rendering interactive viewer ...
"%PYEXE%" scripts\render_hypergraph.py --config configs\default.toml --data-root data\smoke_zh --out out\mmkg_zh_api.html --title "MMGraph-Agent Chinese hypergraph (LLM extraction)"
if errorlevel 1 (
  echo [ERROR] render failed.
  pause
  exit /b 1
)

echo.
echo Opening out\mmkg_zh_api.html ...
start "" "out\mmkg_zh_api.html"
echo Done. Press any key to close.
pause >nul
