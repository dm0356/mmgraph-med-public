@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONPATH=src"

echo ============================================
echo   MMGraph-Agent CPU smoke test
echo ============================================
echo.

set "PYEXE=python"
if defined MMGRAPH_PYTHON set "PYEXE=%MMGRAPH_PYTHON%"

"%PYEXE%" --version >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found: %PYEXE%
  echo Set MMGRAPH_PYTHON to a Python with numpy, or double-click run_smoke_local.bat.
  pause
  exit /b 1
)

"%PYEXE%" -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This Python has no numpy: %PYEXE%
  echo Options:
  echo   1^) double-click run_smoke_local.bat  ^(uses the bundled Python, no install^)
  echo   2^) double-click install_deps.bat      ^(needs internet^)
  pause
  exit /b 1
)

"%PYEXE%" scripts\run_smoke.py --verbose
echo.
echo Done. Press any key to close.
pause >nul
