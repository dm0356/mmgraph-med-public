@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   MMGraph-Agent dependency installer
echo   (installs numpy into your own Python)
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found.
  echo Install Python 3.12+ and tick "Add python.exe to PATH".
  echo Or use a bundled Python via the MMGRAPH_PYTHON environment variable.
  pause
  exit /b 1
)

python --version
echo.

echo Installing numpy (wheel only) ...
python -m pip install --only-binary=:all: numpy
if errorlevel 1 (
  echo.
  echo Retry official PyPI ...
  python -m pip install --only-binary=:all: numpy -i https://pypi.org/simple
)
if errorlevel 1 (
  echo.
  echo Retry Tsinghua mirror ...
  python -m pip install --only-binary=:all: numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
)
if errorlevel 1 (
  echo.
  echo Retry Aliyun mirror ...
  python -m pip install --only-binary=:all: numpy -i https://mirrors.aliyun.com/pypi/simple/
)
if errorlevel 1 (
  echo.
  echo [ERROR] install failed (network blocked, or no numpy wheel for your Python version).
  echo Simplest fix: set MMGRAPH_PYTHON to a Python 3.11+ that already has numpy,
  echo then double-click run_check_api.bat again.
  pause
  exit /b 1
)

echo.
echo Done. You can now double-click run_check_api.bat.
pause >nul
