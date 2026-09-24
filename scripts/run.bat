@echo off
REM Webcam 3D Mapper - Windows launcher.
REM Creates .venv if needed, installs dependencies, starts the local server.

setlocal
cd /d "%~dp0.."

echo.
echo   Webcam 3D Mapper
echo   ----------------
echo.

REM --- Python ---------------------------------------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo   [X] Python was not found on PATH.
  echo       Install Python 3.11 or newer from https://python.org and re-run this script.
  echo.
  pause
  exit /b 1
)

REM --- Virtual environment --------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo   Creating virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo   [X] Could not create the virtual environment.
    pause
    exit /b 1
  )
)

set "VPY=.venv\Scripts\python.exe"

REM Install only when the dependency set changed, so normal starts stay fast.
set "STAMP=.venv\.deps-installed"
set "NEEDS_INSTALL=1"
if exist "%STAMP%" (
  fc /b "%STAMP%" requirements.txt >nul 2>&1 && set "NEEDS_INSTALL=0"
)

if "%NEEDS_INSTALL%"=="1" (
  echo   Installing dependencies...
  "%VPY%" -m pip install --quiet --upgrade pip
  "%VPY%" -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo   [X] Dependency installation failed.
    pause
    exit /b 1
  )
  copy /y requirements.txt "%STAMP%" >nul
)

REM --- COLMAP notice --------------------------------------------------------
REM Detection happens in the app; this is only a friendly heads-up.
where colmap >nul 2>&1 || where COLMAP.bat >nul 2>&1 || (
  if not exist "C:\COLMAP\COLMAP.bat" (
    echo   [!] COLMAP was not found on PATH.
    echo       Scanning will work, but reconstruction needs COLMAP:
    echo       https://github.com/colmap/colmap/releases  ^(download the -windows-no-cuda zip^)
    echo       Then set the path in the app's Settings panel, or set COLMAP_PATH.
    echo.
  )
)

echo   Starting server on http://127.0.0.1:8765
echo   Press Ctrl+C to stop.
echo.
"%VPY%" app.py --open %*

endlocal
