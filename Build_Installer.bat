@echo off
chcp 65001 >nul
title Plan2CAD - Essam Nawafleh
cd /d "%~dp0"
set "PF86=%ProgramFiles(x86)%"
echo ==========================================================
echo    Plan2CAD  -  Essam Nawafleh  -  عصام النوافلة
echo    Building the installer. This takes 5-15 minutes, once.
echo ==========================================================
echo.

rem ---- 1. Python 3.12 -------------------------------------------------
set "PY="
py -3.12 -c "import sys" >nul 2>nul && set "PY=py -3.12"
if not defined PY (
  python -c "import sys; exit(0 if sys.version_info[:2] >= (3,10) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [1/4] Installing Python 3.12 ...
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
  set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
%PY% -c "import sys; print('Python', sys.version.split()[0])" || (echo Python could not be installed. Install Python 3.12 from python.org and run again. & pause & exit /b 1)

rem ---- 2. Inno Setup ---------------------------------------------------
set "ISCC=%PF86%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo [2/4] Installing Inno Setup ...
  winget install -e --id JRSoftware.InnoSetup --silent --accept-package-agreements --accept-source-agreements
  set "ISCC=%PF86%\Inno Setup 6\ISCC.exe"
  if not exist "%PF86%\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
if not exist "%ISCC%" (echo Inno Setup could not be installed. Install it from jrsoftware.org and run again. & pause & exit /b 1)

rem ---- 3. Build the program -----------------------------------------
echo [3/4] Building Plan2CAD.exe ...
if not exist .venv %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt pyinstaller || (echo Package installation failed. Check the internet connection. & pause & exit /b 1)
pip uninstall -y opencv-python opencv-python-headless >nul 2>nul
pip install --force-reinstall --no-deps opencv-contrib-python-headless >nul
pyinstaller --noconfirm Plan2CAD.spec || (echo Build failed. & pause & exit /b 1)

rem ---- 4. Build the installer ---------------------------------------
echo [4/4] Building Plan2CAD_Setup.exe ...
"%ISCC%" installer.iss || (echo Installer build failed. & pause & exit /b 1)

echo.
echo ==========================================================
echo    Done:  output\Plan2CAD_Setup.exe
echo    Double-click it to install Plan2CAD like any program.
echo ==========================================================
explorer output
pause
