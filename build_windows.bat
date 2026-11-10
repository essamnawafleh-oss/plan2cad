@echo off
chcp 65001 >nul
echo ============================================
echo   Plan2CAD - Windows build
echo ============================================
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/  & pause & exit /b 1)
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
rem rapidocr pulls plain opencv; the contrib build must win
pip uninstall -y opencv-python opencv-python-headless
pip install --force-reinstall --no-deps opencv-contrib-python-headless
pyinstaller --noconfirm Plan2CAD.spec
echo.
echo Done:  dist\Plan2CAD\Plan2CAD.exe
pause
