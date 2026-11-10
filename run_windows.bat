@echo off
chcp 65001 >nul
if not exist .venv (python -m venv .venv & call .venv\Scripts\activate.bat & pip install -r requirements.txt & pip uninstall -y opencv-python opencv-python-headless & pip install --force-reinstall --no-deps opencv-contrib-python-headless) else (call .venv\Scripts\activate.bat)
python main.py %*
