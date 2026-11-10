# PyInstaller spec – build with:  pyinstaller Plan2CAD.spec
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
block_cipher = None
a = Analysis(
    ['main.py'],
    pathex=['.'],
    datas=[('p2c/assets', 'p2c/assets')] + collect_data_files('rapidocr_onnxruntime') + collect_data_files('anthropic'),
    binaries=collect_dynamic_libs('onnxruntime'),
    hiddenimports=['ezdxf', 'shapely', 'cv2', 'numpy', 'rapidocr_onnxruntime', 'onnxruntime'] + collect_submodules('anthropic'),
    excludes=['tkinter', 'matplotlib', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.Qt3DCore', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtMultimedia'],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Plan2CAD', console=False,
          icon='p2c/assets/icon.ico', version='version_info.txt', upx=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, name='Plan2CAD', upx=False)
