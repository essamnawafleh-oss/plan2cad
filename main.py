"""Plan2CAD – turn scanned floor plans into layered DXF drawings.

Run:  python main.py [image-or-project]
"""
import os
import sys

if getattr(sys, 'frozen', False):          # PyInstaller
    os.chdir(os.path.dirname(sys.executable))


def main():
    from PySide6.QtCore import Qt, QSettings
    from PySide6.QtWidgets import QApplication
    from p2c.ui import theme
    from p2c.ui.i18n import LANG, is_rtl

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName('Plan2CAD')
    app.setOrganizationName('Plan2CAD')
    LANG['cur'] = QSettings('Plan2CAD', 'Plan2CAD').value('lang', 'ar')
    from PySide6.QtGui import QIcon
    import p2c
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(p2c.__file__), 'assets', 'icon.png')))
    app.setStyle('Fusion')
    app.setPalette(theme.palette())
    app.setFont(theme.font(10))
    app.setStyleSheet(theme.stylesheet())
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft if is_rtl() else Qt.LayoutDirection.LeftToRight)

    from PySide6.QtCore import QTimer
    from p2c.ui.splash import make_splash
    splash = make_splash(); splash.show(); app.processEvents()
    from p2c.ui.main_window import MainWindow
    w = MainWindow()
    app._main = w
    QTimer.singleShot(1800, lambda: (w.show(), splash.finish(w)))
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        w.show()
        arg = sys.argv[1]
        if arg.lower().endswith('.p2c'):
            w.load_project(arg)
        else:
            w.load_image(arg)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
