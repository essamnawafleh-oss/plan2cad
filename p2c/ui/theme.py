"""Dark CAD theme: one place for colours, spacing and type."""
from PySide6.QtGui import QColor, QFont, QPalette

# spacing scale (px)
SP = dict(xs=4, s=8, m=12, l=16, xl=24, xxl=32)

C = dict(
    window='#1E1F22', panel='#26282C', card='#2F3237', input='#24262A',
    border='#3A3D43', border_strong='#5A5F66',
    text='#E8E9EB', text2='#AEB2B9', text3='#8A8F97',
    accent='#3D9BFF', accent_hover='#5AACFF', accent_press='#2C7FD6', accent_soft='#1F3A57',
    success='#3FB950', warning='#D29922', danger='#F85149',
    canvas='#2B2D31', paper='#FFFFFF',
)

# drawing colours per layer (on white paper)
LAYER_COLORS = {
    'A-SCAN': '#9AA0A6', 'A-TRACE': '#E07B39', 'A-WALL': '#111111', 'A-COLUMN': '#111111',
    'A-WINDOW': '#0E8FB8', 'A-DOOR': '#1E9E4A', 'A-STAIR': '#9C27B0', 'A-FURN': '#D9730D',
    'A-SANITARY': '#1D63D6', 'A-KITCHEN': '#0F9D8A', 'A-COURT': '#4C9A2A', 'A-TEXT': '#C62828',
    'A-LEVEL': '#2E7D32', 'A-DIM': '#E53935',
}
SELECT_COLOR = '#FF2D78'
PREVIEW_COLOR = '#3D9BFF'

FONT_FAMILIES = ["Segoe UI Variable", "Segoe UI", "Tahoma", "Noto Sans Arabic", "DejaVu Sans", "Arial"]


def font(pt=10, bold=False):
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPointSizeF(pt)
    if bold:
        f.setWeight(QFont.Weight.DemiBold)
    return f


def palette():
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(C['window']))
    p.setColor(QPalette.ColorRole.Base, QColor(C['input']))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(C['card']))
    p.setColor(QPalette.ColorRole.Text, QColor(C['text']))
    p.setColor(QPalette.ColorRole.WindowText, QColor(C['text']))
    p.setColor(QPalette.ColorRole.Button, QColor(C['card']))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(C['text']))
    p.setColor(QPalette.ColorRole.Highlight, QColor(C['accent']))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor('#FFFFFF'))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(C['card']))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(C['text']))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(C['text3']))
    return p


def stylesheet():
    c = C
    return f"""
* {{ color: {c['text']}; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {c['window']}; }}
QToolTip {{ background: {c['card']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 4px 8px; border-radius: 4px; }}

QToolBar {{ background: {c['panel']}; border: none; border-bottom: 1px solid {c['border']}; padding: 4px 8px; spacing: 4px; }}
QToolBar::separator {{ background: {c['border']}; width: 1px; margin: 6px 6px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px; color: {c['text']}; }}
QToolButton:hover {{ background: {c['card']}; border-color: {c['border']}; }}
QToolButton:pressed {{ background: {c['accent_press']}; }}
QToolButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
QToolButton:disabled {{ color: {c['text3']}; }}
QToolButton#primary {{ background: {c['accent']}; color: #FFFFFF; font-weight: 600; padding: 6px 12px; }}
QToolButton#primary:hover {{ background: {c['accent_hover']}; }}
QToolButton#primary:pressed {{ background: {c['accent_press']}; }}
QToolButton#lang {{ border: 1px solid {c['border']}; padding: 5px 10px; font-weight: 600; }}

QDockWidget {{ color: {c['text2']}; font-weight: 600; }}
QDockWidget::title {{ background: {c['panel']}; padding: 8px 12px; border-bottom: 1px solid {c['border']}; text-align: left; }}
QWidget#panel {{ background: {c['panel']}; }}
QLabel#h1 {{ font-size: 13pt; font-weight: 600; color: {c['text']}; }}
QLabel#section {{ color: {c['text2']}; font-size: 9pt; font-weight: 600; text-transform: uppercase; padding-top: 8px; }}
QLabel#caption {{ color: {c['text2']}; font-size: 9pt; }}
QLabel#hint {{ color: {c['text2']}; background: {c['card']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 8px 12px; }}

QPushButton {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ border-color: {c['border_strong']}; background: #363A40; }}
QPushButton:pressed {{ background: {c['input']}; }}
QPushButton:disabled {{ color: {c['text3']}; border-color: {c['border']}; }}
QPushButton:focus {{ border-color: {c['accent']}; }}
QPushButton#primary {{ background: {c['accent']}; color: #FFFFFF; border: none; font-weight: 600; padding: 7px 18px; }}
QPushButton#primary:hover {{ background: {c['accent_hover']}; }}
QPushButton#primary:pressed {{ background: {c['accent_press']}; }}
QPushButton#primary:disabled {{ background: {c['border']}; color: {c['text3']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 5px 8px; selection-background-color: {c['accent']}; min-height: 20px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{ color: {c['text3']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {c['card']}; border: 1px solid {c['border']}; selection-background-color: {c['accent_soft']}; outline: none; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {c['border_strong']}; background: {c['input']}; }}
QCheckBox::indicator:hover {{ border-color: {c['accent']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}

QSlider::groove:horizontal {{ height: 4px; background: {c['border']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {c['accent']}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 2px; }}

QListWidget, QTreeWidget {{ background: {c['panel']}; border: none; outline: none; }}
QListWidget::item, QTreeWidget::item {{ padding: 5px 6px; border-radius: 6px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {c['card']}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {c['accent_soft']}; color: {c['text']}; }}
QHeaderView::section {{ background: {c['panel']}; border: none; }}

QGroupBox {{ border: 1px solid {c['border']}; border-radius: 8px; margin-top: 14px; padding: 12px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {c['text2']}; font-weight: 600; }}

QStatusBar {{ background: {c['panel']}; border-top: 1px solid {c['border']}; color: {c['text2']}; }}
QStatusBar QLabel {{ color: {c['text2']}; padding: 0 8px; }}
QProgressBar {{ background: {c['input']}; border: 1px solid {c['border']}; border-radius: 6px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 5px; }}

QMenu {{ background: {c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {c['accent_soft']}; }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 4px 8px; }}
QMenuBar {{ background: {c['panel']}; }}
QMenuBar::item:selected {{ background: {c['card']}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #4A4E55; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #5D626A; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #4A4E55; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: #5D626A; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 14px; color: {c['text2']}; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {c['text']}; border-bottom: 2px solid {c['accent']}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {c['text']}; }}
QTabWidget::pane {{ border: none; }}
"""
