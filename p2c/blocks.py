"""Furniture / fixture block library.

Every block is drawn in a normalised 1 x 1 box centred on the origin with the
*front* edge at y = -0.5 and the back at y = +0.5.  Instances scale it to their
real width/depth (metres).
"""
from .blocks_geo import B as GEO, rect, rrect, ell
import numpy as np

# extra blocks not present in the original drawing
GEO['BED_DOUBLE'] = [rect(-.5, -.5, .5, .5), rrect(-.45, .25, -.05, .45, .05), rrect(.05, .25, .45, .45, .05),
                     ('pl', [(-.5, .18), (.5, .18)], False), ('pl', [(.25, .18), (.5, -.1)], False)]
GEO['CHAIR_OFFICE'] = [('pl', [(x*.4, y*.4 - .05) for x, y in [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 25)]], True),
                       rrect(-.42, .3, .42, .5, .08)]
GEO['TV_UNIT'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.4, .2), (.4, .2)], False)]
GEO['SHOWER'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.5, -.5), (.5, .5)], False), ('pl', [(-.5, .5), (.5, -.5)], False),
                 ('circle', (0, 0), .06)]
GEO['URINAL'] = [ell(0, .1, .45, .4, 180, 360, 16, False), ('pl', [(-.45, .1), (-.45, .5), (.45, .5), (.45, .1)], False)]
GEO['DISHWASHER'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.4, -.35), (.4, -.35)], False)]
GEO['TREE'] = [ell(0, 0, .5, .5, n=36), ell(0, 0, .12, .12, n=12)]

# name -> (layer, category, default w, default d, arabic, english)
LIB = {
    'BED_SINGLE':   ('A-FURN', 'furn', 0.95, 1.95, 'سرير مفرد', 'Single bed'),
    'BED_DOUBLE':   ('A-FURN', 'furn', 1.60, 2.00, 'سرير مزدوج', 'Double bed'),
    'NIGHTSTAND':   ('A-FURN', 'furn', 0.50, 0.45, 'كومودينا', 'Nightstand'),
    'WARDROBE':     ('A-FURN', 'furn', 1.80, 0.60, 'خزانة ملابس', 'Wardrobe'),
    'SOFA_3':       ('A-FURN', 'furn', 2.10, 0.90, 'كنباية 3 مقاعد', '3-seat sofa'),
    'SOFA_2':       ('A-FURN', 'furn', 1.60, 0.90, 'كنباية مقعدين', '2-seat sofa'),
    'ARMCHAIR':     ('A-FURN', 'furn', 0.85, 0.85, 'كرسي فردي', 'Armchair'),
    'TABLE_RECT':   ('A-FURN', 'furn', 1.20, 0.70, 'طاولة مستطيلة', 'Rectangular table'),
    'TABLE_ROUND':  ('A-FURN', 'furn', 1.05, 1.05, 'طاولة دائرية', 'Round table'),
    'CHAIR_DINING': ('A-FURN', 'furn', 0.50, 0.50, 'كرسي سفرة', 'Dining chair'),
    'CHAIR_ARM':    ('A-FURN', 'furn', 0.70, 0.60, 'كرسي بذراعين', 'Arm chair'),
    'CHAIR_LIB':    ('A-FURN', 'furn', 0.50, 0.45, 'كرسي مكتبة', 'Library chair'),
    'CHAIR_OFFICE': ('A-FURN', 'furn', 0.60, 0.60, 'كرسي مكتب', 'Office chair'),
    'DESK':         ('A-FURN', 'furn', 1.50, 0.80, 'مكتب', 'Desk'),
    'PEDESTAL':     ('A-FURN', 'furn', 0.45, 0.70, 'وحدة أدراج', 'Pedestal'),
    'CABINET':      ('A-FURN', 'furn', 0.70, 0.45, 'خزانة ملفات', 'Cabinet'),
    'DRAFT_TABLE':  ('A-FURN', 'furn', 1.35, 0.80, 'طاولة رسم', 'Drafting table'),
    'DESK_CURVED':  ('A-FURN', 'furn', 2.20, 1.20, 'مكتب مقوّس', 'Curved desk'),
    'SHELF':        ('A-FURN', 'furn', 2.00, 0.40, 'رف كتب', 'Bookshelf'),
    'TV_UNIT':      ('A-FURN', 'furn', 1.80, 0.45, 'وحدة تلفزيون', 'TV unit'),
    'WET_BAR':      ('A-FURN', 'furn', 1.35, 0.70, 'كاونتر بمغسلة', 'Counter with basin'),
    'WC':           ('A-SANITARY', 'san', 0.45, 0.75, 'مرحاض', 'WC'),
    'BIDET':        ('A-SANITARY', 'san', 0.40, 0.55, 'بيديه', 'Bidet'),
    'BASIN':        ('A-SANITARY', 'san', 0.55, 0.45, 'مغسلة', 'Basin'),
    'BATHTUB':      ('A-SANITARY', 'san', 1.70, 0.75, 'بانيو', 'Bathtub'),
    'SHOWER':       ('A-SANITARY', 'san', 0.90, 0.90, 'دش', 'Shower'),
    'URINAL':       ('A-SANITARY', 'san', 0.40, 0.35, 'مبولة', 'Urinal'),
    'SHAFT':        ('A-SANITARY', 'san', 0.60, 0.90, 'شفت', 'Shaft'),
    'SINK_SINGLE':  ('A-KITCHEN', 'kit', 0.60, 0.50, 'مجلى', 'Sink'),
    'SINK_DRAINER': ('A-KITCHEN', 'kit', 0.45, 0.50, 'مصفاية مجلى', 'Sink drainer'),
    'COOKER':       ('A-KITCHEN', 'kit', 0.60, 0.60, 'غاز', 'Cooker'),
    'FRIDGE':       ('A-KITCHEN', 'kit', 0.70, 0.70, 'ثلاجة', 'Fridge'),
    'DISHWASHER':   ('A-KITCHEN', 'kit', 0.60, 0.60, 'جلاية', 'Dishwasher'),
    'FOUNTAIN':     ('A-COURT', 'court', 1.15, 1.15, 'نافورة', 'Fountain'),
    'STEPS_U3':     ('A-COURT', 'court', 3.00, 1.10, 'درجات', 'Steps'),
    'PLANTER':      ('A-COURT', 'court', 0.55, 0.30, 'حوض زرع', 'Planter'),
    'TREE':         ('A-COURT', 'court', 1.20, 1.20, 'شجرة', 'Tree'),
}
CATEGORIES = [('furn', 'أثاث', 'Furniture'), ('san', 'صحّي', 'Sanitary'),
              ('kit', 'مطبخ', 'Kitchen'), ('court', 'كورت وتراس', 'Court & terrace')]


def block_polylines(name):
    """Normalised geometry as a list of (points Nx2 array, closed, dashed)."""
    out = []
    for g in GEO[name]:
        if g[0] in ('pl', 'dpl'):
            out.append((np.array(g[1], float), g[2], g[0] == 'dpl'))
        elif g[0] == 'circle':
            (cx, cy), r = g[1], g[2]
            a = np.linspace(0, 2*np.pi, 25)
            out.append((np.c_[cx + r*np.cos(a), cy + r*np.sin(a)], True, False))
    return out
