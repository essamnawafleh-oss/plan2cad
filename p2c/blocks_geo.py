"""Block definitions. Geometry in a normalised 1x1 box centred at 0, front edge at y=-0.5, back at y=+0.5.
Each entry: list of ('pl', pts, closed) | ('circle', (x,y), r) | ('arc', (x,y), r, a0, a1) | ('dpl', pts, closed) [dashed]"""
import numpy as np

def rect(x0, y0, x1, y1): return ('pl', [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], True)
def rrect(x0, y0, x1, y1, r, n=4):
    pts = []
    for cx, cy, a0 in ((x1-r, y1-r, 0), (x0+r, y1-r, 90), (x0+r, y0+r, 180), (x1-r, y0+r, 270)):
        for a in np.linspace(a0, a0+90, n):
            pts.append((cx + r*np.cos(np.radians(a)), cy + r*np.sin(np.radians(a))))
    return ('pl', pts, True)
def ell(cx, cy, rx, ry, a0=0, a1=360, n=28, closed=True):
    return ('pl', [(cx + rx*np.cos(np.radians(a)), cy + ry*np.sin(np.radians(a))) for a in np.linspace(a0, a1, n)], closed)

B = {}
# ---- furniture ----
B['BED_SINGLE'] = [rect(-.5, -.5, .5, .5),
                   ('pl', [(-.43, .48), (-.05, .48), (-.05, .28), (-.43, .28)], False),   # sheet turn-down edge
                   ('pl', [(-.43, .28), (-.05, .48)], False)]                              # folded corner
B['NIGHTSTAND'] = [rect(-.5, -.5, .5, .5)]
def sofa(n):
    g = [rrect(-.5, -.5, .5, .5, .06), rect(-.5, .3, .5, .5),                 # outline + back
         rect(-.5, -.5, -.38, .3), rect(.38, -.5, .5, .3)]                     # arms
    for i in range(1, n):
        x = -.38 + .76*i/n; g.append(('pl', [(x, -.45), (x, .3)], False))
    g.append(('pl', [(-.38, -.45), (.38, -.45)], False))
    return g
B['SOFA_3'] = sofa(3); B['SOFA_2'] = sofa(2); B['ARMCHAIR'] = sofa(1)
B['TABLE_RECT'] = [rect(-.5, -.5, .5, .5)]
B['TABLE_ROUND'] = [ell(0, 0, .5, .5, n=48)]
B['CHAIR_DINING'] = [rrect(-.42, -.5, .42, .3, .12), ell(0, .05, .55, .45, 40, 140, 14, False)]
B['CHAIR_ARM'] = [rrect(-.5, -.5, .5, .5, .18), ('pl', [(-.36, -.42), (-.36, .3)], False),
                  ('pl', [(.36, -.42), (.36, .3)], False), ('pl', [(-.44, .3), (.44, .3)], False)]
B['CHAIR_LIB'] = [rrect(-.4, -.5, .4, .35, .15), ('pl', [(-.45, .45), (.45, .45)], False),
                  ('pl', [(-.45, .55), (.45, .55)], False)]
B['DESK'] = [rect(-.5, -.5, .5, .5)]
B['PEDESTAL'] = [('dpl', [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)], True),
                 ('pl', [(-.25, -.2), (.25, .2)], False), ('pl', [(-.25, .2), (.25, -.2)], False)]
B['CABINET'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.5, -.5), (.5, .5)], False), ('pl', [(-.5, .5), (.5, -.5)], False)]
B['DRAFT_TABLE'] = [('pl', [(-.5, .5), (-.5, -.2), (0, -.5), (.5, -.2), (.5, .5)], False),
                    ('pl', [(-.5, -.2), (.5, -.2)], False), ('pl', [(0, -.5), (0, -.32)], False),
                    ('pl', [(-.5, .5), (.5, .5)], False)]
B['SHELF'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.5, 0), (.5, 0)], False)]
B['WARDROBE'] = [rect(-.5, -.5, .5, .5), ('pl', [(0, -.5), (0, .5)], False),
                 ('pl', [(-.5, .5), (0, -.5), (.5, .5)], False)]
# library curved desk (front = convex side)
def arc_pts(cx, cy, r, a0, a1, n=24): return [(cx + r*np.cos(np.radians(a)), cy + r*np.sin(np.radians(a))) for a in np.linspace(a0, a1, n)]
_o = arc_pts(0, 1.45, 1.95, 247, 293); _i = arc_pts(0, 1.45, 1.5, 248, 292)
B['DESK_CURVED'] = [('pl', _o + [(.33, .5), (-.33, .5)], True), ('pl', _i, False)]
B['WET_BAR'] = [rect(-.5, -.5, .5, .5), ell(0, -.05, .3, .22), ell(0, -.05, .18, .1)]
# ---- sanitary ----
B['WC'] = [rect(-.5, .3, .5, .5), ell(0, -.08, .42, .42, 180, 360, 16, False),
           ('pl', [(-.42, -.08), (-.42, .3)], False), ('pl', [(.42, -.08), (.42, .3)], False),
           ell(0, -.05, .28, .3)]
B['BIDET'] = [rrect(-.45, -.5, .45, .5, .4), ell(0, 0, .25, .3), ('circle', (0, .38), .05)]
B['BASIN'] = [rrect(-.5, -.5, .5, .5, .15), ell(0, -.05, .35, .3), ('circle', (0, .35), .05)]
B['BATHTUB'] = [rect(-.5, -.5, .5, .5), rrect(-.44, -.38, .44, .38, .25), ('circle', (-.36, 0), .04)]
B['SHAFT'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.5, -.5), (.5, .5)], False), ('pl', [(-.5, .5), (.5, -.5)], False)]
# ---- kitchen ----
B['SINK_SINGLE'] = [rrect(-.5, -.5, .5, .5, .1), rrect(-.38, -.38, .38, .3, .12), ('circle', (0, .4), .05)]
B['SINK_DRAINER'] = [rect(-.5, -.5, .5, .5)] + [('pl', [(-.38, y), (.38, y)], False) for y in np.linspace(-.38, .38, 7)]
B['COOKER'] = [rect(-.5, -.5, .5, .5)] + [('circle', (x, y), .17) for x in (-.24, .24) for y in (-.24, .24)]
B['FRIDGE'] = [rect(-.5, -.5, .5, .5), ('pl', [(-.5, -.35), (.5, -.35)], False)]
# ---- court / terrace ----
B['FOUNTAIN'] = [rect(-.5, -.5, .5, .5), rect(-.38, -.38, .38, .38)]
B['STEPS_U3'] = [('pl', [(-.5, -.5), (-.5, .5), (.5, .5), (.5, -.5)], False),
                 ('pl', [(-.38, -.5), (-.38, .17), (.38, .17), (.38, -.5)], False),
                 ('pl', [(-.26, -.5), (-.26, -.17), (.26, -.17), (.26, -.5)], False)]
B['PLANTER'] = [rrect(-.5, -.5, .5, .5, .15), rrect(-.35, -.3, .35, .3, .1)]
