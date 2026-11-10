"""Command-line batch conversion:  python cli.py plan.jpg -o plan.dxf [--scale 0.03] [--rotate 90]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2c.model import Project
from p2c.engine import imageio, pipeline
from p2c.dxf_export import export_dxf


def main():
    ap = argparse.ArgumentParser(description='Plan2CAD: scanned plan -> layered DXF')
    ap.add_argument('image')
    ap.add_argument('-o', '--out', help='output .dxf (default: next to the image)')
    ap.add_argument('--scale', type=float, default=0.03, help='metres per pixel (default 0.03)')
    ap.add_argument('--rotate', type=int, default=0, choices=[0, 90, 180, 270], help='rotate image clockwise first')
    ap.add_argument('--no-deskew', action='store_true')
    ap.add_argument('--furniture', action='store_true', help='also try to recognise furniture (experimental)')
    ap.add_argument('--project', help='also save an editable .p2c project')
    a = ap.parse_args()
    g = imageio.imread_gray(a.image)
    if a.rotate:
        g = imageio.rotate_bound(g, -a.rotate)
    if not a.no_deskew:
        sk = imageio.estimate_skew(g)
        if sk:
            g = imageio.rotate_bound(g, sk)
    g = imageio.clean_background(g)
    out = a.out or os.path.splitext(a.image)[0] + '.dxf'
    work = os.path.splitext(out)[0] + '_image.png'
    imageio.imwrite(work, g)
    pr = Project(image_path=os.path.basename(work), scale=a.scale, scale_calibrated=True)
    pipeline.run(pr, g, dict(do_furniture=a.furniture),
                 progress=lambda f, m='': print(f'\r{int(f*100):3d}% {m:12s}', end='', flush=True))
    print()
    export_dxf(pr, out, (g.shape[1], g.shape[0]), work)
    if a.project:
        pr.save(a.project)
    print(f'walls {len(pr.walls)}  columns {len(pr.columns)}  doors {len(pr.doors)}  windows {len(pr.windows)}  -> {out}')


if __name__ == '__main__':
    main()
