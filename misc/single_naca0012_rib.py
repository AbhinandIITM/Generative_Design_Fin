#!/usr/bin/env python3
import math
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import cadquery as cq
except Exception:
    cq = None


def naca0012_half_thickness(x: np.ndarray) -> np.ndarray:
    t = 0.12
    return 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 0.0, 1.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def naca0012_polygon(chord: float, n: int = 250) -> List[Tuple[float, float]]:
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = naca0012_half_thickness(x)
    upper = [(float(chord * xi), float(chord * yi)) for xi, yi in zip(x[::-1], yt[::-1])]
    lower = [(float(chord * xi), float(-chord * yi)) for xi, yi in zip(x[1:], yt[1:])]
    return upper + lower


def build_rib(chord: float = 69.77, thickness_mm: float = 4.0):
    if cq is None:
        raise RuntimeError('CadQuery not installed. Use: pip install cadquery')
    pts = naca0012_polygon(chord)
    rib = cq.Workplane('XY').polyline(pts).close().extrude(thickness_mm)
    
    hole_x = chord * 0.25
    hole_radius = 3.0 / 2.0
    rib = rib.cut(cq.Workplane('XY').center(hole_x, 0).circle(hole_radius).extrude(thickness_mm * 3, both=True))
    
    return rib


def preview(chord: float = 69.77, save_path: str = '', show: bool = True):
    pts = np.array(naca0012_polygon(chord))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(pts[:,0], pts[:,1], color='black', lw=1.5)
    
    hole_x = chord * 0.25
    hole_radius = 3.0 / 2.0
    circle = plt.Circle((hole_x, 0), hole_radius, color='red', fill=False, label='3mm Hole')
    ax.add_patch(circle)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('NACA 0012 rib profile')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chord', type=float, default=69.77)
    ap.add_argument('--extrude', type=float, default=4.0)
    ap.add_argument('--step', default='naca0012_rib.step')
    ap.add_argument('--preview', default='naca0012_rib.png')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    preview(args.chord, save_path=args.preview, show=not args.no_show)
    if cq is not None:
        rib = build_rib(args.chord, args.extrude)
        rib.export(args.step)
        print('Wrote STEP:', args.step)
    else:
        print('CadQuery unavailable; preview only.')


if __name__ == '__main__':
    main()
