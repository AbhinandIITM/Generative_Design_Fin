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


def single_rib(chord: float, extrude_mm: float, hole_diameter: float):
    rib = cq.Workplane('XZ').polyline(naca0012_polygon(chord)).close().extrude(extrude_mm)
    hole_x = 0.25 * chord
    hole = cq.Workplane('XZ').center(hole_x, 0.0).circle(hole_diameter / 2.0).extrude(extrude_mm * 2)
    return rib.cut(hole)


def build_array(chord: float = 69.77, extrude_mm: float = 4.0, hole_diameter: float = 3.0,
                rib_count: int = 5, total_length: float = 105.0):
    if cq is None:
        raise RuntimeError('CadQuery not installed. Use: pip install cadquery')

    if rib_count > 1:
        pitch = (total_length - extrude_mm) / (rib_count - 1)
    else:
        pitch = 0.0
        
    model = None
    rib_positions = []
    for i in range(rib_count):
        y = i * pitch
        rib = single_rib(chord, extrude_mm, hole_diameter).translate((0, y, 0))
        rib_positions.append(y)
        model = rib if model is None else model.union(rib)
    return model, pitch, rib_positions


def preview(chord: float = 69.77, hole_diameter: float = 3.0,
            rib_count: int = 5, total_length: float = 105.0, extrude_mm: float = 4.0, save_path: str = '', show: bool = True):
    pts = np.array(naca0012_polygon(chord))
    hole_x = 0.25 * chord
    theta = np.linspace(0, 2*math.pi, 120)
    cx = hole_x + (hole_diameter/2.0) * np.cos(theta)
    cy = (hole_diameter/2.0) * np.sin(theta)
    
    if rib_count > 1:
        pitch = (total_length - extrude_mm) / (rib_count - 1)
    else:
        pitch = 0.0

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(rib_count):
        y = i * pitch
        ax.plot(pts[:,0], np.full(len(pts), y), pts[:,1], color='black', lw=1.1)
        ax.plot(cx, np.full(len(cx), y), cy, color='tab:blue', lw=1.0)

    ax.set_xlabel('Chord x [mm]')
    ax.set_ylabel('Span y [mm]')
    ax.set_zlabel('Thickness z [mm]')
    ax.set_title(f'{rib_count} NACA 0012 ribs, pitch = {pitch:.2f} mm')
    ax.set_box_aspect((chord, total_length, chord*0.2))
    ax.view_init(elev=18, azim=-60)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chord', type=float, default=69.77)
    ap.add_argument('--extrude', type=float, default=4.0)
    ap.add_argument('--hole-diameter', type=float, default=3.0)
    ap.add_argument('--rib-count', type=int, default=5)
    ap.add_argument('--total-length', type=float, default=105.0)
    ap.add_argument('--step', default='naca0012_rib_array.step')
    ap.add_argument('--preview', default='naca0012_rib_array.png')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    preview(args.chord, args.hole_diameter, args.rib_count, args.total_length, args.extrude,
            save_path=args.preview, show=not args.no_show)

    if cq is not None:
        model, pitch, rib_positions = build_array(args.chord, args.extrude, args.hole_diameter,
                                                  args.rib_count, args.total_length)
        model.export(args.step)
        print('Wrote STEP:', args.step)
        print('Pitch:', pitch)
        print('Rib positions:', rib_positions)
    else:
        print('CadQuery unavailable; preview only.')


if __name__ == '__main__':
    main()
