#!/usr/bin/env python3
import math
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


import cadquery as cq


# =============================================================================
# DESIGN PARAMETERS  — edit here to change the design
# =============================================================================

# -- Airfoil / chord ----------------------------------------------------------
CHORD          = 69.77   # mm  — chord length of the NACA 0012 profile
AIRFOIL_PTS    = 100     # number of control points for the spline (>=50 recommended)

# -- Assembly dimensions -------------------------------------------------------
TOTAL_LENGTH   = 105.0   # mm  — spanwise length of the fin assembly
RIB_COUNT      = 5       # number of ribs along the span
RIB_THICKNESS  = 4.0     # mm  — thickness (spanwise depth) of each rib

# -- Spar hole (at 25 % chord) ------------------------------------------------
HOLE_FRAC      = 0.25    # fraction of chord → hole centre x position
HOLE_DIA       = 3.0     # mm  — diameter of the spar/hole cutout

# -- Leading-edge & trailing-edge solid-link extents (as % of chord) ----------
LE_FRAC        = 0.10    # fraction of chord → LE link ends here (0–10 %)
TE_FRAC        = 0.90    # fraction of chord → TE link starts here (90–100 %)

# -- Stringers ----------------------------------------------------------------
STRINGER_DIA   = 5.0     # mm  — diameter of each cylindrical stringer
# Chordwise centre positions of the three stringer pairs (fraction of chord)
STRINGER_BASE_FRACS = [0.35, 0.55, 0.75]
# Offset applied to each base position: top stringers shift +delta, bottom −delta
STRINGER_DELTA_FRAC = 0.025   # ±2.5 % of chord

# =============================================================================
# DERIVED QUANTITIES  (calculated from the parameters above — do not edit)
# =============================================================================

HOLE_X            = HOLE_FRAC * CHORD
LE_CUT            = LE_FRAC   * CHORD
TE_CUT            = TE_FRAC   * CHORD
TOP_STRINGER_XS   = [(f + STRINGER_DELTA_FRAC) * CHORD for f in STRINGER_BASE_FRACS]
BOT_STRINGER_XS   = [(f - STRINGER_DELTA_FRAC) * CHORD for f in STRINGER_BASE_FRACS]

# =============================================================================


def naca0012_half_thickness(x: np.ndarray) -> np.ndarray:
    """NACA 0012 half-thickness distribution (t/c = 0.12) for normalised x ∈ [0,1]."""
    t = 0.12
    return 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 0.0, 1.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )


def naca0012_polygon(chord: float, n: int = AIRFOIL_PTS) -> List[Tuple[float, float]]:
    """Return upper+lower surface points scaled to *chord* (mm), suitable for a spline."""
    beta  = np.linspace(0.0, math.pi, n)
    x     = 0.5 * (1.0 - np.cos(beta))          # cosine-clustered x ∈ [0,1]
    yt    = naca0012_half_thickness(x)
    upper = [(float(chord * xi), float( chord * yi)) for xi, yi in zip(x[::-1], yt[::-1])]
    lower = [(float(chord * xi), float(-chord * yi)) for xi, yi in zip(x[1:],   yt[1:])]
    return upper + lower


def make_rib(chord: float, rib_thickness: float,
             hole_x: float, hole_dia: float,
             le_cut: float, te_cut: float):
    """Build a single rib: airfoil spline extruded, spar hole cut, LE/TE truncated."""
    pts = naca0012_polygon(chord)
    rib = cq.Workplane('XZ').spline(pts).close().extrude(rib_thickness)

    # Spar hole
    hole = (cq.Workplane('XZ')
              .center(hole_x, 0.0)
              .circle(hole_dia / 2.0)
              .extrude(rib_thickness * 2))
    rib = rib.cut(hole)

    # Remove material inboard of LE cut (truncate leading-edge sliver)
    le_box = (cq.Workplane('XZ')
                .center(le_cut - chord, 0)
                .rect(2 * chord, 2 * chord)
                .extrude(rib_thickness * 2, both=True))
    rib = rib.cut(le_box)

    # Remove material outboard of TE cut (truncate trailing-edge sliver)
    te_box = (cq.Workplane('XZ')
                .center(te_cut + chord, 0)
                .rect(2 * chord, 2 * chord)
                .extrude(rib_thickness * 2, both=True))
    rib = rib.cut(te_box)

    return rib


def make_stringer(x: float, z: float, dia: float,
                  length: float, rib_thickness: float, full_airfoil):
    """Cylindrical stringer clipped to the airfoil envelope via boolean intersect."""
    stringer = (cq.Workplane('XZ')
                  .center(x, z)
                  .circle(dia / 2.0)
                  .extrude(-length)
                  .translate((0, -rib_thickness, 0)))
    return stringer.intersect(full_airfoil)


def build_assembly(chord=CHORD, total_length=TOTAL_LENGTH,
                   rib_count=RIB_COUNT, rib_thickness=RIB_THICKNESS):
    """Assemble ribs + LE/TE links + stringers into one CadQuery solid."""
    if cq is None:
        raise RuntimeError('CadQuery not installed. Use: pip install cadquery')

    # Recalculate derived quantities if caller overrides chord
    hole_x          = HOLE_FRAC          * chord
    le_cut          = LE_FRAC            * chord
    te_cut          = TE_FRAC            * chord
    top_stringer_xs = [(f + STRINGER_DELTA_FRAC) * chord for f in STRINGER_BASE_FRACS]
    bot_stringer_xs = [(f - STRINGER_DELTA_FRAC) * chord for f in STRINGER_BASE_FRACS]

    # Rib pitch (evenly spaced along span)
    pitch = (total_length - rib_thickness) / (rib_count - 1) if rib_count > 1 else 0.0

    # --- Ribs ----------------------------------------------------------------
    model = None
    for i in range(rib_count):
        y   = i * pitch
        rib = make_rib(chord, rib_thickness, hole_x, HOLE_DIA, le_cut, te_cut)
        rib = rib.translate((0, y, 0))
        model = rib if model is None else model.union(rib)

    # Full airfoil solid spanning the assembly — used to clip links & stringers
    pts         = naca0012_polygon(chord)
    full_airfoil = (cq.Workplane('XZ')
                      .spline(pts).close()
                      .extrude(-total_length)
                      .translate((0, -rib_thickness, 0)))

    links = []

    # LE solid link (forward of LE_FRAC)
    le_box = (cq.Workplane('XZ')
                .center(le_cut + chord, 0)
                .rect(2 * chord, 2 * chord)
                .extrude(total_length * 2, both=True))
    links.append(full_airfoil.cut(le_box))

    # TE solid link (aft of TE_FRAC)
    te_box = (cq.Workplane('XZ')
                .center(te_cut - chord, 0)
                .rect(2 * chord, 2 * chord)
                .extrude(total_length * 2, both=True))
    links.append(full_airfoil.cut(te_box))

    # Top stringers
    for x in top_stringer_xs:
        yt = float(naca0012_half_thickness(np.array([x / chord]))[0] * chord)
        links.append(make_stringer(x, yt, STRINGER_DIA, total_length, rib_thickness, full_airfoil))

    # Bottom stringers
    for x in bot_stringer_xs:
        yt = float(naca0012_half_thickness(np.array([x / chord]))[0] * chord)
        links.append(make_stringer(x, -yt, STRINGER_DIA, total_length, rib_thickness, full_airfoil))

    for link in links:
        model = model.union(link)

    return model


def preview(chord=CHORD, save_path='', show=True):
    """2-D matplotlib cross-section preview of the fin."""
    pts         = np.array(naca0012_polygon(chord))
    hole_x      = HOLE_FRAC * chord
    le_cut      = LE_FRAC   * chord
    te_cut      = TE_FRAC   * chord
    top_stringer_xs = [(f + STRINGER_DELTA_FRAC) * chord for f in STRINGER_BASE_FRACS]
    bot_stringer_xs = [(f - STRINGER_DELTA_FRAC) * chord for f in STRINGER_BASE_FRACS]
    th = np.linspace(0, 2 * math.pi, 160)

    fig, ax = plt.subplots(figsize=(11, 3.5))

    # Rib body
    ax.fill(pts[:, 0], pts[:, 1], color='#d9d9d9', edgecolor='black', linewidth=1.2)

    # LE solid link region
    le_pts = pts[pts[:, 0] <= le_cut]
    if len(le_pts) > 0:
        ax.fill(le_pts[:, 0], le_pts[:, 1], color='violet', alpha=0.6, edgecolor='black')

    # TE solid link region
    te_pts = pts[pts[:, 0] >= te_cut]
    if len(te_pts) > 0:
        ax.fill(te_pts[:, 0], te_pts[:, 1], color='violet', alpha=0.6, edgecolor='black')

    # Spar hole
    ax.fill(hole_x + (HOLE_DIA / 2) * np.cos(th),
                      (HOLE_DIA / 2) * np.sin(th),
            color='white', edgecolor='black')

    import matplotlib.patches as patches
    airfoil_poly = patches.Polygon(pts, closed=True, fill=False, edgecolor='none')
    ax.add_patch(airfoil_poly)

    # Top stringers
    for x in top_stringer_xs:
        yt = float(naca0012_half_thickness(np.array([x / chord]))[0] * chord)
        c1 = ax.fill(x  + (STRINGER_DIA / 2) * np.cos(th),
                     yt + (STRINGER_DIA / 2) * np.sin(th),
                     color='violet', alpha=0.6, edgecolor='black')[0]
        c1.set_clip_path(airfoil_poly)

    # Bottom stringers
    for x in bot_stringer_xs:
        yt = float(naca0012_half_thickness(np.array([x / chord]))[0] * chord)
        c2 = ax.fill(x   + (STRINGER_DIA / 2) * np.cos(th),
                     -yt + (STRINGER_DIA / 2) * np.sin(th),
                     color='violet', alpha=0.6, edgecolor='black')[0]
        c2.set_clip_path(airfoil_poly)

    ax.plot(pts[:, 0], pts[:, 1], color='black', lw=1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Cross-section: links only, no LE/TE solid blocks')
    ax.set_xlabel('Chord x [mm]')
    ax.set_ylabel('Thickness y [mm]')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chord',         type=float, default=CHORD)
    ap.add_argument('--total-length',  type=float, default=TOTAL_LENGTH)
    ap.add_argument('--rib-count',     type=int,   default=RIB_COUNT)
    ap.add_argument('--rib-thickness', type=float, default=RIB_THICKNESS)
    ap.add_argument('--step',    default='fin_assembly_links.step')
    ap.add_argument('--preview', default='fin_links_cross_section.png')
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    preview(args.chord, save_path=args.preview, show=not args.no_show)


    model = build_assembly(args.chord, args.total_length,
                            args.rib_count, args.rib_thickness)
    model.export(args.step)
    print('Wrote STEP:', args.step)

if __name__ == '__main__':
    main()
