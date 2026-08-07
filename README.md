# NACA 0012 Fin Assembly — Design Progression

This folder contains the iterative CAD development of a NACA 0012 fin/wing assembly built with **Python + CadQuery**. The design was developed in four stages, each script building on the previous one.

---

## Design Stages

### Stage 1 — Single Rib · `single_naca0012_rib.py`

**Goal:** Generate the basic NACA 0012 airfoil cross-section as a 3-D rib solid.

| Parameter | Value |
|---|---|
| Chord | 69.77 mm |
| Rib thickness (extrusion) | 4.0 mm |
| Spar hole diameter | 3.0 mm at 25 % chord |
| Profile points | 250 (polyline) |

**What it does:**
- Computes the NACA 0012 half-thickness distribution using the standard 4-digit formula.
- Generates upper + lower surface points with cosine clustering (`naca0012_polygon`).
- Extrudes the closed polyline profile to create a solid rib.
- Cuts a circular spar hole at 25 % chord.

**Outputs:** `naca0012_rib.step`, `naca0012_rib.png`

---

### Stage 2 — Rib with Explicit Hole · `single_naca0012_rib_hole.py`

**Goal:** Refine the single-rib script with a cleaner, explicit hole cutout and improved 2-D preview.

**Changes from Stage 1:**
- Hole diameter promoted to a named function argument (`hole_diameter`), making it easy to change.
- Preview switched from a wireframe outline to a **filled cross-section** (grey fill + white hole) — much closer to how the actual part looks.
- Profile extruded on the `XY` plane (corrected orientation).

**Outputs:** `naca0012_rib_hole.step`, `naca0012_rib_hole.png`

---

### Stage 3 — Rib Array (Multi-rib Skeleton) · `naca0012_rib_array.py`

**Goal:** Extend a single rib into a full spanwise skeleton of evenly-spaced ribs.

| Parameter | Value |
|---|---|
| Span (total length) | 105.0 mm |
| Number of ribs | 5 |
| Rib pitch (auto-calculated) | `(total_length - rib_thickness) / (rib_count - 1)` |

**What it does:**
- Introduces `build_array()` — loops over `rib_count`, translates each rib along the span (Y-axis), and boolean-unions all ribs into one solid.
- Adds a **3-D matplotlib preview** showing all ribs and spar holes in perspective.
- Reports rib pitch and absolute Y-positions.

**Outputs:** `naca0012_rib_array.step`, `naca0012_rib_array.png`

---

### Stage 4 — Full Assembly with Structural Links · `fin_assembly_links.py`

**Goal:** Complete the fin by adding **leading-edge / trailing-edge solid links** and **cylindrical stringers** that connect all ribs into a single rigid structure.

| Parameter | Value |
|---|---|
| LE solid link | 0 – 10 % chord (`LE_FRAC = 0.10`) |
| TE solid link | 90 – 100 % chord (`TE_FRAC = 0.90`) |
| Stringer positions | 35 %, 55 %, 75 % chord (3 pairs) |
| Stringer offset (delta) | +/-2.5 % chord (`STRINGER_DELTA_FRAC = 0.025`) |
| Stringer diameter | 5.0 mm |
| Airfoil profile | **B-spline** (replaces polyline — true smooth surface) |

**Key improvements over Stage 3:**
- `polyline` replaced with `spline` — STEP file contains smooth NURBS geometry instead of faceted line segments (matches Fusion 360 sketch quality).
- Profile point count reduced from 250 to 100 — sufficient for a spline, and produces a smaller, cleaner file.
- LE and TE solid blocks extruded from a full-span airfoil envelope and boolean-cut to width.
- Three pairs of cylindrical stringers placed at top/bottom of the airfoil at 35 %, 55 %, 75 % chord, clipped to the airfoil envelope via boolean intersect.
- **All design parameters centralised** at the top of the file — fractions, dimensions, counts, and delta all defined in one place so any developer can modify the design without hunting through functions.

**Outputs:** `fin_assembly_links.step`, `fin_links_cross_section.png`

---

## File Map

```
Fin/
├── single_naca0012_rib.py          # Stage 1 — basic rib solid
├── single_naca0012_rib_hole.py     # Stage 2 — explicit spar hole + better preview
├── naca0012_rib_array.py           # Stage 3 — multi-rib skeleton with pitch calc
├── fin_assembly_links.py           # Stage 4 — full assembly (links + stringers + spline)
│
├── naca0012_rib.step               # STEP output — Stage 1
├── naca0012_rib_hole.step          # STEP output — Stage 2
├── naca0012_rib_array.step         # STEP output — Stage 3
├── fin_assembly_links.step         # STEP output — Stage 4
│
├── naca0012_rib.png                # 2-D preview — Stage 1
├── naca0012_rib_hole.png           # 2-D preview — Stage 2
├── naca0012_rib_array.png          # 3-D preview — Stage 3
├── fin_links_cross_section.png     # 2-D cross-section — Stage 4
├── fin_cross_section.png           # earlier cross-section reference
│
├── wing_skeleton_v3.step           # earlier skeleton reference model
├── v3.pdf                          # design document / reference
└── papers/                         # literature / reference papers
```

---

## Dependencies

```bash
pip install cadquery numpy matplotlib
```

> CadQuery is optional for preview-only runs — each script falls back gracefully and prints the 2-D matplotlib preview if CadQuery is not available.

---

## Quick Start

```bash
# Preview cross-section only (no CadQuery needed)
python fin_assembly_links.py --no-show

# Build full STEP and save preview image
python fin_assembly_links.py --step my_fin.step --preview my_preview.png --no-show

# Override chord and rib count
python fin_assembly_links.py --chord 80 --rib-count 7 --total-length 120
```
