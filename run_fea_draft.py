#!/usr/bin/env python3
"""
run_fea.py  --  Automated FEA pipeline: STEP → Gmsh → CalculiX → Results

Boundary-condition faces are identified by **bounding-box geometry queries**
(not by STEP face names), so the pipeline is fully robust to topological
changes from the parametric / generative design loop.

Units (mm-tonne-s system):
    Length  = mm          Force   = N
    Stress  = MPa (N/mm²) Density = tonne/mm³

Dependencies:
    pip install gmsh numpy

    CalculiX (ccx) must be on PATH:
      Windows  →  https://www.bconverged.com/calculix.php  (prebuilt .exe)
      Linux    →  sudo apt install calculix-ccx
      Conda    →  conda install -c conda-forge calculix

CLI usage:
    python run_fea.py --step fin_assembly_links.step
    python run_fea.py --step fin_assembly_links.step --mesh-size 2.0 --material aluminum
    python run_fea.py --step fin_assembly_links.step --tip-force 10 --frequency

Python / RL-loop usage:
    from run_fea import run_fea
    r = run_fea("fin_assembly_links.step", material="pla", tip_force=5.0)
    print(r["compliance"], r["max_tip_displacement"])

Visualising results:
    The .frd file produced by CalculiX can be viewed in:
      • CalculiX GraphiX (cgx):  cgx -v fea_results/fin_assembly_links.frd
      • FreeCAD:  File → Open  the .frd file in the FEM workbench
      • ParaView: convert first with  ccx2paraview fin_assembly_links.frd vtk
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

try:
    import gmsh
except ImportError:
    sys.exit("ERROR: gmsh not installed.  Run:  pip install gmsh")


# ===============================================================================
# MATERIAL LIBRARY  (mm-tonne-s unit system)
#   E   : Young's modulus  [MPa]
#   nu  : Poisson's ratio  [-]
#   rho : density          [tonne/mm³]   (multiply by 1e9 to get kg/m³)
# ===============================================================================
MATERIALS = {
    "pla":      {"E":   3500.0, "nu": 0.36, "rho": 1.24e-9, "Sy": 50.0},
    "abs":      {"E":   2300.0, "nu": 0.35, "rho": 1.04e-9, "Sy": 40.0},
    "nylon":    {"E":   2400.0, "nu": 0.39, "rho": 1.14e-9, "Sy": 55.0},
    "petg":     {"E":   2020.0, "nu": 0.40, "rho": 1.27e-9, "Sy": 50.0},
    "aluminum": {"E":  70000.0, "nu": 0.33, "rho": 2.70e-9, "Sy": 276.0},
    "steel":    {"E": 210000.0, "nu": 0.30, "rho": 7.85e-9, "Sy": 250.0},
    "titanium": {"E": 116000.0, "nu": 0.34, "rho": 4.43e-9, "Sy": 880.0},
    "cfrp":     {"E": 135000.0, "nu": 0.30, "rho": 1.60e-9, "Sy": 600.0},
}


# ===============================================================================
#  UTILITIES
# ===============================================================================

def _find_ccx() -> str:
    """Locate the CalculiX executable on PATH or in common conda locations."""
    # 1. Check PATH
    for name in ["ccx", "ccx_MT", "ccx.exe", "ccx_MT.exe"]:
        path = shutil.which(name)
        if path:
            return path
    # 2. Check common conda/mamba install locations (Windows)
    for base in [os.path.expanduser("~/miniconda3"),
                 os.path.expanduser("~/anaconda3"),
                 os.path.expanduser("~/mambaforge")]:
        candidate = os.path.join(base, "Library", "bin", "ccx.exe")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _write_nset(f, name: str, node_ids: list):
    """Write a *NSET block in Abaqus/CalculiX format (16 IDs per line)."""
    f.write(f"*NSET, NSET={name}\n")
    per_line = 16
    for i in range(0, len(node_ids), per_line):
        chunk = node_ids[i : i + per_line]
        f.write(", ".join(str(n) for n in chunk) + "\n")


# ===============================================================================
#  STAGE 1 -- MESHING  (Gmsh)
# ===============================================================================

def mesh_step_file(step_path: str, mesh_size: float, element_order: int,
                   job: str, work_dir: str) -> dict:
    """Import a STEP file, identify root/tip faces by bounding box,
    mesh with tetrahedra, and export an Abaqus .inp mesh file.

    Faces are tagged using **spatial / bounding-box queries**, not STEP
    face names, so results are repeatable even when the topology changes.

    Parameters
    ----------
    step_path     : absolute path to the .step file
    mesh_size     : target element edge length [mm]
    element_order : 1 → C3D4 (4-node linear tet),  2 → C3D10 (10-node quadratic tet)
    job           : job name (used for output filenames)
    work_dir      : directory for output files

    Returns
    -------
    dict with keys:
        mesh_inp, root_nodes, tip_nodes, n_nodes, n_elems,
        bb_min, bb_max, elset_name, volume_mm3
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    # -- Import & heal STEP geometry -----------------------------------------
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.healShapes()
    gmsh.model.occ.synchronize()

    # -- Overall bounding box ------------------------------------------------
    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
        gmsh.finalize()
        raise RuntimeError("No 3-D volumes found in the STEP file.")

    bb_min = np.array([ np.inf,  np.inf,  np.inf])
    bb_max = np.array([-np.inf, -np.inf, -np.inf])
    for d, t in volumes:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(d, t)
        bb_min = np.minimum(bb_min, [xmin, ymin, zmin])
        bb_max = np.maximum(bb_max, [xmax, ymax, zmax])

    span = bb_max - bb_min
    tol  = max(0.1, np.min(span[span > 0]) * 1e-3)

    print(f"[mesh] Model bounding box:")
    print(f"       min = [{bb_min[0]:.3f}, {bb_min[1]:.3f}, {bb_min[2]:.3f}]")
    print(f"       max = [{bb_max[0]:.3f}, {bb_max[1]:.3f}, {bb_max[2]:.3f}]")
    print(f"       tol = {tol:.4f} mm")

    # -- CAD volume (for mass estimates later) -------------------------------
    total_volume = 0.0
    for _, t in volumes:
        total_volume += gmsh.model.occ.getMass(3, t)
    print(f"[mesh] CAD volume = {total_volume:.2f} mm^3")

    # -- Geometric face filtering --------------------------------------------
    #    The fin assembly's span axis is Y.
    #    Root = face(s) at Y_min     Tip = face(s) at Y_max
    surfaces = gmsh.model.getEntities(dim=2)
    root_surf_tags = []
    tip_surf_tags  = []
    exterior_surf_tags = []

    for d, t in surfaces:
        smin = gmsh.model.getBoundingBox(d, t)[:3]
        smax = gmsh.model.getBoundingBox(d, t)[3:]

        # Fixed support on the root rib (bottom flat surface at Y_min)
        if abs(smin[1] - bb_min[1]) < tol and abs(smax[1] - bb_min[1]) < tol:
            root_surf_tags.append(t)
        else:
            exterior_surf_tags.append(t)

        # Top flat surface: both Y extents ≈ model Y_max
        if abs(smin[1] - bb_max[1]) < tol and abs(smax[1] - bb_max[1]) < tol:
            tip_surf_tags.append(t)

    print(f"[mesh] Root faces identified: {len(root_surf_tags)}")
    print(f"[mesh] Tip  faces identified: {len(tip_surf_tags)}")

    if not root_surf_tags:
        print("[WARN] No root faces found -- fixed BC will not be applied!")
    if not tip_surf_tags:
        print("[WARN] No tip faces found -- tip load will not be applied!")

    # -- Physical group: VOLUME ONLY -----------------------------------------
    vol_tags   = [t for _, t in volumes]
    elset_name = "EBODY"
    gmsh.model.addPhysicalGroup(3, vol_tags, tag=1, name=elset_name)

    if exterior_surf_tags:
        gmsh.model.addPhysicalGroup(2, exterior_surf_tags, tag=2, name="PLOAD_SURF")

    # -- Mesh settings -------------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.15)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)       # Delaunay
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.HighOrderOptimize", 1) # Untangle high-order elements
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)

    print(f"[mesh] Generating 3-D mesh  (target size = {mesh_size} mm, "
          f"order = {element_order}) ...")
    gmsh.model.mesh.generate(3)

    # -- Collect node tags on root / tip surfaces ----------------------------
    #    getNodes(dim, tag) returns nodes on a specific geometric entity.
    #    includeBoundary=True ensures we also get nodes on shared edges/vertices.
    root_nodes = set()
    for tag in root_surf_tags:
        ntags, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag,
                                             includeBoundary=True)
        root_nodes.update(int(t) for t in ntags)

    tip_nodes = set()
    for tag in tip_surf_tags:
        ntags, _, _ = gmsh.model.mesh.getNodes(dim=2, tag=tag,
                                             includeBoundary=True)
        tip_nodes.update(int(t) for t in ntags)

    # -- Mesh statistics -----------------------------------------------------
    all_nodeTags, _, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(all_nodeTags)
    _, all_elemTags, _ = gmsh.model.mesh.getElements(dim=3)
    n_elems = sum(len(et) for et in all_elemTags)

    print(f"[mesh] Nodes:  {n_nodes:,}")
    print(f"[mesh] Elems:  {n_elems:,}")

    # -- Write mesh to .inp --------------------------------------------------
    mesh_inp = os.path.join(work_dir, f"{job}_mesh.inp")
    gmsh.write(mesh_inp)
    
    # -- Fix element types for calculix shell compatibility
    with open(mesh_inp, "r", encoding="utf-8") as f:
        mesh_text = f.read()
    mesh_text = mesh_text.replace("type=CPS3", "type=S3").replace("type=CPS6", "type=S6")
    with open(mesh_inp, "w", encoding="utf-8") as f:
        f.write(mesh_text)

    # Clean up Gmsh
    gmsh.finalize()

    print(f"[mesh] Nodes: {len(all_nodeTags):6,d}")
    print(f"[mesh] Elems: {len(all_elemTags):6,d}")
    print(f"[mesh] Root node-set: {len(root_nodes):4,d} nodes")
    print(f"[mesh] Tip  node-set: {len(tip_nodes):4,d} nodes")
    print(f"[mesh] Wrote  {mesh_inp}")

    return {
                "mesh_inp": mesh_inp,
        "n_nodes": n_nodes,
        "n_elems": n_elems,
        "elset_name": elset_name,
        "root_nodes": sorted(list(root_nodes)),
        "tip_nodes": sorted(list(tip_nodes)),
        "volume_mm3": total_volume,
        "bb_min":     bb_min.tolist(),
        "bb_max":     bb_max.tolist(),
    }


# ===============================================================================
#  STAGE 2 -- WRITE CALCULIX INPUT DECK
# ===============================================================================

def write_calculix_deck(mesh_info: dict, job: str, work_dir: str,
                        material: str, pressure_mpa: float,
                        run_frequency: bool) -> str:
    """Write the master CalculiX .inp that *INCLUDEs the Gmsh mesh and adds
    node sets, material, BCs, loads, and solver steps.

    Returns the path to the .inp file.
    """
    mat        = MATERIALS[material]
    elset      = mesh_info["elset_name"]
    n_tip      = len(mesh_info["tip_nodes"])
    n_root     = len(mesh_info["root_nodes"])

    ccx_path = os.path.join(work_dir, f"{job}.inp")
    mesh_rel = os.path.basename(mesh_info["mesh_inp"])

    with open(ccx_path, "w", encoding="utf-8") as f:
        # -- Header ----------------------------------------------------------
        f.write(f"** ----------------------------------------------------------\n")
        f.write(f"** CalculiX input deck -- auto-generated by run_fea.py\n")
        f.write(f"** Material       : {material}  (E = {mat['E']:.0f} MPa)\n")
        f.write(f"** Pressure load  : {pressure_mpa:.4f} MPa\n")
        f.write(f"** Root nodes     : {n_root}\n")
        f.write(f"** ----------------------------------------------------------\n")
        f.write(f"**\n")

        # -- Include mesh (nodes + volume elements + ELSET) ------------------
        f.write(f"*INCLUDE, INPUT={mesh_rel}\n")
        f.write(f"**\n")

        # -- Node sets from geometric face filtering -------------------------
        if n_root > 0:
            _write_nset(f, "NFIXED_ROOT", mesh_info["root_nodes"])
        else:
            f.write("** WARNING: no root nodes -- fixed BC skipped\n")

        if n_tip > 0:
            _write_nset(f, "NLOAD_TIP", mesh_info["tip_nodes"])
        else:
            f.write("** WARNING: no tip nodes -- load skipped\n")

        f.write("**\n")

        # -- Material definition ---------------------------------------------
        f.write(f"*MATERIAL, NAME={material.upper()}\n")
        f.write(f"*ELASTIC\n")
        f.write(f"{mat['E']:.1f}, {mat['nu']:.4f}\n")
        f.write(f"*DENSITY\n")
        f.write(f"{mat['rho']:.6e}\n")
        f.write(f"**\n")

        # -- Solid section assignment ----------------------------------------
        f.write(f"*SOLID SECTION, ELSET={elset}, MATERIAL={material.upper()}\n")
        f.write(f"**\n")
        f.write(f"*MATERIAL, NAME=DUMMY_SHELL\n")
        f.write(f"*ELASTIC\n")
        f.write(f"1.0, 0.3\n")
        f.write(f"*SHELL SECTION, ELSET=PLOAD_SURF, MATERIAL=DUMMY_SHELL\n")
        f.write(f"1e-5\n")
        f.write(f"**\n")

        # -- STEP 1: Static analysis (bending under pressure load) ----------
        f.write("** " + "=" * 60 + "\n")
        f.write("** STEP 1 -- Static bending under pressure load\n")
        f.write("** " + "=" * 60 + "\n")
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("**\n")

        # Fixed root  (constrain DOFs 1-3: all translations)
        if n_root > 0:
            f.write("** Fixed support at root rib\n")
            f.write("*BOUNDARY\n")
            f.write("NFIXED_ROOT, 1, 3, 0.0\n")
            f.write("**\n")

        # Distributed pressure load
        f.write(f"** Hydrostatic pressure: {pressure_mpa:.4f} MPa\n")
        f.write("*DLOAD\n")
        f.write(f"PLOAD_SURF, P, {pressure_mpa:.6e}\n")
        f.write("**\n")

        # 450N compressive load on the top surface (spread across top nodes in -Y direction)
        if n_tip > 0:
            f.write("** 450N compressive load (axial -Y direction) spread on top surface\n")
            f.write("*CLOAD\n")
            force_per_node = 450.0 / n_tip
            for n in mesh_info["tip_nodes"]:
                f.write(f"{n}, 2, {-force_per_node}\n")
            f.write("**\n")

        # Output requests
        f.write("** -- Output requests ----------------------------------\n")
        # .frd file (for visualisation in cgx / ParaView / FreeCAD)
        f.write("*NODE FILE\n")
        f.write("U\n")
        f.write("*EL FILE\n")
        f.write("S, ENER\n")
        # .dat file (for programmatic parsing -- tip displacements + compliance)
        if n_tip > 0:
            f.write("*NODE PRINT, NSET=NLOAD_TIP, TOTALS=YES\n")
            f.write("U\n")
        f.write("*EL PRINT, ELSET=EBODY, TOTALS=YES\n")
        f.write("ENER\n")
        f.write("*END STEP\n")

        # -- STEP 2 (optional): Modal / frequency analysis ------------------
        if run_frequency:
            f.write("**\n")
            f.write("** " + "=" * 60 + "\n")
            f.write("** STEP 2 -- Modal analysis (first 10 natural frequencies)\n")
            f.write("** " + "=" * 60 + "\n")
            f.write("*STEP\n")
            f.write("*FREQUENCY\n")
            f.write("10\n")
            f.write("**\n")
            if n_root > 0:
                f.write("*BOUNDARY\n")
                f.write("NFIXED_ROOT, 1, 3, 0.0\n")
            f.write("*NODE FILE\n")
            f.write("U\n")
            f.write("*EL FILE\n")
            f.write("S\n")
            f.write("*END STEP\n")

    print(f"[ccx]  Wrote  {ccx_path}")
    return ccx_path


# ===============================================================================
#  STAGE 3 -- RUN CALCULIX
# ===============================================================================

def run_calculix(job: str, work_dir: str) -> int:
    """Execute  ``ccx -i <job>``  and return the process exit code."""
    ccx = _find_ccx()
    if not ccx:
        print("[ccx]  *** CalculiX (ccx) not found on PATH ***")
        print("       Install options:")
        print("         conda install -c conda-forge calculix")
        print("         https://www.bconverged.com/calculix.php  (Windows)")
        print("         sudo apt install calculix-ccx            (Linux)")
        return -1

    cmd = [ccx, "-i", job]
    
    # Enable multithreading for SPOOLES
    env = os.environ.copy()
    num_threads = str((os.cpu_count()-6) or 4)
    env["OMP_NUM_THREADS"] = num_threads

    print(f"[ccx]  Running:  {' '.join(cmd)}  (cwd: {work_dir}) with {num_threads} threads")
    result = subprocess.run(cmd, cwd=work_dir, env=env,
                            capture_output=True, text=True)

    # Print solver summary (last 40 lines of stdout)
    if result.stdout:
        lines = result.stdout.strip().splitlines()
        for ln in lines[-40:]:
            print(f"       {ln}")

    if result.returncode != 0:
        print(f"\n[ccx]  *** SOLVER FAILED  (exit code {result.returncode}) ***")
        if result.stderr:
            for ln in result.stderr.strip().splitlines()[-20:]:
                print(f"       {ln}")

    return result.returncode


# ===============================================================================
#  STAGE 4 -- PARSE RESULTS
# ===============================================================================

def parse_results(job: str, work_dir: str) -> dict:
    """Parse the CalculiX ``.dat`` file for key FEA outputs.

    Extracted quantities:
        total_strain_energy   -- half the compliance (linear statics)
        compliance            -- 2 × strain energy  =  uᵀ K u
        max_tip_displacement  -- largest ‖u‖ at any tip node  [mm]
        tip_displacements     -- list of [ux, uy, uz] arrays  [mm]
        max_von_mises_mpa     -- maximum Von Mises stress [MPa]
        eigenvalues_hz        -- natural frequencies if a *FREQUENCY step ran
    """
    dat_path = os.path.join(work_dir, f"{job}.dat")
    if not os.path.isfile(dat_path):
        print(f"[parse] {dat_path} not found.")
        return {"error": "no .dat file"}

    results = {
        "total_strain_energy":  None,
        "compliance":           None,
        "max_tip_displacement": None,
        "max_von_mises_mpa":    None,
        "tip_displacements":    [],
        "eigenvalues_hz":       [],
    }

    with open(dat_path, "r") as f:
        lines = f.readlines()

    i = 0
    current_step = 1
    while i < len(lines):
        lower = lines[i].lower().strip()
        
        if "s t e p " in lower or "s t e p\t" in lower or lower.startswith("step "):
            # try to parse the step number
            parts = lower.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                current_step = int(parts[-1])

        # -- Internal energy (density) ----------------------------------------
        #    CalculiX 2.23 outputs: "internal energy density (elem, integ.pnt.,energy)"
        #    We sum all values to get total strain energy. (Only for Step 1)
        if "internal energy" in lower and current_step == 1:
            energy_sum = 0.0
            count = 0
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if row == "":
                    j += 1
                    continue
                if row.lower().startswith("total"):
                    parts = row.split()
                    if len(parts) >= 2:
                        try:
                            energy_sum = float(parts[-1])
                        except ValueError:
                            pass
                    j += 1
                    break
                parts = row.split()
                if len(parts) >= 2:
                    try:
                        val = float(parts[-1])
                        energy_sum += val
                        count += 1
                    except ValueError:
                        break
                else:
                    break
                j += 1
            if count > 0:
                results["total_strain_energy"] = energy_sum
                print(f"[parse] Strain energy summed from {count} integration points: {energy_sum:.6e}")

        # -- Tip displacements -----------------------------------------------
        if "displacements" in lower and "load_tip" in lower and current_step == 1:
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if row == "":
                    j += 1
                    continue
                if not row[0].isdigit() and not row[0] == '-':
                    if row.lower().startswith("total"):
                        j += 1
                        continue
                    break
                parts = row.split()
                if len(parts) >= 4:
                    try:
                        u = np.array([float(parts[1]),
                                      float(parts[2]),
                                      float(parts[3])])
                        results["tip_displacements"].append(u)
                    except (ValueError, IndexError):
                        pass
                j += 1

        # -- Eigenvalues (from *FREQUENCY step) -----------------------------
        if "eigenvalue" in lower and ("output" in lower or "frequency" in lower):
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if row == "" or row.startswith("*") or row.startswith("REAL") or row.startswith("(RAD"):
                    j += 1
                    continue
                parts = row.split()
                # MODE NO, EIGENVALUE, RAD/TIME, CYCLES/TIME, IMAGINARY
                if parts and parts[0].isdigit() and len(parts) >= 4:
                    try:
                        freq_hz = float(parts[3]) # CYCLES/TIME is column 4 (index 3)
                        results["eigenvalues_hz"].append(freq_hz)
                    except ValueError:
                        pass
                    j += 1
                else:
                    break

        i += 1

    # -- Parse .frd for Max Von Mises Stress ---------------------------------
    frd_path = os.path.join(work_dir, f"{job}.frd")
    max_vm = 0.0
    if os.path.isfile(frd_path):
        with open(frd_path, "r") as f:
            in_stress = False
            for line in f:
                if line.startswith(" -4  STRESS"):
                    in_stress = True
                    continue
                if in_stress:
                    if line.startswith(" -3"):
                        break # End of step 1 stress block
                    if line.startswith(" -1"):
                        # Parse fixed width: node(13), sxx(12), syy(12), szz(12), sxy(12), syz(12), szx(12)
                        if len(line) >= 85:
                            try:
                                sxx = float(line[13:25])
                                syy = float(line[25:37])
                                szz = float(line[37:49])
                                sxy = float(line[49:61])
                                syz = float(line[61:73])
                                szx = float(line[73:85])
                                
                                vm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + syz**2 + szx**2)))
                                if vm > max_vm:
                                    max_vm = vm
                            except ValueError:
                                pass
        if max_vm > 0:
            results["max_von_mises_mpa"] = max_vm
            print(f"[parse] Max Von Mises stress: {max_vm:.4f} MPa")

    # -- Derived quantities --------------------------------------------------
    if results["total_strain_energy"] is not None:
        results["compliance"] = 2.0 * results["total_strain_energy"]

    if results["tip_displacements"]:
        mags = [float(np.linalg.norm(u)) for u in results["tip_displacements"]]
        results["max_tip_displacement"] = max(mags)
        print(f"[parse] Tip nodes parsed: {len(results['tip_displacements'])}")
        print(f"[parse] Max tip displacement: {results['max_tip_displacement']:.6e} mm")

    return results


# ===============================================================================
#  PUBLIC API  (call from Python / RL loops)
# ===============================================================================

def run_fea(step_path: str, *,
            mesh_size: float = 2.0,
            element_order: int = 2,
            material: str = "pla",
            pressure_mpa: float = 0.0981,
            frequency: bool = False,
            work_dir: str = "fea_results") -> dict:
    """End-to-end FEA pipeline:  STEP → mesh → solve → results dict.

    Parameters
    ----------
    step_path      : path to the .step CAD file
    mesh_size      : target element edge length  [mm]  (smaller = finer mesh)
    element_order  : 1 → C3D4 (linear tet)   2 → C3D10 (quadratic tet)
    material       : key from MATERIALS dict
    pressure_mpa   : Hydrostatic pressure on the external surfaces [MPa] (default 0.0981 for 10m depth)
    frequency      : if True, also run modal analysis (first 10 natural modes)
    work_dir       : directory for all generated / output files

    Returns
    -------
    dict with keys:
        compliance, total_strain_energy, max_tip_displacement, max_von_mises_mpa,
        tip_displacements, eigenvalues_hz,
        n_nodes, n_elems, bb_min, bb_max, volume_mm3, mass_g,
        root_node_count, tip_node_count
        (plus 'error' if the solver failed)
    """
    step_path = os.path.abspath(step_path)
    work_dir  = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    job = Path(step_path).stem

    header = f"  run_fea  |  {Path(step_path).name}"
    print(f"\n{'=' * 70}")
    print(header)
    print(f"  mesh = {mesh_size} mm   order = {element_order}   "
          f"mat = {material}   P = {pressure_mpa:.4f} MPa")
    print(f"{'=' * 70}\n")

    # 1. Mesh
    mesh_info = mesh_step_file(step_path, mesh_size, element_order,
                               job, work_dir)

    # 2. Write CalculiX input deck
    write_calculix_deck(mesh_info, job, work_dir,
                        material, pressure_mpa, frequency)

    # 3. Solve
    rc = run_calculix(job, work_dir)

    # 4. Parse results
    if rc == 0:
        results = parse_results(job, work_dir)
    else:
        results = {"error": f"ccx exited with code {rc}"}

    # Merge mesh metadata into results
    for key in ("n_nodes", "n_elems", "bb_min", "bb_max", "volume_mm3"):
        results[key] = mesh_info[key]
    results["root_node_count"] = len(mesh_info["root_nodes"])
    results["tip_node_count"]  = len(mesh_info["tip_nodes"])

    # Mass estimate from CAD volume and material density
    mat = MATERIALS.get(material)
    if mat and mesh_info["volume_mm3"]:
        # mass [tonne] = volume [mm³] × density [tonne/mm³]
        mass_tonne = mesh_info["volume_mm3"] * mat["rho"]
        results["mass_g"] = mass_tonne * 1e6   # convert tonne → grams
        
    if mat and results.get("max_von_mises_mpa"):
        if results["max_von_mises_mpa"] > 0:
            results["fos"] = mat["Sy"] / results["max_von_mises_mpa"]
        else:
            results["fos"] = float('inf')

    # -- Summary -------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  RESULTS")
    for k, v in sorted(results.items()):
        if k in ("tip_displacements",):
            continue  # skip large arrays
        print(f"    {k:30s} = {v}")
    print(f"{'-' * 70}\n")

    return results


# ===============================================================================
#  CLI ENTRY POINT
# ===============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Automated FEA:  STEP → Gmsh → CalculiX → results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\\
            Examples
            --------
              python run_fea.py --step fin_assembly_links.step
              python run_fea.py --step fin_assembly_links.step --mesh-size 3.0 --material aluminum
              python run_fea.py --step fin_assembly_links.step --pressure-mpa 0.0981 --frequency

            Available materials:  """ + ", ".join(sorted(MATERIALS.keys()))))

    p.add_argument("--step", required=True,
                   help="Path to the STEP file")
    p.add_argument("--mesh-size", type=float, default=4.0,
                   help="Target element edge length in mm  (default: 4.0)")
    p.add_argument("--element-order", type=int, default=2, choices=[1, 2],
                   help="1 = C3D4 (linear tet)   2 = C3D10 (quadratic tet)")
    p.add_argument("--material", default="pla",
                   choices=sorted(MATERIALS.keys()),
                   help="Material name  (default: pla)")
    p.add_argument("--pressure-mpa", type=float, default=0.0981,
                   help="Hydrostatic pressure on external surfaces in MPa (default: 0.0981 for 10m depth)")
    p.add_argument("--frequency", action="store_true",
                   help="Add a modal-analysis step (first 10 natural modes)")
    p.add_argument("--work-dir", default="fea_results",
                   help="Output directory  (default: fea_results/)")

    args = p.parse_args()

    results = run_fea(
        step_path=args.step,
        mesh_size=args.mesh_size,
        element_order=args.element_order,
        material=args.material,
        pressure_mpa=args.pressure_mpa,
        frequency=args.frequency,
        work_dir=args.work_dir,
    )

    return 0 if "error" not in results else 1


if __name__ == "__main__":
    sys.exit(main())

