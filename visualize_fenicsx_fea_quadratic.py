#!/usr/bin/env python3
"""
visualize_fenicsx_fea.py -- Parallel FEniCSx FEA with Detailed Results Export

This script runs the FEniCSx Finite Element Analysis (FEA) on the 
fin assembly and exports detailed results (mesh, boundary markers, 
displacements, and Von Mises stress fields) to XDMF format so they 
can be visually inspected in ParaView (GUI) similarly to Ansys.

Dependencies:
    conda install -c conda-forge fenics-dolfinx mpich
    pip install gmsh numpy
"""

import time
import os
import argparse
import numpy as np

try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh, fem, io
    from dolfinx.io import gmsh as gmshio
    import ufl
    from petsc4py import PETSc
    from dolfinx.fem.petsc import LinearProblem
except ImportError as e:
    print(f"Failed to import FEniCSx dependencies: {e}")
    print("Please ensure you are in a conda environment with fenics-dolfinx installed.")
    print("Example: conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich")
    pass

import gmsh

def create_and_read_mesh(comm, mesh_size=2.0, step_file="satvik_fin_fixed_rotated.step"):
    """
    Generate a 3D mesh from a STEP file using Gmsh, 
    and read it into a dolfinx mesh in parallel.
    """
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("fin_assembly")
        gmsh.model.occ.importShapes(step_file)
        try:
            gmsh.model.occ.healShapes(tolerance=1e-5)
        except Exception as e:
            print(f"Warning: healShapes failed (often happens if already healed or corrupt): {e}")
            
        gmsh.model.occ.synchronize()
        
        # Check if there are any 3D volumes
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            print("No 3D volumes found! Attempting to sew 2D surfaces into a solid volume...")
            surfaces = gmsh.model.getEntities(dim=2)
            if surfaces:
                surface_tags = [t for _, t in surfaces]
                try:
                    sl = gmsh.model.occ.addSurfaceLoop(surface_tags)
                    gmsh.model.occ.addVolume([sl])
                    gmsh.model.occ.synchronize()
                    volumes = gmsh.model.getEntities(dim=3)
                except Exception as e:
                    print(f"Warning: Failed to sew surfaces into a volume: {e}")

        # Physical groups
        vol_tags = [t for _, t in volumes]
        if vol_tags:
            gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
        else:
            raise ValueError("No 3D volumes could be found or generated from the STEP file. Cannot proceed with 3D FEA.")
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.15)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)       # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.generate(3)
        
    # Import gmsh model into dolfinx mesh
    domain, cell_markers, facet_markers, *_ = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3
    )
    gmsh.finalize()
    return domain

def run_fenicsx_fea(mesh_size=2.0):
    if 'MPI' not in globals():
        print("MPI module not found. Skipping FEniCSx execution.")
        return

    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        print(f"[{rank}] Starting FEniCSx FEA setup with {comm.size} MPI processes")
        
    # ---------------------------------------------------------
    # 1. Meshing
    # ---------------------------------------------------------
    t0_mesh = time.time()
    if 'gmshio' in globals():
        domain = create_and_read_mesh(comm, mesh_size)
    else:
        if rank == 0:
            print("Skipping mesh generation due to missing FEniCSx imports.")
        return
    t1_mesh = time.time()
    mesh_time = t1_mesh - t0_mesh
    
    if rank == 0:
        print(f"[{rank}] Mesh generated and distributed in {mesh_time:.3f} s")
        print(f"[{rank}] Number of cells: {domain.topology.index_map(3).size_global}")
        
    # ---------------------------------------------------------
    # 2. Function Spaces and Variational Problem (Linear Elasticity)
    # ---------------------------------------------------------
    # Using a continuous Galerkin function space of degree 2 (Quadratic Elements)
    V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    
    # Material properties (PLA-like)
    E = 3500.0  # MPa
    nu = 0.36
    lambda_ = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
        
    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
        
    # Weak form
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    
    # Boundary conditions and Loads
    x_coords = domain.geometry.x
    y_min = comm.allreduce(np.min(x_coords[:, 1]), op=MPI.MIN)
    y_max = comm.allreduce(np.max(x_coords[:, 1]), op=MPI.MAX)
    tol = 1e-1
    
    fdim = domain.topology.dim - 1
    
    # Identify facets
    root_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], y_min, atol=tol))
    tip_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], y_max, atol=tol))
    
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    exterior_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Create facet markers (1: root, 2: tip, 3: other exterior)
    facet_indices = np.copy(exterior_facets)
    facet_markers = np.full_like(facet_indices, 3, dtype=np.int32)
    
    sort_idx = np.argsort(facet_indices)
    facet_indices = facet_indices[sort_idx]
    facet_markers = facet_markers[sort_idx]
    
    # Safely mark root and tip
    root_idx = np.searchsorted(facet_indices, root_facets)
    valid_root = (root_idx < len(facet_indices)) & (facet_indices[root_idx % len(facet_indices)] == root_facets)
    facet_markers[root_idx[valid_root]] = 1
    
    tip_idx = np.searchsorted(facet_indices, tip_facets)
    valid_tip = (tip_idx < len(facet_indices)) & (facet_indices[tip_idx % len(facet_indices)] == tip_facets)
    facet_markers[tip_idx[valid_tip]] = 2
    
    mt = mesh.meshtags(domain, fdim, facet_indices, facet_markers)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)
    
    # Dirichlet BC (Fixed at root)
    u_D = np.array([0.0, 0.0, 0.0], dtype=dolfinx.default_scalar_type)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, root_facets), V)
    
    # Loads
    # 1. Hydrostatic Pressure (0.0981 MPa on all exterior faces, including root)
    pressure = fem.Constant(domain, dolfinx.default_scalar_type(0.0981))
    n = ufl.FacetNormal(domain)
    L_pressure = -pressure * ufl.inner(n, v) * (ds(1) + ds(2) + ds(3))
    
    # 2. Tip compressive load (450N in -Y direction)
    # Calculate tip area
    tip_area_form = fem.form(fem.Constant(domain, dolfinx.default_scalar_type(1.0)) * ds(2))
    tip_area = comm.allreduce(fem.assemble_scalar(tip_area_form), op=MPI.SUM)
    T_tip = 450.0 / tip_area if tip_area > 0 else 0.0
    t_tip = fem.Constant(domain, dolfinx.default_scalar_type((0.0, -T_tip, 0.0)))
    L_tip = ufl.inner(t_tip, v) * ds(2)
    
    # Total linear form
    L = L_pressure + L_tip
    
    # ---------------------------------------------------------
    # 3. Solve
    # ---------------------------------------------------------
    t0_solve = time.time()
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "gamg"}, petsc_options_prefix="sys_")
    uh = problem.solve()
    uh.name = "Displacement"
    t1_solve = time.time()
    solve_time = t1_solve - t0_solve
    
    if rank == 0:
        print(f"[{rank}] Linear system solved in {solve_time:.3f} s")
        
    # ---------------------------------------------------------
    # 4. Results
    # ---------------------------------------------------------
    # Calculate compliance
    compliance_form = fem.form(ufl.replace(L, {v: uh}))
    compliance = comm.allreduce(fem.assemble_scalar(compliance_form), op=MPI.SUM)
    
    # Volume
    one = fem.Constant(domain, PETSc.ScalarType(1.0))
    vol_form = fem.form(one * ufl.dx)
    volume = comm.allreduce(fem.assemble_scalar(vol_form), op=MPI.SUM)
    
    # Mass (assuming PLA density from run_fea.py: 1.24e-9 tonne/mm^3)
    rho = 1.24e-9
    mass_g = volume * rho * 1e6  # converted to grams
    
    # Max displacement magnitude
    uh_array = uh.x.array.reshape(-1, domain.geometry.dim)
    disp_mags = np.linalg.norm(uh_array, axis=1)
    max_disp = comm.allreduce(np.max(disp_mags) if len(disp_mags) > 0 else 0.0, op=MPI.MAX)
    
    # Von Mises Stress
    s = sigma(uh) - (1./3)*ufl.tr(sigma(uh))*ufl.Identity(domain.geometry.dim)
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    V_vm = fem.functionspace(domain, ("DG", 0))
    expr = fem.Expression(von_Mises, V_vm.element.interpolation_points)
    vm_func = fem.Function(V_vm)
    vm_func.interpolate(expr)
    vm_func.name = "Von_Mises_Stress"
    max_vm = comm.allreduce(np.max(vm_func.x.array) if len(vm_func.x.array) > 0 else 0.0, op=MPI.MAX)
    
    if rank == 0:
        print("\n" + "="*50)
        print(" FEniCSx FEA Summary ")
        print("="*50)
        print(f" MPI Processes        : {comm.size}")
        print(f" Mesh Time            : {mesh_time:.4f} s")
        print(f" Solve Time           : {solve_time:.4f} s")
        print(f" Volume               : {volume:.2f} mm^3")
        print(f" Mass                 : {mass_g:.2f} g")
        print(f" Compliance           : {compliance:.6e}")
        print(f" Max Displacement     : {max_disp:.6e} mm")
        print(f" Max Von Mises Stress : {max_vm:.3f} MPa")
        print("="*50 + "\n")

    # ---------------------------------------------------------
    # 5. Export for ParaView Visualization
    # ---------------------------------------------------------
    if rank == 0:
        print("[0] Exporting mesh, boundaries, and results to XDMF...")
        
    # We output two files to prevent meshtags from cluttering function space visualizers
    with dolfinx.io.XDMFFile(comm, "fenicsx_boundaries.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(mt, domain.geometry)
        
    with dolfinx.io.XDMFFile(comm, "fenicsx_results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
        xdmf.write_function(vm_func)
        
    if rank == 0:
        print("[0] Export complete. You can open fenicsx_results.xdmf and fenicsx_boundaries.xdmf in ParaView.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel FEniCSx FEA (Quadratic) with Visualization Export")
    parser.add_argument("--mesh-size", type=float, default=2.0, help="Target mesh size")
    args = parser.parse_args()
    
    run_fenicsx_fea(args.mesh_size)
