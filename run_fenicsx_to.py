#!/usr/bin/env python3
"""
run_fenicsx_to.py -- Parallel FEniCSx-based Topology Optimization Codepath

This script serves as a starting point for moving topology optimization
and generative design research to FEniCSx (dolfinx). 
It features MPI-based parallel meshing and solving, which is ideal
for large-scale TO on workstations or clusters.

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
    from dolfinx.io import gmshio
    import ufl
    from petsc4py import PETSc
except ImportError as e:
    print(f"Failed to import FEniCSx dependencies: {e}")
    print("Please ensure you are in a conda environment with fenics-dolfinx installed.")
    print("Example: conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich")
    pass

import gmsh

def create_and_read_mesh(comm, step_path="fin_assembly_links.step", mesh_size=2.0):
    """
    Import a STEP file using Gmsh, generate a 3D mesh, 
    and read it into a dolfinx mesh in parallel.
    """
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("fin_step")
        
        # Import STEP file
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Assign physical groups to all 3D volumes
        volumes = gmsh.model.getEntities(dim=3)
        if volumes:
            vol_tags = [v[1] for v in volumes]
            gmsh.model.addPhysicalGroup(3, vol_tags, tag=1)
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.15)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.mesh.generate(3)
        
    # Import gmsh model into dolfinx mesh
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=3
    )
    gmsh.finalize()
    return domain

def run_fenicsx_fea(step_path="fin_assembly_links.step", mesh_size=2.0):
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
        domain = create_and_read_mesh(comm, step_path=step_path, mesh_size=mesh_size)
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
    # Using a continuous Galerkin function space of degree 1
    V = fem.VectorFunctionSpace(domain, ("CG", 1))
    
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
    
    # Body force (e.g. pressure or gravity)
    f = fem.Constant(domain, PETSc.ScalarType((0.0, -9.81e-3, 0.0)))
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions (Fix at x = 0 plane)
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
        
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, left_boundary)
    u_D = np.array([0.0, 0.0, 0.0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    
    # ---------------------------------------------------------
    # 3. Solve
    # ---------------------------------------------------------
    t0_solve = time.time()
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "gamg"})
    uh = problem.solve()
    t1_solve = time.time()
    solve_time = t1_solve - t0_solve
    
    if rank == 0:
        print(f"[{rank}] Linear system solved in {solve_time:.3f} s")
        
    # ---------------------------------------------------------
    # 4. Results
    # ---------------------------------------------------------
    # Calculate compliance: inner(f, u)
    compliance_form = fem.form(ufl.inner(f, uh) * ufl.dx)
    compliance = comm.allreduce(fem.assemble_scalar(compliance_form), op=MPI.SUM)
    
    if rank == 0:
        print("\n" + "="*50)
        print(" FEniCSx FEA Summary ")
        print("="*50)
        print(f" MPI Processes : {comm.size}")
        print(f" Mesh Time     : {mesh_time:.4f} s")
        print(f" Solve Time    : {solve_time:.4f} s")
        print(f" Compliance    : {compliance:.6e}")
        print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel FEniCSx Topology Optimization Starter")
    parser.add_argument("--step", type=str, default="fin_assembly_links.step", help="Path to the STEP file")
    parser.add_argument("--mesh-size", type=float, default=2.0, help="Target mesh size")
    args = parser.parse_args()
    
    run_fenicsx_fea(args.step, args.mesh_size)
