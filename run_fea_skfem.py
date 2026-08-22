#!/usr/bin/env python3
"""
run_fea_skfem.py -- Native Windows Python in-memory FEA Solver
"""

import time
import numpy as np
import gmsh
import meshio
import skfem
from skfem import MeshTet, Basis, ElementTetP1
from skfem.models.elasticity import linear_elasticity
from pypardiso import spsolve

def create_and_read_mesh(step_file="fin_assembly_links.step", mesh_size=4.0):
    """Loads a STEP file, meshes it, and returns a scikit-fem MeshTet."""
    gmsh.initialize()
    gmsh.model.add("fin_assembly")
    gmsh.model.occ.importShapes(step_file)
    try:
        gmsh.model.occ.healShapes(tolerance=1e-5)
    except:
        pass
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.15)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(3)
    
    # Extract mesh data from Gmsh directly into memory
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(nodeCoords).reshape(-1, 3).T
    
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
    # elemTypes[0] == 4 corresponds to 4-node tetrahedra (Tet4)
    tet_nodes = np.array(elemNodeTags[0]).reshape(-1, 4) - 1  # 0-indexed
    tet_nodes = tet_nodes.T
    
    gmsh.finalize()
    return MeshTet(nodes, tet_nodes)

def run_native_fea(step_file="fin_assembly_links.step", mesh_size=4.0):
    print("=" * 60)
    print(f" Native Python In-Memory 3D FEA (scikit-fem + Pardiso)")
    print("=" * 60)
    
    t0_mesh = time.time()
    mesh = create_and_read_mesh(step_file, mesh_size)
    t1_mesh = time.time()
    print(f"[mesh] Generated {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} tets in {t1_mesh - t0_mesh:.3f} s")
    
    t0_setup = time.time()
    from skfem import ElementVector
    element = ElementVector(ElementTetP1())
    basis = Basis(mesh, element, intorder=1)
    
    E, nu = 3500.0, 0.36
    lambda_ = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    
    # Assemble global stiffness matrix
    form_elasticity = linear_elasticity(lambda_, mu)
    K = skfem.asm(form_elasticity, basis)
    
    # Boundary Conditions based on actual bounding box Y-axis
    y_min = np.min(mesh.p[1])
    y_max = np.max(mesh.p[1])
    tol = 1.0  # mm
    
    boundary_facets = mesh.boundary_facets()
    basis_bnd = skfem.FacetBasis(mesh, element, facets=boundary_facets, intorder=2)
    
    facets_centers = mesh.p[:, mesh.facets].mean(axis=1)
    root_facets = np.where(facets_centers[1] < y_min + tol)[0]
    tip_facets = np.where(facets_centers[1] > y_max - tol)[0]
    
    root_bnd = np.intersect1d(boundary_facets, root_facets)
    tip_bnd = np.intersect1d(boundary_facets, tip_facets)
    
    basis_tip = skfem.FacetBasis(mesh, element, facets=tip_bnd, intorder=2)
    
    # 1. Pressure Load (0.0981 MPa inward)
    p_val = 0.0981
    @skfem.LinearForm
    def pressure_load(v, w):
        return -p_val * skfem.helpers.dot(w.n, v)
        
    # 2. Tip Load (450 N total in -Y direction)
    tip_pts = mesh.p[:, mesh.facets[:, tip_bnd]]
    tip_areas = 0.5 * np.linalg.norm(np.cross((tip_pts[:, 1, :] - tip_pts[:, 0, :]).T, 
                                              (tip_pts[:, 2, :] - tip_pts[:, 0, :]).T), axis=1)
    tip_area = np.sum(tip_areas)
    T_tip = 450.0 / tip_area if tip_area > 0 else 0.0
    
    @skfem.LinearForm
    def tip_load(v, w):
        return -T_tip * v[1]
        
    F = skfem.asm(pressure_load, basis_bnd) + skfem.asm(tip_load, basis_tip)
    
    # Dirichlet BC (Fixed root)
    root_nodes = np.unique(mesh.facets[:, root_bnd].flatten())
    D = basis.get_dofs(root_nodes).all()
    
    I = set(range(K.shape[0])) - set(D)
    I = np.array(list(I), dtype=int)
    
    K_cond = K[I][:, I]
    F_cond = F[I]
    
    t1_setup = time.time()
    print(f"[setup] Matrix assembled ({K.shape[0]} DOFs) in {t1_setup - t0_setup:.3f} s")
    
    t0_solve = time.time()
    u_cond = spsolve(K_cond, F_cond)
    u = np.zeros(K.shape[0])
    u[I] = u_cond
    t1_solve = time.time()
    solve_time = t1_solve - t0_solve
    print(f"[solve] Pardiso factorize & solve in {solve_time:.3f} s")
    
    # Post-processing
    u_disp = u.reshape(-1, 3)
    disp_mag = np.linalg.norm(u_disp, axis=1)
    max_disp = np.max(disp_mag)
    compliance = np.dot(F, u)
    
    # Calculate Von Mises stress at elements
    grad_u = basis.interpolate(u).grad
    sym_grad_u = 0.5 * (grad_u + np.transpose(grad_u, (1, 0, 2, 3)))
    tr = np.trace(sym_grad_u, axis1=0, axis2=1)
    I = np.eye(3)[:, :, None, None]
    stress = lambda_ * tr * I + 2 * mu * sym_grad_u
    s11, s22, s33 = stress[0,0], stress[1,1], stress[2,2]
    s12, s13, s23 = stress[0,1], stress[0,2], stress[1,2]
    vm_quad = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*(s12**2 + s13**2 + s23**2)))
    vm_element = vm_quad[:, 0]  # Constant per element for P1
    max_vm = np.max(vm_element)
    
    # FoS (PLA yield = 40 MPa)
    yield_strength = 40.0
    fos = yield_strength / max_vm if max_vm > 0 else float('inf')
    
    # Volume and Mass (PLA density = 1.24e-9 tonne/mm^3 = 1.24e-3 g/mm^3)
    p1 = mesh.p[:, mesh.t[0]].T
    p2 = mesh.p[:, mesh.t[1]].T
    p3 = mesh.p[:, mesh.t[2]].T
    p4 = mesh.p[:, mesh.t[3]].T
    volumes = np.abs(np.einsum('ij,ij->i', p1-p4, np.cross(p2-p4, p3-p4))) / 6.0
    total_volume = np.sum(volumes)
    mass_g = total_volume * 1.24e-3
    
    mesh_time = t1_mesh - t0_mesh
    
    print("\n" + "="*50)
    print(" In-Memory FEA Summary (scikit-fem) ")
    print("="*50)
    print(f" Mesh Time            : {mesh_time:.4f} s")
    print(f" Setup Time           : {t1_setup - t0_setup:.4f} s")
    print(f" Solve Time           : {solve_time:.4f} s")
    print(f" Volume               : {total_volume:.2f} mm^3")
    print(f" Mass                 : {mass_g:.2f} g")
    print(f" Compliance           : {compliance:.6e}")
    print(f" Max Displacement     : {max_disp:.6e} mm")
    print(f" Max Von Mises Stress : {max_vm:.3f} MPa")
    print(f" Min Factor of Safety : {fos:.3f}")
    print("="*50 + "\n")
    
    mesh_out = meshio.Mesh(
        points=mesh.p.T, 
        cells=[("tetra", mesh.t.T)], 
        point_data={"Displacement": u_disp, "Displacement_Mag": disp_mag},
        cell_data={"Von_Mises": [vm_element]}
    )
    meshio.write("fea_results/in_memory_results.xdmf", mesh_out)
    print("[output] Saved 'fea_results/in_memory_results.xdmf' for ParaView.")
    
    # Render and save images automatically using PyVista
    import pyvista as pv
    import os
    
    # Only render if run locally (avoids errors in headless unless xvfb is used)
    try:
        pv.set_jupyter_backend(None)
        pv.global_theme.window_size = [1200, 800]
        pv.global_theme.background = 'white'
        pv.global_theme.font.color = 'black'
        
        # We need an unstructured grid
        points = mesh.p.T
        # PyVista VTK_TETRA is 10
        cells = np.empty((mesh.t.shape[1], 5), dtype=np.int64)
        cells[:, 0] = 4
        cells[:, 1:] = mesh.t.T
        grid = pv.UnstructuredGrid(cells.flatten(), np.full(mesh.t.shape[1], 10, dtype=np.uint8), points)
        grid.point_data["Displacement Magnitude"] = disp_mag
        grid.cell_data["Von Mises Stress (MPa)"] = vm_element
        
        # Warp by displacement for visualization
        factor = 5.0
        grid.point_data["Vectors"] = u_disp * factor
        warped = grid.warp_by_vector("Vectors", factor=1.0)
        
        # Plot Displacement
        p = pv.Plotter(off_screen=True)
        p.add_mesh(warped, scalars="Displacement Magnitude", cmap="jet", show_edges=False)
        p.add_axes()
        p.view_vector([1.0, 0.5, 1.0], viewup=[0.0, 1.0, 0.0]) # Y-axis is up
        p.screenshot("fea_results/fin_assembly_links_displacement.png")
        p.close()
        
        # Plot Stress
        p = pv.Plotter(off_screen=True)
        p.add_mesh(warped, scalars="Von Mises Stress (MPa)", cmap="jet", show_edges=False)
        p.add_axes()
        p.view_vector([1.0, 0.5, 1.0], viewup=[0.0, 1.0, 0.0]) # Y-axis is up
        p.screenshot("fea_results/fin_assembly_links_von_mises.png")
        p.close()
        
        print("[output] Saved rendered images to fea_results/")
    except Exception as e:
        print(f"[WARN] Failed to render images automatically: {e}")

if __name__ == "__main__":
    run_native_fea(step_file="fin_assembly_links.step", mesh_size=4.0)
