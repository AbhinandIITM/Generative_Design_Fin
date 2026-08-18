import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_frd(frd_path):
    nodes = {}
    disps = {}
    vm_stress = {}
    
    with open(frd_path, "r") as f:
        mode = "NODE"
        for line in f:
            if line.startswith(" -4  DISP"):
                mode = "DISP"
                continue
            elif line.startswith(" -4  STRESS"):
                mode = "STRESS"
                continue
            elif line.startswith(" -3"):
                mode = None
                continue
            
            if mode == "NODE" and line.startswith(" -1"):
                # fixed width: 13, 12, 12, 12
                if len(line) >= 49:
                    try:
                        nid = int(line[3:13])
                        x = float(line[13:25])
                        y = float(line[25:37])
                        z = float(line[37:49])
                        nodes[nid] = (x, y, z)
                    except ValueError:
                        pass
            elif mode == "DISP" and line.startswith(" -1"):
                if len(line) >= 49:
                    try:
                        nid = int(line[3:13])
                        dx = float(line[13:25])
                        dy = float(line[25:37])
                        dz = float(line[37:49])
                        disps[nid] = (dx, dy, dz)
                    except ValueError:
                        pass
            elif mode == "STRESS" and line.startswith(" -1"):
                if len(line) >= 85:
                    try:
                        nid = int(line[3:13])
                        sxx = float(line[13:25])
                        syy = float(line[25:37])
                        szz = float(line[37:49])
                        sxy = float(line[49:61])
                        syz = float(line[61:73])
                        szx = float(line[73:85])
                        vm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*(sxy**2 + syz**2 + szx**2)))
                        vm_stress[nid] = vm
                    except ValueError:
                        pass
    return nodes, disps, vm_stress

def plot_results(job_name, work_dir):
    frd_path = os.path.join(work_dir, f"{job_name}.frd")
    if not os.path.exists(frd_path):
        print(f"Error: {frd_path} not found.")
        return

    nodes, disps, vm_stress = parse_frd(frd_path)
    
    nids = list(nodes.keys())
    if not nids:
        print("No nodes found.")
        return

    # Extract coordinates
    X = np.array([nodes[n][0] for n in nids])
    Y = np.array([nodes[n][1] for n in nids])
    Z = np.array([nodes[n][2] for n in nids])
    
    # Extract displacements
    DX = np.array([disps.get(n, (0,0,0))[0] for n in nids])
    DY = np.array([disps.get(n, (0,0,0))[1] for n in nids])
    DZ = np.array([disps.get(n, (0,0,0))[2] for n in nids])
    D_mag = np.sqrt(DX**2 + DY**2 + DZ**2)
    
    # Extract stresses
    S = np.array([vm_stress.get(n, 0.0) for n in nids])
    
    # Create scale factor for displacement (exaggerate so it's visible)
    # Target 10% of bounding box max dimension
    bb_max = max(X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
    max_disp = D_mag.max()
    scale = (bb_max * 0.1) / max_disp if max_disp > 0 else 1.0
    print(f"Using displacement scale factor: {scale:.1f}")
    
    # Deformed coordinates
    Xd = X + DX * scale
    Yd = Y + DY * scale
    Zd = Z + DZ * scale

    # Map coordinates so CAD Y (vertical span) is the vertical axis in matplotlib
    plot_X, plot_Y, plot_Z = Xd, Zd, Yd

    # Plot Von Mises
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(plot_X, plot_Y, plot_Z, c=S, cmap='jet', s=2, alpha=0.8, edgecolor='none')
    fig.colorbar(p, ax=ax, label='Von Mises Stress (MPa)')
    ax.set_title('Deformed Shape & Von Mises Stress\n(Hydrostatic 10m + 450N Compressive Load)')
    
    # MIN/MAX markers for Von Mises
    if len(S) > 0:
        idx_max = np.argmax(S)
        idx_min = np.argmin(S)
        ax.text(plot_X[idx_max], plot_Y[idx_max], plot_Z[idx_max], f"MAX\n{S[idx_max]:.3f}", color='red', 
                weight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.text(plot_X[idx_min], plot_Y[idx_min], plot_Z[idx_min], f"MIN\n{S[idx_min]:.3f}", color='blue', 
                weight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # Equal aspect ratio trick
    max_range = np.array([plot_X.max()-plot_X.min(), plot_Y.max()-plot_Y.min(), plot_Z.max()-plot_Z.min()]).max() / 2.0
    mid_x = (plot_X.max()+plot_X.min()) * 0.5
    mid_y = (plot_Y.max()+plot_Y.min()) * 0.5
    mid_z = (plot_Z.max()+plot_Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (Chord, mm)')
    ax.set_ylabel('Z (Thickness, mm)')
    ax.set_zlabel('Y (Vertical Span, mm)')
    ax.view_init(elev=20, azim=45)
    
    out_vm = os.path.join(work_dir, f"{job_name}_von_mises.png")
    plt.savefig(out_vm, dpi=300, bbox_inches='tight')
    print(f"Saved {out_vm}")
    plt.close()

    # Plot Displacement Magnitude
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(plot_X, plot_Y, plot_Z, c=D_mag, cmap='plasma', s=2, alpha=0.8, edgecolor='none')
    fig.colorbar(p, ax=ax, label='Displacement Magnitude (mm)')
    ax.set_title(f'Deformed Shape (Scale {scale:.0f}x)')
    
    # MIN/MAX markers for Displacement
    if len(D_mag) > 0:
        idx_max = np.argmax(D_mag)
        idx_min = np.argmin(D_mag)
        ax.text(plot_X[idx_max], plot_Y[idx_max], plot_Z[idx_max], f"MAX\n{D_mag[idx_max]:.4f}", color='red', 
                weight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.text(plot_X[idx_min], plot_Y[idx_min], plot_Z[idx_min], f"MIN\n{D_mag[idx_min]:.4f}", color='blue', 
                weight='bold', bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (Chord, mm)')
    ax.set_ylabel('Z (Thickness, mm)')
    ax.set_zlabel('Y (Vertical Span, mm)')
    ax.view_init(elev=20, azim=45)
    
    out_disp = os.path.join(work_dir, f"{job_name}_displacement.png")
    plt.savefig(out_disp, dpi=300, bbox_inches='tight')
    print(f"Saved {out_disp}")
    plt.close()

    # Plot Factor of Safety (FOS)
    Sy = 50.0  # Yield strength for PLA in MPa
    FOS = np.zeros_like(S)
    mask = S > 1e-6
    FOS[mask] = Sy / S[mask]
    
    if np.any(mask):
        # Calculate actual finite FOS values at nodes carrying non-zero stress.
        fos_values = Sy / S[mask]
        min_fos = np.min(fos_values)
        max_fos = np.max(fos_values)

        idx_mask = np.where(mask)[0]
        idx_min_fos = idx_mask[np.argmin(fos_values)]
        idx_max_fos = idx_mask[np.argmax(fos_values)]

        # Cap only the COLOR MAP, not the reported FOS values.
        fos_cap = max(10.0, min_fos * 2.0)
    else:
        min_fos = float('inf')
        max_fos = float('inf')
        idx_min_fos = None
        idx_max_fos = None
        fos_cap = 10.0

    # Zero-stress nodes have mathematically infinite FOS. They are capped
    # only for visualization; max_fos above is the maximum finite FOS.
    FOS[~mask] = fos_cap
    FOS_plot = np.clip(FOS, 0, fos_cap)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # RdYlGn colormap: Red for low FOS, Green for high FOS
    p = ax.scatter(
        plot_X, plot_Y, plot_Z,
        c=FOS_plot, cmap='RdYlGn', s=2, alpha=0.8, edgecolor='none'
    )
    fig.colorbar(p, ax=ax, label='Factor of Safety (FOS)')
    ax.set_title(
        f'Deformed Shape & Factor of Safety\\n'
        f'(PLA Sy = {Sy:.0f} MPa, display capped at {fos_cap:.1f})'
    )

    # Minimum FOS = critical location.
    if idx_min_fos is not None:
        ax.text(
            plot_X[idx_min_fos], plot_Y[idx_min_fos], plot_Z[idx_min_fos],
            f"MIN FOS\n{min_fos:.2f}",
            color='red', weight='bold',
            bbox=dict(boxstyle="round", fc="white", alpha=0.8)
        )

    # Maximum finite FOS = highest safety margin among stressed nodes.
    if idx_max_fos is not None:
        ax.text(
            plot_X[idx_max_fos], plot_Y[idx_max_fos], plot_Z[idx_max_fos],
            f"MAX FOS\n{max_fos:.2f}",
            color='green', weight='bold',
            bbox=dict(boxstyle="round", fc="white", alpha=0.8)
        )

    print(f"Minimum FOS: {min_fos:.4f}")
    print(f"Maximum finite FOS: {max_fos:.4f}")

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X (Chord, mm)')
    ax.set_ylabel('Z (Thickness, mm)')
    ax.set_zlabel('Y (Vertical Span, mm)')
    ax.view_init(elev=20, azim=45)
    
    out_fos = os.path.join(work_dir, f"{job_name}_fos.png")
    plt.savefig(out_fos, dpi=300, bbox_inches='tight')
    print(f"Saved {out_fos}")
    plt.close()

if __name__ == "__main__":
    plot_results("fin_assembly_links", "fea_results")