# FEniCSx Topology Optimization Plan (WSL)

FEniCSx requires a Linux environment to run natively with MPI support. This guide outlines the steps to set up Windows Subsystem for Linux (WSL), install FEniCSx, and run the `run_fenicsx_to.py` script.

## Phase 1: Setup WSL (Windows Subsystem for Linux)

1. **Install WSL**
   Open your Windows PowerShell (as Administrator) and run:
   ```powershell
   wsl --install
   ```
   *This installs the default Ubuntu distribution. Restart your computer if prompted.*

2. **Access your Windows Files from WSL**
   Once inside the Ubuntu terminal, you can access your Windows files under `/mnt/`. Navigate to your project directory:
   ```bash
   cd /mnt/g/My\ Drive/IISC/BAAUV/Gen.\ Design/Fin
   ```

## Phase 2: Install Conda and FEniCSx Dependencies in WSL

1. **Install Miniconda (Linux)**
   Inside your WSL Ubuntu terminal, download and install Miniconda:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   *Follow the prompts and allow it to initialize.*

2. **Create the FEniCSx Conda Environment**
   Refresh your terminal (`source ~/.bashrc`) and create the environment:
   ```bash
   conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich python=3.10 -y
   ```

3. **Install Auxiliary Packages**
   Activate the environment and install meshing/math tools:
   ```bash
   conda activate fenicsx-env
   pip install gmsh numpy
   ```

## Phase 3: Run the FEniCSx Pipeline

1. **Execute the Script**
   With the environment activated, you can now run the script we generated earlier. To leverage your multi-thread CPU, use `mpirun` to execute it in parallel:
   ```bash
   # Run across 4 MPI processes
   mpirun -n 4 python run_fenicsx_to.py
   ```

2. **Verify the Outputs**
   The script should output the MPI ranks initializing, the parallel meshing times (via Gmsh), the linear solve time, and the final structural compliance of the bounding box. 

## Next Steps for Topology Optimization
Once this baseline runs successfully, you can extend `run_fenicsx_to.py` to:
- Define a material density field (the design variable) over the mesh.
- Implement a SIMP (Solid Isotropic Material with Penalisation) interpolation.
- Formulate an objective function (e.g., minimize compliance subject to a volume constraint).
- Hook it up to an optimizer like `scipy.optimize.minimize` or use dedicated frameworks like `Scikit-Topt`.
