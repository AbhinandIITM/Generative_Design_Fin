# Installation Guide

This project contains two distinct simulation workflows for the fin design:
1. **Gmsh + CalculiX Workflow** (`run_fea.py`)
2. **Parallel FEniCSx Workflow** (`run_fenicsx_to.py`, `visualize_fenicsx_fea.py`)

Below are the instructions to set up the environments for both on Ubuntu.

---

## 1. Setting up the FEniCSx Environment (Topology Optimization & Advanced FEA)

Because FEniCSx requires complex C++ dependencies (PETSc, MPI, dolfinx), it is highly recommended to use `conda` or `mamba`. 

### Step 1: Install Conda / Mamba
If you don't have it installed, we recommend [Miniforge (Mamba)](https://github.com/conda-forge/miniforge) for fast dependency resolution.

### Step 2: Create the Conda Environment
Run the following commands in your terminal to create a dedicated environment named `fenicsx-env`:

```bash
conda create -n fenicsx-env -c conda-forge fenics-dolfinx mpich python=3.10
conda activate fenicsx-env
```
*(Note: If you use mamba, replace `conda` with `mamba` above)*

### Step 3: Install Python Packages
Once the environment is activated, install the remaining dependencies via pip:

```bash
pip install -r requirements.txt
```

You are now ready to run the FEniCSx scripts:
```bash
mpirun -n 4 python visualize_fenicsx_fea.py
mpirun -n 4 python run_fenicsx_to.py
```

---

## 2. Setting up the Gmsh + CalculiX Environment (Baseline FEA)

For the baseline FEA scripts (`run_fea.py`), you need the `ccx` (CalculiX) solver installed on your system.

### Step 1: Install CalculiX (Ubuntu)
CalculiX is available in the default Ubuntu package repositories:

```bash
sudo apt update
sudo apt install calculix-ccx
```

### Step 2: Install Python Dependencies
You can use the same `fenicsx-env` environment created above, or your `base` environment. Just ensure the `requirements.txt` dependencies are installed:

```bash
pip install -r requirements.txt
```

You can now run the CalculiX baseline script:
```bash
python run_fea.py
```

---

## 3. Visualization Tools

To visualize the outputs (such as the `.xdmf` files produced by FEniCSx, or the `.frd` files produced by CalculiX):

**Install ParaView (For FEniCSx `.xdmf` files):**
```bash
sudo apt install paraview
```

*(When opening FEniCSx XDMF files in ParaView, be sure to select the **Xdmf3ReaderT** option!)*

**Install CGX (CalculiX GraphiX):**
```bash
sudo apt install calculix-cgx
```
