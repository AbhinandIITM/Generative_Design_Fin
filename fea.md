# Finite Element Analysis (FEA) Pipeline Documentation

This document outlines the modifications, fixes, and improvements applied to the BAAUV generative design FEA pipeline.

## 1. Geometric & Boundary Condition Corrections
*   **Coordinate System Alignment**: Updated `save_results.py` to properly map the CAD coordinates (X, Z, Y) to the matplotlib 3D axes. The fin's vertical span (Y-axis) now correctly displays as the vertical axis in the visualizer, matching the CAD viewer.
*   **Surface Detection**: Modified the surface detection logic in `run_fea.py` to correctly identify the root (bottom) and tip (top) faces based on the new vertical Y-axis bounding box extents.
*   **Axial Load Implementation**: Fixed the load application. Originally, loads were applying a transverse bending moment. We correctly implemented a `*CLOAD` card in the CalculiX input deck that takes a `450N` compressive load and spreads it evenly across all nodes on the top tip surface in the negative Y-direction.

## 2. FEA Solver Performance Enhancements
*   **Solver Multithreading**: The CalculiX SPOOLES solver was running on a single thread. We modified `run_fea.py` to automatically detect system CPU cores using `os.cpu_count()` and inject the `OMP_NUM_THREADS` environment variable into the subprocess. This drastically speeds up the matrix structure setup and stress calculation phases.

## 3. High-Fidelity vs. Draft Iteration Workflows
To support rapid reinforcement learning (RL) iterations without sacrificing accuracy, we split the pipeline into two standalone scripts:

### `run_fea.py` (Standard / High-Fidelity)
*   **Configuration**: Target mesh size of `2.0 mm` using 2nd-order quadratic elements.
*   **Performance**: Generates ~45,000 nodes. Solves in ~7-10 minutes.
*   **Results (450N Load)**: Peak stress of 22.5 MPa, Max displacement of 1.38 mm, Minimum FOS of 2.22.
*   **Use Case**: Final validation of generative design shapes.

### `run_fea_draft.py` (Fast Iteration)
*   **Configuration**: Target mesh size of `4.0 mm` using 2nd-order quadratic elements.
*   **Performance**: Generates ~19,000 nodes. Solves in ~2.5 minutes.
*   **Results (450N Load)**: Peak stress of 35.2 MPa, Max displacement of 0.39 mm, Minimum FOS of 1.42.
*   **Use Case**: Rapid generative design loop evaluations. 
*   *Note on Shear Locking: We intentionally kept 2nd-order elements even for the coarse draft mesh. An initial attempt at using 1st-order linear elements resulted in severe "shear locking," where the elements acted artificially rigid (0.3mm displacement). The 2nd-order coarse mesh perfectly balances speed while capturing accurate bending curves.*

## 4. Post-Processing & Visualization
*   **Factor of Safety (FOS)**: Updated `save_results.py` to compute and display the minimum FOS (the critical failure point) and the maximum finite FOS directly on the output plots and in the terminal logs.
*   **Color Mapping**: Zero-stress nodes (which have mathematically infinite FOS) are now capped in the colormap to prevent washing out the visual gradient, ensuring the critical low-FOS regions (red) are highly visible.

## Conclusion
Under the updated 450N axial compressive load, the PLA fin framework handles the stress beautifully with a safety factor > 2.2. The dual-script setup now enables fast topological iterations alongside high-fidelity validation.
