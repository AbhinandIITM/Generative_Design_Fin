# Research Statement: Generative, Parametric, and Topology Optimization of Internal Structural Skeletons

## 1. Vision & Executive Summary

This research focuses on developing an end-to-end computational framework combining **Generative Design**, **Parametric CAD Modeling**, **Topology Optimization (TO)**, and **Machine Learning / Reinforcement Learning (ML/RL)** specifically targeting the **internal load-bearing structural skeleton** of engineering components.

The primary benchmark application is the automated synthesis and structural optimization of the **rigid internal wing/fin skeleton** bounded by a fixed **NACA 0012** outer mold line ($105\text{ mm}$ span $\times 7\text{ mm}$ thickness envelope). Because a soft, compliant outer skin covers the skeleton to define aerodynamic flow, aerodynamic CFD shape optimization is decoupled; the structural optimization relies **purely on Finite Element Analysis (FEA)** under specified loading conditions. This methodology directly extends to internal ribbing, stiffeners, and cutouts for broader engineering members such as beams, columns, and stringers to enhance manufacturability (DFAM).

---

## 2. Research Objectives & Methodology

```
                                  [ Generative AI / RL Agent ]
                                               │
                                               ▼
[ Fixed Aero Loads / FEA BCs ] ──► [ Parametric CadQuery Skeleton ] ──► [ Pure FEA Analysis ]
                                               │                           (Stiffness/Stress/Modes)
                                               ▼                                   │
                                  [ Internal Topology Opt (TO) ] ◄─────────────────┘
                                  (Ribs / Spars / Stringers / Cutouts)
                                               │
                                               ▼
                                  [ Manufacturable STEP / AM ]
```

### Pillar 1 — Fixed OML & Internal Skeleton Parametric Modeling
- **Outer Mold Line (OML):** Fixed NACA 0012 envelope ($105\text{ mm} \times 7\text{ mm}$) draped with a soft skin.
- **Internal Skeleton Geometry:** Parametric synthesis of internal ribs, leading/trailing edge solid attachment blocks, longitudinal stringers, and internal lightning holes using NURBS B-splines.
- **Cross-Sectional Variation:** Parametric variation of internal member thickness, web cutout dimensions, and stringer spatial distributions along the span.

### Pillar 2 — Pure FEA Structural & Topology Optimization
- **Structural Objectives:** Minimize total compliance (maximize bending and torsional stiffness) and structural weight under critical aerodynamic shear and bending loads.
- **FEA Performance Criteria:** Evaluate von Mises stress distributions, local buckling modes, and natural vibration frequencies.
- **Internal Member Synthesis:** Density-based (SIMP) and evolutionary (BESO) topology optimization applied strictly within the interior design domain to place material where strain energy density is highest.

### Pillar 3 — ML/RL-Driven Design Space Exploration & Surrogates
- **Reinforcement Learning (RL):** Markov Decision Process (MDP) formulation where the RL agent acts on discrete topological alterations (adding/removing stringers, resizing rib cutouts).
- **Fast FEA Surrogates:** Deep neural networks and Broad Learning Systems trained on FEA output to instantly predict stress and compliance, accelerating RL exploration loops without expensive repeated meshing.
- **Parametric CAD Validity:** Automated determination of valid geometric parameter ranges to prevent B-rep kernel failures during aggressive optimization sweeps.

---

## 3. Generalization & Manufacturability Focus

Focusing on internal skeleton topology under pure FEA allows the framework to directly generalize to other structural engineering components:

* **Internal Web & Stiffener Synthesis:** Applying the parametric topology optimization routine to internal webs of I-beams, box girders, and tubular columns to optimize shear/bending capacity and prevent localized buckling.
* **Manufacturability & DFAM Constraints:** Incorporating self-supporting overhang constraints ($< 45^\circ$), minimum wall thickness, and CNC tool clearance for internal cutouts.
* **Multi-Constraint FEA Objectives:** Optimizing stiffness-to-weight ratios while simultaneously managing natural frequencies to avoid resonance.
