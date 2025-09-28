# [ICLR 2026] Disco: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring

This repository contains the supplementary materials for our ICLR 2026 submission, "Beyond Bipartite Constraints: Dynamic Conflict-aware Coloring for Complex Cell Instance Segmentation (Disco)".

Our work introduces a novel framework, `Disco`, designed to address the fundamental challenges of instance segmentation in dense, topologically complex cellular tissues. The core of `Disco` is a "divide and conquer" strategy, featuring two innovative mechanisms: **"Explicit Marking"** for conflict-aware label generation and **"Implicit Disambiguation"** for robust, end-to-end constrained optimization.

---

## 📂 Repository Contents

This repository is organized into two primary directories, which will be made fully public upon acceptance of the paper.

### 1. `/DataSet/`

This directory contains the four benchmark datasets used in our study. Most notably, it includes our newly introduced **GBC-FS 2025** dataset, a large-scale, high-density benchmark specifically designed to evaluate segmentation performance in challenging, real-world pathological scenarios.

Each dataset is structured with:
-   Image patches (`.png`)
-   Instance-level annotations (`.npy`)
-   Pre-computed cell adjacency graphs (`.yaml`)

### 2. `/Code/`

This directory contains the complete source code for our proposed **`Disco`** framework, implemented in PyTorch. The code is organized to be fully reproducible and includes:

-   **Data Processing:** Scripts for our **"Explicit Marking"** label generation algorithm.
-   **Model Implementation:** The `DiscoNet` architecture and our decoupled loss system, including the crucial **Adjacency Constraint Loss ($\mathcal{L}_{adj}$)**.
-   **Experiment Scripts:** All necessary tools and configuration files to replicate the quantitative results (Tables 2-7) and qualitative visualizations (Figures 1-6) presented in the paper.

---

## 📝 Reproducibility

We are committed to ensuring full reproducibility of our research. Upon acceptance, this repository will be made public, complete with detailed setup instructions, pre-trained model weights, and scripts to reproduce all figures and tables. Further details on our experimental setup are provided in Section 5.1 and 5.2 of the main paper.

---

## Citation

If you find our work useful in your research, please consider citing our paper (BibTeX to be provided upon publication).
