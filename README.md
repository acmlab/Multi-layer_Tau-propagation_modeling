# Understanding Mechanistic Role of Structural and Functional Connectivity in Tau Propagation Through Multi-Layer Modeling



# 📊 Statistical Models

This folder contains MATLAB scripts for statistical analysis of tau propagation patterns across the brain using structural and functional connectivity, gene expression, and demographic factors. Each script targets a specific aspect of modeling, correlation, or group difference testing.

---

## 📁 File Descriptions

### `GAM.m`
Fits **Generalized Additive Models (GAMs)** to model the effect of **age**, **sex**, and **subject ID** on tau SUVR values across ROIs.

- Outputs:
  - Partial age effects per ROI
  - ROI-wise p-values (model comparison: with vs. without age term)

---

### `corrNeighbor_tau.m`
Computes correlations between regional tau accumulation and the **mean tau of its neighbors** (based on FC or SC).

- Outputs:
  - ROI-wise adjusted R and p-values
  - Regression plots

---

### `corrNeighbor_lobes.m`
Analyzes **lobe-level associations** between each region’s tau and the mean tau in neighbors from each **7** or **13** cortical system.

- Outputs:
  - 7×7 and 13×13 correlation matrices
  - FDR-corrected p-values
  - Correlation heatmap visualizations

---

### `gene_us_uf_analysis.m`
Explores associations between **gene expression patterns** and tau propagation differences in **US vs. UF regions**.

- Likely includes:
  - Gene-wise correlation or regression with propagation metrics

---

### `lasso_gengens.m`
Performs **non-negative LASSO regression with bootstrapping** to identify genes associated with tau propagation patterns.

- Outputs:
  - Stability path plots
  - Selection frequency matrix
  - Top-ranked gene list

---

### `mixed_model_group_diff.m`
Applies **linear mixed-effects models** to test group differences (e.g., AD vs. CN) in tau propagation, accounting for subject-level variability.

- Suitable for:
  - Longitudinal or hierarchical data
  - Repeated-measures tau SUVR analysis

---

### `us_uf_sex_apoe.m`
Stratified analysis of tau propagation in **US vs. UF regions** across **sex** and **APOE genotype**.

- Outputs:
  - Interaction effects
  - Group-level comparisons

---

## 📝 Notes

- Make sure required inputs (e.g., SC/FC matrices, tau SUVR files, region labels, gene data) are available in the appropriate paths.
- Output results are typically saved to `Results/` or `Gene_results/` directories.
- Scripts assume MATLAB R2021a+ with `fitrgam`, `fitlme`, etc.

---

# 🧠 SC-FC Multi-layer Tau Propagation Model

This folder contains the implementation of the **closed-loop feedback multi-layer neural transport model** for tau propagation modeling, as illustrated in **Fig. 9** of the manuscript.

## 🔍 Overview

The model integrates:

- **Graph Convolutional Networks (GCN)** to extract network-informed features from structural and functional connectivity
- **Linear Quadratic Regulators (LQR)** for feedback control over propagation dynamics
- **PDE-based transport modeling** for SC-specific and FC-specific tau diffusion
- **Multi-Layer Perceptron (MLP)** for prediction of tau accumulation at the next time point

## 🧱 Key Components

- `GCN_layers.py` – Graph convolutional layers for SC/FC networks  
- `control_constrints.py` – Feedback control using LQR  
- `model_prediction.py` – Core model forward propagation combining PDE solver and control  
- `train.py` – Training script with loss functions and optimizer  
- `dataset.py` – Custom dataset loader for input SUVRs and network matrices  
- `5fold.py` – Cross-validation framework  
- `utils.py` – Helper functions for training, evaluation, etc.  
- `optimal_control/` & `wirings/` – Additional modules for control system implementation and parameter setup

## 📥 Input

- Baseline tau SUVR vector `x⁰`
- SC and FC adjacency matrices `F and S`

## 📤 Output

- Predicted tau SUVR vector `x¹` at the next time point
- Disentangled SC-driven and FC-driven propagation terms

## 🧪 Usage

```bash
python model_prediction.py

