# Multi-layer_Tau-propagation_modeling



# 📊 Statistical Models for Multi-layer Tau Propagation Modeling

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

## 📌 Citation
If you use this code, please cite the corresponding publication (to be added).
