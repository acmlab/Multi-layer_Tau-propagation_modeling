import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from brainsmash.mapgen.base import Base


# ============================================================
# 0. User paths
# ============================================================
coords_path = r"continuity_region.node"
excel_path = r"Tau_SUVR.xlsx"
lobe_path  = r"region_labels_mapped.csv"

out_dir = r".\BrainSMASH_results"
os.makedirs(out_dir, exist_ok=True)

n_surrogates = 1000
num_lobes = 7
lobe_names = ['Frontal', 'Insula', 'Temporal', 'Occipital', 'Parietal', 'Limbic', 'Sub-cortical']

# ROI colors for plotting
lobe_colors = np.array([
    [253, 175, 125],   # Frontal
    [133, 192, 154],   # Insula
    [69, 149, 201],    # Temporal
    [238, 177, 179],   # Occipital
    [199, 196, 141],   # Parietal
    [178, 178, 0],     # Limbic
    [168, 95, 91],     # Sub-cortical
]) / 255.0


# ============================================================
# 1. Helpers
# ============================================================
def extract_node_num(col):
    m = re.search(r'Node[_ ]?(\d+)', str(col))
    return int(m.group(1)) if m else 10**9


def annualized_delta_tau_map(df, node_cols):
    """
    Compute subject-level annualized delta tau using first two timepoints,
    then return group-average ROI map (1 x 160).
    """
    tau_avg_map = np.zeros(len(node_cols), dtype=float)
    valid_count = 0

    for ptid, group in df.groupby('PTID'):
        if len(group) < 2:
            continue

        group = group.sort_values('EXAMDATE').iloc[:2]

        tau_t1 = group.iloc[0][node_cols].values.astype(float)
        tau_t2 = group.iloc[1][node_cols].values.astype(float)

        age_t1 = float(group.iloc[0]['AGE'])
        age_t2 = float(group.iloc[1]['AGE'])

        # avoid zero interval
        if np.isclose(age_t2 - age_t1, 0):
            continue

        delta_tau = (tau_t2 - tau_t1) / (age_t2 - age_t1)

        if np.any(np.isnan(delta_tau)):
            continue

        tau_avg_map += delta_tau
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid subjects found for annualized delta tau computation.")

    empirical_map = tau_avg_map / valid_count
    return empirical_map, valid_count


def compute_lobe_means_from_roi_map(roi_map, lobe_index, num_lobes):
    out = np.zeros(num_lobes, dtype=float)
    for l in range(1, num_lobes + 1):
        idx = (lobe_index == l)
        out[l - 1] = np.nanmean(roi_map[idx])
    return out


def compute_null_distribution_lobe(surrogate_maps, lobe_index, num_lobes):
    n_perm = surrogate_maps.shape[0]
    null_distribution = np.zeros((n_perm, num_lobes), dtype=float)

    for p in range(n_perm):
        surrogate_map = surrogate_maps[p, :]
        for l in range(1, num_lobes + 1):
            idx = (lobe_index == l)
            null_distribution[p, l - 1] = np.nanmean(surrogate_map[idx])

    return null_distribution


def compute_p_and_z(true_lobe_means, null_distribution):
    num_lobes = true_lobe_means.shape[0]
    p_values = np.zeros(num_lobes, dtype=float)
    z_scores = np.zeros(num_lobes, dtype=float)
    null_means = np.zeros(num_lobes, dtype=float)
    null_stds = np.zeros(num_lobes, dtype=float)

    for l in range(num_lobes):
        null_vals = null_distribution[:, l]
        true_val = true_lobe_means[l]

        null_means[l] = np.mean(null_vals)
        null_stds[l] = np.std(null_vals, ddof=1)

        # two-sided p-value with +1 correction
        p_values[l] = (np.sum(np.abs(null_vals) >= np.abs(true_val)) + 1) / (len(null_vals) + 1)

        if null_stds[l] > 0:
            z_scores[l] = (true_val - null_means[l]) / null_stds[l]
        else:
            z_scores[l] = np.nan

    return p_values, z_scores, null_means, null_stds


def plot_lobe_zscores(z_scores, p_values, title, save_prefix):
    plt.figure(figsize=(4, 2.5), dpi=300)
    bars = plt.bar(
        range(len(z_scores)),
        z_scores,
        color=lobe_colors,
        edgecolor='black',
        linewidth=1.0
    )

    plt.axhline(0, color='black', linewidth=1.2)
    plt.xticks(range(len(z_scores)), lobe_names, rotation=35, ha='right', fontsize=11)
    plt.ylabel('Z-score', fontsize=13, fontweight='bold')
    if title: 
        plt.title(title, fontsize=14, fontweight='bold')

    # significance stars
    for i, (z, p) in enumerate(zip(z_scores, p_values)):
        stars = ""
        if p < 0.05: stars = "*"
        if p < 0.01: stars = "**"
        if p < 0.001: stars = "***"
        if stars:
            offset = np.sign(z) * (0.15 + 0.06 * abs(z))
            plt.text(i, z + offset, stars, ha='center', va='center',
                     fontsize=16, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.svg", transparent=True, bbox_inches='tight')
    plt.savefig(f"{save_prefix}.pdf", transparent=True, bbox_inches='tight')
    plt.close()


# ======  Baseline  Tau SUVR  ======
def plot_empirical_raw_tau(df, node_cols, lobe_index, num_lobes, lobe_names, lobe_colors, out_dir):
    print("\nComputing subject-level BASELINE raw tau means for empirical plotting...")
    subject_raw_taus = []
    
    for ptid, group in df.groupby('PTID'):
        if len(group) < 2:
            continue
        group = group.sort_values('EXAMDATE').iloc[:2]
        
        tau_t1 = group.iloc[0][node_cols].values.astype(float)
        
        if np.any(np.isnan(tau_t1)):
            continue
            
        subject_raw_taus.append(tau_t1)
        

    subject_roi_matrix = np.vstack(subject_raw_taus)
    n_subjects = subject_roi_matrix.shape[0]
    
    lobe_means = np.zeros(num_lobes)
    lobe_sems = np.zeros(num_lobes) 
    
    for l in range(1, num_lobes + 1):
        idx = (lobe_index == l)
        
        subj_lobe_vals = np.nanmean(subject_roi_matrix[:, idx], axis=1)
        
        lobe_means[l-1] = np.mean(subj_lobe_vals)
        lobe_sems[l-1] = np.std(subj_lobe_vals, ddof=1) / np.sqrt(n_subjects)
        
    
    plt.figure(figsize=(4, 2.5), dpi=300)
    bars = plt.bar(
        range(num_lobes),
        lobe_means,
        yerr=lobe_sems,          
        capsize=3,                
        color=lobe_colors,
        edgecolor='black',
        linewidth=1.0,
        error_kw={'linewidth': 1.0, 'markeredgewidth': 1.0}
    )

    plt.axhline(0, color='black', linewidth=1.2)
    plt.xticks(range(num_lobes), lobe_names, rotation=35, ha='right', fontsize=11)
    
    # Mean Tau Concentration 
    plt.ylabel('Mean Tau Concentration', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_prefix = os.path.join(out_dir, "Empirical_Raw_Tau_7lobes")
    plt.savefig(f"{save_prefix}.svg", transparent=True, bbox_inches='tight')
    plt.savefig(f"{save_prefix}.pdf", transparent=True, bbox_inches='tight')
    plt.close()
    print(f"Saved empirical raw baseline tau plot to: {save_prefix}.svg/.pdf")
# =========================================================================


def run_brainsmash_experiment(empirical_map, distance_matrix, lobe_index, label, out_prefix):
    print(f"\nRunning BrainSMASH experiment: {label}")

    # fit BrainSMASH to ROI-level empirical map
    base = Base(x=empirical_map, D=distance_matrix)
    surrogates = base(n=n_surrogates)

    # observed lobe means from the SAME empirical map
    true_lobe_means = compute_lobe_means_from_roi_map(empirical_map, lobe_index, num_lobes)

    # null distribution at lobe level
    null_distribution = compute_null_distribution_lobe(surrogates, lobe_index, num_lobes)

    # stats
    p_values, z_scores, null_means, null_stds = compute_p_and_z(true_lobe_means, null_distribution)

    # save table
    result_df = pd.DataFrame({
        "Lobe": lobe_names,
        "TrueMean": true_lobe_means,
        "NullMean": null_means,
        "NullStd": null_stds,
        "Zscore": z_scores,
        "Pvalue": p_values
    })
    result_csv = os.path.join(out_dir, f"{out_prefix}_results.csv")
    result_df.to_csv(result_csv, index=False)

    # save surrogates
    surrogate_csv = os.path.join(out_dir, f"{out_prefix}_surrogates_{n_surrogates}x160.csv")
    np.savetxt(surrogate_csv, surrogates, delimiter=',')

    # plot
    plot_lobe_zscores(
        z_scores=z_scores,
        p_values=p_values,
        title="",
        save_prefix=os.path.join(out_dir, out_prefix)
    )

    print(result_df)
    print(f"Saved results to: {result_csv}")
    print(f"Saved surrogates to: {surrogate_csv}")

    return result_df, surrogates


# ============================================================
# 2. Load coordinates
# ============================================================
print("1. Loading ROI centroid coordinates...")
coords_df = pd.read_csv(coords_path, delim_whitespace=True, header=None)
coords = coords_df.iloc[:, :3].values.astype(float)

if coords.shape[0] != 160:
    raise ValueError(f"Expected 160 ROI coordinates, got {coords.shape[0]}")

print("2. Computing 160x160 Euclidean distance matrix...")
distance_matrix = squareform(pdist(coords, metric='euclidean'))


# ============================================================
# 3. Load tau data and compute empirical ROI map
# ============================================================
print("3. Loading longitudinal tau data...")
df = pd.read_excel(excel_path)

node_cols = [col for col in df.columns if 'Node' in str(col)]
node_cols = sorted(node_cols, key=extract_node_num)

if len(node_cols) != 160:
    raise ValueError(f"Expected 160 node columns, got {len(node_cols)}")

empirical_map_raw, valid_count = annualized_delta_tau_map(df, node_cols)
print(f"Computed empirical ROI-level annualized delta tau map from {valid_count} subjects.")

empirical_map_centered = empirical_map_raw - np.mean(empirical_map_raw)


# ============================================================
# 4. Load lobe labels
# ============================================================
print("4. Loading lobe labels...")
lobe_df = pd.read_csv(lobe_path, header=0)
lobe_index = lobe_df.iloc[:, 0].values

if len(lobe_index) != 160:
    raise ValueError(f"Expected 160 lobe labels, got {len(lobe_index)}")


# ============================================================
# 5. Run two BrainSMASH experiments
# ============================================================
raw_df, raw_surrogates = run_brainsmash_experiment(
    empirical_map=empirical_map_raw,
    distance_matrix=distance_matrix,
    lobe_index=lobe_index,
    label="Raw map",
    out_prefix="BrainSMASH_raw_map_7lobes"
)

centered_df, centered_surrogates = run_brainsmash_experiment(
    empirical_map=empirical_map_centered,
    distance_matrix=distance_matrix,
    lobe_index=lobe_index,
    label="Centered map",
    out_prefix="BrainSMASH_centered_map_7lobes"
)

# ============================================================
# 6. Mean Tau Concentration 
# ============================================================
plot_empirical_raw_tau(
    df=df, 
    node_cols=node_cols, 
    lobe_index=lobe_index, 
    num_lobes=num_lobes, 
    lobe_names=lobe_names, 
    lobe_colors=lobe_colors, 
    out_dir=out_dir
)

print("\nDone.")
print("Interpretation:")
print("- Raw map: tests whether observed lobe-level tau progression is elevated relative to geometry-preserving nulls.")
print("- Centered map: tests which lobes are relatively more/less prominent after removing the global mean.")
print("- Empirical RAW map: shows the baseline cross-sectional Mean Tau SUVR ± SEM per lobe across all subjects.")
