import os
import re
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bct  
from scipy.stats import pearsonr, wilcoxon
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
warnings.filterwarnings("ignore")

# ============================================================
# 0. USER CONFIG & THEME SETTINGS
# ============================================================
SC_DIR = "SC"
FC_DIR = "FC"
TAU_EXCEL = "Tau.xlsx"

COORDS_PATH = "continuity_region.node_160.node"
MODULE_LABELS_CSV = "region_functional_mapped.csv"

OUT_DIR = "./Topology_Final_Results"
os.makedirs(OUT_DIR, exist_ok=True)

N_NODES = 160
N_SURROGATES = 1000

COLOR_SC = '#4dbcd5' 
COLOR_FC = '#e64a35' 

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

def get_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

# ============================================================
# 1. HELPERS & TAU EXTRACTION
# ============================================================
def normalize_subject_id(x): return str(x).strip().upper()
def extract_node_num(col):
    m = re.search(r"Node[_ ]?(\d+)", str(col))
    return int(m.group(1)) if m else 10**9

def subject_id_from_filename(path: Path) -> str:
    stem = path.stem
    m = re.search(r"(\d+_S_\d+)", stem, flags=re.IGNORECASE)
    if m: return normalize_subject_id(m.group(1))
    return normalize_subject_id(stem.split("_")[0])

def load_numeric_csv(path: str) -> np.ndarray:
    for header in [None, 0]:
        try:
            df = pd.read_csv(path, header=header)
            df_num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
            arr = df_num.to_numpy(dtype=float)
            if arr.size > 0: return arr
        except Exception: pass
    try:
        df = pd.read_csv(path, index_col=0)
        arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if arr.size > 0: return arr
    except Exception: pass
    raise ValueError(f"Could not parse numeric csv: {path}")

def load_matrix_160(path: str) -> np.ndarray:
    arr = load_numeric_csv(path)
    if arr.shape[0] >= N_NODES and arr.shape[1] >= N_NODES: arr = arr[:N_NODES, :N_NODES]
    return arr

def load_modules(path: str) -> np.ndarray:
    df = pd.read_csv(path, header=0)
    vals = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=int)
    remap = {11: 10, 12: 11, 13: 12, 14: 13}
    vals = np.array([remap.get(int(x), int(x)) for x in vals], dtype=int)
    return vals

def preprocess_matrix(W: np.ndarray, is_sc: bool = True) -> np.ndarray:
    W = np.asarray(W, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    if is_sc:
        W[W < 0] = 0.0
        W = np.log1p(W)
    else:
        W = np.clip(W, a_min=0, a_max=None)
    mx = np.max(W)
    if mx > 0: W = W / mx
    return W

def load_coords(path: str) -> np.ndarray:
    coords_df = pd.read_csv(path, sep=r"\s+", header=None)
    return coords_df.iloc[:, :3].values.astype(float)

def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = prev
    q_corrected = np.empty(n, dtype=float)
    q_corrected[order] = np.minimum(q, 1.0)
    return q_corrected

def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12: return np.nan, np.nan
    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p)

def build_subject_tau_from_excel(excel_path: str) -> Dict[str, np.ndarray]:
    df = pd.read_excel(excel_path)
    node_cols = sorted([c for c in df.columns if "Node" in str(c)], key=extract_node_num)
    subject_tau = {}
    for ptid, group in df.groupby("PTID"):
        sid = normalize_subject_id(ptid)
        if len(group) < 2: continue
        group = group.sort_values("EXAMDATE")
        g = group.iloc[[0, -1]]
        tau_1, tau_2 = g.iloc[0][node_cols].values.astype(float), g.iloc[1][node_cols].values.astype(float)
        age_1, age_2 = float(g.iloc[0]["AGE"]), float(g.iloc[1]["AGE"])
        if not np.isfinite(age_1) or not np.isfinite(age_2) or age_2 <= age_1: continue
        dtau = (tau_2 - tau_1) / (age_2 - age_1)
        if not np.any(np.isnan(dtau)): subject_tau[sid] = dtau
    return subject_tau

# ============================================================
# 2. TOPOLOGY & BRAINSMASH
# ============================================================
def nodal_efficiency_weighted(W):
    n = W.shape[0]
    D = np.full_like(W, np.inf, dtype=float)
    pos = W > 0
    D[pos] = 1.0 / W[pos]
    np.fill_diagonal(D, 0.0)
    sp = dijkstra(D, directed=False, unweighted=False)
    eff = np.zeros(n, dtype=float)
    for i in range(n):
        d = sp[i].copy()
        valid = np.isfinite(d) & (d > 0)
        if np.any(valid): eff[i] = np.mean(1.0 / d[valid])
    return eff

def compute_topology(W: np.ndarray, modules: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "strength": np.asarray(np.sum(W, axis=1), dtype=float),
        "clustering": np.asarray(bct.clustering_coef_wu(W), dtype=float),
        "efficiency": np.asarray(nodal_efficiency_weighted(W), dtype=float),
        "participation": np.asarray(bct.participation_coef(W, modules), dtype=float),
    }

def run_brainsmash_test(metric_map, tau_map, dist_mat, n_surrogates=1000):
    from brainsmash.mapgen.base import Base
    base = Base(x=tau_map, D=dist_mat)
    surrogates = base(n=n_surrogates)
    true_r, _ = safe_corr(metric_map, tau_map)
    null_rs = np.array([safe_corr(metric_map, s)[0] for s in surrogates], dtype=float)
    null_rs = null_rs[np.isfinite(null_rs)]
    p_null = (np.sum(np.abs(null_rs) >= np.abs(true_r)) + 1) / (len(null_rs) + 1)
    return float(true_r), float(p_null)

def fit_ols(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    if len(y) < X.shape[1] + 5: return {"r2": np.nan}
    mu, sd = np.mean(X, axis=0), np.std(X, axis=0)
    sd[sd<1e-12]=1.0
    Xz = (X - mu) / sd
    Xd = np.column_stack([np.ones(len(y)), Xz])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yhat = Xd @ beta
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {"r2": 1 - (np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else np.nan}

# ============================================================
# 3. MAIN WORKFLOW
# ============================================================
print("\n--- Starting Topology Analysis Beyond Strength ---")
modules = load_modules(MODULE_LABELS_CSV)
coords = load_coords(COORDS_PATH)
dist_mat = squareform(pdist(coords, metric="euclidean"))
subject_tau = build_subject_tau_from_excel(TAU_EXCEL)

sc_files = {subject_id_from_filename(p): str(p) for p in Path(SC_DIR).glob("*.csv")}
fc_files = {subject_id_from_filename(p): str(p) for p in Path(FC_DIR).glob("*.csv")}
common_subs = sorted(set(sc_files.keys()) & set(fc_files.keys()) & set(subject_tau.keys()))
print(f"Matched {len(common_subs)} subjects.")

metric_names = ["strength", "clustering", "efficiency", "participation"]
subject_r2_rows, subject_ablation_rows = [], []
all_sc_mats, all_fc_mats = [], []

for sid in common_subs:
    try:
        Wsc = preprocess_matrix(load_matrix_160(sc_files[sid]), is_sc=True)
        Wfc = preprocess_matrix(load_matrix_160(fc_files[sid]), is_sc=False)
        all_sc_mats.append(Wsc); all_fc_mats.append(Wfc)
        tau_y = subject_tau[sid]
        topo_sc, topo_fc = compute_topology(Wsc, modules), compute_topology(Wfc, modules)
        
        for modality, topo_dict in [("SC", topo_sc), ("FC", topo_fc)]:
            r2_strength = fit_ols(tau_y, np.column_stack([topo_dict["strength"]]))["r2"]
            X_full = np.column_stack([topo_dict["strength"], topo_dict["clustering"], topo_dict["efficiency"], topo_dict["participation"]])
            r2_full = fit_ols(tau_y, X_full)["r2"]
            subject_r2_rows.append({"Subject": sid, "Modality": modality, "Delta_R2": r2_full - r2_strength})
            
            for em in ["clustering", "efficiency", "participation"]:
                r2_pair = fit_ols(tau_y, np.column_stack([topo_dict["strength"], topo_dict[em]]))["r2"]
                subject_ablation_rows.append({"Subject": sid, "Modality": modality, "ExtraMetric": em, "Delta_R2_pair": r2_pair - r2_strength})
    except Exception as e:
        pass

r2_df = pd.DataFrame(subject_r2_rows)
ablation_df = pd.DataFrame(subject_ablation_rows)

print("Running Group-level BrainSMASH...")
sc_group = np.mean(np.stack(all_sc_mats, axis=0), axis=0)
fc_group = np.mean(np.stack(all_fc_mats, axis=0), axis=0)
tau_group = np.mean(np.stack([subject_tau[sid] for sid in common_subs], axis=0), axis=0)

topo_sc_group, topo_fc_group = compute_topology(sc_group, modules), compute_topology(fc_group, modules)
group_rows = []
for modality, topo_dict in [("SC", topo_sc_group), ("FC", topo_fc_group)]:
    for m in metric_names:
        r_true, p_null = run_brainsmash_test(topo_dict[m], tau_group, dist_mat, N_SURROGATES)
        group_rows.append({"Modality": modality, "Metric": m, "Group_r": r_true, "P_null": p_null})

group_df = pd.DataFrame(group_rows)
group_df["P_FDR"] = np.nan
for modality in ["SC", "FC"]:
    idx = group_df["Modality"] == modality
    group_df.loc[idx, "P_FDR"] = fdr_bh(group_df.loc[idx, "P_null"].values)

# ============================================================
# 4. PLOTTING (1x5 Grid for Appendix Figure S1)
# ============================================================
print("Generating Figure 1 (1x5 Grid)...")

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

labels = ["clustering", "efficiency", "participation"]

# ---- Subplot 1: SC Specific Contributions Barplot ----
ax = axes[0]
vals = [np.mean(ablation_df[(ablation_df["Modality"] == "SC") & (ablation_df["ExtraMetric"] == m)]["Delta_R2_pair"]) for m in labels]
errs = [np.std(ablation_df[(ablation_df["Modality"] == "SC") & (ablation_df["ExtraMetric"] == m)]["Delta_R2_pair"]) for m in labels]
ax.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=COLOR_SC, alpha=0.8, edgecolor="black")
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.capitalize() for l in labels], rotation=20)
ax.set_title("SC Specific Contributions", fontweight='bold')
ax.set_ylabel(r"$\Delta R^2$ (Strength + Metric - Strength Only)", fontweight='bold')

# ---- Subplot 2: FC Specific Contributions Barplot ----
ax = axes[1]
vals = [np.mean(ablation_df[(ablation_df["Modality"] == "FC") & (ablation_df["ExtraMetric"] == m)]["Delta_R2_pair"]) for m in labels]
errs = [np.std(ablation_df[(ablation_df["Modality"] == "FC") & (ablation_df["ExtraMetric"] == m)]["Delta_R2_pair"]) for m in labels]
ax.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=COLOR_FC, alpha=0.8, edgecolor="black")
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([l.capitalize() for l in labels], rotation=20)
ax.set_title("FC Specific Contributions", fontweight='bold')

# ---- Subplot 3: SC Macro-scale Spatial Corr Barplot ----
ax = axes[2]
df_sc = group_df[group_df["Modality"] == "SC"]
vals = df_sc["Group_r"].values
bars = ax.bar(range(len(metric_names)), vals, color=COLOR_SC, alpha=0.8, edgecolor="black")
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(range(len(metric_names)))
ax.set_xticklabels([m.capitalize() for m in metric_names], rotation=20)
ax.set_title("SC Macro-scale Spatial Corr", fontweight='bold')
ax.set_ylabel("Correlation with Group Mean $\Delta$Tau", fontweight='bold')
for i, bar in enumerate(bars):
    q = df_sc["P_FDR"].values[i]
    stars = get_stars(q)
    if stars:
        yval = bar.get_height()
        offset = np.sign(yval) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2, yval + offset, stars, ha='center', va='bottom' if yval > 0 else 'top', color='red', fontweight='bold', fontsize=16)

# ---- Subplot 4: FC Macro-scale Spatial Corr Barplot ----
ax = axes[3]
df_fc = group_df[group_df["Modality"] == "FC"]
vals = df_fc["Group_r"].values
bars = ax.bar(range(len(metric_names)), vals, color=COLOR_FC, alpha=0.8, edgecolor="black")
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticks(range(len(metric_names)))
ax.set_xticklabels([m.capitalize() for m in metric_names], rotation=20)
ax.set_title("FC Macro-scale Spatial Corr", fontweight='bold')
for i, bar in enumerate(bars):
    q = df_fc["P_FDR"].values[i]
    stars = get_stars(q)
    if stars:
        yval = bar.get_height()
        offset = np.sign(yval) * 0.02
        ax.text(bar.get_x() + bar.get_width()/2, yval + offset, stars, ha='center', va='bottom' if yval > 0 else 'top', color='black', fontweight='bold', fontsize=16)

# ---- Subplot 5: Delta R2 Boxplot (Micro-scale) ----
ax = axes[4]
sc_dr2 = r2_df[r2_df["Modality"] == "SC"]["Delta_R2"].dropna().values
fc_dr2 = r2_df[r2_df["Modality"] == "FC"]["Delta_R2"].dropna().values

bp = ax.boxplot([sc_dr2, fc_dr2], labels=["SC", "FC"], widths=0.5, patch_artist=True)
for patch, color in zip(bp['boxes'], [COLOR_SC, COLOR_FC]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in bp['medians']: median.set(color="white", linewidth=2.5)

ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_ylabel(r"$\Delta R^2$ (Full Topology - Strength Only)", fontweight="bold")
ax.set_title("Higher-Order Topology Explains\nAdditional Variance", fontweight="bold")

_, p_sc = wilcoxon(sc_dr2)
_, p_fc = wilcoxon(fc_dr2)
ax.text(1, np.max(sc_dr2)*0.9, get_stars(p_sc), ha="center", color="red", fontsize=16, fontweight='bold')
ax.text(2, np.max(fc_dr2)*0.9, get_stars(p_fc), ha="center", color="red", fontsize=16, fontweight='bold')

y_min_ab = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
y_max_ab = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(y_min_ab, y_max_ab)
axes[1].set_ylim(y_min_ab, y_max_ab)

y_min_mc = min(axes[2].get_ylim()[0], axes[3].get_ylim()[0])
y_max_mc = max(axes[2].get_ylim()[1], axes[3].get_ylim()[1])
axes[2].set_ylim(y_min_mc, y_max_mc)
axes[3].set_ylim(y_min_mc, y_max_mc)

plt.suptitle("Appendix Figure S1: Connectome Nodal Topology Contributes Beyond Nodal Strength", fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_S1_Appendix.svg"))
plt.close()

print("Figure_S1_Appendix.svg generated successfully in:", OUT_DIR)
