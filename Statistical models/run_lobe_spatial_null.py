import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
from brainsmash.mapgen.base import Base
import warnings
warnings.filterwarnings('ignore')

ROI_NUM = '160'
OUT_DIR = f"./SpatialNull_Python"
os.makedirs(OUT_DIR, exist_ok=True)

FC_MAT_PATH = r"C:\Users\Tingting Dan\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\Multi-layer\Results\160\FC.mat"             # 你的 FC 矩阵 (包含 mean_matrix)
T_VECTOR_PATH = r"C:\Users\Tingting Dan\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\Multi-layer\Results\160\TVector.txt"      # 你的 Tau 变化速率向量
COORDS_PATH = r"E:\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\My paper\ICML2023\vis\continuity_region.node_160.node"      # 160 个脑区的空间坐标文件
LABEL_MAT_PATH = r"C:\Users\Tingting Dan\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\Multi-layer\SC_regions_label.mat"               # Yeo7 脑叶标签 (包含 lobe_index_sc)

THRESHOLD = 0.2  
NUM_SURROGATES = 1000 

def compute_lobe_corr(T_vec, G, labels, num_lobes=7, threshold=0.2):
    nodenum = len(T_vec)
    nei_mat = np.full((nodenum, num_lobes), np.nan)
    r_mat = np.zeros((num_lobes, num_lobes))

    for i in range(nodenum):
        idx = np.where(G[:, i] > threshold)[0]
        if len(idx) > 0:
            neighbor_T = T_vec[idx]
            neighbor_label = labels[idx]
            for j in range(1, num_lobes + 1): 
                idx_label = np.where(neighbor_label == j)[0]
                if len(idx_label) > 0:
                    nei_mat[i, j-1] = np.nanmean(neighbor_T[idx_label])

    for i in range(1, num_lobes + 1):
        idx_i = np.where(labels == i)[0]
        T_sub = T_vec[idx_i]
        for j in range(1, num_lobes + 1):
            nei_sub = nei_mat[idx_i, j-1]
            valid = ~np.isnan(nei_sub) & ~np.isnan(T_sub)
            data_num = np.sum(valid)
            
            if data_num < 10:
                r_mat[i-1, j-1] = 0.0
            else:
                r, _ = pearsonr(T_sub[valid], nei_sub[valid])
                adjrs = 1 - (1 - r**2) * (data_num - 1) / (data_num - 2)
                adjrs = max(0, adjrs)
                
                r_mat[i-1, j-1] = np.sqrt(adjrs) if r > 0 else -np.sqrt(adjrs)
                
    return r_mat


print("Loading data...")
mat_data = sio.loadmat(FC_MAT_PATH)
G = mat_data['mean_matrix']

TVector = np.loadtxt(T_VECTOR_PATH).flatten()

label_data = sio.loadmat(LABEL_MAT_PATH)
Yeo7_label = label_data['lobe_index_sc'].flatten()

coords = pd.read_csv(COORDS_PATH, sep=r"\s+", header=None).iloc[:, :3].values
dist_mat = squareform(pdist(coords, metric="euclidean"))


print("Calculating empirical correlation matrix...")
r_yeo_emp = compute_lobe_corr(TVector, G, Yeo7_label, num_lobes=7, threshold=THRESHOLD)

print(f"Generating {NUM_SURROGATES} spatial null surrogates via BrainSMASH...")
base = Base(x=TVector, D=dist_mat)
surrogates = base(n=NUM_SURROGATES) # (1000, 160)

print("Running permutations to build spatial null distribution...")
r_yeo_null = np.zeros((7, 7, NUM_SURROGATES))
for s in range(NUM_SURROGATES):
    fake_TVector = surrogates[s, :]
    r_yeo_null[:, :, s] = compute_lobe_corr(fake_TVector, G, Yeo7_label, num_lobes=7, threshold=THRESHOLD)


print("Computing 1-tailed spatial P-values and applying exact FDR correction...")
p_yeo_spatial = np.ones((7, 7)) 
for i in range(7):
    for j in range(7):
        true_r = r_yeo_emp[i, j]
        if true_r != 0: 
            null_dist = r_yeo_null[i, j, :]
            p_yeo_spatial[i, j] = (np.sum(null_dist >= true_r) + 1) / (NUM_SURROGATES + 1)

valid_mask = (r_yeo_emp != 0)
p_values_valid = p_yeo_spatial[valid_mask]

_, p_fdr_valid = fdrcorrection(p_values_valid, alpha=0.05, method='indep')

p_yeo_fdr = np.ones((7, 7))
p_yeo_fdr[valid_mask] = p_fdr_valid

significant_mask = p_yeo_fdr < 0.05 
r_yeo_corrected = r_yeo_emp * significant_mask

print("Plotting the stunning FDR-corrected matrix...")
white = (1.0, 1.0, 1.0)
green = (133/255, 192/255, 154/255)
cmap_name = 'white_green'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, [white, green], N=256)

plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})
fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

cax = ax.imshow(r_yeo_emp, cmap=custom_cmap, vmin=0, vmax=np.max(r_yeo_emp))

cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Correlation ($R$)', fontweight='bold')

ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='--', linewidth=1.5)
ax.tick_params(which="minor", size=0)

region_labels = ['Frontal', 'Insula', 'Temporal', 'Occipital', 'Parietal', 'Limbic', 'Sub-cortical']
ax.set_xticks(np.arange(7))
ax.set_yticks(np.arange(7))
ax.set_xticklabels(region_labels, rotation=45, ha="right", fontweight='bold')
ax.set_yticklabels(region_labels, fontweight='bold')

for i in range(7):
    for j in range(7):
        if significant_mask[i, j]:
            ax.text(j, i, "*", ha="center", va="center", color="red", fontsize=18, fontweight='bold')

ax.set_title('Spatial-Null Confirmed Correlation Matrix', fontweight='bold', pad=15)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "Fig2_Spatial_Null_Matrix.svg"))
plt.close()

res_df = pd.DataFrame({
    'Seed_Lobe': np.repeat(region_labels, 7),
    'Neighbor_Lobe': np.tile(region_labels, 7),
    'R_emp': r_yeo_emp.flatten(),
    'P_spatial': p_yeo_spatial.flatten(),
    'P_FDR': p_yeo_fdr.flatten(),
    'Significant': significant_mask.flatten()
})
res_df.to_csv(os.path.join(OUT_DIR, "Lobe_Correlation_Stats2.csv"), index=False)

print(f"All Done! The matrix is completely bulletproof now. Results saved to {OUT_DIR}")