import os
import re
import numpy as np
import pandas as pd
import torch
import bct  # pip install bctpy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from brainsmash.mapgen.base import Base
from scipy.spatial.distance import pdist, squareform

# ============================================================
# 1. 路径设置 (请替换为你的实际路径)
# ============================================================
# 直接读取你刚刚生成的极其方便的 .pt 文件！
pt_file_path = r"C:\Users\Tingting Dan\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\Multi-layer\Statistical models\group_mean_SC_FC_torch.pt"

# 计算 BrainSMASH 和 提取 Delta Tau 用的数据
coords_path = r"E:\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\My paper\ICML2023\vis\continuity_region.node_160.node"
excel_path = r"E:\ACMLab_Data\ADNI-data_tau_amyloid_FDG_CT_info\amyloid_tau_FDG_CT\Tau_SUVR_Swapped_update_age.xlsx"
network_label_path = r"C:\Users\Tingting Dan\OneDrive - University of North Carolina at Chapel Hill\Tingting_Dan\UNC_Work\Multi-layer\region_functional_mapped.csv"

out_dir = r".\Topology_Results"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 2. 数据准备与预处理函数
# ============================================================
def extract_node_num(col):
    m = re.search(r'Node[_ ]?(\d+)', str(col))
    return int(m.group(1)) if m else 10**9

def get_empirical_delta_tau(excel_path):
    print("Extracting empirical Delta Tau from excel...")
    df = pd.read_excel(excel_path)
    node_cols = sorted([col for col in df.columns if 'Node' in str(col)], key=extract_node_num)
    
    tau_avg_map = np.zeros(len(node_cols), dtype=float)
    valid_count = 0
    
    for ptid, group in df.groupby('PTID'):
        if len(group) < 2: continue
        group = group.sort_values('EXAMDATE').iloc[:2]
        tau_t1 = group.iloc[0][node_cols].values.astype(float)
        tau_t2 = group.iloc[1][node_cols].values.astype(float)
        age_t1, age_t2 = float(group.iloc[0]['AGE']), float(group.iloc[1]['AGE'])
        
        if np.isclose(age_t2 - age_t1, 0): continue
        delta_tau = (tau_t2 - tau_t1) / (age_t2 - age_t1)
        if np.any(np.isnan(delta_tau)): continue
        
        tau_avg_map += delta_tau
        valid_count += 1
        
    return tau_avg_map / valid_count

def load_data():
    # 1. 直接从 .pt 文件加载矩阵！
    print("Loading SC and FC matrices from .pt file...")
    pt_data = torch.load(pt_file_path)
    
    # 提取 Ws 和 Wf 并转为 numpy 数组 (bctpy 需要 numpy)
    SC = pt_data['Ws'].numpy()
    FC = pt_data['Wf'].numpy()
    
    # 为了拓扑计算更稳定，我们对 FC 取绝对值并截断负连接，SC 本身是正的
    FC = np.clip(FC, a_min=0, a_max=None)
    np.fill_diagonal(SC, 0)
    np.fill_diagonal(FC, 0)
    
    # 2. 坐标和距离矩阵
    coords_df = pd.read_csv(coords_path, delim_whitespace=True, header=None)
    coords = coords_df.iloc[:, :3].values.astype(float)
    dist_mat = squareform(pdist(coords, metric='euclidean'))
    
    # 3. 网络标签 (1-13)，处理缺口 10
    lobe_df = pd.read_csv(network_label_path, header=None)
    network_labels = lobe_df.iloc[:, 0].values.astype(int)
    network_labels[network_labels == 11] = 10
    network_labels[network_labels == 12] = 11
    network_labels[network_labels == 13] = 12
    network_labels[network_labels == 14] = 13
    
    # 4. 经验 Tau 变化率
    tau_map = get_empirical_delta_tau(excel_path)
    
    return SC, FC, dist_mat, network_labels, tau_map

# ============================================================
# 3. 核心图论拓扑计算
# ============================================================
def compute_graph_metrics(adj_matrix, community_labels):
    # 1. Clustering Coefficient (Segregation)
    CC = bct.clustering_coef_wu(adj_matrix)
    
    # 2. Betweenness Centrality (Integration)
    # 将权重转为距离 (距离越短，连接越强)
    dist_matrix = bct.weight_conversion(adj_matrix, 'lengths')
    BC = bct.betweenness_wei(dist_matrix)
    
    # 3. Participation Coefficient (Mesoscale Hub)
    PC = bct.participation_coef(adj_matrix, community_labels)
    
    return CC, BC, PC

# ============================================================
# 4. BrainSMASH 零模型相关性计算
# ============================================================
def spatial_correlation_with_null(metric_map, tau_map, surrogates):
    true_r, _ = pearsonr(metric_map, tau_map)
    null_rs = np.zeros(surrogates.shape[0])
    
    for i in range(surrogates.shape[0]):
        null_rs[i], _ = pearsonr(metric_map, surrogates[i, :])
        
    p_null = (np.sum(np.abs(null_rs) >= np.abs(true_r)) + 1) / (len(null_rs) + 1)
    return true_r, p_null

# ============================================================
# 5. 可视化绘图函数
# ============================================================
def plot_correlation(metric_map, tau_map, metric_name, network_name, r_val, p_null, out_path):
    plt.figure(figsize=(4.5, 4), dpi=300)
    
    # 散点颜色区分 SC 和 FC
    color = 'teal' if network_name == 'FC' else 'coral'
    
    plt.scatter(metric_map, tau_map, color=color, alpha=0.7, edgecolors='k', s=40)
    
    # 线性拟合
    m, b = np.polyfit(metric_map, tau_map, 1)
    plt.plot(metric_map, m*metric_map + b, color='red', linewidth=2, linestyle='--')
    
    plt.xlabel(f'{metric_name} ({network_name})', fontsize=12, fontweight='bold')
    plt.ylabel('Tau Accumulation Rate (\u0394Tau)', fontsize=12, fontweight='bold')
    
    # 显著性标星号颜色
    p_color = 'red' if p_null < 0.05 else 'black'
    
    plt.text(0.05, 0.85, f'$r = {r_val:.3f}$\n$p_{{null}} = {p_null:.4f}$', 
             transform=plt.gca().transAxes, fontsize=12, color=p_color, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================================================
# 6. 主程序运行
# ============================================================
print("\n--- Starting Topology Analysis ---")
SC, FC, dist_mat, network_labels, tau_map = load_data()

print("\n1. Computing Graph Metrics for 160 ROIs...")
CC_sc, BC_sc, PC_sc = compute_graph_metrics(SC, network_labels)
CC_fc, BC_fc, PC_fc = compute_graph_metrics(FC, network_labels)

print("\n2. Generating 1000 BrainSMASH Spatial Null Surrogates...")
base = Base(x=tau_map, D=dist_mat)
surrogates = base(n=1000)

print("\n3. Testing Spatial Correlations...")
results = []
metrics = [
    ('Clustering_Coef', CC_sc, CC_fc),
    ('Betweenness', BC_sc, BC_fc),
    ('Participation_Coef', PC_sc, PC_fc)
]

for name, sc_val, fc_val in metrics:
    r_sc, p_sc = spatial_correlation_with_null(sc_val, tau_map, surrogates)
    r_fc, p_fc = spatial_correlation_with_null(fc_val, tau_map, surrogates)
    
    results.append({'Metric': name, 'Network': 'SC', 'r': r_sc, 'p_null': p_sc})
    results.append({'Metric': name, 'Network': 'FC', 'r': r_fc, 'p_null': p_fc})
    
    print(f"{name:18s} | SC: r={r_sc:6.3f}, p_null={p_sc:.4f}  |  FC: r={r_fc:6.3f}, p_null={p_fc:.4f}")
    
    # 画图
    plot_correlation(sc_val, tau_map, name.replace('_', ' '), 'SC', r_sc, p_sc, os.path.join(out_dir, f"{name}_SC_Tau.pdf"))
    plot_correlation(fc_val, tau_map, name.replace('_', ' '), 'FC', r_fc, p_fc, os.path.join(out_dir, f"{name}_FC_Tau.pdf"))

# 保存结果表
df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(out_dir, "Topology_Tau_Correlation_Results.csv"), index=False)
print(f"\n✅ All done! Results and figures saved to: {out_dir}")