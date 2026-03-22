import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import argparse
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ============================================================
# 1. Reproducibility & Graph Utils
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

parser = argparse.ArgumentParser(description="Train PINN for regression")
parser.add_argument("--sample", type=int, default=300, help="Sample")
parser.add_argument("--ratio", type=float, default=0.4, help="Ratio of SC/FC")
parser.add_argument("--decay", type=float, default=1e-7, help="Decay")
parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
args = parser.parse_args()

def normalize_graph_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
    A = torch.clamp(adj_matrix, min=0.0)
    A = 0.5 * (A + A.t())
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    A = A + I
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D.clamp(min=1e-12), -0.5)
    D_inv_sqrt_mat = torch.diag(D_inv_sqrt)
    L = I - D_inv_sqrt_mat @ A @ D_inv_sqrt_mat
    L = 0.5 * (L + L.t())
    eigenvalues = torch.linalg.eigvalsh(L)
    max_eig = torch.max(eigenvalues)
    if max_eig > 1e-5:
        L = L / max_eig
        
    return L

def make_sparse_symmetric_graph(n_nodes: int, density: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n_nodes, n_nodes, generator=g).abs()
    mask = (torch.rand(n_nodes, n_nodes, generator=g) < density).float()
    W = W * mask
    W = torch.triu(W, diagonal=1)
    return W + W.t()

# ============================================================
# 2. Synthetic Data Configuration
# ============================================================
@dataclass
class SyntheticConfig:
    n_subjects: int = args.sample  
    n_nodes: int = 160
    n_seed_rois: int = 4
    total_time: float = 1.0
    dt: float = 0.1
    obs_noise: float = 0.05
    init_noise: float = 0.02
    rho_s: float = args.ratio 
    c_s: float = 0.3
    c_f: float = 0.3
    lambda_s: float = 0.12
    lambda_f: float = 0.12
    coupling_rank: int = 5
    seed: int = 42

def euler_integrate_coupled(us0, uf0, Ls, Lf, Ms, Mf, c_s, c_f, lambda_s, lambda_f, total_time, dt):
    steps = int(round(total_time / dt))
    us, uf = us0.clone(), uf0.clone()
    for _ in range(steps):
        dus = -c_s * (Ls @ us) + lambda_s * (Ms @ uf)
        duf = -c_f * (Lf @ uf) + lambda_f * (Mf @ us)
        us, uf = us + dt * dus, uf + dt * duf
    return us, uf

class SyntheticTauDataset(Dataset):
    def __init__(self, x0, x1, usT, ufT, us0, uf0):
        self.x0, self.x1 = x0, x1
        self.usT, self.ufT = usT, ufT
        self.us0, self.uf0 = us0, uf0
    def __len__(self): return self.x0.shape[0]
    def __getitem__(self, idx):
        return {"x0": self.x0[idx], "x1": self.x1[idx], "usT": self.usT[idx], "ufT": self.ufT[idx], "us0": self.us0[idx], "uf0": self.uf0[idx]}

class GroundTruthEmissionMLP(nn.Module):
    def __init__(self, n_nodes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes, hidden_dim),
            nn.Tanh(), 
            nn.Linear(hidden_dim, n_nodes),
            nn.Softplus()
        )
        
        self.net[0].weight.data.copy_(torch.eye(hidden_dim, n_nodes))
        
        self.net[2].weight.data.copy_(torch.eye(n_nodes, hidden_dim))

    def forward(self, x):
        return self.net(x)

def generate_synthetic_dataset(Ls, Lf, config):
    set_seed(config.seed)
    n = config.n_nodes
    g = torch.Generator().manual_seed(config.seed)
    Ms_gt = (torch.randn(n, config.coupling_rank, generator=g)*0.08) @ (torch.randn(n, config.coupling_rank, generator=g)*0.08).t()
    Mf_gt = (torch.randn(n, config.coupling_rank, generator=g)*0.08) @ (torch.randn(n, config.coupling_rank, generator=g)*0.08).t()
    
    gt_emission = GroundTruthEmissionMLP(n)
    gt_emission.eval() 
    
    x0_list, x1_list, us0_list, uf0_list, usT_list, ufT_list = [], [], [], [], [], []
    for _ in range(config.n_subjects):
        x0 = torch.zeros(n)
        x0[np.random.choice(n, size=config.n_seed_rois, replace=False)] = torch.rand(config.n_seed_rois) * 0.8 + 0.4
        x0 = (x0 + 0.05 * torch.rand(n)).clamp(min=0.0)

        us0 = (config.rho_s * x0 + config.init_noise * torch.randn(n)).clamp(min=0.0)
        uf0 = ((1.0 - config.rho_s) * x0 + config.init_noise * torch.randn(n)).clamp(min=0.0)

        usT, ufT = euler_integrate_coupled(us0, uf0, Ls, Lf, Ms_gt, Mf_gt, config.c_s, config.c_f, config.lambda_s, config.lambda_f, config.total_time, config.dt)
        
        clean_x1 = gt_emission(usT + ufT).detach()
        heterogeneous_noise_scale = torch.rand(n) * config.obs_noise + 0.01
        x1 = (clean_x1 + heterogeneous_noise_scale * torch.randn(n)).clamp(min=0.0)
        
        x0_list.append(x0); x1_list.append(x1); us0_list.append(us0); uf0_list.append(uf0); usT_list.append(usT); ufT_list.append(ufT)

    dataset = SyntheticTauDataset(torch.stack(x0_list), torch.stack(x1_list), torch.stack(usT_list), torch.stack(ufT_list), torch.stack(us0_list), torch.stack(uf0_list))
    return dataset

# ============================================================
# 3. Model Definition
# ============================================================
class GatedInitializer(nn.Module):
    def __init__(self, n_nodes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_nodes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_nodes))
    def forward(self, x0):
        gate = torch.sigmoid(self.net(x0))
        return gate * x0, (1.0 - gate) * x0, gate

class CoupledTauModel(nn.Module):
    def __init__(self, n_nodes, rank=5, hidden_dim=128, total_time=1.0, dt=0.1):
        super().__init__()
        self.total_time, self.dt = total_time, dt
        self.init_split = GatedInitializer(n_nodes, hidden_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(n_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_nodes),
            nn.Softplus() 
        )
        
        self.Ms_A, self.Ms_B = nn.Parameter(0.05 * torch.randn(n_nodes, rank)), nn.Parameter(0.05 * torch.randn(n_nodes, rank))
        self.Mf_A, self.Mf_B = nn.Parameter(0.05 * torch.randn(n_nodes, rank)), nn.Parameter(0.05 * torch.randn(n_nodes, rank))
        self.raw_cs, self.raw_cf = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_s, self.raw_lambda_f = nn.Parameter(torch.tensor(-1.0)), nn.Parameter(torch.tensor(-1.0))

    def forward(self, x0, Ls, Lf):
        us, uf, gate = self.init_split(x0)
        c_s, c_f = F.softplus(self.raw_cs) + 1e-4, F.softplus(self.raw_cf) + 1e-4
        l_s, l_f = F.softplus(self.raw_lambda_s) + 1e-4, F.softplus(self.raw_lambda_f) + 1e-4
        Ms, Mf = self.Ms_A @ self.Ms_B.t(), self.Mf_A @ self.Mf_B.t()
        
        for _ in range(int(round(self.total_time / self.dt))):
            dus = -c_s * (us @ Ls.t()) + l_s * (uf @ Ms.t())
            duf = -c_f * (uf @ Lf.t()) + l_f * (us @ Mf.t())
            us, uf = (us + self.dt * dus).clamp(min=0.0), (uf + self.dt * duf).clamp(min=0.0)
            
        latent_sum = us + uf
        x1_hat = self.decoder(latent_sum)
        
        return {"x1_hat": x1_hat, "usT_hat": us, "ufT_hat": uf}

# ============================================================
# 4. Global Attribution Metrics & Training
# ============================================================
def corrcoef_torch(x, y):
    vx, vy = x.flatten() - x.mean(), y.flatten() - y.mean()
    return ((vx * vy).sum() / torch.sqrt((vx.pow(2).sum() + 1e-8) * (vy.pow(2).sum() + 1e-8))).item()

def compute_metrics(batch, outputs):
    gt_s_sum = batch["usT"].sum().clamp(min=1e-8)
    gt_f_sum = batch["ufT"].sum().clamp(min=1e-8)
    pred_s_sum = outputs["usT_hat"].sum().clamp(min=1e-8)
    pred_f_sum = outputs["ufT_hat"].sum().clamp(min=1e-8)
    
    gt_sc_weight = (gt_s_sum / (gt_s_sum + gt_f_sum)).item()
    pred_sc_weight = (pred_s_sum / (pred_s_sum + pred_f_sum)).item()

    return {
        "us_corr": corrcoef_torch(outputs["usT_hat"].detach(), batch["usT"]),
        "uf_corr": corrcoef_torch(outputs["ufT_hat"].detach(), batch["ufT"]),
        "gt_sc_weight": gt_sc_weight,
        "pred_sc_weight": pred_sc_weight,
    }

def train_eval_split(dataset, train_idx, val_idx, Ls, Lf, n_epochs=150, device="cpu"):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=len(val_idx), shuffle=False)
    
    model = CoupledTauModel(n_nodes=160).to(device)
    Ls, Lf = Ls.to(device), Lf.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x0, x1 = batch["x0"].to(device), batch["x1"].to(device)
            out = model(x0, Ls, Lf)
            loss = F.mse_loss(out["x1_hat"], x1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        v_batch = next(iter(val_loader))
        out_val = model(v_batch["x0"].to(device), Ls, Lf)
        v_batch_device = {k: v.to(device) for k, v in v_batch.items()}
        return compute_metrics(v_batch_device, out_val)

def run_5fold_cv(dataset, Ls, Lf, device="cpu"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    print(f"\n--- Starting 5-Fold Cross Validation (N={len(dataset)}) ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        metrics = train_eval_split(dataset, train_idx, val_idx, Ls, Lf, n_epochs=120, device=device)
        all_metrics.append(metrics)
        print(f"  Fold {fold+1} | SC Recov: {metrics['pred_sc_weight']:.3f} (GT:{metrics['gt_sc_weight']:.3f}) | Corr: {metrics['us_corr']:.3f}")
    
    summary = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        summary[key] = {"mean": np.mean(vals), "std": np.std(vals)}
    return summary

# ============================================================
# 5. Execution & Plotting
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = "/ram/USERS/bendan/ACMLab_DATA/ADNI/ADNI_tau_FC_SC/group_average_results/group_mean_SC_FC_torch.pt"
    # Ws = make_sparse_symmetric_graph(160, density=0.08, seed=1)
    # Wf = make_sparse_symmetric_graph(160, density=0.25, seed=2)
    # Ls, Lf = normalize_graph_matrix(Ws), normalize_graph_matrix(Wf)

    data = torch.load(data_path, map_location="cpu")
    Ws = data["Ws"].float()
    Wf = data["Wf"].float()
    Ls = data["Ls"].float()
    Lf = data["Lf"].float()

    n_nodes = Ws.shape[0]
    config = SyntheticConfig()
    dataset = generate_synthetic_dataset(Ls, Lf, config)
    cv_summary = run_5fold_cv(dataset, Ls, Lf, device=device)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 5), dpi=300)

    gt_sc, gt_fc = cv_summary['gt_sc_weight']['mean'], 1 - cv_summary['gt_sc_weight']['mean']
    pred_sc, pred_fc = cv_summary['pred_sc_weight']['mean'], 1 - cv_summary['pred_sc_weight']['mean']
    pred_sc_std, pred_fc_std = cv_summary['pred_sc_weight']['std'], cv_summary['pred_sc_weight']['std']

    bar_width = 0.35
    x_pos = np.arange(2)
    
    ax1.bar(x_pos - bar_width/2, [gt_sc, gt_fc], bar_width, label='Ground Truth', color='#D3D3D3', edgecolor='black', hatch='//')
    ax1.bar(x_pos + bar_width/2, [pred_sc, pred_fc], bar_width, yerr=[pred_sc_std, pred_fc_std], 
            label='Model Recovered', color=[ '#4dbcd5','#e64a35'], capsize=5, edgecolor='black')
    
    ax1.set_ylabel('Global Attribution Weight', fontsize=12, fontweight='bold')
    ax1.set_title('A. Modality Attribution Recovery', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['SC Contribution ($u_s$)', 'FC Contribution ($u_f$)'], fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right')

    for i, (val, std) in enumerate(zip([pred_sc, pred_fc], [pred_sc_std, pred_fc_std])):
        ax1.text(x_pos[i] + bar_width/2, val + 0.05, f'{val:.3f}', ha='center', fontweight='bold')

    corrs = [cv_summary['us_corr']['mean'], cv_summary['uf_corr']['mean']]
    stds = [cv_summary['us_corr']['std'], cv_summary['uf_corr']['std']]
    
    ax2.bar(x_pos, corrs, yerr=stds, color=['#4dbcd5','#e64a35'], width=0.5, capsize=5, edgecolor='black', alpha=0.85)
    
    ax2.set_ylabel('Pearson Correlation (r)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Spatial Pattern Disentanglement', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['SC Pattern', 'FC Pattern'], fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.1)

    for i, val in enumerate(corrs):
        ax2.text(x_pos[i], val + 0.05, f'{val:.3f}', ha='center', fontweight='bold')

    plt.tight_layout()
    
    plot_filename = f'Fig_S_Synthetic_Validation_SC{args.ratio}_{args.sample}_{args.lr}_{args.decay}.svg'
    plt.savefig(plot_filename, format='svg', bbox_inches='tight',transparent=True)
    
