import os
import gc
import psutil
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from dataset import ClassifyGraphDataset
from network import MSClassifyNet
from utils import AverageMeter
from wirings import wirings
from control_constrints import new_ctrb

# ---------------------------- Configuration ----------------------------

in_features = 160
out_features = 160
num_classes = 2
batch_size = 1
num_workers = 0
total_epochs = 150
learning_rate = 0.0008
use_cuda = True

num_folds = 5
random_seed = 42

inputs_dir = "**/input"
targets_dir = "**/truth"
sc_dir = "**/SC"
fc_dir = "**/FC"
result_path = './train_results_kfold'
model_save_path = result_path
os.makedirs(result_path, exist_ok=True)

device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

mae_ratio = 1
l1_lambda = 0.001
l2_lambda = 1

# ---------------------------- Utility Functions ----------------------------

def print_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 ** 3)
    print(f"[{stage}] CPU Memory Usage: {memory_gb:.3f} GB")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{stage}] GPU Memory: Allocated={allocated:.3f} GB, Reserved={reserved:.3f} GB")


def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = AverageMeter()
    maes = AverageMeter()
    bar = tqdm(data_loader, desc="Training", leave=False)

    for batch_idx, (inputs, targets, sc, fc) in enumerate(bar):
        inputs = inputs.squeeze(0).to(device)
        targets = targets.squeeze(0).to(device)
        sc = sc.squeeze(0).to(device)
        fc = fc.squeeze(0).to(device)

        optimizer.zero_grad()

        outputs, _, _, B1, B2, _ = model(inputs, sc, fc)
        predict_loss = F.l1_loss(outputs, targets)
        l1_norm = torch.norm(model.wm_sc.laplacian, 1) + torch.norm(model.wm_fc.laplacian, 1)

        with torch.no_grad():
            r1 = torch.linalg.matrix_rank(new_ctrb(sc.detach(), B1.detach())).item()
            r2 = torch.linalg.matrix_rank(new_ctrb(fc.detach(), B2.detach())).item()
        rank_scalar = r1 + r2
        l2_control_loss = torch.tensor(in_features * 2 - rank_scalar, device=device, dtype=torch.float32)

        loss = mae_ratio * predict_loss + l1_lambda * l1_norm + l2_lambda * l2_control_loss

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), 1)
        maes.update(predict_loss.item(), 1)
        bar.set_postfix(loss=f'{losses.avg:.4f}', mae=f'{maes.avg:.4f}')

        del inputs, targets, sc, fc, outputs, B1, B2, predict_loss, l1_norm, l2_control_loss, loss
        if batch_idx % 50 == 0:
            gc.collect()

    return maes.avg


@torch.no_grad()
def validate(model, data_loader, device):
    model.eval()
    maes = AverageMeter()
    bar = tqdm(data_loader, desc="Validation", leave=False)

    for inputs, targets, sc, fc in bar:
        inputs = inputs.squeeze(0).to(device)
        targets = targets.squeeze(0).to(device)
        sc = sc.squeeze(0).to(device)
        fc = fc.squeeze(0).to(device)

        outputs, *_ = model(inputs, sc, fc)
        mae_val = F.l1_loss(outputs, targets).item()
        maes.update(mae_val, 1)
        bar.set_postfix(mae=f'{maes.avg:.4f}')

        del inputs, targets, sc, fc, outputs

    gc.collect()
    return maes.avg


# ---------------------------- Main Loop ----------------------------

if __name__ == '__main__':
    print("Initial memory usage:")
    print_memory_usage()

    full_dataset = ClassifyGraphDataset(inputs_dir, targets_dir, sc_dir, fc_dir)
    print(f"Total number of samples: {len(full_dataset)}")

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    fold_results_mae = []
    global_best_mae = float('inf')
    best_model_info = {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f'\n{"="*25} FOLD {fold + 1}/{num_folds} {"="*25}')
        print_memory_usage(f"Start of Fold {fold+1}")

        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_ids))
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_ids))

        wiring = wirings.FullyConnected(in_features, out_features)
        model = MSClassifyNet(num_classes=num_classes, in_features=in_features, wiring=wiring, out_features=out_features)
        model.to(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        best_fold_mae = float('inf')

        for epoch in range(total_epochs):
            print(f"\n--- Fold {fold+1}, Epoch {epoch+1}/{total_epochs} ---")
            train_mae = train_one_epoch(model, train_loader, optimizer, device)
            val_mae = validate(model, val_loader, device)
            scheduler.step(val_mae)
            print(f"Epoch {epoch+1}: Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_mae < best_fold_mae:
                best_fold_mae = val_mae
                print(f"✨ New best MAE for this Fold: {best_fold_mae:.4f}")

            if val_mae < global_best_mae:
                global_best_mae = val_mae
                best_model_info = {'fold': fold + 1, 'epoch': epoch + 1, 'mae': global_best_mae}
                save_path = os.path.join(model_save_path, 'global_best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f"🏆 New GLOBAL BEST MAE: {global_best_mae:.4f} — Model saved to {save_path}")

            gc.collect()
            torch.cuda.empty_cache()

        fold_results_mae.append(best_fold_mae)
        print(f'--- Fold {fold + 1} finished. Best Validation MAE: {best_fold_mae:.4f} ---')
        del model, optimizer, train_loader, val_loader, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage(f"End of Fold {fold+1}")

    # ---------------------------- Final Results ----------------------------

    print(f'\n{"="*30} FINAL RESULTS {"="*30}')
    print("--- K-Fold Cross-Validation Performance ---")
    fold_results_mae = np.array(fold_results_mae)
    print(f'All fold best MAEs: {fold_results_mae}')
    print(f'Average MAE over {num_folds} folds: {fold_results_mae.mean():.4f} ± {fold_results_mae.std():.4f}')

    print("\n--- Global Best Model ---")
    if best_model_info:
        print(f"Best model found in Fold {best_model_info['fold']} at Epoch {best_model_info['epoch']} with MAE: {best_model_info['mae']:.4f}")
    else:
        print("No best model found.")

    results_df = pd.DataFrame({
        'Fold': [f'Fold_{i+1}' for i in range(num_folds)] + ['Average', 'Std_Dev'],
        'MAE': list(fold_results_mae) + [fold_results_mae.mean(), fold_results_mae.std()]
    })
    results_df.to_csv(os.path.join(result_path, 'kfold_mae_results.csv'), index=False)
    print(f"Results saved to {os.path.join(result_path, 'kfold_mae_results.csv')}")
