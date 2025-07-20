import os
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy.io as scio
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torchinfo import summary

from dataset import ClassifyGraphDataset
from network import MSClassifyNet
from utils import AverageMeter, plot_epochs
from wirings import wirings
from control_constrints import new_ctrb

warnings.filterwarnings("ignore")
mpl.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Config
in_features = 160
out_features = 160
num_classes = 2
batch_size = 1
num_workers = 0
use_cuda = True
start_epoch = 1
total_epochs = 100

# Data paths
inputs_dir = "**input"
targets_dir = "**truth"
sc_dir = "**SC"
fc_dir = "**FC"

# Output paths
result_path = './train_results'
models_save_path = os.path.join(result_path, 'models_save')
os.makedirs(result_path, exist_ok=True)
os.makedirs(models_save_path, exist_ok=True)

# Device
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Dataset Splits
num_sample = sum(1 for f in os.scandir(inputs_dir) if f.name.endswith('.csv'))
num_train = num_sample * 9 // 20
num_val = num_sample // 2 - num_train
num_test = num_sample - num_train - num_val

train_dataset = ClassifyGraphDataset(inputs_dir, targets_dir, sc_dir, fc_dir, slice=slice(num_train))
val_dataset = ClassifyGraphDataset(inputs_dir, targets_dir, sc_dir, fc_dir, slice=slice(num_train, num_train + num_val))
test_dataset = ClassifyGraphDataset(inputs_dir, targets_dir, sc_dir, fc_dir, slice=slice(num_train + num_val, num_sample))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Model Setup
wiring = wirings.FullyConnected(in_features, out_features)
model = MSClassifyNet(num_classes=num_classes, in_features=in_features, wiring=wiring, out_features=out_features).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

# Regularization Coefficients
mae_ratio = 1
l1_lambda = 0.001
l2_lambda = 1

# Train Loop
def train(loader):
    model.train()
    losses, maes, ranks = AverageMeter(), AverageMeter(), AverageMeter()
    bar = tqdm(loader, desc='Training')
    for inputs, targets, sc, fc in bar:
        inputs, targets, sc, fc = [x.squeeze().to(device) for x in (inputs, targets, sc, fc)]
        optimizer.zero_grad()
        outputs, _, _, B1, B2, _ = model(inputs, sc, fc)
        predict_loss = F.l1_loss(outputs, targets)
        l1 = torch.norm(model.wm_sc.laplacian, 1) + torch.norm(model.wm_fc.laplacian, 1)
        rank = torch.linalg.matrix_rank(new_ctrb(sc, B1.detach())) + torch.linalg.matrix_rank(new_ctrb(fc, B2.detach()))
        l2 = in_features * 2 - rank
        loss = mae_ratio * predict_loss + l1_lambda * l1 + l2_lambda * l2
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        maes.update(predict_loss.item())
        ranks.update(rank.item())
        bar.set_postfix(loss=losses.avg, mae=maes.avg)
    return losses.avg, maes.avg, ranks.avg

@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses, maes, ranks = AverageMeter(), AverageMeter(), AverageMeter()
    bar = tqdm(loader, desc='Evaluating')
    for inputs, targets, sc, fc in bar:
        inputs, targets, sc, fc = [x.squeeze().to(device) for x in (inputs, targets, sc, fc)]
        outputs, _, _, B1, B2, _ = model(inputs, sc, fc)
        predict_loss = F.l1_loss(outputs, targets)
        l1 = torch.norm(model.wm_sc.laplacian, 1) + torch.norm(model.wm_fc.laplacian, 1)
        rank = torch.linalg.matrix_rank(new_ctrb(sc, B1.detach())) + torch.linalg.matrix_rank(new_ctrb(fc, B2.detach()))
        l2 = in_features * 2 - rank
        loss = mae_ratio * predict_loss + l1_lambda * l1 + l2_lambda * l2
        losses.update(loss.item())
        maes.update(predict_loss.item())
        ranks.update(rank.item())
        bar.set_postfix(loss=losses.avg, mae=maes.avg)
    return losses.avg, maes.avg, ranks.avg

# Training Loop
best_mae = float('inf')
metrics = {'train': [], 'val': [], 'test': []}

for epoch in range(start_epoch, start_epoch + total_epochs):
    print(f"\nEpoch {epoch}")
    tr_loss, tr_mae, tr_rank = train(train_loader)
    te_loss, te_mae, te_rank = evaluate(test_loader)
    va_loss, va_mae, va_rank = evaluate(val_loader)

    metrics['train'].append((tr_loss, tr_mae, tr_rank))
    metrics['test'].append((te_loss, te_mae, te_rank))
    metrics['val'].append((va_loss, va_mae, va_rank))

    plot_epochs(os.path.join(result_path, 'loss.svg'), [list(zip(*metrics[m]))[0] for m in ['train', 'test', 'val']],
                range(start_epoch, epoch+1), xlabel='epoch', ylabel='loss', legends=['train', 'test', 'val'])
    plot_epochs(os.path.join(result_path, 'mae.svg'), [list(zip(*metrics[m]))[1] for m in ['train', 'test', 'val']],
                range(start_epoch, epoch+1), xlabel='epoch', ylabel='mae', legends=['train', 'test', 'val'])
    plot_epochs(os.path.join(result_path, 'rank.svg'), [list(zip(*metrics[m]))[2] for m in ['train', 'test', 'val']],
                range(start_epoch, epoch+1), xlabel='epoch', ylabel='rank', legends=['train', 'test', 'val'])

    pd.DataFrame([[epoch, tr_loss, tr_mae, te_loss, te_mae, va_loss, va_mae]]).to_csv(
        os.path.join(result_path, 'log.csv'), mode='a', index=False, header=False)

    torch.save({'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()},
               os.path.join(models_save_path, f"{epoch}_{te_mae:.4f}.pt"))
