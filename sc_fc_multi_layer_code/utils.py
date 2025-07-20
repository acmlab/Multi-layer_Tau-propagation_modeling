import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as scio
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.covariance import ledoit_wolf
from numpy.lib.stride_tricks import sliding_window_view


def fc2vector(fc, offset=-1):
    index = np.tril_indices(fc.shape[-2], offset=offset)
    return fc[..., index[0], index[1]]


def get_fc(bold, normalization=True, relu=False, estimate=False):
    def _process(signal):
        fc = np.corrcoef(signal)
        if estimate:
            X = signal.T
            X = (X - X.mean(axis=0)) / X.std(axis=0)
            fc, _ = ledoit_wolf(X)
        if relu:
            fc[fc < 0] = 0
        if normalization:
            if relu:
                eigvals, eigvecs = np.linalg.eigh(fc)
                eigvals[eigvals <= 1e-4] = 1e-4
                fc = eigvecs @ np.diag(eigvals) @ eigvecs.T
            else:
                fc += 1e-4 * np.eye(fc.shape[0])
        return fc

    if isinstance(bold, list):
        return [_process(sig) for sig in bold]
    return _process(bold)


def sliding_window(bold, window, step, pad=True):
    if pad:
        left = (window - 1) // 2
        right = window - 1 - left
        bold = np.concatenate((bold[..., :left], bold, bold[..., -right:]), axis=-1)
    return [bold[..., pos:pos + window].copy() for pos in range(0, bold.shape[-1] - window + 1, step)]


def corrcoef(X):
    X = X - X.mean(-1, keepdims=True)
    c = X @ X.swapaxes(-2, -1)
    stddev = np.sqrt(np.diagonal(c, axis1=-2, axis2=-1))
    c /= stddev[..., None]
    c /= stddev[..., None, :]
    np.clip(c, -1, 1, out=c)
    return c


def sliding_window_corrcoef(X, window, padding=True):
    if padding:
        left = (window - 1) // 2
        right = window - 1 - left
        X = np.concatenate((X[..., :left], X, X[..., -right:]), axis=-1)
    X_window = sliding_window_view(X, (X.shape[-2], window), (-2, -1)).squeeze()
    return corrcoef(X_window)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return sorted(data, key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])


def _read_file(path, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError("The file doesn't exist")
    if path.endswith('.mat'):
        return scio.loadmat(path)['rawdata']
    elif path.endswith('.csv') or path.endswith('.txt'):
        return np.loadtxt(path, dtype=dtype, delimiter=',' if path.endswith('.csv') else None)
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError("Unsupported file type")


def read_data(path, dtype=np.float32, startswith=None, endswith=None):
    if isinstance(path, list):
        return [_read_file(p, dtype) for p in path if (startswith is None or os.path.basename(p).startswith(startswith)) and (endswith is None or p.endswith(endswith))]
    elif os.path.isdir(path):
        files = [os.path.join(path, name) for name in sorted_alphanumeric(os.listdir(path)) if (startswith is None or name.startswith(startswith)) and (endswith is None or name.endswith(endswith))]
        return [_read_file(f, dtype) for f in tqdm(files, desc='Reading files')]
    else:
        return _read_file(path, dtype)


def print_heatmap(fc, cmap='RdYlBu_r', square=True, xticklabels=10, yticklabels=10, xrotation=0, yrotation=0,
                  annot=False, show=True, save=None, figsize=None, cbar=True, bbox_inches=None):
    plt.figure(figsize=figsize)
    sns.heatmap(fc, cmap=cmap, square=square, xticklabels=xticklabels, yticklabels=yticklabels,
                annot=annot, cbar=cbar)
    plt.xticks(rotation=xrotation)
    plt.yticks(rotation=yrotation)
    if save:
        plt.savefig(save, bbox_inches=bbox_inches)
    if show:
        plt.show()
    else:
        plt.close()


def print_cluster(X, labels):
    tsne_result = TSNE().fit_transform(X)
    for label in np.unique(labels):
        members = labels == label
        plt.scatter(tsne_result[members, 0], tsne_result[members, 1], label=label)
    plt.legend()
    plt.show()


def purity_score(y_true, y_pred):
    return np.sum(np.amax(metrics.cluster.contingency_matrix(y_true, y_pred), axis=0)) / np.sum(metrics.cluster.contingency_matrix(y_true, y_pred))


def cluster_score(labels_true, labels_pred):
    return purity_score(labels_true, labels_pred), accuracy_score(labels_true, labels_pred), normalized_mutual_info_score(labels_true, labels_pred)


def random_spd(n):
    A = np.random.rand(n, n)
    return A @ A.T + np.random.rand() * np.eye(n)


def spd_pad(matrix, padding, padding_val=1e-4):
    shape = list(matrix.shape)
    shape[-2] += padding
    shape[-1] += padding
    out = np.zeros(shape, dtype=np.float32)
    np.fill_diagonal(out[..., -padding:, -padding:], padding_val)
    out[..., :matrix.shape[-2], :matrix.shape[-1]] = matrix
    return out


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def plot_epochs(fname, data, epochs, xlabel, ylabel, legends, max=True):
    plt.figure()
    for i, series in enumerate(data):
        val = np.max(series) if max else np.min(series)
        idx = np.argmax(series) + 1 if max else np.argmin(series) + 1
        plt.plot(epochs, series, label=legends[i])
        plt.plot(idx, val, 'ko')
        plt.annotate(f'({idx},{val:.4f})', xy=(idx, val))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(fname)
    plt.close()


def write_to_file(filename, data_list):
    mean_val = np.mean(data_list)
    std_val = np.std(data_list)
    with open(filename, 'w') as f:
        for val in data_list:
            f.write(f"{val}\n")
        f.write(f"mean: {mean_val:.4f}\n")
        f.write(f"std: {std_val:.4f}\n")
