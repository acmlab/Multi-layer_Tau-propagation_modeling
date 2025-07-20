import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class ClassifyGraphDataset(Dataset):
    def __init__(self, inputs_dir, targets_dir, sc_dir, fc_dir, slice=None):
        super().__init__()

        # Load and preprocess all paths
        self.inputs_path = self._get_sorted_paths(inputs_dir, slice)
        self.targets_path = self._get_sorted_paths(targets_dir, slice)
        self.sc_path = self._get_sorted_paths(sc_dir, slice)
        self.fc_path = self._get_sorted_paths(fc_dir, slice)

        # Load data into memory
        self.inputs = [np.loadtxt(p, delimiter=',', dtype=np.float32) for p in self.inputs_path]
        self.targets = [np.loadtxt(p, delimiter=',', dtype=np.float32) for p in self.targets_path]
        self.sc = [pd.read_csv(p, header=None).values.astype(np.float32) for p in self.sc_path]
        self.fc = [pd.read_csv(p, header=None).values.astype(np.float32) for p in self.fc_path]

        # Total number of data samples
        self.n_data = len(self.inputs_path)

        # Debug print
        print(f"Total samples: {self.n_data}\nExample path: {inputs_dir}\nRange: {self.inputs_path[0]} ... {self.inputs_path[-1]}")

    def _get_sorted_paths(self, directory, slice):
        path_obj = Path(directory)
        sorted_paths = sorted(path_obj.glob('*'), key=self._alphanum_key)
        return sorted_paths[slice] if slice else sorted_paths

    def _alphanum_key(self, path):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        return [convert(c) for c in re.split('([0-9]+)', path.name)]

    def __len__(self):
        return self.n_data

    def __add__(self, other):
        self.inputs += other.inputs
        self.targets += other.targets
        self.sc += other.sc
        self.fc += other.fc
        self.n_data += other.n_data
        return self

    def __getitem__(self, index):
        inputs = torch.tensor(self.inputs[index], dtype=torch.float32)
        targets = torch.tensor(self.targets[index], dtype=torch.float32)
        sc = torch.tensor(self.sc[index], dtype=torch.float32)
        fc = torch.tensor(self.fc[index], dtype=torch.float32)

        return inputs, targets, sc, fc