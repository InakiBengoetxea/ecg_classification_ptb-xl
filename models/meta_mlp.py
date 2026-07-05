import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# ------------------------------
# Metadata Dataset
# ------------------------------
class MetaDataset(Dataset):
    def __init__(self, meta, y):
        # Defensively handle both Pandas DataFrames/Series and raw NumPy arrays
        if isinstance(meta, (pd.DataFrame, pd.Series)):
            meta_np = meta.values
        else:
            meta_np = np.asarray(meta)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.values
        else:
            y_np = np.asarray(y)

        self.meta = torch.tensor(meta_np, dtype=torch.float32)
        self.y = torch.tensor(y_np, dtype=torch.float32)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx], self.y[idx]


# ------------------------------
# Metadata MLP
# ------------------------------
class MetaMLP(nn.Module):
    def __init__(self, in_features, hidden=64, out_features=32):
        super().__init__()
        # Store out_features explicitly as an attribute so fusion.py can read it easily
        self.out_features = out_features
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            # Removed final ReLU activation to prevent dead activations and ensure
            # continuous, expressive gradients/shapley attributions for metadata features.
            nn.Linear(hidden, out_features) 
        )

    def forward(self, x):
        return self.net(x)
