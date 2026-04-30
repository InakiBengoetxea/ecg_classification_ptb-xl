import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ------------------------------
# Metadata Dataset
# ------------------------------
class MetaDataset(Dataset):
    def __init__(self, meta, y):
        self.meta = torch.tensor(meta.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

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
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
