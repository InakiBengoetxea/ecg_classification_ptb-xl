"""
data_loader.py
--------------
Utilities for loading and preprocessing the PTB-XL dataset across multiple tasks.
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    """Load raw ECG waveform data from disk."""
    base_path = Path(path)

    if sampling_rate == 100:
        data = [wfdb.rdsamp(str(base_path / f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(str(base_path / f)) for f in df.filename_hr]
        
    return np.array([signal for signal, _ in data])


def aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame, task_type: str) -> list:
    """Map raw SCP codes to either superclass or subclass lists."""
    if task_type == 'superclass':
        tmp = [agg_df.loc[key].diagnostic_class for key in y_dic.keys() if key in agg_df.index]
    elif task_type == 'subclass':
        tmp = [agg_df.loc[key].diagnostic_subclass for key in y_dic.keys() if key in agg_df.index]
    else:
        # For 'scp-statement', we keep the raw keys themselves if they represent diagnostic statements
        tmp = [key for key in y_dic.keys() if key in agg_df.index]
        
    # Clean out any nulls or missing mappings safely
    return list(set([item for item in tmp if pd.notna(item)]))


# ---------------------------------------------------------------------------
# Full dataset preparation pipeline
# ---------------------------------------------------------------------------

def load_ptbxl(path: str, sampling_rate: int = 100, task: str = 'superclass'):
    """
    Load and split PTB-XL into train / val / test sets for a specific target task.
    Automatically restricts subclasses and scp-statements to the top 10 most frequent items.
    """
    # 1. Load basic tables
    Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1] # isolate true diagnostic rows
    
    # 2. Map codes based on selected task
    Y["target_labels"] = Y.scp_codes.apply(lambda y_dic: aggregate_diagnostic(y_dic, agg_df, task))

    # 3. Handle data splits (Folds 1-8 are Train, 9 is Val, 10 is Test)
    train_folds = list(range(1, 9))
    val_fold = 9
    test_fold = 10

    Y_train_all = Y[Y.strat_fold.isin(train_folds)]
    
    # 4. Determine Active Categories
    if task == 'superclass':
        # Superclass has 5 default categories
        classes = sorted({c for sub in Y_train_all.target_labels for c in sub})
    else:
        # Calculate frequencies in the training set to find the top 10 categories
        all_train_labels = [label for sublist in Y_train_all.target_labels for label in sublist]
        counts = pd.Series(all_train_labels).value_counts()
        classes = sorted(list(counts.head(10).index))
        print(f"  [Loader] Sliced {task} down to top 10 categories: {classes}")
        
        # Filter target labels so only the selected top 10 remain active
        Y["target_labels"] = Y.target_labels.apply(lambda sublist: [item for item in sublist if item in classes])

    # 5. Extract raw ECG matrices
    X = load_raw_data(Y, sampling_rate, path)

    X_train = X[Y.strat_fold.isin(train_folds)]
    y_train = Y[Y.strat_fold.isin(train_folds)].target_labels

    X_val = X[Y.strat_fold == val_fold]
    y_val = Y[Y.strat_fold == val_fold].target_labels

    X_test = X[Y.strat_fold == test_fold]
    y_test = Y[Y.strat_fold == test_fold].target_labels

    # 6. Binarize labels
    mlb = MultiLabelBinarizer(classes=classes)
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin = mlb.transform(y_val)
    y_test_bin = mlb.transform(y_test)

    # Return Y table as the 8th argument so the XAI notebook can match indices for meta features
    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, classes, Y


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

class PTBXL_Dataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int = 32):
    train_ds = PTBXL_Dataset(X_train, y_train)
    val_ds = PTBXL_Dataset(X_val, y_val)
    test_ds = PTBXL_Dataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Metadata preprocessing
# ---------------------------------------------------------------------------

def load_metadata(Y: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["age", "sex", "height", "weight", "pacemaker", "extra_beats", "heart_axis"]
    meta = Y[meta_cols].copy()

    meta["sex"] = meta["sex"].map({"male": 0, "female": 1, 0: 0, 1: 1, "0": 0, "1": 1})

    def check_presence(val):
        val_str = str(val).lower()
        if "pacemaker" in val_str or "ja" in val_str or "yes" in val_str:
            return 1
        return 0

    meta["pacemaker"] = meta["pacemaker"].apply(check_presence)
    meta["extra_beats"] = meta["extra_beats"].apply(check_presence)
    meta["heart_axis"] = pd.to_numeric(meta["heart_axis"], errors='coerce').fillna(0).astype(int)
    meta = meta.fillna(0).astype(int)
    
    return meta


# ---------------------------------------------------------------------------
# Fusion dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    def __init__(self, X, meta, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

        if isinstance(meta, (pd.DataFrame, pd.Series)):
            meta_np = meta.values
        else:
            meta_np = np.asarray(meta)
        self.meta = torch.tensor(meta_np, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.meta[idx], self.y[idx]