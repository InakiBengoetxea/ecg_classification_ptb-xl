"""
data_loader.py
--------------
Utilities for loading and preprocessing the PTB-XL dataset.
"""

import os
import ast
import zipfile
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
    """Load raw ECG waveform data from disk.

    Args:
        df: DataFrame with filename_lr / filename_hr columns.
        sampling_rate: 100 or 500 Hz.
        path: Root path of the PTB-XL dataset.

    Returns:
        numpy array of shape (N, seq_len, 12).
    """
    # 1. Force path to be a Path object so the '/' operator always works
    base_path = Path(path)

    # 2. Use the '/' operator to join, then convert to str for wfdb compatibility
    if sampling_rate == 100:
        data = [wfdb.rdsamp(str(base_path / f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(str(base_path / f)) for f in df.filename_hr]
        
    return np.array([signal for signal, _ in data])


def aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame) -> list:
    """Map SCP codes to diagnostic superclasses.

    Args:
        y_dic: Dictionary of SCP codes and their confidence values.
        agg_df: Filtered scp_statements DataFrame (diagnostic == 1).

    Returns:
        List of unique diagnostic superclasses.
    """
    tmp = [
        agg_df.loc[key].diagnostic_class
        for key in y_dic.keys()
        if key in agg_df.index
    ]
    return list(set(tmp))


# ---------------------------------------------------------------------------
# Full dataset preparation pipeline
# ---------------------------------------------------------------------------

def load_ptbxl(path: str, sampling_rate: int = 100):
    """Load and split PTB-XL into train / val / test sets.

    Args:
        path: Root directory of the extracted PTB-XL dataset.
        sampling_rate: 100 or 500 Hz.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, classes).
    """
    # ---- Load and convert annotation data ----
    Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)

    # ---- diagnostic aggregation ----
    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y["diagnostic_superclass"] = Y.scp_codes.apply(
        lambda y_dic: aggregate_diagnostic(y_dic, agg_df)
    )

    # ---- Load raw signal data ----
    X = load_raw_data(Y, sampling_rate, path)

    # ---- Split data into train, validation, and test according to PTB-XL folds ----
    train_folds = list(range(1, 9)) # 1–8
    val_fold = 9                    # 9
    test_fold = 10                  # 10

    # Train
    X_train = X[np.where(Y.strat_fold.isin(train_folds))]
    y_train = Y[Y.strat_fold.isin(train_folds)].diagnostic_superclass

    # Validation
    X_val = X[np.where(Y.strat_fold == val_fold)]
    y_val = Y[Y.strat_fold == val_fold].diagnostic_superclass

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    # ---- binarise labels ----
    classes = sorted({c for sub in y_train for c in sub})
    mlb = MultiLabelBinarizer(classes=classes)
    y_train_bin = mlb.fit_transform(y_train)
    y_val_bin = mlb.transform(y_val)
    y_test_bin = mlb.transform(y_test)

    return X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, classes


# ---------------------------------------------------------------------------
# PyTorch Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

#  ECGs are shaped (N, 1000, 12) but PyTorch expects (N, 12, 1000)

class PTBXL_Dataset(Dataset):
    """PTB-XL PyTorch Dataset.

    Transposes ECG signals from (N, seq_len, 12) → (N, 12, seq_len) so that
    PyTorch Conv1d can treat the 12 leads as channels.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_dataloaders(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    batch_size: int = 32,
):
    """Create train / val / test DataLoaders.

    Args:
        X_train, X_val, X_test: ECG arrays.
        y_train, y_val, y_test: Binary label arrays.
        batch_size: Mini-batch size.

    Returns:
        Tuple of (train_dl, val_dl, test_dl).
    """
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

def load_metadata(Y):
    meta_cols = ["age", "sex", "height", "weight", "pacemaker", "extra_beats", "heart_axis"]
    meta = Y[meta_cols].copy()

    # 1. Sex: Handle common string/numeric variants
    meta["sex"] = meta["sex"].map({"male": 0, "female": 1, 0: 0, 1: 1, "0": 0, "1": 1})

    # 2. Presence Features: 
    # Logic: If the string contains the relevant keyword, it is 1, else 0.
    def check_presence(val):
        # Convert to string to safely handle different types
        val_str = str(val).lower()
        # Look for indicators of presence
        if "pacemaker" in val_str or "ja" in val_str or "yes" in val_str:
            return 1
        return 0

    meta["pacemaker"] = meta["pacemaker"].apply(check_presence)
    meta["extra_beats"] = meta["extra_beats"].apply(check_presence)

    # 3. Heart Axis: Force numeric conversion, setting non-numeric to 0
    meta["heart_axis"] = pd.to_numeric(meta["heart_axis"], errors='coerce').fillna(0).astype(int)

    # 4. Final cleaning: Handle any other missing numerical values
    meta = meta.fillna(0).astype(int)
    
    return meta


# ---------------------------------------------------------------------------
# Fusion dataset
# ---------------------------------------------------------------------------
class FusionDataset(Dataset):
    def __init__(self, X, meta, y):
        # ECG: (batch, 12, 1000)
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)

        # Metadata: float32
        self.meta = torch.tensor(meta.values, dtype=torch.float32)

        # Labels: MUST be long (int64)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.meta[idx], self.y[idx]