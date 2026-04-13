"""
trainer.py
----------
Training loop, evaluation, early stopping, and model checkpointing utilities.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# One-epoch functions
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cuda",
) -> float:
    """Run one training epoch.

    Args:
        model: PyTorch model.
        loader: Training DataLoader.
        optimizer: Optimiser instance.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> float:
    """Run one evaluation epoch (no gradient updates).

    Args:
        model: PyTorch model.
        loader: Validation or test DataLoader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        Mean loss over all batches.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int = 50,
    patience: int = 5,
    checkpoint_path: str = "best_ecg_model.pt",
    device: str = "cuda",
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 2,
):
    """Train with early stopping and learning-rate scheduling.

    Args:
        model: PyTorch model.
        train_dl: Training DataLoader.
        val_dl: Validation DataLoader.
        optimizer: Optimiser instance.
        criterion: Loss function.
        num_epochs: Maximum number of epochs.
        patience: Early-stopping patience (epochs without improvement).
        checkpoint_path: File path for the best model checkpoint.
        device: 'cuda' or 'cpu'.
        scheduler_factor: LR reduction factor for ReduceLROnPlateau.
        scheduler_patience: Plateau patience for the scheduler.

    Returns:
        Tuple of (train_losses, val_losses) lists.
    """
    #To reduce overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=scheduler_factor, patience=scheduler_patience
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_dl, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:3d}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
):
    """Collect sigmoid predictions and ground-truth labels over a DataLoader.

    Args:
        model: PyTorch model.
        loader: DataLoader to iterate over.
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of (all_preds, all_targets) as numpy arrays.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = torch.sigmoid(model(X)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    return np.vstack(all_preds), np.vstack(all_targets)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_losses(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """Plot training vs. validation loss curves.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses: List of per-epoch validation losses.
        save_path: If provided, save the figure to this path.
        figsize: Figure size (width, height) in inches.
    """
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
