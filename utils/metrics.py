"""
metrics.py
----------
Evaluation metrics and visualisation helpers for multi-label ECG classification.
"""

from typing import Dict, List, Optional

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score


# ---------------------------------------------------------------------------
# AUC helpers
# ---------------------------------------------------------------------------

def compute_auc_per_class(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    classes: List[str],
) -> Dict[str, float]:
    """Compute per-class ROC-AUC scores."""
    auc_scores = {}
    for i, cls in enumerate(classes):
        try:
            auc_scores[cls] = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc_scores[cls] = np.nan
    return auc_scores


def compute_macro_auc(auc_scores: Dict[str, float]) -> float:
    """Compute macro-average AUC, ignoring NaN entries."""
    return float(np.nanmean(list(auc_scores.values())))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_auc_bar(
    auc_scores: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Bar chart of per-class AUC scores."""
    plt.figure(figsize=figsize)
    plt.bar(auc_scores.keys(), auc_scores.values(), color="skyblue")
    plt.ylabel("AUC")
    plt.title("AUC per Diagnostic Superclass")
    plt.ylim(0, 1)
    plt.grid(axis="y")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    macro_auc = compute_macro_auc(auc_scores)
    print(f"Macro AUC: {macro_auc:.4f}")


def plot_prediction_vs_truth(
    preds: np.ndarray,
    targets: np.ndarray,
    classes: List[str],
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Side-by-side bar chart comparing predictions vs ground truth for one sample."""
    print(f"Ground truth: {targets[sample_idx]}")
    print(f"Predictions : {preds[sample_idx]}")

    x = np.arange(len(classes))
    width = 0.35  # Width of the individual bars

    plt.figure(figsize=figsize)
    plt.bar(x - width/2, preds[sample_idx], width, alpha=0.8, label="Predicted", color="royalblue")
    plt.bar(x + width/2, targets[sample_idx], width, alpha=0.6, label="True", color="orange")
    
    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Probability / Label")
    plt.title(f"Prediction vs Ground Truth (Sample {sample_idx})")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_confusion_matrices(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    classes: List[str],
    threshold: float = 0.5,
    save_dir: Optional[str] = None,
    figsize: tuple = (4, 3),
):
    """Plot one confusion matrix per class."""
    pred_bin = (all_preds >= threshold).astype(int)

    for i, cls in enumerate(classes):
        cm = confusion_matrix(all_targets[:, i], pred_bin[:, i])

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(f"Confusion Matrix – {cls}")
        plt.xlabel("Prediction")
        plt.ylabel("True Label")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"cm_{cls}.png"), bbox_inches="tight")

        plt.show()


# ---------------------------------------------------------------------------
# Full evaluation metrics
# ---------------------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

def compute_all_metrics(
    all_targets: np.ndarray,
    all_preds: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute full set of evaluation metrics."""
    pred_bin = (all_preds >= threshold).astype(int)
    metrics = {}

    metrics["Accuracy"] = accuracy_score(all_targets, pred_bin)
    metrics["F1"] = f1_score(all_targets, pred_bin, average="macro")
    metrics["Precision (PPV)"] = precision_score(
        all_targets, pred_bin, average="macro", zero_division=0
    )
    metrics["Sensitivity (Recall)"] = recall_score(
        all_targets, pred_bin, average="macro"
    )

    # Specificity and MCC calculated tracking per-class distributions
    specificity_per_class = []
    mcc_per_class = []
    
    for i in range(all_targets.shape[1]):
        y_true = all_targets[:, i]
        y_pred = pred_bin[:, i]

        # Component isolation
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        spec = tn / (tn + fp + 1e-8)
        specificity_per_class.append(spec)

        # Isolated class MCC calculations to avoid global TN inflation artifact distortions
        mcc_per_class.append(matthews_corrcoef(y_true, y_pred))

    metrics["Specificity"] = float(np.mean(specificity_per_class))
    metrics["MCC"] = float(np.nanmean(mcc_per_class))

    # AUC calculation
    auc_scores = compute_auc_per_class(all_targets, all_preds, list(range(all_targets.shape[1])))
    metrics["AUC"] = compute_macro_auc(auc_scores)

    return metrics

def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Plot training and validation loss curves side-by-side over epochs."""
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(val_losses, label="Val Loss", color="orange", lw=2)
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        # Guarantee parent directories exist dynamically
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()