"""
metrics.py
----------
Evaluation metrics and visualisation helpers for multi-label ECG classification.
"""

from typing import Dict, List, Optional

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
    """Compute per-class ROC-AUC scores.

    Args:
        all_targets: Ground-truth binary array of shape (N, C).
        all_preds: Predicted probability array of shape (N, C).
        classes: List of class names.

    Returns:
        Dictionary mapping class name → AUC (or NaN if class is missing).
    """
    auc_scores = {}
    for i, cls in enumerate(classes):
        try:
            auc_scores[cls] = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except ValueError:
            auc_scores[cls] = np.nan
    return auc_scores


def compute_macro_auc(auc_scores: Dict[str, float]) -> float:
    """Compute macro-average AUC, ignoring NaN entries.

    Args:
        auc_scores: Output of :func:`compute_auc_per_class`.

    Returns:
        Macro-average AUC.
    """
    return float(np.nanmean(list(auc_scores.values())))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_auc_bar(
    auc_scores: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Bar chart of per-class AUC scores.

    Args:
        auc_scores: Output of :func:`compute_auc_per_class`.
        save_path: If provided, save the figure to this path.
        figsize: Figure size (width, height) in inches.
    """
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
    figsize: tuple = (8, 4),
):
    """Bar chart comparing predicted probabilities vs. ground truth for one sample.

    Args:
        preds: Predicted probabilities array of shape (N, C).
        targets: Ground-truth binary array of shape (N, C).
        classes: List of class names.
        sample_idx: Index of the sample to visualise.
        save_path: If provided, save the figure to this path.
        figsize: Figure size (width, height) in inches.
    """
    print(f"Ground truth: {targets[sample_idx]}")
    print(f"Predictions : {preds[sample_idx]}")

    plt.figure(figsize=figsize)
    plt.bar(classes, preds[sample_idx], alpha=0.6, label="Predicted")
    plt.bar(classes, targets[sample_idx], alpha=0.6, label="True")
    plt.xticks(rotation=45)
    plt.ylabel("Probability / Label")
    plt.title("Prediction vs Ground Truth")
    plt.legend()

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
    """Plot one confusion matrix per class.

    Args:
        all_targets: Ground-truth binary array of shape (N, C).
        all_preds: Predicted probability array of shape (N, C).
        classes: List of class names.
        threshold: Decision threshold for converting probabilities to binary.
        save_dir: If provided, save each figure to this directory.
        figsize: Per-class figure size (width, height) in inches.
    """
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
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"cm_{cls}.png"), bbox_inches="tight")

        plt.show()


# ---------------------------------------------------------------------------
# Full evaluation metrics (for multi-label classification)
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
    """Compute full set of evaluation metrics.

    Includes:
        - Accuracy
        - F1-score (macro)
        - Precision (PPV)
        - Sensitivity (Recall)
        - Specificity
        - MCC
        - AUC (macro)

    Args:
        all_targets: Ground-truth binary array (N, C)
        all_preds: Predicted probabilities (N, C)
        threshold: Threshold for binarisation

    Returns:
        Dictionary of metrics
    """
    pred_bin = (all_preds >= threshold).astype(int)

    metrics = {}

    # Basic metrics
    metrics["Accuracy"] = accuracy_score(all_targets, pred_bin)
    metrics["F1"] = f1_score(all_targets, pred_bin, average="macro")
    metrics["Precision (PPV)"] = precision_score(
        all_targets, pred_bin, average="macro", zero_division=0
    )
    metrics["Sensitivity (Recall)"] = recall_score(
        all_targets, pred_bin, average="macro"
    )

    # Specificity (per class → mean)
    specificity_per_class = []
    for i in range(all_targets.shape[1]):
        y_true = all_targets[:, i]
        y_pred = pred_bin[:, i]

        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        spec = tn / (tn + fp + 1e-8)
        specificity_per_class.append(spec)

    metrics["Specificity"] = float(np.mean(specificity_per_class))

    # MCC (flattened)
    metrics["MCC"] = matthews_corrcoef(
        all_targets.ravel(), pred_bin.ravel()
    )

    # AUC (macro using existing functions)
    auc_scores = compute_auc_per_class(all_targets, all_preds, list(range(all_targets.shape[1])))
    metrics["AUC"] = compute_macro_auc(auc_scores)

    return metrics