# XResNet1d-101 on PTB-XL

Multi-label ECG classification using a 1-D XResNet-101 and multi-modal clinical metadata fusion on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/).

---

## Project structure

```
ptbxl_project/
├── models/
│   ├── __init__.py
│   ├── xresnet1d.py               # XResNet1d architecture (101)
│   ├── meta_mlp.py                # Metadata processing network (MetaMLP)
│   └── fusion.py                  # Multimodal fusion model (ECG + Meta)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # PTB-XL loading, Dataset, DataLoader helpers
│   ├── trainer.py                 # Training loop, early stopping, inference
│   ├── metrics.py                 # Performance metrics (AUC, confusion matrices, loss curves)
│   └── robustness.py              # Synthetic noise injection pipelines (EMG, PLI, BW)
├── xai_pipeline.py                # XAI Framework (LRP, DTD, Grad-CAM, SHAP, PFI)
├── notebooks/
│   ├── fusion_training.ipynb      # End-to-end multi-modal model training across task horizons
│   ├── noisy_training_robustness.ipynb  # Implements a targeted data augmentation pipeline to evaluate model robustness under controlled training corruption
│   ├── robustness_analysis.ipynb  # Performance degradation evaluation under noise stress-tests
│   ├── xai_implementation.ipynb   # Local and global explainability generation pipeline
│   └── xresnet1d101_ptbxl.ipynb   # Baseline single-modality ECG network training
├── outputs/                       # Checkpoints, loss curves, figures (git-ignored)
│   ├── confusion_matrices/
│   ├── scp-statement/             # Outputs and evaluation metrics for SCP task
│   ├── subclass/                  # Outputs and evaluation metrics for Subclass task
│   └── superclass/                # Outputs and evaluation metrics for Superclass task
├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/  # Raw dataset root
├── data/                          # Place raw PTB-XL data here (git-ignored)
├── .gitignore                     # Ignores large datasets and outputs
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/InakiBengoetxea/ecg_classification_ptb-xl.git
cd ptbxl-xresnet1d
pip install -r requirements.txt
```

### 2. Download the dataset

This project is designed to be portable. The dataset is not included in the repository. To run the code:

    1. Download the PTB-XL dataset from PhysioNet.

    2. Place the extracted folder in your project root or a preferred local directory.

    3. Important: Open the first cell of the notebooks and update the PATH variable to point to your local dataset location.

Recommended local structure:

ptbxl_project/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ├── ptbxl_database.csv
    └── records100/
```
### 3. Run the notebook

```bash

# 1. Train or evaluate the xresnet1d101 architecture
jupyter notebook notebooks/xresnet1d101_ptbxl.ipynb

# 2. Train or evaluate the fusion architecture
jupyter notebook notebooks/fusion_training.ipynb

# 3. Stress-test the models under simulated clinical artifacts
jupyter notebook notebooks/robustness_analysis.ipynb

# 4. Generate post-hoc local/global explanations
jupyter notebook notebooks/xai_implementation.ipynb
```

---

## Module overview

### `models/xresnet1d.py`
| Symbol | Description |
|--------|-------------|
| `xresnet1d101(**kwargs)` | Convenience constructors |
| `build_xresnet1d101(num_classes, input_channels, device)` | Build + move to device |
| `XResNet1d` | 1-D Residual Network backbone (101 layers) |
| `ResBlock` | Bottleneck residual block (1-D) 

### `models/MetaMLP.py`
| `MetaMLP` | 3-layer MLP for clinical metadata embeddings |

### `models/fusion.py`
| `FusionModel` | Combines ECG logits and Meta features for final classification |

### `utils/data_loader.py`
| Symbol | Description |
|--------|-------------|
| `load_ptbxl(path, sampling_rate)` | Full pipeline → split arrays + class list |
| `build_dataloaders(...)` | Wrap arrays in `DataLoader` |
| `PTBXL_Dataset` | PyTorch `Dataset` (transposes leads to channel dim) |
| `load_metadata(Y)` | Standardizes age/weight and encodes categorical metadata |
| `FusionDataset` | PyTorch Dataset returning (ECG, Meta, Labels) |
| `mount_and_extract_colab(...)` | Colab Drive helper |

### `utils/trainer.py`
| Symbol | Description |
|--------|-------------|
| `train(model, ...)` | Full loop with early stopping + LR scheduling |
| `train_epoch / eval_epoch` | Single epoch functions |
| `get_predictions(model, loader)` | Collect sigmoid preds & targets |
| `plot_losses(train_losses, val_losses)` | Loss curve visualisation |

### `utils/robustness.py`
| Symbol | Description |
|--------|-------------|
| `FusionStressTestWrapper` | Stateful tracking wrapper designed to feed multi-modal evaluation inputs into single-input stress evaluation functions. |

### `utils/metrics.py`
| Symbol | Description |
|--------|-------------|
| `compute_auc_per_class(targets, preds, classes)` | Per-class ROC-AUC |
| `compute_macro_auc(auc_scores)` | Macro-average AUC |
| `plot_auc_bar(auc_scores)` | Bar chart of per-class AUC |
| `plot_confusion_matrices(targets, preds, classes)` | One CM per class |
| `plot_prediction_vs_truth(preds, targets, classes)` | Single-sample bar chart |

### `xai_pipeline.py`
| Symbol | Description |
|--------|-------------|
| `LRPExplainer` | Layer-wise Relevance Propagation for signal interpretation |
| `DTDExplainer` | Deep Taylor Decomposition for sharper signal attribution |
| `SHAPExplainer` | KernelSHAP for evaluating clinical metadata impact |
| `PFIExplainer` | Permutation Feature Importance for global modality ranking |
| `XAIPipeline` | Unified entry point for all XAI methods and dashboards |

---

## Results


================================================================================
Task Matrix Category | Test Accuracy   | Macro F1-Score  | Macro ROC-AUC  
---------------------------------------------------------------------------
superclass           |          0.6283 |          0.7285 |          0.9212
subclass             |          0.5965 |          0.5704 |          0.9464
scp-statement        |          0.6051 |          0.5749 |          0.9347
===========================================================================

Explainability & Robustness Summary

ECG Attributions: LRP and DTD maps successfully highlight target diagnostic features corresponding directly to the P-wave, QRS complex, and T-wave boundaries.

Metadata Impact: Resolving the zero-variance bottleneck reveals clear global demographic distributions across the superclass tasks.

Noise Tolerance: Stress matrix testing reveals high resilience against Power Line Interference and Baseline Wander, while highlighting specific vulnerability vectors under severe 0 dB Muscle Artifact (EMG) noise contamination.
---

## References

- Strodthoff et al. (2020) — *PTB-XL, a large publicly available electrocardiography dataset*
- He et al. (2019) — *Bag of Tricks for Image Classification with CNNs* (XResNet)
- fastai XResNet implementation
- SHAP (Lundberg et al.) for model-agnostic feature importance.
