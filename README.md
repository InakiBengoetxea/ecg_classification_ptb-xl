# XResNet1d-101 on PTB-XL

Multi-label ECG classification using a 1-D XResNet-101 on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/).

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
│   └── metrics.py                 # AUC, confusion matrices, plots
├── xai_pipeline.py                # XAI Framework (LRP, DTD, Grad-CAM, SHAP, PFI)
├── notebooks/
│   └── xresnet1d101_ptbxl.ipynb   # Baseline ECG training
│   ├── fusion_training.ipynb      # Multimodal model training
│   └── xai_implementation.ipynb   # XAI generation and dashboard visualization
├── outputs/                       # Checkpoints, loss curves, figures (git-ignored)
├── data/                          # Place raw PTB-XL data here (git-ignored)
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

Download PTB-XL from PhysioNet and extract it into `data/`:

```
data/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    └── records100/
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/xresnet1d101_ptbxl.ipynb
jupyter notebook notebooks/fusion_training.ipynb
jupyter notebook notebooks/xai_implementation.ipynb
```

Or open it in **Google Colab** — the notebook contains a commented-out cell that mounts Google Drive and unzips the dataset automatically.

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


| Metric               | ECG Model | Fusion Model |
|----------------------|-----------|--------------|
| Accuracy             |	0.6260 |	0.6274    |
| F1 Score             |	0.7251 |	0.7310    |
| Precision (PPV)      |	0.7918 |	0.7882    |
| Sensitivity (Recall) |	0.6862 |	0.6931    |
| Specificity          |	0.9265 |	0.9295    |
| MCC                  |	0.6938 |	0.6967    |
| AUC                  |	0.9224 |	0.9234    |

Explainability Observations (XAI):

The failure cases provided illustrate instances where the model's prediction confidence did not align with the actual labels:

Sample 33 (False Negative): The model predicted CD with high confidence, while the actual labels were NORM and CD.

Sample 32 (False Positive): The model predicted CD with high confidence, but the actual label was STTC.

Sample 37 (False Negative): The model predicted HYP as the top class, while the actual label was CD.

---

## References

- Strodthoff et al. (2020) — *PTB-XL, a large publicly available electrocardiography dataset*
- He et al. (2019) — *Bag of Tricks for Image Classification with CNNs* (XResNet)
- fastai XResNet implementation
- Zennit Framework for attribution-based XAI.
- SHAP (Lundberg et al.) for model-agnostic feature importance.
