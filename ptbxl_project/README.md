# XResNet1d-101 on PTB-XL

Multi-label ECG classification using a 1-D XResNet-101 on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/).

---

## Project structure

```
ptbxl_project/
├── models/
│   ├── __init__.py
│   └── xresnet1d.py          # XResNet1d architecture (18 / 34 / 50 / 101)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py         # PTB-XL loading, Dataset, DataLoader helpers
│   ├── trainer.py             # Training loop, early stopping, inference
│   └── metrics.py             # AUC, confusion matrices, plots
├── notebooks/
│   └── xresnet1d101_ptbxl.ipynb   # Clean notebook — imports from libraries only
├── outputs/                   # Checkpoints, loss curves, figures (git-ignored)
├── data/                      # Place raw PTB-XL data here (git-ignored)
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone & install

```bash
git clone https://github.com/<your-handle>/ptbxl-xresnet1d.git
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
```

Or open it in **Google Colab** — the notebook contains a commented-out cell that mounts Google Drive and unzips the dataset automatically.

---

## Module overview

### `models/xresnet1d.py`
| Symbol | Description |
|--------|-------------|
| `xresnet1d18/34/50/101(**kwargs)` | Convenience constructors |
| `build_xresnet1d101(num_classes, input_channels, device)` | Build + move to device |
| `XResNet1d` | Full architecture class |
| `ResBlock` | Bottleneck residual block (1-D) |

### `utils/data_loader.py`
| Symbol | Description |
|--------|-------------|
| `load_ptbxl(path, sampling_rate)` | Full pipeline → split arrays + class list |
| `build_dataloaders(...)` | Wrap arrays in `DataLoader` |
| `PTBXL_Dataset` | PyTorch `Dataset` (transposes leads to channel dim) |
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

---

## Results

| Metric | Value |
|--------|-------|
| Macro AUC | *run notebook* |

---

## References

- Strodthoff et al. (2020) — *PTB-XL, a large publicly available electrocardiography dataset*
- He et al. (2019) — *Bag of Tricks for Image Classification with CNNs* (XResNet)
- fastai XResNet implementation
