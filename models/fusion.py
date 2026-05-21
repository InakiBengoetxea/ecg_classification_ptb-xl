"""
fusion.py
---------
FusionModel combining xresnet1d101 (ECG branch) and MetaMLP (metadata branch).

Design:
  - ECG branch  : xresnet1d101 WITH its full Linear head intact (256 → num_classes).
                  Features are extracted BEFORE the final Linear so the head
                  remains intact for LRP backpropagation.
  - Meta branch : MetaMLP  (in_features → 64 → 32)
  - Fusion      : concat(ecg_features[256], meta_features[32]) → 288-dim
                  → classifier (288 → 64 → num_classes)
"""
    
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, ecg_model, meta_model, num_classes: int):
        super().__init__()
        self.ecg         = ecg_model
        self.meta        = meta_model
        self.num_classes = num_classes

        meta_feat_dim = 32                           # MetaMLP out_features
        fusion_dim    = num_classes + meta_feat_dim  # 5 + 32 = 37

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, ecg: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # Feature extraction from both branches
        ecg_feat  = self.ecg(ecg)    # Output: (Batch, 5)
        meta_feat = self.meta(meta)  # Output: (Batch, 32)

        # Tensor flattening to ensure compatible dimensions for concatenation
        if ecg_feat.dim() > 2:
            ecg_feat = ecg_feat.view(ecg_feat.size(0), -1)
        if meta_feat.dim() > 2:
            meta_feat = meta_feat.view(meta_feat.size(0), -1)

        # Fusion by concatenation (Total features: 37)
        fusion = torch.cat([ecg_feat, meta_feat], dim=1)
        return self.classifier(fusion)