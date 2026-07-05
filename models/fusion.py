"""
fusion.py
---------
FusionModel combining xresnet1d101 (ECG branch) and MetaMLP (metadata branch).

Design:
  - ECG branch  : xresnet1d101 WITH its full Linear head intact.
                  Features are extracted (logits) so the head remains intact 
                  for LRP backpropagation.
  - Meta branch : MetaMLP dynamically sized based on metadata features.
  - Fusion      : concat(ecg_logits [num_classes], meta_features [dynamic])
                  -> classifier (fusion_dim -> 64 -> num_classes)
"""

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, ecg_model, meta_model, num_classes: int):
        super().__init__()
        self.ecg         = ecg_model
        self.meta        = meta_model
        self.num_classes = num_classes

        # Safely find the output shape of the meta_model by scanning for Linear layers
        linear_layers = [m for m in self.meta.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            meta_feat_dim = linear_layers[-1].out_features
        else:
            # Fallback guard: track shape by performing a mock forward pass through the meta block
            with torch.no_grad():
                mock_input = torch.zeros(1, list(self.meta.modules())[1].in_features if hasattr(list(self.meta.modules())[1], 'in_features') else 1)
                meta_feat_dim = self.meta(mock_input).shape[-1]
        
        # Calculate the combined dimension size (ECG Logits + Meta Features)
        fusion_dim = num_classes + meta_feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, ecg: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # Feature extraction from both branches
        ecg_feat  = self.ecg(ecg)    # Output shape: (Batch, num_classes)
        meta_feat = self.meta(meta)  # Output shape: (Batch, meta_feat_dim)

        # Secure flattening across the batch dimension using flatten instead of view
        if ecg_feat.dim() > 2:
            ecg_feat = ecg_feat.flatten(1)
        if meta_feat.dim() > 2:
            meta_feat = meta_feat.flatten(1)

        # Fusion by concatenation
        fusion = torch.cat([ecg_feat, meta_feat], dim=1)
        return self.classifier(fusion)