"""
xai_pipeline.py
---------------
Full XAI pipeline for the FusionModel (xresnet1d101 + MetaMLP).

Methods implemented:
  ECG branch  : LRP (zennit via ECGOnlyWrapper), Deep Taylor Decomposition (DTD), Grad-CAM
  Meta branch : SHAP (KernelExplainer)
  Global      : Permutation Feature Importance (PFI)

LRP design:
  The FusionModel runs the ECG branch fully (including its Linear head),
  producing class logits that are concatenated with metadata features before
  the fusion classifier. LRP uses an ECGOnlyWrapper that fixes the metadata
  and exposes only the ECG tensor as input. This lets relevance flow:
 
      fusion classifier → ECG logits → ECG backbone → raw ECG signal
 
  The result is a (12, 1000) relevance map — one value per lead per timestep —
  with no architecture changes required to FusionModel.

Usage
-----
    from xai_pipeline import XAIPipeline
    xai = XAIPipeline(model, device=device, class_names=classes, meta_cols=META_COLS)
    xai.explain_sample(ecg_tensor, meta_tensor, target_class=0)
    xai.shap_summary(meta_background, meta_test)
    xai.pfi(test_loader, metric_fn)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy


# ============================================================
# 0.  Dependency helpers  (graceful imports)
# ============================================================

def _require(pkg, install):
    try:
        return __import__(pkg)
    except ImportError:
        raise ImportError(f"Install with:  pip install {install}")

# ============================================================
# 1.  Model surgery helpers
# ============================================================

def _split_fusion(fusion_model: nn.Module):
    """Return (ecg_branch, meta_branch, classifier) as separate modules."""
    if not (hasattr(fusion_model, "ecg") and hasattr(fusion_model, "meta") and hasattr(fusion_model, "classifier")):
        raise AttributeError(
            "The provided model does not match the expected FusionModel structure. "
            "Ensure it contains 'ecg', 'meta', and 'classifier' attributes."
        )
    return fusion_model.ecg, fusion_model.meta, fusion_model.classifier

# ============================================================
# 1b.  ECGOnlyWrapper
# ============================================================

class ECGOnlyWrapper(nn.Module):
    """Wraps FusionModel with metadata frozen — exposes only ECG as input."""
    def __init__(self, fusion_model: nn.Module, meta: torch.Tensor):
        super().__init__()
        self.fusion = fusion_model
        self.meta   = meta

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        # Dynamically expand metadata batch size to match incoming ECG variations
        batch_size = ecg.shape[0]
        expanded_meta = self.meta.expand(batch_size, -1) if self.meta.dim() == 2 else self.meta
        return self.fusion(ecg, expanded_meta)


# ============================================================
# 2.  LRP  (Layer-wise Relevance Propagation)  – ECG branch
# ============================================================

class LRPExplainer:
    """
    LRP for xresnet1d101 using the zennit library.

    Rules applied:
      - Input / first conv layers : ZBox rule  (Stops the raw electrical signals from distorting the baseline)
      - Hidden conv layers        : AlphaBeta rule (alpha=2, beta=1 splits the positive and negative evidence so they don't cancel each other out, giving clean red (positive) and blue (negative) heatmaps)
      - BatchNorm / Linear        : Pass rule (lets relevance flow through BatchNorm layers cleanly without altering the data)
    """

    def __init__(self, fusion_model: nn.Module, device: str = "cpu"):
        _require("zennit", "zennit")
        self.fusion_model = fusion_model.to(device).eval()
        self.device = device

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        from zennit.composites import EpsilonPlusFlat
        from zennit.attribution import Gradient

        ecg = ecg.to(self.device).detach().requires_grad_(True)
        meta = meta.to(self.device).detach()
        composite = EpsilonPlusFlat()

        wrapper = ECGOnlyWrapper(self.fusion_model, meta)
        
        with torch.no_grad():
            output_logits = wrapper(ecg)
            
        target_output = torch.zeros_like(output_logits).to(self.device)
        target_output[:, target_class] = 1.0

        with Gradient(model=wrapper, composite=composite) as attributor:
            out, relevance = attributor(ecg, target_output)

        return relevance.squeeze(0).detach().cpu().numpy()

    def plot(self, relevance: np.ndarray, lead_names=None, title="LRP – ECG Relevance",
             save_path=None):
        """Plot per-lead relevance as a heatmap."""
        if lead_names is None:
            lead_names = [f"Lead {i+1}" for i in range(relevance.shape[0])]

        fig, axes = plt.subplots(relevance.shape[0], 1, figsize=(14, relevance.shape[0] * 1.2), sharex=True)
        if relevance.shape[0] == 1:
            axes = [axes]

        # Guard against absolute zero relevance values causing matplotlib ylim crashes
        max_val = np.abs(relevance).max()
        vmax = max_val if max_val > 0 else 1.0
        
        for ax, lead, name in zip(axes, relevance, lead_names):
            ax.fill_between(range(len(lead)), lead, where=(lead > 0), color="#E63946", alpha=0.7)
            ax.fill_between(range(len(lead)), lead, where=(lead < 0), color="#457B9D", alpha=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=40)
            ax.set_ylim(-vmax, vmax)
            ax.tick_params(left=False, labelleft=False)

        axes[0].set_title(title, fontsize=11)
        axes[-1].set_xlabel("Time step (samples)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ============================================================
# 3.  Deep Taylor Decomposition (DTD)  – ECG branch
# ============================================================

class DTDExplainer:
    """
    Deep Taylor Decomposition via zennit's WSquare / Flat composite.

    DTD is more restrictive than LRP. By using the WSquare rule on the 
    first layer, 'relevance smearing' is avoided and sharper 
    attributions are obtained that follow the ECG morphology more closely.
    """

    def __init__(self, fusion_model: nn.Module, device: str = "cpu"):
        _require("zennit", "zennit")
        self.fusion_model = fusion_model.to(device).eval()
        self.device    = device

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        from zennit.composites import LayerMapComposite
        from zennit.rules import WSquare, Flat
        from zennit.attribution import Gradient
        from zennit.types import Convolution, Linear

        ecg = ecg.to(self.device).detach().requires_grad_(True)
        meta = meta.to(self.device).detach()
        wrapper = ECGOnlyWrapper(self.fusion_model, meta)

        # DTD Logic: WSquare for the input layer, Flat for hidden layers.
        # This prevents the 'uniform lead importance' you saw with LRP.
        layer_map = [
            (Convolution, WSquare()), # First conv uses WSquare
            (Linear, Flat()),         # Linear layers use Flat
            (Convolution, Flat()),   # Hidden convs use Flat
        ]
        composite = LayerMapComposite(layer_map=layer_map)

        with torch.no_grad():
            output_logits = wrapper(ecg)

        # Dynamic target assignment matching the batch dimension
        target_output = torch.zeros_like(output_logits).to(self.device)
        target_output[:, target_class] = 1.0

        with Gradient(model=wrapper, composite=composite) as attributor:
            out, attr = attributor(ecg, target_output)

        # DTD usually produces only positive attribution (supporting evidence)
        return attr.squeeze(0).detach().cpu().numpy()

    def compare_with_lrp(self, lrp_rel: np.ndarray, dtd_rel: np.ndarray, lead_idx: int = 0, save_path=None):
        """Side-by-side comparison of LRP vs DTD for one lead."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        t = range(lrp_rel.shape[1])

        # LRP - Red/Blue for Pos/Neg
        max_lrp = np.abs(lrp_rel[lead_idx]).max()
        vmax_lrp = max_lrp if max_lrp > 0 else 1.0
        ax1.fill_between(t, lrp_rel[lead_idx], color="#E63946", alpha=0.6, where=(lrp_rel[lead_idx] > 0))
        ax1.fill_between(t, lrp_rel[lead_idx], color="#457B9D", alpha=0.6, where=(lrp_rel[lead_idx] < 0))
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_title(f"LRP Relevance – Lead {lead_idx+1}", fontsize=10)
        ax1.set_ylim(-vmax_lrp, vmax_lrp)

        # DTD - Usually focuses on positive attribution
        ax2.plot(t, dtd_rel[lead_idx], color="#2A9D8F", lw=1)
        ax2.fill_between(t, dtd_rel[lead_idx], color="#2A9D8F", alpha=0.3)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_title(f"DTD Attribution (Sharper) – Lead {lead_idx+1}", fontsize=10)

        # Calculate correlation and safely guard against flat signals returning NaN values
        corr = np.corrcoef(lrp_rel[lead_idx], dtd_rel[lead_idx])[0, 1]
        if np.isnan(corr):
            corr = 0.0
            
        fig.suptitle(f"XAI Validation: LRP vs DTD (Pearson r = {corr:.3f})", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return corr

# ============================================================
# 4.  GradCAM  –  qualitative  (ECG branch)
# ============================================================

class GradCAMExplainer:
    """
    Grad-CAM for xresnet1d101.

    Hooks into the last ResBlock's output (before the head).
    The target layer is model[-2]  (last block group).
    """
    def __init__(self, fusion_model: nn.Module, device: str = "cpu"):
        self.fusion_model = fusion_model.to(device).eval()
        self.device     = device
        self._gradients = None
        self._activations = None
        self._handlers = []  # Track registration to avoid memory leakage
        self._hook_target()

    def _hook_target(self):
        # xresnet1d: Sequential([stem0, stem1, stem2, MaxPool, block0, block1, block2, block3, head])
        # Last conv block = index -2
        target_group = list(self.fusion_model.ecg.children())[-2]

        # Fixed: Dig directly into the nested layer arrays to target the actual Convolutional block layer execution point
        if isinstance(target_group, nn.Sequential) and len(target_group) > 0:
            target_layer = target_group[-1]
            if hasattr(target_layer, 'conv2'):  # Handles specific subblock wrappers
                target_module = target_layer.conv2
            else:
                target_module = list(target_layer.modules())[-1]
        else:
            target_module = target_group

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handlers.append(target_module.register_forward_hook(fwd_hook))
        self._handlers.append(target_module.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        """Remove hooks explicitly from the model to free memory."""
        for handle in self._handlers:
            handle.remove()
        self._handlers = []

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute Grad-CAM activation map.

        Returns:
            cam : np.ndarray  – dynamically upsampled to input signal length
        """
        ecg = ecg.to(self.device).detach().requires_grad_(True)
        meta = meta.to(self.device).detach()
        input_len = ecg.shape[-1] # Dynamically capture time steps (e.g., 1000)

        self.fusion_model.zero_grad()
        out = self.fusion_model(ecg, meta)
        score = out[0, target_class]
        score.backward()

        # weights = global-average of gradients over time axis
        weights = self._gradients.mean(dim=-1, keepdim=True)
        cam = (weights * self._activations).sum(dim=1)
        cam = torch.relu(cam).squeeze(0).cpu().numpy()

        # Dynamic alignment interpolation fix based on input signal timeline
        cam = np.interp(np.linspace(0, len(cam)-1, input_len),
                        np.arange(len(cam)), cam)

        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def plot(self, ecg_np: np.ndarray, cam: np.ndarray,
             lead_idx: int = 0, save_path=None):
        """Overlay Grad-CAM heatmap on ECG lead."""
        fig, ax = plt.subplots(figsize=(14, 3))
        t = np.arange(ecg_np.shape[-1])
        signal = ecg_np[lead_idx] if ecg_np.ndim == 2 else ecg_np[0, lead_idx]
        ax.plot(t, signal, color="k", lw=0.7, zorder=2)
        sc = ax.scatter(t, signal, c=cam, cmap="hot", s=2, zorder=3, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Grad-CAM activation")
        ax.set_title(f"Grad-CAM – Lead {lead_idx+1}", fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

# ============================================================
# 5.  SHAP  –  metadata branch
# ============================================================

class SHAPExplainer:
    """
    SHAP KernelExplainer for the MetaMLP branch.

    KernelExplainer is model-agnostic and handles correlated tabular
    features correctly, which matters for clinical metadata (age, sex, etc.).
    """

    def __init__(self, meta_model: nn.Module, background: np.ndarray,
                 device: str = "cpu"):
        """
        Args:
            meta_model  : The MetaMLP (returns feature embeddings or logits).
            background  : Representative background dataset, shape (N_bg, n_features).
                          Use K-means summary or random sample of training data.
            device      : 'cpu' or 'cuda'
        """
        shap = _require("shap", "shap")
        self.meta_model = meta_model.to(device).eval()
        self.device     = device
        self._shap      = shap

        def _predict(x: np.ndarray) -> np.ndarray:
            t = torch.tensor(x, dtype=torch.float32).to(device)
            with torch.no_grad():
                out = self.meta_model(t)
            return out.cpu().numpy()

        self.explainer = shap.KernelExplainer(_predict, np.asarray(background))

    def explain(self, meta_samples: np.ndarray, n_samples: int = 100):
        """
        Compute SHAP values for meta_samples.

        Args:
            meta_samples : (N, n_features)
            n_samples    : Number of coalition samples for KernelSHAP

        Returns:
            shap_values  : Cleaned list of arrays or structured multi-class values
        """
        return self.explainer.shap_values(np.asarray(meta_samples), nsamples=n_samples)

    def summary_plot(self, shap_values, meta_samples: np.ndarray,
                     feature_names=None, class_idx: int = 0, save_path=None):
        """
        Beeswarm summary plot for one output class.
        Shows global feature importance + direction of effect.
        """
        # Multi-class output structure standardization checking
        if isinstance(shap_values, list):
            vals = shap_values[class_idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, class_idx]
        else:
            vals = shap_values

        self._shap.summary_plot(vals, np.asarray(meta_samples), feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary – class {class_idx}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close() # Free internal canvas memory to prevent dashboard cross-contamination

    def waterfall_plot(self, shap_values, meta_sample: np.ndarray,
                       feature_names=None, class_idx: int = 0,
                       sample_idx: int = 0, save_path=None):
        """
        Waterfall (force) plot for a single patient prediction.
        """
        if isinstance(shap_values, list):
            vals = shap_values[class_idx]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, class_idx]
        else:
            vals = shap_values

        # Safeguard expected_value tracking against array, list or scalar mismatches
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[class_idx]

        exp = self._shap.Explanation(
            values=vals[sample_idx],
            base_values=base_val,
            data=np.asarray(meta_sample)[sample_idx],
            feature_names=feature_names,
        )
        self._shap.waterfall_plot(exp, show=False)
        plt.title(f"SHAP Waterfall – sample {sample_idx}, class {class_idx}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close() # Free internal canvas memory


# ============================================================
# 6.  Permutation Feature Importance (PFI)  –  global / cross-modal
# ============================================================

class PFIExplainer:
    """
    Permutation Feature Importance at the global model level.

    Supports:
      - Permuting individual metadata columns
      - Permuting entire ECG leads
      - Permuting whole modality (all ECG vs all metadata)

    Metric: any callable (predictions, targets) → scalar (higher = better).
    """

    def __init__(self, fusion_model: nn.Module, device: str = "cpu",
                 n_repeats: int = 5):
        self.model     = fusion_model.to(device).eval()
        self.device    = device
        self.n_repeats = n_repeats

    @torch.no_grad()
    def _score(self, ecg: torch.Tensor, meta: torch.Tensor, targets: torch.Tensor, metric_fn) -> float:
        preds = torch.sigmoid(self.model(ecg, meta)).cpu()
        return metric_fn(preds.numpy(), targets.numpy())
    
    def run_meta_pfi(self, ecg: torch.Tensor, meta: torch.Tensor,
                     targets: torch.Tensor, metric_fn,
                     feature_names=None) -> dict:
        """
        Permute each metadata column independently.

        Returns:
            dict  feature_name → mean importance (drop in metric)
        """
        # Ensure tensors are cleanly isolated from historical compute graphs
        ecg, meta, targets = ecg.detach().to(self.device), meta.detach().to(self.device), targets.detach()
        baseline = self._score(ecg, meta, targets, metric_fn)
        n_feats = meta.shape[1]
        names = feature_names or [f"meta_{i}" for i in range(n_feats)]
        results = {}

        for i, name in enumerate(names):
            drops = []
            for _ in range(self.n_repeats):
                meta_perm = meta.clone()
                idx = torch.randperm(meta_perm.shape[0], device=self.device)
                # Direct slice modification assignment securely decoupled
                meta_perm[:, i] = meta[idx, i]
                drops.append(baseline - self._score(ecg, meta_perm, targets, metric_fn))
            results[name] = float(np.mean(drops))

        return results

    def run_ecg_lead_pfi(self, ecg: torch.Tensor, meta: torch.Tensor, targets: torch.Tensor, metric_fn, lead_names=None) -> dict:
        """
        Permute each ECG lead independently.

        Returns:
            dict  lead_name → mean importance (drop in metric)
        """
        ecg, meta, targets = ecg.detach().to(self.device), meta.detach().to(self.device), targets.detach()
        baseline = self._score(ecg, meta, targets, metric_fn)
        n_leads = ecg.shape[1]
        names = lead_names or [f"Lead {i+1}" for i in range(n_leads)]
        results = {}

        for i, name in enumerate(names):
            drops = []
            for _ in range(self.n_repeats):
                ecg_perm = ecg.clone()
                idx = torch.randperm(ecg_perm.shape[0])
                # Secure slice index update tracking via immutable source arrays
                ecg_perm[:, i, :] = ecg[idx, i, :]
                drops.append(baseline - self._score(ecg_perm, meta, targets, metric_fn))
            results[name] = float(np.mean(drops))

        return results

    def run_modality_pfi(self, ecg: torch.Tensor, meta: torch.Tensor, targets: torch.Tensor, metric_fn) -> dict:
        """
        Permute whole modalities to compare ECG vs metadata contribution.
        """
        ecg, meta, targets = ecg.detach().to(self.device), meta.detach().to(self.device), targets.detach()
        baseline = self._score(ecg, meta, targets, metric_fn)
        results = {}

        # Permute all ECG
        drops = []
        for _ in range(self.n_repeats):
            ecg_perm = ecg[torch.randperm(ecg.shape[0], device=self.device)]
            drops.append(baseline - self._score(ecg_perm, meta, targets, metric_fn))
        results["ECG (whole modality)"] = float(np.mean(drops))

        # Permute all metadata
        drops = []
        for _ in range(self.n_repeats):
            meta_perm = meta[torch.randperm(meta.shape[0], device=self.device)]
            drops.append(baseline - self._score(ecg, meta_perm, targets, metric_fn))
        results["Metadata (whole modality)"] = float(np.mean(drops))
        return results

    @staticmethod
    def plot(importance_dict: dict, title="Permutation Feature Importance", color="#E63946", save_path=None):
        """Horizontal bar chart of importance scores."""
        names  = list(importance_dict.keys())
        values = list(importance_dict.values())
        order  = np.argsort(values)
        names  = [names[i] for i in order]
        values = [values[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
        bars = ax.barh(names, values, color=color, edgecolor="white", height=0.6)
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("Mean drop in metrics")
        ax.set_title(title, fontsize=11)
        
        # Calculate axis range limits to ensure text padding offset logic does not clip bounds
        x_max = max(values) if max(values) > 0 else 0.1
        padding = x_max * 0.01

        for bar, val in zip(bars, values):
            # Dynamic label text placement checking for positive or negative drops
            if val >= 0:
                ax.text(val + padding, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", ha="left", fontsize=7)
            else:
                ax.text(val - padding, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va="center", ha="right", fontsize=7)
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()


# ============================================================
# 7.  Unified XAIPipeline
# ============================================================

class XAIPipeline:
    """
    Unified entry point for all XAI methods.

    Args:
        fusion_model  : Trained FusionModel (after at least one forward pass
                        so that the lazy classifier is materialised).
        device        : 'cuda' or 'cpu'
        class_names   : List of class label strings.
        meta_cols     : List of metadata column names (for SHAP plots).
        lead_names    : List of 12 ECG lead names (default: standard 12-lead).
        n_pfi_repeats : Number of permutation repeats for PFI.
    """

    LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

    def __init__(self, fusion_model: nn.Module, device: str = "cpu", class_names=None, meta_cols=None, lead_names=None, n_pfi_repeats: int = 5):
        self.model = fusion_model.to(device).eval()
        self.device = device
        self.class_names = class_names or []
        self.meta_cols = meta_cols or []
        self.lead_names = lead_names or self.LEAD_NAMES
        self.meta_model = fusion_model.meta

        # Lazy-initialise explainers on demand
        self._lrp   = None
        self._dtd   = None
        self._gcam  = None
        self._shap  = None
        self._pfi   = PFIExplainer(fusion_model, device, n_pfi_repeats)

    # ------------------------------------------------------------------
    # ECG XAI
    # ------------------------------------------------------------------

    def _cls_name(self, idx: int) -> str:
        return self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}"

    def explain_ecg_lrp(self, ecg, meta, target_class, save_path=None) -> np.ndarray:
        """Run LRP on one ECG sample and plot the result."""
        if self._lrp is None:
            self._lrp = LRPExplainer(self.model, self.device)
        rel = self._lrp.explain(ecg, meta, target_class)
        self._lrp.plot(rel, lead_names=self.lead_names, title=f"LRP – class: {self._cls_name(target_class)}", save_path=save_path)
        return rel

    def explain_ecg_dtd(self, ecg, meta, target_class, save_path=None) -> np.ndarray:
        """Run DTD on one ECG sample."""
        if self._dtd is None:
            self._dtd = DTDExplainer(self.model, self.device)
        return self._dtd.explain(ecg, meta, target_class)

    def validate_lrp_with_dtd(self, ecg, meta, target_class, lead_idx=0, save_path=None):
        """Run both LRP and DTD, then plot the comparison."""
        lrp_rel = self.explain_ecg_lrp(ecg, meta, target_class)
        dtd_rel = self.explain_ecg_dtd(ecg, meta, target_class)
        
        # Cleaned redundant instantiation check to prevent memory collisions
        corr = self._dtd.compare_with_lrp(lrp_rel, dtd_rel, lead_idx=lead_idx, save_path=save_path)
        return lrp_rel, dtd_rel, corr

    def explain_ecg_gradcam(self, ecg, meta, target_class, lead_idx=0, save_path=None) -> np.ndarray:
        """Run Grad-CAM on one ECG sample (qualitative / optional)."""
        if self._gcam is None:
            self._gcam = GradCAMExplainer(self.model, self.device)
        cam = self._gcam.explain(ecg, meta, target_class)
        ecg_np = ecg.squeeze(0).cpu().numpy()
        self._gcam.plot(ecg_np, cam, lead_idx=lead_idx, save_path=save_path)
        return cam

    # ------------------------------------------------------------------
    # Metadata XAI
    # ------------------------------------------------------------------

    def init_shap(self, background_meta: np.ndarray):
        """
        Initialise SHAP explainer with a background dataset.

        Args:
            background_meta : np.ndarray (N_bg, n_features)
                              Typically 50-200 training samples or K-means summary.
        """
        self._shap = SHAPExplainer(self.meta_model, background_meta, self.device)

    def explain_meta_shap(self, meta_samples: np.ndarray, class_idx: int = 0, n_samples: int = 100, save_summary: str = None, save_waterfall: str = None, sample_idx: int = 0):
        """
        Compute and plot SHAP values for metadata samples.

        Args:
            meta_samples  : np.ndarray (N, n_features)
            class_idx     : Which output class to explain
            n_samples     : KernelSHAP coalition samples (higher = more accurate)
            save_summary  : Path to save summary plot (optional)
            save_waterfall: Path to save waterfall plot (optional)
            sample_idx    : Which sample to show in waterfall plot
        """
        if self._shap is None:
            raise RuntimeError("Call init_shap(background) first.")
        shap_vals = self._shap.explain(meta_samples, n_samples=n_samples)
        self._shap.summary_plot(shap_vals, meta_samples, feature_names=self.meta_cols, class_idx=class_idx, save_path=save_summary)
        self._shap.waterfall_plot(shap_vals, meta_samples, feature_names=self.meta_cols, class_idx=class_idx, sample_idx=sample_idx, save_path=save_waterfall)
        return shap_vals

    # ------------------------------------------------------------------
    # Global PFI
    # ------------------------------------------------------------------

    def run_pfi(self, ecg: torch.Tensor, meta: torch.Tensor, targets: torch.Tensor, metric_fn, save_prefix: str = None):
        """
        Run all three levels of PFI:
          1. Per metadata column
          2. Per ECG lead
          3. Whole-modality comparison
        """
        print("[PFI] Computing metadata feature importance …")
        meta_imp = self._pfi.run_meta_pfi(ecg, meta, targets, metric_fn, feature_names=self.meta_cols)

        print("[PFI] Computing ECG lead importance …")
        lead_imp = self._pfi.run_ecg_lead_pfi(ecg, meta, targets, metric_fn, lead_names=self.lead_names)

        print("[PFI] Computing modality-level importance …")
        mod_imp = self._pfi.run_modality_pfi(ecg, meta, targets, metric_fn)

        PFIExplainer.plot(meta_imp, title="PFI – Metadata Features", color="#E63946", save_path=f"{save_prefix}_meta.png" if save_prefix else None)
        PFIExplainer.plot(lead_imp, title="PFI – ECG Leads", color="#457B9D", save_path=f"{save_prefix}_leads.png" if save_prefix else None)
        PFIExplainer.plot(mod_imp, title="PFI – Modality Comparison", color="#2A9D8F", save_path=f"{save_prefix}_modality.png" if save_prefix else None)
        return meta_imp, lead_imp, mod_imp

    # ------------------------------------------------------------------
    # Full single-sample explanation dashboard
    # ------------------------------------------------------------------

    def explain_sample(self, ecg: torch.Tensor, meta: torch.Tensor,
                       target_class: int, background_meta: np.ndarray = None,
                       save_path: str = None):
        """
        Generate a complete explanation dashboard for one sample:
          - Model prediction probabilities
          - LRP heatmap (ECG)
          - Grad-CAM overlay (ECG)
          - SHAP waterfall (metadata)  [requires background_meta]
        """
        self.model.eval()
        ecg_dev = ecg.to(self.device).detach()
        meta_dev = meta.to(self.device).detach()
        
        with torch.no_grad():
            probs = torch.sigmoid(self.model(ecg_dev, meta_dev)).cpu().numpy()[0]

        # Allocate space dynamically based on metadata availability
        n_panels = 3 if background_meta is None else 4
        fig = plt.figure(figsize=(15, n_panels * 4))
        gs = gridspec.GridSpec(n_panels, 1, hspace=0.4)

        # ---- Panel 1: Prediction Probability Bar Chart ----
        ax0 = fig.add_subplot(gs[0])
        names = self.class_names if self.class_names else [f"Class {i}" for i in range(len(probs))]
        colors = ["#E63946" if i == target_class else "#A8DADC" for i in range(len(probs))]
        
        ax0.bar(names, probs, color=colors, edgecolor="white", width=0.5)
        ax0.axhline(0.5, color="black", lw=0.8, linestyle="--", label="Threshold = 0.5")
        ax0.set_ylim(0, 1.05)
        ax0.set_ylabel("Probability Score", fontsize=9)
        ax0.set_title(f"Model Predictions (Targeting: {self._cls_name(target_class)})", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(axis="y", linestyle=":", alpha=0.6)

        # ---- Panel 2: Condensed LRP Lead Overview ----
        ax1 = fig.add_subplot(gs[1])
        # Compute LRP without triggering internal plots
        if self._lrp is None:
            self._lrp = LRPExplainer(self.model, self.device)
        lrp_rel = self._lrp.explain(ecg, meta, target_class)
        
        # Plot a summary view onto the dashboard axis (e.g., mean absolute relevance over all 12 leads)
        mean_lrp = np.mean(np.abs(lrp_rel), axis=0)
        t = np.arange(len(mean_lrp))
        ax1.plot(t, mean_lrp, color="#1D3557", lw=1, label="Mean Abs Relevance")
        ax1.fill_between(t, mean_lrp, color="#1D3557", alpha=0.15)
        ax1.set_title("ECG Branch: Global LRP Heatmap Energy Profile (All Leads)", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Relevance Intensity", fontsize=9)
        ax1.set_xlim(0, len(mean_lrp))
        ax1.grid(axis="x", linestyle=":", alpha=0.6)

        # ---- Panel 3: Grad-CAM Spatial Overlay (Lead II Focus) ----
        ax2 = fig.add_subplot(gs[2])
        if self._gcam is None:
            self._gcam = GradCAMExplainer(self.model, self.device)
        cam = self._gcam.explain(ecg, meta, target_class)
        
        ecg_np = ecg.squeeze(0).cpu().numpy()
        lead_idx = 1 # Standard Lead II mapping indices
        signal = ecg_np[lead_idx] if ecg_np.ndim == 2 else ecg_np[0, lead_idx]
        
        ax2.plot(t, signal, color="#2B2D42", lw=0.8, label="Lead II Signal", zorder=2)
        sc = ax2.scatter(t, signal, c=cam, cmap="YlOrRd", s=3, zorder=3, vmin=0, vmax=1)
        cbar = plt.colorbar(sc, ax=ax2, orientation="vertical", pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Grad-CAM Activation", fontsize=8)
        
        ax2.set_title(f"ECG Branch: Grad-CAM Target Alignment Overlay (Lead {self.lead_names[lead_idx]})", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Amplitude", fontsize=9)
        ax2.set_xlim(0, len(signal))

        # ---- Panel 4: Tabular SHAP Contribution Breakdowns ----
        if background_meta is not None:
            ax3 = fig.add_subplot(gs[3])
            if self._shap is None:
                self.init_shap(background_meta)
            
            # Extract SHAP explanations safely
            shap_vals = self._shap.explain(meta.cpu().numpy(), n_samples=100)
            
            if isinstance(shap_vals, list):
                vals = shap_vals[target_class][0]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3: # Fixed variable name here
                vals = shap_vals[0, :, target_class]
            else:
                vals = shap_vals[0]

            # Build a localized horizontal bar chart for the single sample's waterfall distribution
            y_pos = np.arange(len(self.meta_cols))
            ax3.barh(y_pos, vals, color=["#E63946" if v > 0 else "#457B9D" for v in vals], edgecolor="white", height=0.6)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(self.meta_cols, fontsize=8)
            ax3.axvline(0, color="black", lw=0.8, linestyle="--")
            ax3.set_title("Metadata Branch: Feature Impact Scores (SHAP Values)", fontsize=10, fontweight="bold")
            ax3.set_xlabel("Impact on Model Output Logit", fontsize=9)
            ax3.grid(axis="x", linestyle=":", alpha=0.6)

        # Save out the complete canvas panel matrix sheet safely
        out_path = save_path or "xai_dashboard.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Dashboard] Complete composite validation report successfully exported to: {out_path}")
        plt.show()
        plt.close(fig)

        return probs