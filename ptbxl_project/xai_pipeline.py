"""
xai_pipeline.py
---------------
Full XAI pipeline for the FusionModel (xresnet1d101 + MetaMLP).

Methods implemented:
  ECG branch  : LRP (zennit via ECGOnlyWrapper), Deep Taylor Decomposition (DTD), optional Grad-CAM
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
    """
    Return (ecg_branch, meta_branch, classifier) as separate modules.
    Assumes FusionModel structure from fusion.py.
    """
    ecg  = fusion_model.ecg
    meta = fusion_model.meta
    clf  = fusion_model.classifier   # already materialised after training
    return ecg, meta, clf

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
        return self.fusion(ecg, self.meta)


# ============================================================
# 2.  LRP  (Layer-wise Relevance Propagation)  – ECG branch
# ============================================================

class LRPExplainer:
    """
    LRP for xresnet1d101 using the zennit library.

    Rules applied:
      - Input / first conv layers : ZBox rule  (handles raw signals well)
      - Hidden conv layers        : AlphaBeta rule (alpha=2, beta=1)
      - BatchNorm / Linear        : Pass rule
    """

    def __init__(self, fusion_model: nn.Module, device: str = "cpu"):
        zennit = _require("zennit", "zennit")
        from zennit.composites import EpsilonPlusFlat
        from zennit.attribution import Gradient

        self.fusion_model = fusion_model.to(device).eval()  # hau kendu itebau: self.ecg_model = ecg_model.to(device)
        self.device    = device
        self._composite_cls = EpsilonPlusFlat   # sensible default for ResNets
        self._Gradient      = Gradient

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        from zennit.composites import EpsilonPlusFlat
        from zennit.attribution import Gradient

        ecg = ecg.to(self.device).requires_grad_(True)
        composite = EpsilonPlusFlat()

        # Run once without composite to get output shape
        meta    = meta.to(self.device)
        wrapper = ECGOnlyWrapper(self.fusion_model, meta)
        with torch.no_grad():
            out_shape = wrapper(ecg).shape[1]
        with Gradient(model=wrapper, composite=composite) as attributor:
            out, relevance = attributor(
            ecg,
            torch.eye(out_shape, device=self.device)[target_class].unsqueeze(0)
        )

        return relevance.squeeze(0).detach().cpu().numpy()

    def plot(self, relevance: np.ndarray, lead_names=None, title="LRP – ECG Relevance",
             save_path=None):
        """Plot per-lead relevance as a heatmap."""
        if lead_names is None:
            lead_names = [f"Lead {i+1}" for i in range(relevance.shape[0])]

        fig, axes = plt.subplots(relevance.shape[0], 1,
                                 figsize=(14, relevance.shape[0] * 1.2), sharex=True)
        if relevance.shape[0] == 1:
            axes = [axes]

        vmax = np.abs(relevance).max()
        for ax, lead, name in zip(axes, relevance, lead_names):
            ax.fill_between(range(len(lead)), lead,
                            where=(lead > 0), color="#E63946", alpha=0.7, label="positive")
            ax.fill_between(range(len(lead)), lead,
                            where=(lead < 0), color="#457B9D", alpha=0.7, label="negative")
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

    DTD is theoretically equivalent to LRP with specific root-point rules.
    It validates the LRP explanations by providing a complementary perspective.
    """

    def __init__(self, fusion_model: nn.Module, device: str = "cpu"):
        _require("zennit", "zennit")
        self.fusion_model = fusion_model.to(device).eval()
        self.device    = device

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute DTD attribution for one ECG sample.

        Args:
            ecg          : Tensor (1, 12, 1000)
            target_class : Class index

        Returns:
            attribution  : np.ndarray (12, 1000)
        """
        from zennit.composites import EpsilonFlat       # DTD ≈ LRP with Flat/WSquare at input
        from zennit.attribution import Gradient

        ecg = ecg.to(self.device).requires_grad_(True)
        composite = EpsilonFlat()

        meta    = meta.to(self.device)
        wrapper = ECGOnlyWrapper(self.fusion_model, meta)
        with torch.no_grad():
            out_shape = wrapper(ecg).shape[1]

        with Gradient(model=wrapper, composite=composite) as attributor:
            _, attr = attributor(ecg, torch.eye(out_shape, device=self.device)[target_class].unsqueeze(0))

        return attr.squeeze(0).detach().cpu().numpy()

    def compare_with_lrp(self, lrp_rel: np.ndarray, dtd_rel: np.ndarray,
                         lead_idx: int = 0, save_path=None):
        """
        Side-by-side comparison of LRP vs DTD for one lead.
        Useful to validate consistency between the two methods.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
        t = range(lrp_rel.shape[1])

        ax1.plot(t, lrp_rel[lead_idx], color="#E63946", lw=0.8)
        ax1.axhline(0, color="k", lw=0.5)
        ax1.set_title(f"LRP  –  Lead {lead_idx+1}", fontsize=10)
        ax1.set_ylabel("Relevance")

        ax2.plot(t, dtd_rel[lead_idx], color="#2A9D8F", lw=0.8)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_title(f"DTD  –  Lead {lead_idx+1}", fontsize=10)
        ax2.set_ylabel("Attribution")
        ax2.set_xlabel("Time step (samples)")

        corr = np.corrcoef(lrp_rel[lead_idx], dtd_rel[lead_idx])[0, 1]
        fig.suptitle(f"LRP vs DTD validation  (Pearson r = {corr:.3f})", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return corr


# ============================================================
# 4.  GradCAM  –  optional / qualitative  (ECG branch)
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
        self._hook_target()

    def _hook_target(self):
        # xresnet1d: Sequential([stem0, stem1, stem2, MaxPool, block0, block1, block2, block3, head])
        # Last conv block = index -2
        target = list(self.fusion_model.ecg.children())[-2]

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        target.register_forward_hook(fwd_hook)
        target.register_full_backward_hook(bwd_hook)

    def explain(self, ecg: torch.Tensor, meta: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute Grad-CAM activation map.

        Returns:
            cam : np.ndarray (1000,)  – upsampled to input length
        """
        ecg = ecg.to(self.device).requires_grad_(True)
        meta = meta.to(self.device)
        self.fusion_model.zero_grad()
        out = self.fusion_model(ecg, meta)

        score = out[0, target_class]
        score.backward()

        # weights = global-average of gradients over time axis
        weights = self._gradients.mean(dim=-1, keepdim=True)   # (1, C, 1)
        cam = (weights * self._activations).sum(dim=1)         # (1, T_feat)
        cam = torch.relu(cam).squeeze(0).cpu().numpy()

        # Upsample to input length (1000)
        cam = np.interp(np.linspace(0, len(cam)-1, 1000),
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
        sc = ax.scatter(t, signal, c=cam, cmap="hot", s=2, zorder=3,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Grad-CAM activation")
        ax.set_title(f"Grad-CAM  –  Lead {lead_idx+1}", fontsize=10)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Amplitude")
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

        self.explainer = shap.KernelExplainer(_predict, background)

    def explain(self, meta_samples: np.ndarray, n_samples: int = 100) -> object:
        """
        Compute SHAP values for meta_samples.

        Args:
            meta_samples : (N, n_features)
            n_samples    : Number of coalition samples for KernelSHAP

        Returns:
            shap_values  : List of arrays, one per output dimension
        """
        return self.explainer.shap_values(meta_samples, nsamples=n_samples)

    def summary_plot(self, shap_values, meta_samples: np.ndarray,
                     feature_names=None, class_idx: int = 0, save_path=None):
        """
        Beeswarm summary plot for one output class.
        Shows global feature importance + direction of effect.
        """
        vals = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
        self._shap.summary_plot(vals, meta_samples,
                                feature_names=feature_names,
                                show=False)
        plt.title(f"SHAP Summary  –  class {class_idx}", fontsize=11)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def waterfall_plot(self, shap_values, meta_sample: np.ndarray,
                       feature_names=None, class_idx: int = 0,
                       sample_idx: int = 0, save_path=None):
        """
        Waterfall (force) plot for a single patient prediction.
        """
        vals = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
        exp = self._shap.Explanation(
            values=vals[sample_idx],
            base_values=self.explainer.expected_value[class_idx]
                if isinstance(self.explainer.expected_value, list)
                else self.explainer.expected_value,
            data=meta_sample[sample_idx],
            feature_names=feature_names,
        )
        self._shap.waterfall_plot(exp, show=False)
        plt.title(f"SHAP Waterfall  –  sample {sample_idx}, class {class_idx}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


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
    def _score(self, ecg: torch.Tensor, meta: torch.Tensor,
               targets: torch.Tensor, metric_fn) -> float:
        ecg, meta = ecg.to(self.device), meta.to(self.device)
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
        baseline = self._score(ecg, meta, targets, metric_fn)
        n_feats  = meta.shape[1]
        names    = feature_names or [f"meta_{i}" for i in range(n_feats)]
        results  = {}

        for i, name in enumerate(names):
            drops = []
            for _ in range(self.n_repeats):
                meta_perm = meta.clone()
                idx = torch.randperm(meta_perm.shape[0])
                meta_perm[:, i] = meta_perm[idx, i]
                drops.append(baseline - self._score(ecg, meta_perm, targets, metric_fn))
            results[name] = float(np.mean(drops))

        return results

    def run_ecg_lead_pfi(self, ecg: torch.Tensor, meta: torch.Tensor,
                         targets: torch.Tensor, metric_fn,
                         lead_names=None) -> dict:
        """
        Permute each ECG lead independently.

        Returns:
            dict  lead_name → mean importance (drop in metric)
        """
        baseline  = self._score(ecg, meta, targets, metric_fn)
        n_leads   = ecg.shape[1]
        names     = lead_names or [f"Lead {i+1}" for i in range(n_leads)]
        results   = {}

        for i, name in enumerate(names):
            drops = []
            for _ in range(self.n_repeats):
                ecg_perm = ecg.clone()
                idx = torch.randperm(ecg_perm.shape[0])
                ecg_perm[:, i, :] = ecg_perm[idx, i, :]
                drops.append(baseline - self._score(ecg_perm, meta, targets, metric_fn))
            results[name] = float(np.mean(drops))

        return results

    def run_modality_pfi(self, ecg: torch.Tensor, meta: torch.Tensor,
                         targets: torch.Tensor, metric_fn) -> dict:
        """
        Permute whole modalities to compare ECG vs metadata contribution.
        """
        baseline = self._score(ecg, meta, targets, metric_fn)
        results  = {}

        # Permute all ECG
        drops = []
        for _ in range(self.n_repeats):
            ecg_perm = ecg[torch.randperm(ecg.shape[0])]
            drops.append(baseline - self._score(ecg_perm, meta, targets, metric_fn))
        results["ECG (whole modality)"] = float(np.mean(drops))

        # Permute all metadata
        drops = []
        for _ in range(self.n_repeats):
            meta_perm = meta[torch.randperm(meta.shape[0])]
            drops.append(baseline - self._score(ecg, meta_perm, targets, metric_fn))
        results["Metadata (whole modality)"] = float(np.mean(drops))

        return results

    @staticmethod
    def plot(importance_dict: dict, title="Permutation Feature Importance",
             color="#E63946", save_path=None):
        """Horizontal bar chart of importance scores."""
        names  = list(importance_dict.keys())
        values = list(importance_dict.values())
        order  = np.argsort(values)
        names  = [names[i] for i in order]
        values = [values[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
        bars = ax.barh(names, values, color=color, edgecolor="white", height=0.6)
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("Mean drop in AUC (higher = more important)")
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=7)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


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

    def __init__(self, fusion_model: nn.Module, device: str = "cpu",
                 class_names=None, meta_cols=None,
                 lead_names=None, n_pfi_repeats: int = 5):
        self.model       = fusion_model.to(device).eval()
        self.device      = device
        self.class_names = class_names or []
        self.meta_cols   = meta_cols   or []
        self.lead_names  = lead_names  or self.LEAD_NAMES
        self.meta_model = fusion_model.meta   # only need meta_model for SHAP

        # Lazy-initialise explainers on demand
        self._lrp   = None
        self._dtd   = None
        self._gcam  = None
        self._shap  = None
        self._pfi   = PFIExplainer(fusion_model, device, n_pfi_repeats)

    # ------------------------------------------------------------------
    # ECG XAI
    # ------------------------------------------------------------------

    def explain_ecg_lrp(self, ecg, meta, target_class, save_path=None) -> np.ndarray:
        """Run LRP on one ECG sample and plot the result."""
        if self._lrp is None:
            self._lrp = LRPExplainer(self.model, self.device)
        rel = self._lrp.explain(ecg, meta, target_class)
        self._lrp.plot(rel, lead_names=self.lead_names,
                       title=f"LRP – class: {self._cls_name(target_class)}",
                       save_path=save_path)
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
        if self._dtd is None:
            self._dtd = DTDExplainer(self.model, self.device)
        corr = self._dtd.compare_with_lrp(lrp_rel, dtd_rel,
                                           lead_idx=lead_idx, save_path=save_path)
        print(f"[DTD/LRP validation] Pearson r on lead {lead_idx}: {corr:.4f}")
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
        print(f"[SHAP] Explainer initialised with {len(background_meta)} background samples.")

    def explain_meta_shap(self, meta_samples: np.ndarray,
                          class_idx: int = 0, n_samples: int = 100,
                          save_summary: str = None,
                          save_waterfall: str = None,
                          sample_idx: int = 0):
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
        self._shap.summary_plot(shap_vals, meta_samples,
                                feature_names=self.meta_cols,
                                class_idx=class_idx,
                                save_path=save_summary)
        self._shap.waterfall_plot(shap_vals, meta_samples,
                                  feature_names=self.meta_cols,
                                  class_idx=class_idx,
                                  sample_idx=sample_idx,
                                  save_path=save_waterfall)
        return shap_vals

    # ------------------------------------------------------------------
    # Global PFI
    # ------------------------------------------------------------------

    def run_pfi(self, ecg: torch.Tensor, meta: torch.Tensor,
                targets: torch.Tensor, metric_fn,
                save_prefix: str = None):
        """
        Run all three levels of PFI:
          1. Per metadata column
          2. Per ECG lead
          3. Whole-modality comparison

        Args:
            ecg        : (N, 12, 1000) tensor
            meta       : (N, n_features) tensor
            targets    : (N, n_classes) tensor
            metric_fn  : callable (preds_np, targets_np) → scalar
            save_prefix: If provided, save plots with this prefix

        Returns:
            Tuple of (meta_imp, lead_imp, modality_imp) dicts
        """
        print("[PFI] Computing metadata feature importance …")
        meta_imp = self._pfi.run_meta_pfi(
            ecg, meta, targets, metric_fn, feature_names=self.meta_cols)

        print("[PFI] Computing ECG lead importance …")
        lead_imp = self._pfi.run_ecg_lead_pfi(
            ecg, meta, targets, metric_fn, lead_names=self.lead_names)

        print("[PFI] Computing modality-level importance …")
        mod_imp = self._pfi.run_modality_pfi(ecg, meta, targets, metric_fn)

        PFIExplainer.plot(meta_imp,
                          title="PFI – Metadata Features",
                          color="#E63946",
                          save_path=f"{save_prefix}_meta.png" if save_prefix else None)

        PFIExplainer.plot(lead_imp,
                          title="PFI – ECG Leads",
                          color="#457B9D",
                          save_path=f"{save_prefix}_leads.png" if save_prefix else None)

        PFIExplainer.plot(mod_imp,
                          title="PFI – Modality Comparison",
                          color="#2A9D8F",
                          save_path=f"{save_prefix}_modality.png" if save_prefix else None)

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

        Args:
            ecg            : (1, 12, 1000) tensor
            meta           : (1, n_features) tensor
            target_class   : Class to explain
            background_meta: np.ndarray for SHAP (required for metadata panel)
            save_path      : Path to save the dashboard figure
        """
        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                self.model(ecg.to(self.device), meta.to(self.device))
            ).cpu().numpy()[0]

        n_panels = 3 if background_meta is None else 4
        fig = plt.figure(figsize=(16, n_panels * 3 + 1))
        gs  = gridspec.GridSpec(n_panels, 1, hspace=0.5)

        # ---- Panel 0: prediction bar chart ----
        ax0 = fig.add_subplot(gs[0])
        names = self.class_names if self.class_names else [f"Class {i}" for i in range(len(probs))]
        colors = ["#E63946" if i == target_class else "#A8DADC" for i in range(len(probs))]
        ax0.bar(names, probs, color=colors, edgecolor="white")
        ax0.set_ylim(0, 1)
        ax0.set_ylabel("Probability")
        ax0.set_title("Model Predictions", fontsize=10)
        ax0.axhline(0.5, color="k", lw=0.8, linestyle="--", label="threshold=0.5")
        ax0.legend(fontsize=7)

        plt.savefig(save_path or "xai_dashboard.png", dpi=150, bbox_inches="tight")
        print(f"[Dashboard] Saved to {save_path or 'xai_dashboard.png'}")

        # LRP + GradCAM plotted separately (they use their own figures)
        print("[Dashboard] Running LRP …")
        lrp_save = save_path.replace(".png", "_lrp.png") if save_path else "xai_lrp.png"
        self.explain_ecg_lrp(ecg, target_class, save_path=lrp_save)

        print("[Dashboard] Running Grad-CAM …")
        gcam_save = save_path.replace(".png", "_gcam.png") if save_path else "xai_gcam.png"
        self.explain_ecg_gradcam(ecg, target_class, save_path=gcam_save)

        if background_meta is not None:
            print("[Dashboard] Running SHAP …")
            if self._shap is None:
                self.init_shap(background_meta)
            shap_save = save_path.replace(".png", "_shap.png") if save_path else "xai_shap.png"
            self.explain_meta_shap(meta.cpu().numpy(), class_idx=target_class,
                                   save_waterfall=shap_save)

        plt.show()
        return probs

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _cls_name(self, idx: int) -> str:
        if self.class_names and idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)
