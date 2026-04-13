"""
xresnet1d.py
------------
PyTorch implementation of XResNet-1D for ECG / time-series classification.

Based on the fastai XResNet architecture, adapted for 1-D signals.
"""

import inspect
import re
from enum import Enum

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def delegates(to=None, keep=False):
    """Copy signature parameters from *to* into the decorated callable."""
    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop("kwargs")
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd["kwargs"] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f


def store_attr(self, nms: str):
    """Set instance attributes from the caller's local variables."""
    mod = inspect.currentframe().f_back.f_locals
    for n in re.split(", *", nms):
        setattr(self, n, mod[n])


def _conv_func(ndim: int = 2, transpose: bool = False):
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> nn.Module:
    if func and hasattr(m, "weight"):
        func(m.weight)
    with torch.no_grad():
        if getattr(m, "bias", None) is not None:
            m.bias.fill_(0.0)
    return m


def _get_norm(prefix: str, nf: int, ndim: int = 2, zero: bool = False, **kwargs) -> nn.Module:
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf: int, ndim: int = 2, norm_type=NormType.Batch, **kwargs) -> nn.Module:
    return _get_norm("BatchNorm", nf, ndim, zero=(norm_type == NormType.BatchZero), **kwargs)


def AdaptiveAvgPool(sz: int = 1, ndim: int = 2) -> nn.Module:
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)


def AvgPool(ks: int = 2, stride=None, padding: int = 0, ndim: int = 2, ceil_mode: bool = False) -> nn.Module:
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


# ---------------------------------------------------------------------------
# ConvLayer
# ---------------------------------------------------------------------------

class ConvLayer(nn.Sequential):
    """Conv → (optional BN) → (optional activation) layer."""

    def __init__(
        self,
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        padding=None,
        bias=None,
        ndim: int = 2,
        norm_type=NormType.Batch,
        bn_1st: bool = True,
        act_cls=nn.ReLU,
        transpose: bool = False,
        init=nn.init.kaiming_normal_,
        xtra=None,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (bn or inn)

        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(
            conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs),
            init,
        )

        layers = [conv]
        act_bn = []
        if act_cls is not None:
            act_bn.append(act_cls())
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra:
            layers.append(xtra)

        super().__init__(*layers)


# ---------------------------------------------------------------------------
# ResBlock (1-D)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block supporting expansion factors of 1 (basic) or >1 (bottleneck)."""

    @delegates(ConvLayer.__init__)
    def __init__(
        self,
        expansion: int,
        ni: int,
        nf: int,
        stride: int = 1,
        kernel_size: int = 3,
        groups: int = 1,
        reduction=None,
        nh1=None,
        nh2=None,
        dw: bool = False,
        g2: int = 1,
        sa: bool = False,
        sym: bool = False,
        norm_type=NormType.Batch,
        act_cls=nn.ReLU,
        ndim: int = 1,
        pool=AvgPool,
        pool_first: bool = True,
        **kwargs,
    ):
        super().__init__()
        norm2 = (
            NormType.BatchZero
            if norm_type == NormType.Batch
            else NormType.InstanceZero
            if norm_type == NormType.Instance
            else norm_type
        )

        if nh2 is None:
            nh2 = nf
        if nh1 is None:
            nh1 = nh2

        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)

        if expansion == 1:
            layers = [
                ConvLayer(ni, nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                ConvLayer(nh2, nf, kernel_size, groups=g2, **k1),
            ]
        else:
            layers = [
                ConvLayer(ni, nh1, 1, **k0),
                ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                ConvLayer(nh2, nf, 1, groups=g2, **k1),
            ]

        self.convs = nn.Sequential(*layers)
        self.idpath = nn.Sequential()

        if ni != nf:
            self.idpath = nn.Sequential(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride != 1:
            self.idpath = nn.Sequential(pool(2, ndim=ndim, ceil_mode=True))

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.idpath(x))


# ---------------------------------------------------------------------------
# XResNet1d
# ---------------------------------------------------------------------------

def init_cnn(m: nn.Module):
    """Kaiming-normal initialisation for Conv and Linear layers."""
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for child in m.children():
        init_cnn(child)


class XResNet1d(nn.Sequential):
    """XResNet architecture adapted for 1-D time-series / ECG signals."""

    @delegates(ResBlock)
    def __init__(
        self,
        block,
        expansion: int,
        layers: list,
        p: float = 0.0,
        input_channels: int = 3,
        num_classes: int = 1000,
        stem_szs: tuple = (32, 32, 64),
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        widen: float = 1.0,
        sa: bool = False,
        act_cls=nn.ReLU,
        lin_ftrs_head=None,
        ps_head: float = 0.5,
        bn_final_head: bool = False,
        bn_head: bool = True,
        act_head: str = "relu",
        concat_pooling: bool = True,
        **kwargs,
    ):
        store_attr(self, "block,expansion,act_cls")

        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(
                stem_szs[i],
                stem_szs[i + 1],
                ks=kernel_size_stem,
                stride=2 if i == 0 else 1,
                act_cls=act_cls,
                ndim=1,
            )
            for i in range(3)
        ]

        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs

        blocks = [
            self._make_layer(
                ni=block_szs[i],
                nf=block_szs[i + 1],
                blocks=l,
                stride=1 if i == 0 else 2,
                kernel_size=kernel_size,
                sa=sa and i == (len(layers) - 4),
                ndim=1,
                **kwargs,
            )
            for i, l in enumerate(layers)
        ]

        head = nn.Sequential(
            AdaptiveAvgPool(1, ndim=1),
            nn.Flatten(),
            nn.Linear(block_szs[-1] * expansion, num_classes),
        )

        super().__init__(*stem, nn.MaxPool1d(3, 2, 1), *blocks, head)
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        return nn.Sequential(
            *[
                self.block(
                    self.expansion,
                    ni if i == 0 else nf,
                    nf,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                    sa=sa and i == (blocks - 1),
                    act_cls=self.act_cls,
                    **kwargs,
                )
                for i in range(blocks)
            ]
        )

    def get_output_layer(self) -> nn.Module:
        return self[-1][-1]

    def set_output_layer(self, x: nn.Module):
        self[-1][-1] = x


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def xresnet1d101(**kwargs) -> XResNet1d:
    """XResNet1d-101: expansion=4, layers=[3,4,23,3]."""
    return XResNet1d(ResBlock, 4, [3, 4, 23, 3], **kwargs)


def build_xresnet1d101(num_classes: int, input_channels: int = 12, device: str = "cuda") -> XResNet1d:
    """Convenience wrapper: build and move to device.

    Args:
        num_classes: Number of output classes.
        input_channels: Number of ECG leads (default 12).
        device: 'cuda' or 'cpu'.

    Returns:
        XResNet1d model on the specified device.
    """
    model = xresnet1d101(
        input_channels=input_channels,
        num_classes=num_classes,
        kernel_size=5,
        kernel_size_stem=5,
    )
    return model.to(device)
