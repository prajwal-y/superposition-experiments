"""Shared helpers: the original 2014 InceptionV1 (Lucid's inception5h, via Lucent).

Neuron ids follow the Circuits thread: "4e:55" = channel 55 of mixed4e. For mixed4e that
channel sits in the 1x1 branch (channels 0-255), so its pre-ReLU value is
mixed4e_1x1_pre_relu_conv[:, 55]. Pre-ReLU values let us see a neuron's negative side.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

EXP = Path(__file__).resolve().parents[1]           # the repository root
OUT = EXP / "out"
DATA = Path(os.environ.get("SUPERPOSITION_DATA", EXP / "data"))   # Open Images and C4 downloads (see README)

# Branch layout of each mixed layer (1x1, 3x3, 5x5, pool_reduce), from Lucid's InceptionV1.
BRANCHES = {
    "mixed3a": [("1x1", 64), ("3x3", 128), ("5x5", 32), ("pool_reduce", 32)],
    "mixed3b": [("1x1", 128), ("3x3", 192), ("5x5", 96), ("pool_reduce", 64)],
    "mixed4a": [("1x1", 192), ("3x3", 204), ("5x5", 48), ("pool_reduce", 64)],
    "mixed4b": [("1x1", 160), ("3x3", 224), ("5x5", 64), ("pool_reduce", 64)],
    "mixed4c": [("1x1", 128), ("3x3", 256), ("5x5", 64), ("pool_reduce", 64)],
    "mixed4d": [("1x1", 112), ("3x3", 288), ("5x5", 64), ("pool_reduce", 64)],
    "mixed4e": [("1x1", 256), ("3x3", 320), ("5x5", 128), ("pool_reduce", 128)],
    "mixed5a": [("1x1", 256), ("3x3", 320), ("5x5", 128), ("pool_reduce", 128)],
    "mixed5b": [("1x1", 384), ("3x3", 384), ("5x5", 128), ("pool_reduce", 128)],
}


def branch_of(layer: str, channel: int) -> tuple[str, int]:
    """Map a concatenated-layer channel to (pre-relu conv module name, channel within it)."""
    start = 0
    for name, width in BRANCHES[layer]:
        if channel < start + width:
            return f"{layer}_{name}_pre_relu_conv", channel - start
        start += width
    raise ValueError(f"{layer}:{channel} out of range")


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(dev: torch.device | None = None):
    from lucent.modelzoo import inceptionv1
    m = inceptionv1(pretrained=True).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(dev or device())


def preprocess(img: Image.Image, size: int = 224, center_crop: bool = True) -> torch.Tensor:
    """PIL image -> 1x3xHxW tensor in Lucid's value range (x*255 - 117), RGB."""
    img = img.convert("RGB")
    if center_crop:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x * 255.0 - 117.0).unsqueeze(0)


def from_unit(x01: torch.Tensor) -> torch.Tensor:
    """Bx3xHxW tensor in [0,1] -> Lucid's value range."""
    return x01 * 255.0 - 117.0


class Recorder:
    """Record outputs of named modules during a forward pass."""

    def __init__(self, model, names: list[str]):
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = []
        mods = dict(model.named_modules())
        for n in names:
            self.handles.append(mods[n].register_forward_hook(self._hook(n)))

    def _hook(self, name):
        def f(_m, _i, out):
            self.acts[name] = out
        return f

    def close(self):
        for h in self.handles:
            h.remove()


def neuron_maps(model, x: torch.Tensor, specs: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
    """Pre-ReLU spatial maps (BxHxW) for neurons like ("mixed4e", 55)."""
    names = sorted({branch_of(l, c)[0] for l, c in specs})
    rec = Recorder(model, names)
    try:
        model(x)
    finally:
        rec.close()
    out = {}
    for l, c in specs:
        mod, k = branch_of(l, c)
        out[f"{l[5:]}:{c}"] = rec.acts[mod][:, k]
    return out


def max_spatial(maps: torch.Tensor) -> torch.Tensor:
    return maps.flatten(1).max(1).values


def min_spatial(maps: torch.Tensor) -> torch.Tensor:
    return maps.flatten(1).min(1).values


LAYERS = ["mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a"]


class _Stop(Exception):
    pass


MAX_BATCH = 128  # one batch of 432 at 224 px asked MPS for 18.7 GB; keep batches small


def layer_prerelu(model, x: torch.Tensor, layers: list[str]) -> dict[str, torch.Tensor]:
    """Full pre-ReLU maps (BxCxHxW) of whole mixed layers, branches concatenated in Lucid order.
    The forward pass stops once the deepest requested layer is done."""
    assert len(x) <= MAX_BATCH, f"batch {len(x)} > {MAX_BATCH}"
    names = [f"{l}_{b}_pre_relu_conv" for l in layers for b, _ in BRANCHES[l]]
    rec = Recorder(model, names)
    order = list(BRANCHES)  # mixed3a ... mixed5b, in network order
    deepest = max(layers, key=order.index)
    mods = dict(model.named_modules())

    def stop(_m, _i, _o):
        raise _Stop

    h = mods[deepest].register_forward_hook(stop)  # the CatLayer runs after all four branches
    try:
        model(x)
    except _Stop:
        pass
    finally:
        rec.close()
        h.remove()
    return {l: torch.cat([rec.acts[f"{l}_{b}_pre_relu_conv"] for b, _ in BRANCHES[l]], 1) for l in layers}


class Ablate:
    """Zero chosen channels of mixed layers' post-ReLU outputs (the CatLayer outputs)."""

    def __init__(self, model, chans: dict[str, list[int]]):
        mods = dict(model.named_modules())
        self.handles = []
        for layer, cs in chans.items():
            if not cs:
                continue
            idx = torch.tensor(sorted(cs))

            def hook(_m, _i, out, idx=idx):
                out = out.clone()
                out[:, idx.to(out.device)] = 0
                return out
            self.handles.append(mods[layer].register_forward_hook(hook))

    def close(self):
        for h in self.handles:
            h.remove()
