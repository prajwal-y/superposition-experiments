"""E0a: orientation tuning of InceptionV1 3b neurons to synthetic curves (as in Curve Detectors, 2020).

A curve is a 120-degree arc of a circle; its orientation phi is the direction its bulge points.
Saves every 3b neuron's spatial-max pre-ReLU response for each (phi, radius, polarity).
usage: python e0_curves.py <out_npz>
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402

S, K = 224, 4


def arc(phi_deg, radius, width, dark_on_light, span=120):
    bg, fg = (217, 0) if dark_on_light else (38, 255)
    img = Image.new("L", (S * K, S * K), bg)
    d = ImageDraw.Draw(img)
    # put the arc's midpoint at the image centre; the circle centre sits opposite the bulge
    ph = math.radians(phi_deg)
    cx, cy = S * K / 2 - radius * K * math.cos(ph), S * K / 2 - radius * K * math.sin(ph)
    pts = [(cx + radius * K * math.cos(a), cy + radius * K * math.sin(a))
           for a in np.linspace(ph - math.radians(span / 2), ph + math.radians(span / 2), 90)]
    d.line(pts, fill=fg, width=int(width * K), joint="curve")
    h = width * K / 2
    for x, y in pts:
        d.ellipse([x - h, y - h, x + h, y + h], fill=fg)
    return img.resize((S, S), Image.LANCZOS).convert("RGB")


phis = list(range(0, 360, 5))
radii = [12, 20, 32]
pols = [True, False]
dev = incv1.device()
model = incv1.load_model(dev)
grid = [(p, r, pol) for r in radii for pol in pols for p in phis]
xs = torch.cat([incv1.preprocess(arc(p, r, 3, pol), center_crop=False) for p, r, pol in grid])
with torch.no_grad():
    a = torch.cat([incv1.layer_prerelu(model, xs[i:i + 64].to(dev), ["mixed3b"])["mixed3b"].cpu()
                   for i in range(0, len(xs), 64)])
# value at the image centre (where the curve sits) and the spatial max
c = a.shape[-1] // 2
center = a[:, :, c - 1:c + 1, c - 1:c + 1].flatten(2).max(2).values.cpu().numpy()
smax = a.flatten(2).max(2).values.cpu().numpy()
np.savez(sys.argv[1], phi=np.array([g[0] for g in grid]), radius=np.array([g[1] for g in grid]),
         dark=np.array([g[2] for g in grid]), center=center, smax=smax)
arc(45, 20, 3, True).save(Path(sys.argv[1]).with_suffix(".example.png"))
print("saved", len(grid), "curves")
