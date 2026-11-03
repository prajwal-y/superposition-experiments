"""E0d: what is the curve neuron 3b:379 built from? (tool 4, taught on the easy neuron)
1. Attribution of 3b:379 (spatial max) to each mixed3a channel (gradient x activation, summed over space),
   averaged over its 192 strongest pool photos -> its 10 strongest inputs.
2. Knock out those 10 (zero their outputs) vs 20 draws of 10 random 3a channels: share of response lost.
3. Are the inputs themselves curve/line detectors? Their orientation tuning on the same synthetic curves (E0a).
usage: python e0_curve_knockout.py <pool_npz> <openimages_dir> <out_json>
"""
import json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402
import math  # noqa: E402
from PIL import ImageDraw  # noqa: E402


def arc(phi_deg, radius, width, dark_on_light, span=120, S=224, K=4):
    """Copied from e0_curves.py (importing that script runs it)."""
    bg, fg = (217, 0) if dark_on_light else (38, 255)
    img = Image.new("L", (S * K, S * K), bg)
    d = ImageDraw.Draw(img)
    ph = math.radians(phi_deg)
    cx, cy = S * K / 2 - radius * K * math.cos(ph), S * K / 2 - radius * K * math.sin(ph)
    pts = [(cx + radius * K * math.cos(a), cy + radius * K * math.sin(a))
           for a in np.linspace(ph - math.radians(span / 2), ph + math.radians(span / 2), 90)]
    d.line(pts, fill=fg, width=int(width * K), joint="curve")
    h = width * K / 2
    for x, y in pts:
        d.ellipse([x - h, y - h, x + h, y + h], fill=fg)
    return img.resize((S, S), Image.LANCZOS).convert("RGB")

pool, oi, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
P = np.load(pool); ids = P["ids"]; photo = oi / "val384"
dev = torch.device("cpu"); model = incv1.load_model(dev)
N, L = 379, "mixed3b"
top = np.argsort(-P["max_mixed3b"][:, N].astype(np.float32))[:192]
mods = dict(model.named_modules()); mod_n, k = incv1.branch_of(L, N)
got = {}
def keep(_m, _i, o): o.retain_grad(); got["a"] = o
h1 = mods["mixed3a"].register_forward_hook(keep)
h2 = mods[mod_n].register_forward_hook(lambda _m, _i, o: got.__setitem__("n", o))
attr = torch.zeros(256)
for s in range(0, len(top), 32):
    x = torch.cat([incv1.preprocess(Image.open(photo / f"{ids[j]}.jpg")) for j in top[s:s + 32]])
    with torch.enable_grad():
        model(x.requires_grad_(True))
        got["n"][:, k].flatten(1).max(1).values.sum().backward()
    attr += (got["a"].grad * got["a"]).sum((2, 3)).sum(0).detach()
h1.remove(); h2.remove()
attr /= len(top)
inputs = [int(c) for c in torch.argsort(attr, descending=True)[:10]]

def resp(ablate=None):
    ab = incv1.Ablate(model, {"mixed3a": ablate}) if ablate else None
    vals = []
    try:
        with torch.no_grad():
            for s in range(0, len(top), 64):
                x = torch.cat([incv1.preprocess(Image.open(photo / f"{ids[j]}.jpg")) for j in top[s:s + 64]])
                a = incv1.layer_prerelu(model, x, [L])[L][:, N]
                vals.append(a.flatten(1).max(1).values.clamp(min=0))
    finally:
        if ab: ab.close()
    return float(torch.cat(vals).mean())
base = resp(); drop_top = 1 - resp(inputs) / base
rng = random.Random(0)
drops = [1 - resp(rng.sample(range(256), 10)) / base for _ in range(20)]
# orientation tuning of the inputs on the same curves as E0a (radius 20, both polarities, centre value)
phis = list(range(0, 360, 5))
xs = torch.cat([incv1.preprocess(arc(p, 20, 3, pol), center_crop=False) for pol in (True, False) for p in phis])
with torch.no_grad():
    a3 = torch.cat([incv1.layer_prerelu(model, xs[i:i + 72], ["mixed3a"])["mixed3a"] for i in range(0, len(xs), 72)])
c = a3.shape[-1] // 2
cen = a3[:, :, c - 1:c + 1, c - 1:c + 1].flatten(2).max(2).values.numpy()
tun = np.maximum(cen[:72], cen[72:])          # best polarity per angle
tuning = {}
for ch in inputs:
    t = tun[:, ch]; med = np.median(t); pk = t.max()
    width = int(((t - med) > 0.5 * (pk - med)).sum() * 5)
    tuning[ch] = {"preferred_angle": phis[int(t.argmax())], "half_width_deg": width,
                  "tuned": bool(pk - med > 0.5 * abs(pk) and width < 180)}
r379 = np.load(Path(__file__).parents[1] / "out/e0/curves.npz")
res = {"neuron": "3b:379", "photos": len(top), "top10_inputs_3a": inputs,
       "attribution": [round(float(attr[c]), 1) for c in inputs],
       "drop_knocking_out_top10": round(drop_top, 3),
       "drop_random10_mean": round(float(np.mean(drops)), 3), "drop_random10_max": round(float(np.max(drops)), 3),
       "input_tuning": tuning,
       "3b379_preferred_angle_r20": int(np.array(phis)[np.argmax(np.stack([r379['center'][(r379['phi'] == p) & (r379['radius'] == 20)][:, N].max() for p in phis]))])}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
