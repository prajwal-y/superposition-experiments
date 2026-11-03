"""E0c: mirror test. A neuron that detects a left- or right-facing pose should lose much of its response
when its strongest photos are flipped left to right; a neuron for an object should not care.

retention = mean(relu(response to mirrored photo)) / mean(relu(response to original)), over the
neuron's 100 strongest pool photos.
usage: python e0_mirror.py <pool_npz> <openimages_dir> <out_json>
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402

pool_npz, oi, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
P = np.load(pool_npz)
ids = P["ids"]
photo = oi / "val384"
dev = incv1.device()
model = incv1.load_model(dev)
named = [("mixed4d", 352), ("mixed4d", 417), ("mixed4d", 277), ("mixed4d", 241), ("mixed4b", 414), ("mixed4e", 55)]
rng = random.Random(0)
baseline = [("mixed4b", n) for n in rng.sample(range(512), 64)] + [("mixed4d", n) for n in rng.sample(range(528), 64)]


def retention(layer, n, k=100):
    top = np.argsort(-P[f"max_{layer}"][:, n].astype(np.float32))[:k]
    o, m = [], []
    with torch.no_grad():
        for s in range(0, k, 50):
            ims = [Image.open(photo / f"{ids[j]}.jpg").convert("RGB") for j in top[s:s + 50]]
            x = torch.cat([incv1.preprocess(im) for im in ims] + [incv1.preprocess(ImageOps.mirror(im)) for im in ims])
            a = incv1.layer_prerelu(model, x.to(dev), [layer])[layer][:, n].flatten(1).max(1).values.clamp(min=0).cpu()
            o.append(a[:len(ims)])
            m.append(a[len(ims):])
    return float(torch.cat(m).mean() / torch.cat(o).mean())


res = {f"{l[5:]}:{n}": round(retention(l, n), 3) for l, n in named}
base = [retention(l, n) for l, n in baseline]
res["baseline_128_random_4b_4d"] = {"median": round(float(np.median(base)), 3), "p5": round(float(np.percentile(base, 5)), 3),
                                    "p95": round(float(np.percentile(base, 95)), 3)}
for k in [k for k in res if ":" in k]:
    res.setdefault("share_of_random_neurons_that_keep_less", {})[k] = round(float(np.mean(np.array(base) < res[k])), 3)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
