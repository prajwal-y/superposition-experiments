"""A small sparse autoencoder trained on layer 4e of InceptionV1, over the 41,144-photo pool.

A photo's "layer numbers" are the 832 values of layer 4e for that photo, each neuron's value being its strongest
response anywhere in the photo (spatial max, pre-ReLU, from pool_acts.py). TopK SAE: 832 -> 4,096 concepts, exactly
8 active per photo, -> 832 rebuilt. Adam, 40 epochs, seed 0, CPU (about a minute on a laptop).
Reports the share of variance rebuilt across all photos, and the 8 concepts active on 4e:55's strongest photo with each
concept's top 4 photos (Open Images ids and authors).

usage: python sae_layer4e.py <pool_npz> <openimages_dir> <out_dir>      (writes <out_dir>/sae.json)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

POOL, OI, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
HERO = "0f58ef7a6b7c2d90"            # 4e:55's strongest photo (rank 1, the fox on the contact sheet)
N_LAT, K, EPOCHS, BATCH = 4096, 8, 40, 1024

torch.manual_seed(0)
P = np.load(POOL)
ids = P["ids"]
X = torch.from_numpy(P["max_mixed4e"].astype(np.float32))
mu = X.mean(0)
scale = float((X - mu).norm(dim=1).mean()) / np.sqrt(X.shape[1])
Xn = (X - mu) / scale                                                          # mean row norm = sqrt(832)
D = X.shape[1]

W_dec = torch.randn(N_LAT, D)
W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
W_dec.requires_grad_(True)
W_enc = W_dec.detach().t().clone().requires_grad_(True)
b_enc = torch.zeros(N_LAT, requires_grad=True)
b_dec = torch.zeros(D, requires_grad=True)
opt = torch.optim.Adam([W_dec, W_enc, b_enc, b_dec], lr=1e-3)


def encode(x):
    pre = (x - b_dec) @ W_enc + b_enc
    v, i = pre.topk(K, dim=1)
    z = torch.zeros_like(pre).scatter_(1, i, torch.relu(v))
    return z


def decode(z):
    return z @ W_dec + b_dec


n = len(Xn)
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0
    for s in range(0, n, BATCH):
        xb = Xn[perm[s:s + BATCH]]
        loss = ((decode(encode(xb)) - xb) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=1, keepdim=True)
        tot += float(loss) * len(xb)
    if ep % 10 == 9 or ep == 0:
        print(f"epoch {ep + 1}: mse {tot / n:.2f}")

with torch.no_grad():
    Z = torch.cat([encode(Xn[s:s + 4096]) for s in range(0, n, 4096)])
    R = decode(Z)
    fvu = float(((R - Xn) ** 2).sum() / ((Xn - Xn.mean(0)) ** 2).sum())
    alive = int((Z > 0).any(0).sum())
print(f"variance explained {1 - fvu:.3f}; concepts ever active {alive} of {N_LAT}")

h = int(np.where(ids == HERO)[0][0])
z = Z[h]
act = torch.nonzero(z).flatten().tolist()
act = sorted(act, key=lambda j: -float(z[j]))
x_orig = X[h].numpy()
x_reb = (R[h] * scale + mu).numpy()
peak = P["max_mixed4e"].astype(np.float32).max(0)                             # each neuron's pool peak
share = lambda v: [round(float(a), 3) for a in np.clip(v / peak, -1, 1.5)]
corr = float(np.corrcoef(x_orig, x_reb)[0, 1])
print("hero active", act, [round(float(z[j]), 2) for j in act], "corr", round(corr, 3))

csv_path = OI / "validation-images-with-rotation.csv"
authors = {r["ImageID"]: r["Author"] for r in csv.DictReader(open(csv_path))} if csv_path.exists() else {}
OUT.mkdir(parents=True, exist_ok=True)
concepts = []
for j in act:
    col = Z[:, j].numpy()
    o = np.argsort(-col)[:4]
    tops = []
    for q in o:
        s = str(ids[q])
        tops.append({"id": s, "author": authors.get(s, "")})
    concepts.append({"index": int(j), "value": round(float(z[j]), 3), "share_of_its_max": round(float(z[j] / col.max()), 3),
                     "photos_where_it_is_active": int((col > 0).sum()), "top_photos": tops})

data = {
    "source": "out/pool/pool.npz (max_mixed4e); trained by code/sae_layer4e.py",
    "credit": "Photos: Open Images, CC BY 2.0",
    "layer": "4e", "n_neurons": D, "n_concepts": N_LAT, "k_active": K, "epochs": EPOCHS, "n_photos": int(n),
    "variance_explained_all_photos": round(1 - fvu, 3), "concepts_ever_active": alive,
    "hero": {"id": HERO, "author": authors.get(HERO, ""),
             "layer_share_of_neuron_peak": share(x_orig), "rebuilt_share_of_neuron_peak": share(x_reb),
             "corr_layer_vs_rebuilt": round(corr, 3)},
    "active_concepts": concepts,
}
json.dump(data, open(OUT / "sae.json", "w"))
print(json.dumps({k: v for k, v in data.items() if k not in ("hero",)}, indent=1)[:3000])
