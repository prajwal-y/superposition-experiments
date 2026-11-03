"""E2: toy models for Part 1 and Part 3 (CPU, seconds to minutes).

E2a  Toy Models of Superposition (Elhage et al. 2022), the 5-in-2 case: x' = ReLU(W^T W x + b), 5 features of equal
     importance, 2 hidden dimensions. Sweep how rare the features are; count how many get an arrow, save the arrows,
     and save training snapshots of the sparse case (for the "snap into a pentagon" animation).
E2b  How much room many dimensions give: typical and worst overlap (|cosine|) between random arrows as dimensions grow.
E2c  Which things share a neuron depends on what fires together. 16 features, h = ReLU(W1 x + b1) with k neurons,
     x' = ReLU(W2 h + b2):
       ring    - the features are curve orientations on a ring; one orientation at a time appears and its
                 neighbours fire with it (like real curve detectors)
       onehot  - at most one feature at a time, no neighbours (features that never fire together)
     For each hidden neuron we record its response to every feature alone and measure whether the features it
     responds to are neighbours on the ring.
usage: python e2_toy.py <out_dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
R = {}


# ---------------- E2a ----------------
def train_5in2(density, steps=12000, snap_every=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(2, 5, generator=g) * 0.3)
    b = torch.nn.Parameter(torch.zeros(5))
    opt = torch.optim.Adam([W, b], lr=3e-3)
    snaps = []
    for s in range(steps):
        x = torch.rand(1024, 5, generator=g) * (torch.rand(1024, 5, generator=g) < density)
        xh = torch.relu(x @ W.T @ W + b)
        loss = ((x - xh) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if snap_every and (s % snap_every == 0 or s == steps - 1):
            snaps.append({"step": s, "W": W.detach().T.tolist(), "loss": float(loss)})
    return W.detach(), b.detach(), float(loss), snaps


rows = []
for dens in [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]:
    best = None
    for seed in range(3):  # keep the best of three starts, as toy-model sweeps usually do
        W, b, loss, _ = train_5in2(dens, seed=seed)
        if best is None or loss < best[2]:
            best = (W, b, loss)
    W, b, loss = best
    norms = W.norm(dim=0)
    rep = int((norms > 0.5).sum())
    ang = sorted(float(math.degrees(math.atan2(W[1, i], W[0, i])) % 360) for i in range(5) if norms[i] > 0.5)
    gaps = [round((ang[(i + 1) % len(ang)] - ang[i]) % 360, 1) for i in range(len(ang))] if len(ang) > 1 else []
    rows.append({"density": dens, "features_with_an_arrow": rep, "arrow_lengths": [round(float(n), 2) for n in norms],
                 "angle_gaps_deg": gaps, "arrows": W.T.tolist(), "bias": b.tolist(), "loss": round(loss, 5)})
R["E2a_5in2"] = rows
_, _, _, snaps = train_5in2(0.05, steps=6000, snap_every=50)
json.dump(snaps, open(out / "E2a_snapshots_density0.05.json", "w"))

# ---------------- E2b ----------------
ov = {}
g = torch.Generator().manual_seed(1)
for d in [2, 3, 10, 100, 1000, 10000]:
    n = 4000
    v = torch.randn(n, d, generator=g)
    v = v / v.norm(dim=1, keepdim=True)
    c = (v @ v.T).abs()
    iu = torch.triu_indices(n, n, 1)
    vals = c[iu[0], iu[1]]
    ov[d] = {"median_abs_cos": round(float(vals.median()), 4), "max_abs_cos_among_4000": round(float(vals.max()), 3),
             "angle_of_median_overlap_deg": round(math.degrees(math.acos(float(vals.median()))), 1)}
R["E2b_random_arrow_overlap"] = ov


# ---------------- E2c ----------------
K = 16
theta = torch.arange(K) * (2 * math.pi / K)


def batch(kind, B, g, p=0.15):
    present = (torch.rand(B, generator=g) < p).float()
    amp = torch.rand(B, generator=g) * present
    if kind == "ring":
        phi = torch.rand(B, generator=g) * 2 * math.pi
        tune = torch.clamp(torch.cos(theta[None, :] - phi[:, None]), min=0) ** 4   # neighbours fire together
        return amp[:, None] * tune
    k = torch.randint(0, K, (B,), generator=g)
    return amp[:, None] * torch.nn.functional.one_hot(k, K).float()


def train_squeeze(kind, k_hidden, steps=20000, seed=0):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.nn.Parameter(torch.randn(k_hidden, K, generator=g) * 0.2)
    b1 = torch.nn.Parameter(torch.zeros(k_hidden))
    W2 = torch.nn.Parameter(torch.randn(K, k_hidden, generator=g) * 0.2)
    b2 = torch.nn.Parameter(torch.zeros(K))
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=3e-3)
    for _ in range(steps):
        x = batch(kind, 2048, g)
        xh = torch.relu(torch.relu(x @ W1.T + b1) @ W2.T + b2)
        loss = ((x - xh) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [t.detach() for t in (W1, b1, W2, b2)], float(loss)


def ring_dist(i, j):
    d = abs(i - j) % K
    return min(d, K - d)


res = {}
for kind in ["ring", "onehot"]:
    for kh in [4, 6]:
        (W1, b1, W2, b2), loss = train_squeeze(kind, kh)
        # response of each hidden neuron to each feature alone (amplitude 1) and, for the ring, to every orientation
        single = torch.relu(torch.eye(K) @ W1.T + b1)              # (K, kh)
        sweep_phi = torch.linspace(0, 2 * math.pi, 145)[:-1]
        tune_in = torch.clamp(torch.cos(theta[None, :] - sweep_phi[:, None]), min=0) ** 4
        sweep = torch.relu(tune_in @ W1.T + b1)                      # (144, kh)
        neurons = []
        for n in range(kh):
            r = single[:, n]
            if r.max() <= 1e-6:
                neurons.append({"dead": True})
                continue
            strong = [int(i) for i in torch.nonzero(r > 0.5 * r.max()).flatten()]
            pair_d = [ring_dist(a, b) for ai, a in enumerate(strong) for b in strong[ai + 1:]]
            neurons.append({"features_above_half_max": strong,
                            "mean_ring_distance_between_them": round(float(np.mean(pair_d)), 2) if pair_d else 0.0,
                            "single_feature_response": [round(float(v), 3) for v in r],
                            "orientation_sweep": [round(float(v), 3) for v in sweep[:, n]]})
        live = [nn for nn in neurons if not nn.get("dead")]
        multi = [nn for nn in live if len(nn["features_above_half_max"]) > 1]
        # random baseline: mean ring distance of random sets of the same sizes
        rng = np.random.default_rng(0)
        base = []
        for nn in multi:
            m = len(nn["features_above_half_max"])
            for _ in range(200):
                s = rng.choice(K, m, replace=False)
                base.append(np.mean([ring_dist(a, b) for ai, a in enumerate(s) for b in s[ai + 1:]]))
        res[f"{kind}_k{kh}"] = {
            "loss": round(loss, 5), "neurons": neurons,
            "summary": {"live_neurons": len(live), "neurons_sharing_2plus_features": len(multi),
                        "mean_ring_distance_within_shared": round(float(np.mean([nn["mean_ring_distance_between_them"] for nn in multi])), 2) if multi else None,
                        "random_sets_mean_ring_distance": round(float(np.mean(base)), 2) if base else None}}
        # Fourier check for the ring: how much of each neuron's orientation tuning sits in each frequency
        if kind == "ring":
            for nn in live:
                f = np.abs(np.fft.rfft(np.array(nn["orientation_sweep"]) - np.mean(nn["orientation_sweep"])))
                nn["power_by_frequency_0to4"] = [round(float(v), 3) for v in (f[:5] ** 2 / (f ** 2).sum())]
R["E2c_squeeze"] = {k: {"loss": v["loss"], "summary": v["summary"]} for k, v in res.items()}
json.dump(res, open(out / "E2c_neurons.json", "w"))
json.dump(R, open(out / "E2_results.json", "w"), indent=1)
print(json.dumps({"E2a": [{k: r[k] for k in ["density", "features_with_an_arrow", "angle_gaps_deg", "loss"]} for r in rows],
                  "E2b": ov, "E2c": R["E2c_squeeze"]}, indent=1))
