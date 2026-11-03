"""E0b: does feature visualization drift into dogs, and what does "guiding" it along real-photo routes do?

For each reader of the car neuron 4c:447 (4d:352, 4d:417, 4d:277, 4d:241) we optimise four images at once:
  free   - plain feature visualization (mean activation of the channel), as in 2019
  car    - for the first 64 steps, also reward the 4c route that real car photos take into the neuron
  dog    - same, with the route real dog photos take
  other  - same, with the route its top photos that show neither vehicles nor animals take
A route is the neuron's attribution to each 4c channel (gradient x activation, summed over space),
averaged over that group's 48 strongest photos and clipped at zero. After step 64 every image follows
the plain objective, as the essay describes ("often only for the first 64 steps").
Finally each image is scored by the neuron's spatial max, as a fraction of its pool max.

usage: python e0_drift.py <pool_npz> <openimages_dir> <out_dir> [readers=352,417,277,241] [guide_steps=64] [lam_mult=1]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402
from lucent.optvis import param, transform  # noqa: E402

pool_npz, oi, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cpu")  # Lucent's transforms are only tested on CPU; batch of 4 keeps it quick
torch.manual_seed(0)
model = incv1.load_model(dev)
P = np.load(pool_npz)
ids = P["ids"]
photo = oi / "val384"
names = {r[0]: r[1] for r in csv.reader(open(oi / "class-descriptions-boxable.csv"))}
labels = defaultdict(set)
for r in csv.DictReader(open(oi / "validation-annotations-human-imagelabels-boxable.csv")):
    if r["Confidence"] == "1":
        labels[r["ImageID"]].add(names.get(r["LabelName"], r["LabelName"]))
VEH = {"Car", "Van", "Truck", "Taxi", "Bus", "Limousine", "Land vehicle", "Vehicle", "Ambulance", "Motorcycle"}
DOG = {"Dog"}
ANIMAL = {"Dog", "Cat", "Animal", "Mammal", "Carnivore", "Fox", "Horse", "Bird", "Rabbit", "Insect"}
READERS = [int(n) for n in sys.argv[4].split(",")] if len(sys.argv) > 4 else [352, 417, 277, 241]
SNAP = [1, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
STEPS = 512
GUIDE_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 64
LAM_MULT = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
TAG = "" if (GUIDE_STEPS, LAM_MULT) == (64, 1.0) else f"_g{GUIDE_STEPS}_x{LAM_MULT:g}"


def group_ids(n, kind, k=48):
    act = P["max_mixed4d"][:, n].astype(np.float32)
    order = np.argsort(-act)
    sel = []
    for j in order:
        labs = labels.get(ids[j], set())
        ok = {"car": bool(labs & VEH), "dog": bool(labs & DOG),
              "other": not (labs & VEH) and not (labs & ANIMAL)}[kind]
        if ok:
            sel.append(j)
        if len(sel) == k:
            break
    return sel


def route(n, idxs):
    """Mean attribution of 4d:n to each 4c channel (gradient x activation, summed over space)."""
    mods = dict(model.named_modules())
    mod_d, k = incv1.branch_of("mixed4d", n)
    got = {}

    def keep_c(_m, _i, o):
        o.retain_grad()
        got["c"] = o

    hc = mods["mixed4c"].register_forward_hook(keep_c)
    hd = mods[mod_d].register_forward_hook(lambda _m, _i, o: got.__setitem__("d", o))
    tot, acts, cm, dm = torch.zeros(512), [], torch.zeros(512), 0.0
    try:
        for s in range(0, len(idxs), 16):
            x = torch.cat([incv1.preprocess(Image.open(photo / f"{ids[j]}.jpg")) for j in idxs[s:s + 16]]).to(dev)
            with torch.enable_grad():
                model(x.requires_grad_(True))
                d = got["d"][:, k].flatten(1).max(1).values
                d.sum().backward()
            c = got["c"]
            tot += (c.grad * c).sum((2, 3)).sum(0).detach().cpu()
            acts.append(d.detach().cpu())
            cm += c.detach().mean((2, 3)).sum(0).cpu()
            dm += float(got['d'][:, k].detach().mean((1, 2)).sum())
    finally:
        hc.remove()
        hd.remove()
    r = (tot / len(idxs)).clamp(min=0)
    r = r / (r.norm() + 1e-8)
    # weight so that, on these photos, the route term equals the neuron's mean activation
    lam = (dm / len(idxs)) / max(float((cm / len(idxs)) @ r), 1e-6)
    return r, float(torch.cat(acts).mean()), lam


tfs = transform.standard_transforms + [transform.preprocess_inceptionv1()]
results = {}
nat4d = P["max_mixed4d"].astype(np.float32).max(0)
for n in READERS:
    groups = {k: group_ids(n, k) for k in ["car", "dog", "other"]}
    routes = {k: route(n, v) for k, v in groups.items()}
    params, image_f = param.image(224, batch=4, device=dev)
    opt = torch.optim.Adam(params, lr=0.05)
    mods = dict(model.named_modules())
    store = {}
    hs = [mods["mixed4c"].register_forward_hook(lambda _m, _i, o: store.__setitem__("c", o)),
          mods["mixed4d"].register_forward_hook(lambda _m, _i, o: store.__setitem__("d", o))]
    snaps = defaultdict(list)
    conds = ["free", "car", "dog", "other"]
    try:
        for step in range(1, STEPS + 1):
            opt.zero_grad()
            img = image_f()
            x = img
            for t in tfs:
                x = t(x)
            model(x)
            obj = store["d"][:, n].mean((1, 2))          # per image
            loss = -obj.sum()
            if step <= GUIDE_STEPS:
                c = store["c"]
                for b, cond in enumerate(conds[1:], start=1):
                    r, _nat, lam_c = routes[cond]
                    g = (c[b] * r.to(dev)[:, None, None]).sum(0).mean()
                    loss = loss - LAM_MULT * lam_c * g
            loss.backward()
            opt.step()
            if step in SNAP:
                with torch.no_grad():
                    snaps[step] = (image_f().detach().cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    finally:
        for h in hs:
            h.remove()
    # score the finished images without transforms
    with torch.no_grad():
        fin = image_f().detach()
        a = incv1.layer_prerelu(model, incv1.from_unit(fin), ["mixed4d"])["mixed4d"][:, n]
        score = (a.flatten(1).max(1).values / float(nat4d[n])).tolist()
    results[f"4d:{n}"] = {c: round(s, 3) for c, s in zip(conds, score)}
    results[f"4d:{n}"]["group_sizes"] = {k: len(v) for k, v in groups.items()}
    results[f"4d:{n}"]["route_photo_mean_act_over_pool_max"] = {k: round(routes[k][1] / float(nat4d[n]), 3) for k in routes}
    # strips: rows = conditions, columns = snapshots
    rows = [np.concatenate([snaps[s][b] for s in SNAP], 1) for b in range(4)]
    Image.fromarray(np.concatenate(rows, 0)).save(out / f"drift_4d{n}{TAG}.jpg", quality=90)
    Image.fromarray(np.concatenate([snaps[STEPS][b] for b in range(4)], 1)).save(out / f"final_4d{n}{TAG}.png")
    print(n, results[f"4d:{n}"], flush=True)

json.dump({"snapshots": SNAP, "rows": ["free", "car-guided", "dog-guided", "other-guided"], "scores": results},
          open(out / f"E0_drift{TAG}.json", "w"), indent=1)
