"""E1: the 4e:55 battery. Reads pool and stimulus activations, writes numbers to out/e1/E1_results.json
and contact sheets to out/e1/*.jpg.

usage: python e1_analyze.py <pool_npz> <stimuli_npz> <openimages_dir> <out_dir>
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402

pool_npz, stim_npz, oi_dir, out = map(Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
P = np.load(pool_npz)
ST = np.load(stim_npz)
ids = P["ids"]
photo = oi_dir / "val384"
R: dict = {}

# ---------- labels ----------
names = {r[0]: r[1] for r in csv.reader(open(oi_dir / "class-descriptions-boxable.csv"))}
labels = defaultdict(set)
for r in csv.DictReader(open(oi_dir / "validation-annotations-human-imagelabels-boxable.csv")):
    if r["Confidence"] == "1":
        labels[r["ImageID"]].add(names.get(r["LabelName"], r["LabelName"]))
GROUPS = {
    "cat": {"Cat"}, "dog": {"Dog"}, "fox/wild canid": {"Fox"}, "other mammal": {"Lynx", "Raccoon", "Tiger", "Lion", "Bear", "Horse", "Squirrel", "Rabbit", "Hamster", "Mouse", "Leopard", "Cheetah", "Jaguar (Animal)", "Red panda", "Otter", "Polar bear", "Brown bear", "Deer", "Goat", "Sheep", "Cattle", "Pig", "Monkey", "Bat (Animal)"},
    "car/van/truck": {"Car", "Van", "Truck", "Taxi", "Bus", "Limousine", "Land vehicle"},
    "human face": {"Human face"}, "bird": {"Bird"},
}


def group_of(i):
    labs = labels.get(i, set())
    hits = [g for g, s in GROUPS.items() if labs & s]
    return hits


def composition(sel):
    c = Counter()
    for i in sel:
        for g in group_of(i) or ["other"]:
            c[g] += 1
    return {k: round(v / len(sel), 3) for k, v in c.most_common()}


# ---------- A. the whole range ----------
mx = P["max_mixed4e"][:, 55].astype(np.float32)
mn = P["min_mixed4e"][:, 55].astype(np.float32)
nat_max = float(mx.max())
order = np.argsort(-mx)
R["natural"] = {"n_photos": int(len(mx)), "max": nat_max, "p99": float(np.percentile(mx, 99)),
                "p50": float(np.median(mx))}
bands = {"top 25": order[:25], "rank 26-100": order[25:100], "rank 101-400": order[100:400],
         "rank 401-1600": order[400:1600], "random 400": np.random.default_rng(0).choice(len(mx), 400, replace=False)}
R["composition_by_rank"] = {k: composition(ids[v]) for k, v in bands.items()}
neg = np.argsort(mn)
R["composition_most_negative"] = {"bottom 25": composition(ids[neg[:25]]), "bottom 100": composition(ids[neg[:100]])}
# where do the first cars appear?
car_ranks = [k for k, j in enumerate(order[:2000]) if "car/van/truck" in group_of(ids[j])]
animal_ranks = [k for k, j in enumerate(order[:2000]) if set(group_of(ids[j])) & {"cat", "dog", "fox/wild canid", "other mammal"}]
R["first_car_rank"] = car_ranks[:10]
R["first_animal_rank"] = animal_ranks[:10]


def font(sz):
    for f in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def crop224(i):
    im = Image.open(photo / f"{i}.jpg").convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((224, 224), Image.BICUBIC)


def overlay(im, m, vmax):
    """Tint where 4e:55 fires (positive part of its 14x14 map)."""
    m = np.clip(m / vmax, 0, 1)
    heat = Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255
    h = np.asarray(heat).astype(np.float32)[..., None] / 255
    tint = a * (0.35 + 0.65 * h) + np.array([1.0, 0.85, 0.2]) * 0.0
    return Image.fromarray((tint * 255).clip(0, 255).astype(np.uint8))


def sheet(idxs, path, title, values=None, maps=None, cols=8):
    t = 28
    rows = (len(idxs) + cols - 1) // cols
    W = Image.new("RGB", (cols * 228, rows * 248 + t), (250, 250, 248))
    d = ImageDraw.Draw(W)
    d.text((6, 4), title, fill=(0, 0, 0), font=font(18))
    for k, j in enumerate(idxs):
        im = crop224(ids[j])
        if maps is not None:
            im = overlay(im, maps[j].astype(np.float32), nat_max * 0.6)
        x, y = (k % cols) * 228 + 2, (k // cols) * 248 + t
        W.paste(im, (x, y))
        lab = ",".join(group_of(ids[j])) or "-"
        v = f"{values[j] / nat_max:+.2f} " if values is not None else ""
        d.text((x, y + 225), f"{v}{lab}"[:34], fill=(0, 0, 0), font=font(13))
    W.save(path, quality=88)


sheet(order[:48], out / "A1_top48.jpg", "4e:55 top 48 photos (value = spatial max / pool max)", mx)
sheet(order[:48], out / "A2_top48_where.jpg", "4e:55 top 48: bright = where it fires", mx, P["map_4e55"])
for lo, hi, name in [(100, 132, "A3_rank100"), (400, 432, "A4_rank400"), (1600, 1632, "A5_rank1600")]:
    sheet(order[lo:hi], out / f"{name}.jpg", f"4e:55 ranks {lo + 1}-{hi}", mx)
sheet(neg[:48], out / "A6_most_negative48.jpg", "4e:55 most negative (value = spatial min / pool max)", mn)

# ---------- B. synthetic V and O ----------
kind, idx, rho = ST["kind"], ST["idx"], ST["rho"]
S4e = ST["max_mixed4e"].astype(np.float32)        # (n_stim, 832)
nat4e = P["max_mixed4e"].astype(np.float32).max(0)  # per-neuron pool max
sharp = (kind == "V") & (rho == 0)
best_v = S4e[sharp].max(0)
ratio = best_v / nat4e
rank55 = int((ratio > ratio[55]).sum()) + 1
top_v = np.argsort(-ratio)[:10]
R["synthetic"] = {
    "4e55_best_V_over_pool_max": float(ratio[55]),
    "4e55_rank_of_832": rank55,
    "n_neurons_with_higher_ratio": rank55 - 1,
    "top10_V_neurons": [[int(n), round(float(ratio[n]), 3)] for n in top_v],
    "4e55_best_O_over_pool_max": float(S4e[kind == "O"][:, 55].max() / nat4e[55]),
    "4e55_best_V_percentile_in_pool": float((mx < best_v[55]).mean() * 100),
}
# rounding curve: the 20 sharp specs that drive each neuron hardest, followed through rounding
ROUND = sorted(set(rho[kind == "V"].tolist()))
v_idx_sharp = idx[sharp]


def rounding_curve(n, k=20):
    best = v_idx_sharp[np.argsort(-S4e[sharp][:, n])[:k]]
    curve = []
    for r in ROUND:
        m = (kind == "V") & (rho == r) & np.isin(idx, best)
        curve.append(float(S4e[m][:, n].mean() / nat4e[n]))
    return curve


best_v_neuron = int(top_v[0])
R["rounding"] = {"rho": ROUND, "4e:55": rounding_curve(55), f"4e:{best_v_neuron}": rounding_curve(best_v_neuron),
                 "4e:798": rounding_curve(798)}
# which V does 4e:55 like best?
bi = v_idx_sharp[np.argsort(-S4e[sharp][:, 55])[:5]]
R["4e55_favourite_Vs"] = [{k: (ST[f"v_{k}"][i].item() if hasattr(ST[f"v_{k}"][i], "item") else ST[f"v_{k}"][i])
                           for k in ["L", "theta", "phi", "style", "dark"]} for i in bi]

# ---------- C. upstream V detectors: ablate them, does 4e:55 care? ----------
up_layers = ["mixed4a", "mixed4b", "mixed4c", "mixed4d"]
vsel = {}
for l in up_layers:
    Sl = ST[f"max_{l}"].astype(np.float32)
    natl = P[f"max_{l}"].astype(np.float32).max(0)
    bv = Sl[sharp].max(0) / natl
    # same top specs, fully rounded: a V detector should drop
    rr = np.zeros(Sl.shape[1])
    for n in range(Sl.shape[1]):
        best = v_idx_sharp[np.argsort(-Sl[sharp][:, n])[:10]]
        m = (kind == "V") & (rho == max(ROUND)) & np.isin(idx, best)
        rr[n] = Sl[m][:, n].mean() / natl[n]
    vsel[l] = (bv, bv - rr)
cands = [(l, n, vsel[l][0][n]) for l in up_layers for n in range(len(vsel[l][0]))
         if vsel[l][0][n] > 0.4 and vsel[l][1][n] > 0.5 * vsel[l][0][n]]
cands.sort(key=lambda t: -t[2])
chosen = cands[:40]
R["upstream_V_detectors"] = {"n_candidates": len(cands), "chosen": [f"{l[5:]}:{n} ({v:.2f})" for l, n, v in chosen]}
dev = incv1.device()
model = incv1.load_model(dev)


def mean_resp(targets, img_idx, ablate=None):
    vals = {t: [] for t in targets}
    abl = incv1.Ablate(model, ablate) if ablate else None
    try:
        with torch.no_grad():
            for s in range(0, len(img_idx), 64):
                x = torch.cat([incv1.preprocess(Image.open(photo / f"{ids[j]}.jpg")) for j in img_idx[s:s + 64]])
                a = incv1.layer_prerelu(model, x.to(dev), ["mixed4e"])["mixed4e"]
                for t in targets:
                    vals[t].append(a[:, t].flatten(1).max(1).values.cpu())
    finally:
        if abl:
            abl.close()
    return {t: float(torch.cat(v).clamp(min=0).mean()) for t, v in vals.items()}


targets = [55, best_v_neuron] + ([798] if best_v_neuron != 798 else [])
top_imgs = {t: np.argsort(-P["max_mixed4e"][:, t].astype(np.float32))[:192] for t in targets}
abl_v = defaultdict(list)
for l, n, _ in chosen:
    abl_v[l].append(n)
res = {}
for t in targets:
    base = mean_resp([t], top_imgs[t])[t]
    with_v = mean_resp([t], top_imgs[t], abl_v)[t]
    rand = []
    rng = random.Random(t)
    for _ in range(20):
        abl_r = {l: rng.sample(range(len(vsel[l][0])), len(ns)) for l, ns in abl_v.items()}
        rand.append(mean_resp([t], top_imgs[t], abl_r)[t])
    res[f"4e:{t}"] = {"drop_V_ablation": round(1 - with_v / base, 3),
                      "drop_random_mean": round(1 - float(np.mean(rand)) / base, 3),
                      "drop_random_p95": round(1 - float(np.percentile(rand, 5)) / base, 3)}
R["upstream_ablation"] = res

# ---------- D. who reads 4e:55? ablate it and watch mixed5a ----------
top500 = order[:384]
base5, abl5 = [], []
with torch.no_grad():
    for s in range(0, len(top500), 64):
        x = torch.cat([incv1.preprocess(Image.open(photo / f"{ids[j]}.jpg")) for j in top500[s:s + 64]]).to(dev)
        base5.append(incv1.layer_prerelu(model, x, ["mixed5a"])["mixed5a"].flatten(2).max(2).values.cpu())
        ab = incv1.Ablate(model, {"mixed4e": [55]})
        try:
            abl5.append(incv1.layer_prerelu(model, x, ["mixed5a"])["mixed5a"].flatten(2).max(2).values.cpu())
        finally:
            ab.close()
b5, a5 = torch.cat(base5).clamp(min=0), torch.cat(abl5).clamp(min=0)
nat5 = torch.from_numpy(P["max_mixed5a"].astype(np.float32).max(0))
drop = ((b5 - a5).mean(0) / nat5).numpy()
readers = np.argsort(-drop)[:8]
R["readers_by_ablation"] = [[int(n), round(float(drop[n]), 4)] for n in readers]
for n in [94, 352, 317]:
    R.setdefault("essay_readers_rank", {})[f"5a:{n}"] = int((drop > drop[n]).sum()) + 1
# expanded weights 4e:55 -> 5a (1x1 direct; 3x3/5x5 through their bottlenecks, ReLU ignored)
W1 = model.mixed5a_1x1_pre_relu_conv.weight[:, 55, 0, 0].cpu()
B3 = model.mixed5a_3x3_bottleneck_pre_relu_conv.weight[:, 55, 0, 0].cpu()
W3 = model.mixed5a_3x3_pre_relu_conv.weight.cpu().sum((2, 3)) @ B3
B5 = model.mixed5a_5x5_bottleneck_pre_relu_conv.weight[:, 55, 0, 0].cpu()
W5 = model.mixed5a_5x5_pre_relu_conv.weight.cpu().sum((2, 3)) @ B5
exp = torch.cat([W1, W3, W5, torch.zeros(128)]).numpy()
R["readers_by_expanded_weight"] = [[int(n), round(float(exp[n]), 4)] for n in np.argsort(-exp)[:8]]
mx5 = P["max_mixed5a"].astype(np.float32)
for n in sorted(set([int(r) for r in readers[:4]] + [94, 352, 317] + [int(x) for x in np.argsort(-exp)[:3]])):
    o5 = np.argsort(-mx5[:, n])[:16]
    sheet(o5, out / f"D_reader_5a{n}.jpg", f"5a:{n} top 16 (reads 4e:55)", mx5[:, n], cols=8)
    R.setdefault("reader_top16_composition", {})[f"5a:{n}"] = composition(ids[o5])

# ---------- E. Cammarata's version: pointed vs round object classes (box crops) ----------
POINTED = ["Fox", "Cat", "Car", "Train", "Rocket", "Dagger", "Christmas tree", "Candle", "Tent", "Billiard table", "Cocktail", "Wine glass"]
ROUNDED = ["Ball", "Golf ball", "Tennis ball", "Clock", "Pumpkin", "Wheel", "Snail", "Orange", "Apple", "Tomato", "Doughnut"]
inv = {v: k for k, v in names.items()}
boxes = defaultdict(list)
want = {inv[c]: c for c in POINTED + ROUNDED if c in inv}
for r in csv.DictReader(open(oi_dir / "validation-annotations-bbox.csv")):
    c = want.get(r["LabelName"])
    if c and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0" and (photo / f"{r['ImageID']}.jpg").exists():
        bw, bh = float(r["XMax"]) - float(r["XMin"]), float(r["YMax"]) - float(r["YMin"])
        if bw * bh > 0.04:
            boxes[c].append((r["ImageID"], float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])))


def box_crop(b):
    i, x0, x1, y0, y1 = b
    im = Image.open(photo / f"{i}.jpg").convert("RGB")
    W_, H_ = im.size
    cx, cy, s = (x0 + x1) / 2 * W_, (y0 + y1) / 2 * H_, max((x1 - x0) * W_, (y1 - y0) * H_) * 1.15
    return im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))


cls = {}
for c in POINTED + ROUNDED:
    bs = boxes.get(c, [])
    random.Random(0).shuffle(bs)
    bs = bs[:120]
    if len(bs) < 8:
        continue
    vals = []
    with torch.no_grad():
        for s in range(0, len(bs), 64):
            x = torch.cat([incv1.preprocess(box_crop(b), center_crop=False) for b in bs[s:s + 64]]).to(dev)
            a = incv1.layer_prerelu(model, x, ["mixed4e"])["mixed4e"][:, 55]
            vals.append(torch.stack([a.flatten(1).max(1).values, a.flatten(1).mean(1), a.flatten(1).min(1).values], 1).cpu())
    v = torch.cat(vals).numpy() / nat_max
    cls[c] = {"group": "pointed" if c in POINTED else "round", "n": len(v),
              "median_max": round(float(np.median(v[:, 0])), 3), "median_mean": round(float(np.median(v[:, 1])), 3),
              "median_min": round(float(np.median(v[:, 2])), 3), "share_max_above_0.25": round(float((v[:, 0] > 0.25).mean()), 3)}
R["classes"] = cls

# ---------- F. whole photos by label (all 20k-class labels), from the saved 14x14 maps ----------
names6 = {r[0]: r[1] for r in csv.reader(open(oi_dir / "oidv6-class-descriptions.csv"))}
lab_all = defaultdict(set)
for r in csv.DictReader(open(oi_dir / "validation-annotations-human-imagelabels.csv")):
    if r["Confidence"] == "1":
        lab_all[names6.get(r["LabelName"], "?")].add(r["ImageID"])
pos = {i: k for k, i in enumerate(ids)}
M = P["map_4e55"].astype(np.float32).reshape(len(ids), -1)
WHOLE = ["Cat", "Fox", "Coyote", "Wolf", "Dog", "Pug", "Japanese chin", "Pomeranian", "Car", "Bullet train",
         "High-speed rail", "Fire", "Flame", "Martini", "Billiard table", "Golf ball", "Clock", "Snail", "Ball", "Bullfrog"]
whole = {}
for c in WHOLE:
    js = [pos[i] for i in lab_all.get(c, ()) if i in pos]
    if len(js) < 5:
        continue
    m = M[js] / nat_max
    whole[c] = {"n": len(js), "median_max": round(float(np.median(m.max(1))), 3),
                "median_mean": round(float(np.median(m.mean(1))), 3), "median_min": round(float(np.median(m.min(1))), 3),
                "pool_percentile_of_median_max": round(float((mx / nat_max < np.median(m.max(1))).mean() * 100), 1)}
R["whole_photo_labels"] = whole
R["whole_photo_pool_reference"] = {"median_max": round(float(np.median(mx) / nat_max), 3),
                                   "median_mean": round(float(np.median(M.mean(1)) / nat_max), 3)}
# do the essay's readers respond to their named subjects? (pool percentile of median response)
mx5all = P["max_mixed5a"].astype(np.float32)
chk = {}
for n, c in [(94, "Fire"), (94, "Flame"), (352, "Coyote"), (352, "Cat"), (317, "Martini"), (317, "Dog")]:
    js = [pos[i] for i in lab_all.get(c, ()) if i in pos]
    if js:
        chk[f"5a:{n} on {c}"] = round(float((mx5all[:, n] < np.median(mx5all[js, n])).mean() * 100), 1)
R["essay_readers_on_their_subjects_pool_percentile"] = chk
bull = [pos[i] for i in lab_all.get("Bullfrog", ()) if i in pos]
if bull:
    R["5a_neurons_most_driven_by_bullfrog_photos"] = [int(n) for n in np.argsort(-np.median(mx5all[bull] / mx5all.max(0), 0))[:5]]

# ---------- G. head-on vs side-on cars (box shape as a rough guide to viewpoint) ----------
carb = []
inv_all = {v: k for k, v in names.items()}
for r in csv.DictReader(open(oi_dir / "validation-annotations-bbox.csv")):
    if r["LabelName"] == inv_all["Car"] and r["IsGroupOf"] == "0" and r["IsOccluded"] == "0" and r["IsTruncated"] == "0":
        bw, bh = float(r["XMax"]) - float(r["XMin"]), float(r["YMax"]) - float(r["YMin"])
        if bw * bh > 0.06 and (photo / f"{r['ImageID']}.jpg").exists():
            carb.append((r["ImageID"], float(r["XMin"]), float(r["XMax"]), float(r["YMin"]), float(r["YMax"])))
random.Random(1).shuffle(carb)
aspect, cv = [], []
with torch.no_grad():
    for s in range(0, min(len(carb), 768), 64):
        bs = carb[s:s + 64]
        ims = [box_crop(b) for b in bs]
        for b in bs:
            W_, H_ = Image.open(photo / f"{b[0]}.jpg").size
            aspect.append((b[2] - b[1]) * W_ / ((b[4] - b[3]) * H_))
        x = torch.cat([incv1.preprocess(im, center_crop=False) for im in ims]).to(dev)
        cv.append(incv1.layer_prerelu(model, x, ["mixed4e"])["mixed4e"][:, 55].flatten(1).max(1).values.cpu())
aspect, cv = np.array(aspect), torch.cat(cv).numpy() / nat_max
front, side = cv[aspect < 1.35], cv[aspect > 2.0]
R["car_views"] = {"n_front_like(w/h<1.35)": int(len(front)), "median_front": round(float(np.median(front)), 3),
                  "n_side_like(w/h>2.0)": int(len(side)), "median_side": round(float(np.median(side)), 3)}
fi = np.argsort(-cv)[:16]
car_sel = [carb[k] for k in fi]

json.dump(R, open(out / "E1_results.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float)[:6000])
