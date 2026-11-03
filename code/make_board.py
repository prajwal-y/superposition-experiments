"""One review board of the E0/E1 evidence (internal, not a film frame).
usage: python make_board.py <pool_npz> <stimuli_npz> <openimages_dir> <out_jpg>"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402
import stimuli  # noqa: E402

pool, stim, oi, out = map(Path, sys.argv[1:5])
P, ST = np.load(pool), np.load(stim)
ids = P["ids"]
photo = oi / "val384"
mx = P["max_mixed4e"][:, 55].astype(np.float32)
mn = P["min_mixed4e"][:, 55].astype(np.float32)
nat = float(mx.max())
F = lambda s: ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", s)  # noqa: E731
T = 180  # tile size


def crop(i):
    im = Image.open(photo / f"{i}.jpg").convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((T, T), Image.BICUBIC)


def heat(im, m):
    m = np.clip(m / max(m.max(), 1e-6), 0, 1) ** 1.5
    h = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((T, T), Image.BICUBIC)) / 255.0
    return Image.fromarray((np.asarray(im) * (0.25 + 0.75 * h[..., None])).astype(np.uint8))


rows = []


def row(title, tiles, labels):
    W = Image.new("RGB", (8 * (T + 6) + 6, T + 64), (250, 249, 246))
    d = ImageDraw.Draw(W)
    d.text((8, 6), title, fill=(20, 20, 20), font=F(20))
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        x = 6 + k * (T + 6)
        W.paste(t.resize((T, T)), (x, 34))
        d.text((x + 2, 34 + T + 4), lab, fill=(40, 40, 40), font=F(14))
    rows.append(W)


o = np.argsort(-mx)
row("1. 4e:55's strongest photos (bright = where it fires). Value = share of its peak",
    [heat(crop(ids[j]), P["map_4e55"][j].astype(np.float32)) for j in o[:8]], [f"{mx[j] / nat:.2f}" for j in o[:8]])
n = np.argsort(mn)
row("2. What pushes it furthest below zero (most negative value)",
    [crop(ids[j]) for j in n[:8]], [f"{mn[j] / nat:+.2f}" for j in n[:8]])
# rounding: 4e:55's favourite sharp V, rounded step by step
kind, idx, rho = ST["kind"], ST["idx"], ST["rho"]
S = ST["max_mixed4e"].astype(np.float32)
sharp = (kind == "V") & (rho == 0)
specs = list(stimuli.v_specs())
best = idx[sharp][np.argmax(S[sharp][:, 55])]
tiles, labs = [], []
for r in stimuli.ROUND:
    m = (kind == "V") & (idx == best) & (rho == r)
    tiles.append(stimuli.render_v(specs[best], r))
    labs.append(f"4e:55 {S[m][0, 55] / nat:.2f}")
o_best = np.argmax(np.where(kind == "O", S[:, 55], -1e9))
tiles.append(stimuli.render_o(list(stimuli.o_specs())[idx[o_best]]))
labs.append(f"4e:55 {S[o_best, 55] / nat:.2f}")
row("3. Its favourite drawn V, rounded step by step, then its favourite O (photos reach 1.00)", tiles, labs)
# drift strip for 4d:352
strip = Image.open(out.parent / "e0" / "drift_4d352.jpg")
snaps = [1, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
pick = [1, 16, 32, 64, 128, 512]
tiles = []
for r in [0, 1]:
    for s in pick[:4]:
        k = snaps.index(s)
        tiles.append(strip.crop((k * 224, r * 224, (k + 1) * 224, (r + 1) * 224)))
row("4. Feature visualization of car-circuit reader 4d:352. Left 4: free, steps 1/16/32/64. Right 4: car-guided, same steps",
    tiles, ["free 1", "free 16", "free 32", "free 64", "car 1", "car 16", "car 32", "car 64"])
fin = Image.open(out.parent / "e0" / "final_4d352.png")
row("5. 4d:352 after 512 steps: free, car-guided, dog-guided, other-guided (score = x its best photo)",
    [fin.crop((k * 224, 0, (k + 1) * 224, 224)) for k in range(4)] + [Image.new("RGB", (T, T), (250, 249, 246))] * 4,
    ["free 1.68", "car 1.89", "dog 1.83", "other 1.76", "", "", "", ""])
e0 = out.parent / "e0"
t6, l6 = [], []
for n in [417, 241, 277]:
    im = Image.open(e0 / f"final_4d{n}_g128_x16.png")
    t6 += [im.crop((0, 0, 224, 224)), im.crop((224, 0, 448, 224))]
    l6 += [f"4d:{n} free", f"4d:{n} car-guided"]
row("6. Strong car guidance (x16, 128 steps): 417 turns into cars; 241 and 277 stay animals",
    t6 + [Image.new("RGB", (T, T), (250, 249, 246))] * 2, l6 + ["", ""])
H = sum(r.height for r in rows)
B = Image.new("RGB", (rows[0].width, H), (250, 249, 246))
y = 0
for r in rows:
    B.paste(r, (0, y))
    y += r.height
B.save(out, quality=90)
print("saved", out, B.size)
