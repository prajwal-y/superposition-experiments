"""Sheets of car crops by box shape (w/h < 1.35 'front-like', > 2.0 'side-like') with 4e:55's map overlaid.
usage: python e1_carviews.py <pool_npz> <openimages_dir> <out_dir>"""
import csv, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).parent))
import incv1
pool, oi, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
P = np.load(pool); nat = float(P["max_mixed4e"][:, 55].astype(np.float32).max())
print("pool median of 4e:55 spatial min / max:", round(float(np.median(P["min_mixed4e"][:, 55].astype(np.float32))) / nat, 3))
photo = oi / "val384"
names = {r[0]: r[1] for r in csv.reader(open(oi / "class-descriptions-boxable.csv"))}
car = [k for k, v in names.items() if v == "Car"][0]
boxes = []
for r in csv.DictReader(open(oi / "validation-annotations-bbox.csv")):
    if r["LabelName"] == car and r["IsGroupOf"] == "0" and r["IsOccluded"] == "0" and r["IsTruncated"] == "0":
        x0, x1, y0, y1 = (float(r[k]) for k in ["XMin", "XMax", "YMin", "YMax"])
        if (x1 - x0) * (y1 - y0) > 0.06 and (photo / f"{r['ImageID']}.jpg").exists():
            boxes.append((r["ImageID"], x0, x1, y0, y1))
random.Random(1).shuffle(boxes); boxes = boxes[:768]
dev = incv1.device(); model = incv1.load_model(dev)
def crop(b):
    i, x0, x1, y0, y1 = b; im = Image.open(photo / f"{i}.jpg").convert("RGB"); W, H = im.size
    cx, cy, s = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H, max((x1 - x0) * W, (y1 - y0) * H) * 1.15
    return im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((224, 224), Image.BICUBIC), (x1 - x0) * W / ((y1 - y0) * H)
ims, asp, vals, maps = [], [], [], []
with torch.no_grad():
    for s in range(0, len(boxes), 64):
        cs = [crop(b) for b in boxes[s:s + 64]]
        x = torch.cat([incv1.preprocess(c[0], center_crop=False) for c in cs]).to(dev)
        a = incv1.layer_prerelu(model, x, ["mixed4e"])["mixed4e"][:, 55].cpu()
        ims += [c[0] for c in cs]; asp += [c[1] for c in cs]; vals += a.flatten(1).max(1).values.tolist(); maps += list(a)
asp, vals = np.array(asp), np.array(vals) / nat
def sheet(idx, path, title):
    W = Image.new("RGB", (8 * 228, ((len(idx) + 7) // 8) * 248 + 26), (250, 250, 248)); d = ImageDraw.Draw(W); d.text((6, 6), title, fill=0)
    for k, j in enumerate(idx):
        m = np.clip(maps[j].numpy() / (nat * 0.6), 0, 1); h = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)) / 255.0
        im = Image.fromarray((np.asarray(ims[j]) * (0.35 + 0.65 * h[..., None])).astype(np.uint8))
        x, y = (k % 8) * 228 + 2, (k // 8) * 248 + 26; W.paste(im, (x, y)); d.text((x, y + 226), f"{vals[j]:+.2f}  w/h {asp[j]:.2f}", fill=0)
    W.save(path, quality=88)
side = np.where(asp > 2.0)[0]; front = np.where(asp < 1.35)[0]
rng = np.random.default_rng(0)
sheet(side[np.argsort(-vals[side])][:16], out / "G_side_like_top16.jpg", "side-like cars (w/h > 2), strongest 16")
sheet(rng.choice(side, 16, replace=False), out / "G_side_like_random16.jpg", "side-like cars, random 16")
sheet(rng.choice(front, 16, replace=False), out / "G_front_like_random16.jpg", "front-like cars (w/h < 1.35), random 16")
