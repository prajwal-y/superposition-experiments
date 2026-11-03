"""E1b: 96 random car crops on numbered sheets WITHOUT values, for sorting by viewing angle by eye.
Values are saved separately (out/e1/e1b_cars.json) and joined only after labelling.
usage: python e1b_car_sheets.py <pool_npz> <openimages_dir> <out_dir>"""
import csv, json, random, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
import incv1
pool, oi, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nat = float(np.load(pool)["max_mixed4e"][:, 55].astype(np.float32).max())
photo = oi / "val384"
names = {r[0]: r[1] for r in csv.reader(open(oi / "class-descriptions-boxable.csv"))}
car = [k for k, v in names.items() if v == "Car"][0]
boxes = []
for r in csv.DictReader(open(oi / "validation-annotations-bbox.csv")):
    if r["LabelName"] == car and r["IsGroupOf"] == "0" and r["IsOccluded"] == "0" and r["IsTruncated"] == "0" and r["IsInside"] == "0":
        x0, x1, y0, y1 = (float(r[k]) for k in ["XMin", "XMax", "YMin", "YMax"])
        if (x1 - x0) * (y1 - y0) > 0.08 and (photo / f"{r['ImageID']}.jpg").exists():
            boxes.append((r["ImageID"], x0, x1, y0, y1))
random.Random(7).shuffle(boxes); boxes = boxes[:96]
def crop(b):
    i, x0, x1, y0, y1 = b; im = Image.open(photo / f"{i}.jpg").convert("RGB"); W, H = im.size
    cx, cy, s = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H, max((x1 - x0) * W, (y1 - y0) * H) * 1.15
    return im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((224, 224), Image.BICUBIC)
ims = [crop(b) for b in boxes]
dev = incv1.device(); model = incv1.load_model(dev); vals = []
with torch.no_grad():
    for s in range(0, 96, 48):
        x = torch.cat([incv1.preprocess(im, center_crop=False) for im in ims[s:s + 48]]).to(dev)
        vals += (incv1.layer_prerelu(model, x, ["mixed4e"])["mixed4e"][:, 55].flatten(1).max(1).values.cpu() / nat).tolist()
json.dump([{"n": k, "image": b[0], "box": b[1:], "value": round(v, 3)} for k, (b, v) in enumerate(zip(boxes, vals))], open(out / "e1b_cars.json", "w"), indent=0)
f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
for sh in range(4):
    W = Image.new("RGB", (6 * 230, 4 * 230), (255, 255, 255)); d = ImageDraw.Draw(W)
    for k in range(24):
        n = sh * 24 + k; x, y = (k % 6) * 230 + 3, (k // 6) * 230 + 3
        W.paste(ims[n], (x, y)); d.rectangle([x, y, x + 34, y + 26], fill=(255, 255, 0)); d.text((x + 3, y + 1), str(n), fill=0, font=f)
    W.save(out / f"e1b_sheet{sh}.jpg", quality=85)
print("ok", len(boxes))
