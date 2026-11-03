"""Download Open Images V6 validation photos (CC BY 2.0) and keep 384 px (short side) copies.

usage: python fetch_openimages.py <data_dir>
<data_dir> must hold validation-images-with-rotation.csv; photos go to <data_dir>/val384/.
Photos with a non-zero rotation are skipped. Progress is printed every 500 photos.
"""
import concurrent.futures as cf
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

data = Path(sys.argv[1])
out = data / "val384"
out.mkdir(exist_ok=True)
rows = [r for r in csv.DictReader(open(data / "validation-images-with-rotation.csv"))
        if r["Rotation"] in ("", "0", "0.0")]
todo = [r for r in rows if not (out / f"{r['ImageID']}.jpg").exists()]
print(f"{len(rows)} upright photos, {len(todo)} to fetch", flush=True)


def fetch(r):
    url = f"https://open-images-dataset.s3.amazonaws.com/validation/{r['ImageID']}.jpg"
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=30).read()
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            w, h = im.size
            s = 384 / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            im.save(out / f"{r['ImageID']}.jpg", quality=90)
            return True
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1 + attempt)
    print("FAIL", r["ImageID"], str(err)[:80], flush=True)
    return False


t0 = time.time()
done = ok = 0
with cf.ThreadPoolExecutor(24) as ex:
    for res in ex.map(fetch, todo):
        done += 1
        ok += res
        if done % 500 == 0:
            print(f"{done}/{len(todo)} ok={ok} {time.time() - t0:.0f}s", flush=True)
print(f"DONE {ok}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
