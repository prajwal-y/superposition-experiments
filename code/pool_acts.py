"""Run every pool photo through InceptionV1 (224 centre crop) and save, per photo and per neuron,
the spatial max and min of the pre-ReLU value for mixed3b..mixed5a, plus 4e:55's full 14x14 map.

usage: python pool_acts.py <photo_dir> <out_npz>
"""
import sys
import threading
import queue
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402

photo_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(photo_dir.glob("*.jpg"))
dev = incv1.device()
model = incv1.load_model(dev)
B = 64
q: queue.Queue = queue.Queue(maxsize=8)


def loader():
    for i in range(0, len(files), B):
        xs = []
        for f in files[i:i + B]:
            try:
                xs.append(incv1.preprocess(Image.open(f)))
            except Exception:  # noqa: BLE001
                xs.append(torch.zeros(1, 3, 224, 224) - 117)
        q.put(torch.cat(xs))
    q.put(None)


threading.Thread(target=loader, daemon=True).start()
mx = {l: [] for l in incv1.LAYERS}
mn = {l: [] for l in incv1.LAYERS}
maps55 = []
done = 0
with torch.no_grad():
    while (x := q.get()) is not None:
        acts = incv1.layer_prerelu(model, x.to(dev), incv1.LAYERS)
        for l, a in acts.items():
            flat = a.flatten(2)
            mx[l].append(flat.max(2).values.half().cpu())
            mn[l].append(flat.min(2).values.half().cpu())
        maps55.append(acts["mixed4e"][:, 55].half().cpu())
        done += len(x)
        if done % 2048 < B:
            print(f"{done}/{len(files)}", flush=True)

np.savez(out_path, ids=np.array([f.stem for f in files]),
         **{f"max_{l}": torch.cat(mx[l]).numpy() for l in incv1.LAYERS},
         **{f"min_{l}": torch.cat(mn[l]).numpy() for l in incv1.LAYERS},
         map_4e55=torch.cat(maps55).numpy())
print("saved", out_path, flush=True)
