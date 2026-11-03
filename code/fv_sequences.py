"""Feature visualization from random noise (Lucent, CPU), as in Feature Visualization (Olah et al., 2017): start from
Lucent's decorrelated FFT noise and step the pixels uphill on the neuron's mean value, with Lucent's standard
transforms. Each saved picture is scored on the untransformed image by the neuron's spatial max (pre-ReLU) as a
multiple of its strongest photo in the pool (out/pool/pool.npz), as in E0.

  seq   the curve detector 3b:379, snapshots at steps 0, 4, 16, 32, 64, 128, 256, 512   -> fv_seq_3b379.json + PNGs
  deep  seven deeper neurons (4a:489, 4b:411, 4b:493, 4d:458, 4b:373, 4b:360, 4c:447), 256 steps -> fv_deep.json + PNGs

usage: python fv_sequences.py <pool_npz> <out_dir> seq|deep [initial_noise_sd=0.5]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import incv1  # noqa: E402
from lucent.optvis import param, transform, objectives  # noqa: E402

OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
dev = torch.device("cpu")
model = incv1.load_model(dev)
P = np.load(sys.argv[1])
tfs = transform.standard_transforms + [transform.preprocess_inceptionv1()]


def score(img01, layer, n):
    with torch.no_grad():
        a = incv1.layer_prerelu(model, incv1.from_unit(img01), [layer])[layer][:, n]
    return (a.flatten(1).max(1).values / float(P[f"max_{layer}"][:, n].astype(np.float32).max())).tolist()


def run(specs, steps, snaps, tag, sd=None):
    params, image_f = param.image(224, sd=sd, batch=len(specs), device=dev)
    obj = objectives.Objective.sum([objectives.channel(l, n, batch=b) for b, (l, n) in enumerate(specs)])
    opt = torch.optim.Adam(params, lr=0.05)
    from lucent.optvis.render import hook_model
    hook = hook_model(model, image_f)
    out = {}
    for step in range(0, steps + 1):
        if step in snaps:
            with torch.no_grad():
                im = image_f().detach()
            sc = score(im, specs[0][0], specs[0][1]) if len({l for l, _ in specs}) == 1 else [score(im[b:b + 1], l, n)[0] for b, (l, n) in enumerate(specs)]
            for b, (l, n) in enumerate(specs):
                f = OUT / f"{l[5:]}_{n}_step{step}.png"
                Image.fromarray((im[b].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)).save(f)
                out.setdefault(f"{l[5:]}:{n}", {})[step] = round(sc[b], 3)
            print(tag, step, [round(s, 2) for s in sc], flush=True)
        if step == steps:
            break
        def closure():
            opt.zero_grad()
            x = image_f()
            for t in tfs:
                x = t(x)
            model(x)
            loss = obj(hook)
            loss.backward()
            return loss
        opt.step(closure)
    return out


mode = sys.argv[3]
SD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
if mode == "seq":
    res = run([("mixed3b", 379)], 512, {0, 4, 16, 32, 64, 128, 256, 512}, "seq", sd=SD)  # visible noise at step 0
    json.dump({"what": "3b:379 feature visualization from noise; score = spatial max pre-ReLU / strongest pool photo",
               "source": f"code/fv_sequences.py (Lucent, seed 0, initial noise sd {SD}, Adam lr 0.05, standard transforms); pool max from out/pool/pool.npz",
               "scores": res}, open(OUT / "fv_seq_3b379.json", "w"), indent=1)
else:
    specs = [("mixed4a", 489), ("mixed4b", 411), ("mixed4b", 493), ("mixed4d", 458), ("mixed4b", 373), ("mixed4b", 360), ("mixed4c", 447)]
    res = run(specs, 256, {256}, "deep")
    json.dump({"what": "deeper-layer feature visualizations (256 steps); score = spatial max pre-ReLU / strongest pool photo",
               "source": "code/fv_sequences.py (Lucent, seed 0, Adam lr 0.05, standard transforms); pool max from out/pool/pool.npz",
               "scores": res}, open(OUT / "fv_deep.json", "w"), indent=1)
