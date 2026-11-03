"""Forward every synthetic V (all roundings) and O; save each neuron's spatial-max pre-ReLU value.

usage: python e1_stimuli.py <out_npz>
"""
import sys
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import incv1  # noqa: E402
import stimuli  # noqa: E402

out = Path(sys.argv[1])
dev = incv1.device()
model = incv1.load_model(dev)
specs = list(stimuli.v_specs())
jobs = [("V", i, r) for i in range(len(specs)) for r in stimuli.ROUND]
ospecs = list(stimuli.o_specs())
jobs += [("O", i, 0.0) for i in range(len(ospecs))]


def draw(job):
    kind, i, r = job
    im = stimuli.render_v(specs[i], r) if kind == "V" else stimuli.render_o(ospecs[i])
    return incv1.preprocess(im, center_crop=False)


res = {l: [] for l in incv1.LAYERS}
B = 128
with cf.ThreadPoolExecutor(8) as ex, torch.no_grad():
    for s in range(0, len(jobs), B):
        x = torch.cat(list(ex.map(draw, jobs[s:s + B]))).to(dev)
        acts = incv1.layer_prerelu(model, x, incv1.LAYERS)
        for l, a in acts.items():
            res[l].append(a.flatten(2).max(2).values.half().cpu())
        if s % 2048 == 0:
            print(s, "/", len(jobs), flush=True)

np.savez(out,
         kind=np.array([j[0] for j in jobs]), idx=np.array([j[1] for j in jobs]), rho=np.array([j[2] for j in jobs]),
         v_L=np.array([s["L"] for s in specs]), v_theta=np.array([s["theta"] for s in specs]),
         v_phi=np.array([s["phi"] for s in specs]), v_style=np.array([s["style"] + str(s["width"]) for s in specs]),
         v_dark=np.array([s["dark_on_light"] for s in specs]),
         o_R=np.array([s["R"] for s in ospecs]), o_style=np.array([s["style"] + str(s["width"]) for s in ospecs]),
         o_dark=np.array([s["dark_on_light"] for s in ospecs]),
         **{f"max_{l}": torch.cat(res[l]).numpy() for l in incv1.LAYERS})
print("saved", out, len(jobs), "stimuli", flush=True)
