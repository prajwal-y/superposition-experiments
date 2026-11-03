"""E4 stage 1: draw 40 random Qwen3-0.6B MLP neurons (pre-registered seed) and record their values on C4 text.

Saves out/e4/neurons.json (written BEFORE the model runs) and out/e4/acts.npz with, for every non-pad token:
tokens (int32), doc (int32), pos (int16) and acts (float16, n_tokens x 40). Docs 0-9,999 = explanation split,
10,000-10,399 = held-out split (see E4_PREREG.md).
usage: python e4_collect.py <c4_json_gz> <out_dir>
"""
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

c4, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
L, N, K = 28, 3072, 40
rng = np.random.default_rng(20261024)
flat = rng.choice(L * N, K, replace=False)
neurons = sorted([(int(f // N), int(f % N)) for f in flat])
json.dump({"seed": 20261024, "model": "Qwen/Qwen3-0.6B", "neurons": [{"layer": l, "index": i} for l, i in neurons]},
          open(out / "neurons.json", "w"), indent=1)
print("neurons drawn:", neurons, flush=True)

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).to(dev).eval()
by_layer = {}
for k, (l, i) in enumerate(neurons):
    by_layer.setdefault(l, []).append((k, i))
cap = {}


def make_hook(l):
    idx = torch.tensor([i for _, i in by_layer[l]], device=dev)
    cols = [k for k, _ in by_layer[l]]

    def hook(_m, inputs):
        cap[l] = (cols, inputs[0][..., idx].float().cpu())
    return hook


hs = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l)) for l in by_layer]
docs = []
with gzip.open(c4, "rt") as f:
    for line in f:
        docs.append(json.loads(line)["text"])
        if len(docs) == 10400:
            break
enc = [tok(d, add_special_tokens=False)["input_ids"][:256] for d in docs]
T, D, P, A = [], [], [], []
t0 = time.time()
B = 16
with torch.no_grad():
    for s in range(0, len(enc), B):
        batch = enc[s:s + B]
        m = max(len(e) for e in batch)
        ids = torch.full((len(batch), m), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros((len(batch), m), dtype=torch.long)
        for r, e in enumerate(batch):
            ids[r, :len(e)] = torch.tensor(e)
            att[r, :len(e)] = 1
        model(input_ids=ids.to(dev), attention_mask=att.to(dev))
        full = torch.zeros(len(batch), m, K)
        for l, (cols, v) in cap.items():
            full[:, :, cols] = v
        for r, e in enumerate(batch):
            n = len(e)
            T.append(np.array(e, dtype=np.int32))
            D.append(np.full(n, s + r, dtype=np.int32))
            P.append(np.arange(n, dtype=np.int16))
            A.append(full[r, :n].numpy().astype(np.float16))
        if (s // B) % 50 == 0:
            done = sum(len(x) for x in T)
            print(f"docs {s + len(batch)}/{len(enc)} tokens {done} {done / (time.time() - t0):.0f} tok/s", flush=True)
for h in hs:
    h.remove()
np.savez(out / "acts.npz", tokens=np.concatenate(T), doc=np.concatenate(D), pos=np.concatenate(P), acts=np.concatenate(A))
print("saved", out / "acts.npz", sum(len(x) for x in T), "tokens", flush=True)
