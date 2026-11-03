"""E4 stage 2a: readable examples per neuron (explanation split only).
For each neuron: its 40 strongest positive tokens, 40 strongest negative tokens and 40 random tokens, each shown with
up to 64 tokens before it and 8 after, the token itself marked [[like this]]. Writes out/e4/contexts/n{k:02d}_L{l}_N{i}.txt
and out/e4/tokens_explain.json (decoded token strings per explanation doc, for rule fitting).
usage: python e4_contexts.py <out_e4_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

o = Path(sys.argv[1])
Z = np.load(o / "acts.npz")
tokens, doc, pos, acts = Z["tokens"], Z["doc"], Z["pos"], Z["acts"].astype(np.float32)
neur = json.load(open(o / "neurons.json"))["neurons"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
vocab = {}


def s(t):
    if t not in vocab:
        vocab[t] = tok.decode([int(t)])
    return vocab[t]


ex = doc < 10000
starts = np.searchsorted(doc, np.arange(doc.max() + 2))
(o / "contexts").mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def ctx(j):
    d, p = doc[j], pos[j]
    a = starts[d]
    left = "".join(s(t) for t in tokens[a + max(0, p - 64):a + p])
    right = "".join(s(t) for t in tokens[a + p + 1:min(starts[d + 1], a + p + 9)])
    return (left + "[[" + s(tokens[j]) + "]]" + right).replace("\n", "\\n")


for k, nd in enumerate(neur):
    v = np.where(ex, acts[:, k], np.nan)
    order = np.argsort(np.nan_to_num(v, nan=0.0))
    vmax, vmin = np.nanmax(v), np.nanmin(v)
    lines = [f"neuron {k}: layer {nd['layer']} index {nd['index']}  |  explanation split: {ex.sum()} tokens  |  "
             f"max {vmax:.3f}  min {vmin:.3f}  mean {np.nanmean(v):.4f}  p99 {np.nanpercentile(v, 99):.3f}  p1 {np.nanpercentile(v, 1):.3f}", ""]
    lines.append("== STRONGEST POSITIVE ==")
    for j in order[::-1][:40]:
        lines.append(f"{v[j]:+.3f}  {ctx(j)}")
    lines.append("")
    lines.append("== STRONGEST NEGATIVE ==")
    for j in order[:40]:
        lines.append(f"{v[j]:+.3f}  {ctx(j)}")
    lines.append("")
    lines.append("== RANDOM ==")
    for j in rng.choice(np.nonzero(ex)[0], 40, replace=False):
        lines.append(f"{v[j]:+.3f}  {ctx(j)}")
    (o / "contexts" / f"n{k:02d}_L{nd['layer']}_N{nd['index']}.txt").write_text("\n".join(lines))
print("wrote", len(neur), "context files")
