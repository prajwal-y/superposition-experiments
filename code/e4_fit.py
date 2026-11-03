"""E4 helper for the rule writer: fit and check ONE rule on the EXPLANATION split only (never the held-out split).

A rule file out/e4/rules/n{k:02d}.py defines   rule(toks: list[str], i: int) -> str
  toks = the document's decoded tokens, i = current position; the rule may read toks[max(0, i-64): i+1] only.
  It returns a category name; "none" is the default. Category values are set automatically to the category's mean
  real activation on the explanation split.
Prints category counts and means, the correlation between prediction and real value, and the share of the
neuron's variance the rule explains, on a 400-document sample of the explanation split.
usage: python e4_fit.py <out_e4_dir> <k>
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

o, k = Path(sys.argv[1]), int(sys.argv[2])
Z = np.load(o / "acts.npz")
tokens, doc, acts = Z["tokens"], Z["doc"], Z["acts"][:, k].astype(np.float32)
spec = importlib.util.spec_from_file_location("r", o / "rules" / f"n{k:02d}.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cache = {}
dec = lambda t: cache.setdefault(int(t), tok.decode([int(t)]))  # noqa: E731
starts = np.searchsorted(doc, np.arange(doc.max() + 2))
docs = np.random.default_rng(1).choice(10000, 400, replace=False)
cats, vals = [], []
for d in docs:
    a, b = starts[d], starts[d + 1]
    toks = [dec(t) for t in tokens[a:b]]
    for i in range(len(toks)):
        cats.append(mod.rule(toks, i))
    vals.append(acts[a:b])
vals = np.concatenate(vals)
cats = np.array(cats)
pred = np.zeros_like(vals)
print(f"neuron {k}: {len(vals)} tokens from 400 explanation docs")
for c in sorted(set(cats)):
    m = cats == c
    pred[m] = vals[m].mean()
    print(f"  {c:>12s}: n={m.sum():6d}  mean={vals[m].mean():+.4f}  sd={vals[m].std():.4f}")
r = np.corrcoef(pred, vals)[0, 1] if pred.std() > 0 else 0.0
print(f"  correlation {r:.3f}   variance explained {r * r:.3f}")
