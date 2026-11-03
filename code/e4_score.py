"""E4 stage 3: score every rule on the HELD-OUT split (docs 10,000-10,199), Gao's way.

For each neuron, run the model (fp32) on held-out text with the neuron's value replaced and measure the average
next-token KL divergence from the unmodified model:
  ablate      value -> its mean on the explanation split
  rule        value -> the rule's prediction (category means from the explanation split)
  ablate_pos  positive part -> 0 (negative part kept);   rule_pos  positive part -> max(prediction, 0)
  ablate_neg  negative part -> 0 (positive part kept);   rule_neg  negative part -> min(prediction, 0)
recovery = 1 - KL(rule) / KL(ablate), and the same per side. Only the layers from the neuron's layer onward are rerun,
starting from the saved input to that layer.
usage: python e4_score.py <out_e4_dir>
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

o = Path(sys.argv[1])
Z = np.load(o / "acts.npz")
tokens, doc, acts = Z["tokens"], Z["doc"], Z["acts"].astype(np.float32)
neur = json.load(open(o / "neurons.json"))["neurons"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cache = {}
dec = lambda t: cache.setdefault(int(t), tok.decode([int(t)]))  # noqa: E731
starts = np.searchsorted(doc, np.arange(doc.max() + 2))
HELD = list(range(10000, 10200))
CAL = np.random.default_rng(2).choice(10000, 1000, replace=False)


def load_rule(k):
    spec = importlib.util.spec_from_file_location(f"r{k}", o / "rules" / f"n{k:02d}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.rule


def doc_cats(rule, d):
    a, b = starts[d], starts[d + 1]
    toks = [dec(t) for t in tokens[a:b]]
    return [rule(toks, i) for i in range(len(toks))]


# calibrate category means on the explanation split; predict held-out tokens
pred = {}
means = {}
for k in range(len(neur)):
    rule = load_rule(k)
    sums, cnt = {}, {}
    for d in CAL:
        a = starts[d]
        for i, c in enumerate(doc_cats(rule, d)):
            sums[c] = sums.get(c, 0.0) + float(acts[a + i, k])
            cnt[c] = cnt.get(c, 0) + 1
    cm = {c: sums[c] / cnt[c] for c in sums}
    ex_mask = doc < 10000
    means[k] = float(acts[ex_mask, k].mean())
    pred[k] = {d: np.array([cm.get(c, cm.get("none", means[k])) for c in doc_cats(rule, d)], dtype=np.float32) for d in HELD}
    print("calibrated", k, {c: round(v, 4) for c, v in cm.items()}, flush=True)

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(dev).eval()
layers = model.model.layers
CONDS = ["ablate", "rule", "ablate_pos", "rule_pos", "ablate_neg", "rule_neg"]
kl_sum = {k: {c: 0.0 for c in CONDS} for k in range(len(neur))}
n_tok = 0
B = 2  # 4 left only ~20% of memory free on the 16 GB laptop
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(HELD), B):
        ds = HELD[s:s + B]
        enc = [tokens[starts[d]:starts[d + 1]] for d in ds]
        m = max(len(e) for e in enc)
        ids = torch.zeros((len(ds), m), dtype=torch.long)
        att = torch.zeros((len(ds), m), dtype=torch.long)
        for r, e in enumerate(enc):
            ids[r, :len(e)] = torch.from_numpy(e.astype(np.int64))
            att[r, :len(e)] = 1
        ids, att = ids.to(dev), att.to(dev)
        saved = {}

        def grab(j):
            def h(_m, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                saved[j] = (hs.detach(), {kk: v for kk, v in kwargs.items() if kk not in ("hidden_states", "past_key_values", "past_key_value", "use_cache")})
            return h
        hs = [layers[j].register_forward_pre_hook(grab(j), with_kwargs=True) for j in range(len(layers))]
        try:
            base = torch.log_softmax(model(input_ids=ids, attention_mask=att, use_cache=False).logits.float(), -1)
        finally:
            for h in hs:
                h.remove()
        valid = att.bool()
        n_tok += int(valid.sum())
        for k, nd in enumerate(neur):
            l, i = nd["layer"], nd["index"]
            P = torch.zeros(len(ds), m, device=dev)
            for r, d in enumerate(ds):
                P[r, :len(pred[k][d])] = torch.from_numpy(pred[k][d])
            for cond in CONDS:
                def rep(_m, inputs, cond=cond):
                    x = inputs[0].clone()
                    v = x[..., i]
                    pos, neg = v.clamp(min=0), v.clamp(max=0)
                    new = {"ablate": torch.full_like(v, means[k]), "rule": P,
                           "ablate_pos": neg, "rule_pos": neg + P.clamp(min=0),
                           "ablate_neg": pos, "rule_neg": pos + P.clamp(max=0)}[cond]
                    x[..., i] = new
                    return (x,) + tuple(inputs[1:])
                hk = layers[l].mlp.down_proj.register_forward_pre_hook(rep)
                try:
                    h, kw = saved[l]
                    for j in range(l, len(layers)):
                        out = layers[j](h, **kw)
                        h = out[0] if isinstance(out, tuple) else out
                    lp = torch.log_softmax(model.lm_head(model.model.norm(h)).float(), -1)
                finally:
                    hk.remove()
                kl = (base.exp() * (base - lp)).sum(-1)
                kl_sum[k][cond] += float(kl[valid].sum())
                del lp, kl, h
            if dev.type == 'mps':
                torch.mps.empty_cache()
        print(f"batch {s // B + 1}/{(len(HELD) + B - 1) // B}  {time.time() - t0:.0f}s", flush=True)

res = []
for k, nd in enumerate(neur):
    kl = {c: kl_sum[k][c] / n_tok for c in CONDS}
    rec = lambda a, r: (1 - kl[r] / kl[a]) if kl[a] > 1e-12 else None  # noqa: E731
    r_all = rec("ablate", "rule")
    cat = ("no measurable effect" if kl["ablate"] < 1e-7 else
           "explained" if r_all >= 0.5 else "partly explained" if r_all >= 0.2 else "not explained")
    res.append({"k": k, **nd, "kl": kl, "recovery": r_all, "recovery_pos": rec("ablate_pos", "rule_pos"),
                "recovery_neg": rec("ablate_neg", "rule_neg"), "category": cat})
json.dump({"held_out_tokens": n_tok, "neurons": res}, open(o / "E4_scores.json", "w"), indent=1)
from collections import Counter  # noqa: E402
print("categories:", Counter(r["category"] for r in res), flush=True)
