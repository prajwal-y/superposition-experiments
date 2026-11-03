"""One worked example of Gao's test, from the E4 setup exactly (code/e4_score.py): by default neuron n27 (Qwen3-0.6B
layer 20, neuron 1273; its rule out/e4/rules/n27.py), on the HELD-OUT split (docs 10,000-10,199).
For the ten held-out documents where the neuron goes most negative, run the model three ways (fp32, CPU):
  real    unmodified
  rule    the neuron's value replaced by the rule's prediction (category means calibrated on the explanation split,
          the same 1,000 calibration documents e4_score.py draws)
  off     the neuron's value replaced by its mean on the explanation split (e4_score.py "ablate")
and keep the position where switching the neuron off changes the next-word distribution most. Writes the line of text,
the neuron's real and rule values on it, the top next-word probabilities under each run and the KLs.

usage: python e4_worked_example.py <out_e4_dir> [k=27]        (writes <out_e4_dir>/worked_example_n<k>.json)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

O = Path(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 27
torch.set_grad_enabled(False)

Z = np.load(O / "acts.npz")
tokens, doc, acts = Z["tokens"], Z["doc"], Z["acts"].astype(np.float32)
nd = json.load(open(O / "neurons.json"))["neurons"][K]
L, I = nd["layer"], nd["index"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
cache = {}
dec = lambda t: cache.setdefault(int(t), tok.decode([int(t)]))  # noqa: E731
starts = np.searchsorted(doc, np.arange(doc.max() + 2))
spec = importlib.util.spec_from_file_location(f"r{K}", O / "rules" / f"n{K:02d}.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rule = m.rule


def cats(d):
    a, b = starts[d], starts[d + 1]
    toks = [dec(t) for t in tokens[a:b]]
    return [rule(toks, i) for i in range(len(toks))]


# calibration exactly as e4_score.py
CAL = np.random.default_rng(2).choice(10000, 1000, replace=False)
sums, cnt = {}, {}
for d in CAL:
    a = starts[d]
    for i, c in enumerate(cats(d)):
        sums[c] = sums.get(c, 0.0) + float(acts[a + i, K])
        cnt[c] = cnt.get(c, 0) + 1
cm = {c: sums[c] / cnt[c] for c in sums}
mean = float(acts[doc < 10000, K].mean())
print("category means", {c: round(v, 3) for c, v in cm.items()}, "mean", round(mean, 4), flush=True)

# candidate held-out documents: where the neuron fires hardest
held = np.arange(starts[10000], starts[10200])
mins = {}
for j in held:
    d = int(doc[j])
    mins[d] = min(mins.get(d, 0.0), float(acts[j, K]))
cand = sorted(mins, key=lambda d: mins[d])[:10]

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).eval()
down = model.model.layers[L].mlp.down_proj


def run(ids, new=None):
    h = None
    if new is not None:
        def rep(_m, inputs):
            x = inputs[0].clone()
            x[0, :, I] = new
            return (x,) + tuple(inputs[1:])
        h = down.register_forward_pre_hook(rep)
    try:
        return torch.log_softmax(model(input_ids=ids, use_cache=False).logits.float()[0], -1)
    finally:
        if h:
            h.remove()


best = None
for d in cand:
    a, b = starts[d], starts[d + 1]
    ids = torch.from_numpy(tokens[a:b].astype(np.int64))[None]
    real_vals = torch.from_numpy(acts[a:b, K])
    pred = torch.tensor([cm.get(c, cm.get("none", mean)) for c in cats(d)], dtype=torch.float32)
    base = run(ids)
    # check the hook reproduces the recorded values: replacing with the recorded values must change almost nothing
    same = run(ids, real_vals)
    lr, lo = run(ids, pred), run(ids, torch.full_like(real_vals, mean))
    kl = lambda lp: (base.exp() * (base - lp)).sum(-1)  # noqa: E731
    k_same, k_rule, k_off = kl(same), kl(lr), kl(lo)
    p = int(torch.argmax(k_off))
    print(d, "min act", round(mins[d], 1), "pos", p, "KL off", float(k_off[p]), "rule", float(k_rule[p]), "same", float(k_same.max()), flush=True)
    if best is None or float(k_off[p]) > best["kl_off"]:
        best = {"doc": d, "pos": p, "kl_off": float(k_off[p]), "kl_rule": float(k_rule[p]), "kl_same_max": float(k_same.max()),
                "lp": {"real": base[p], "rule": lr[p], "off": lo[p]}, "a": a, "b": b, "pred": pred, "real_vals": real_vals}

d, p, a = best["doc"], best["pos"], best["a"]
w0 = max(0, p - 11)
line = [{"text": dec(tokens[a + i]), "real": round(float(best["real_vals"][i]), 2), "rule": round(float(best["pred"][i]), 2)}
        for i in range(w0, p + 1)]
P = {c: best["lp"][c].exp() for c in best["lp"]}
top = torch.topk(P["real"], 5).indices.tolist()
for c in ("rule", "off"):
    for t in torch.topk(P[c], 3).indices.tolist():
        if t not in top:
            top.append(t)
nxt = int(tokens[a + p + 1]) if a + p + 1 < best["b"] else None
out = {
    "source": f"recomputed from out/e4 (acts.npz, rules/n{K:02d}.py, neurons.json) with the method of "
              "code/e4_score.py; script code/e4_worked_example.py",
    "neuron": {"k": K, "layer": L, "index": I, "id": f"{L}:{I}"},
    "model": "Qwen/Qwen3-0.6B, fp32",
    "held_out_doc": d, "position_in_doc": p,
    "line": line,
    "category_means": {c: round(v, 3) for c, v in cm.items()}, "mean_explanation_split": round(mean, 4),
    "next_token_actual": dec(nxt) if nxt is not None else None,
    "next_words": [{"text": dec(t), "real": round(float(P["real"][t]), 4), "rule": round(float(P["rule"][t]), 4),
                    "off": round(float(P["off"][t]), 4)} for t in top],
    "kl_at_position": {"rule": best["kl_rule"], "off": best["kl_off"]},
    "hook_check_max_kl_with_recorded_values": best["kl_same_max"],
}
(O / f"worked_example_n{K:02d}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
