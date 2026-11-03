# E4 pre-registration: neurons nobody picked (written 2026-10-24, before any neuron was looked at)

Question: in a small open language model, what share of RANDOMLY chosen neurons can be explained by one plain written rule, judged by Leo Gao's standard (the rule stands in for the neuron and the model's output barely changes)?

## Fixed choices
- **Model:** Qwen/Qwen3-0.6B (Apache-2.0), 28 layers x 3,072 MLP neurons = 86,016.
  - A neuron's value is the input to `down_proj`: `silu(gate) * up`. It is signed, so each neuron has a positive side and a negative side.
- **Sample:** 40 neurons drawn uniformly from all 86,016 with `numpy.random.default_rng(20261024)`, listed in `out/e4/neurons.json` before collection. No neuron is swapped out after the draw, for any reason.
- **Text:**
  - allenai/c4 English validation shard 00000 (public web text), the first 256 tokens of each document.
  - Documents 0-9,999 are the **explanation split**, used to read examples and calibrate rules.
  - Documents 10,000-10,199 are the **held-out split**, used only for scoring. (Reduced from 400 documents to 200 for laptop time, decided on 10-24 before any rule was written. Documents 10,200-10,399 were collected but are not used.)
- **Rules:**
  - One rule per neuron, written as Python after reading its 40 strongest positive and 40 strongest negative contexts, plus 40 random ones from the explanation split.
  - A rule sees only the current token and the up to 64 tokens before it, as text.
  - A rule returns a category per token (for example "none", "weak", "strong", or a negative category). Each category's predicted value is its mean real value on the explanation split, set automatically, so rules aren't judged on hand-picked numbers.
  - A rule must describe one thing. If a neuron seems to need two unrelated descriptions, that's recorded as the finding, and the rule keeps the larger one.
  - Rules are written by Claude, as Goodfire's were written by agents. The rule writer may check a rule's fit on the explanation split (`code/e4_fit.py`) and revise it, but never sees the held-out split. The scoring is mechanical.
- **Scoring on the held-out split**, in fp32, as the average next-token KL divergence from the unmodified model:
  - **Ablated:** the neuron's value is replaced by its mean on the explanation split.
  - **Rule:** the neuron's value is replaced by the rule's prediction.
  - **Recovery** = 1 - KL(rule) / KL(ablated). 1 means the rule fully stands in for the neuron; 0 means it does no better than switching the neuron off.
  - The positive and negative sides are also scored separately: replace only one side, by zero (ablated) or by the rule's prediction for that side, and keep the other side's real values.
- **Categories:**
  - **Explained:** recovery >= 0.5.
  - **Partly explained:** 0.2 to 0.5.
  - **Not explained:** < 0.2.
  - **No measurable effect:** ablation KL below 1e-7 nats per token.
- **Reported in full:** every neuron's rule, score and category, including the failures.

## Known limits (said in the film if used)
- 0.6B parameters and web text only. It says nothing directly about Qwen3-32B or about code.
- Rules as code are less flexible than an AI simulator, which biases the result toward "not explained".
- Goodfire's 90% was an agent judgement per neuron side, not this score. The two numbers must not be put side by side as like for like.
