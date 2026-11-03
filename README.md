# superposition-experiments

Code and results for the experiments in the video **"Can We Actually Read an AI's Neurons?"**

In October 2026 Nick Cammarata's essay for Goodfire, *Neural Networks Are Already Clean*, argued that most neurons do one main job. Its best-known example was InceptionV1's neuron 4e:55, long described as responding to cat faces and car fronts, which the essay read as a detector for one shape, a V. Liv Gorton replied with tests on 4e:55 that point the other way, and Leo Gao with a stricter standard for calling a neuron explained.

We re-ran their tests, and some of our own, on two networks:
- **InceptionV1** (GoogLeNet, 2014), the image network that 4e:55 is from.
- **Qwen3-0.6B**, a small open language model, using Leo Gao's standard for when a neuron counts as explained.

Everything here ran on a 16 GB M1 Pro laptop.

## Results at a glance

| | Question | Result | Whose test |
|---|---|---|---|
| **E0a** | Are the curve detectors from *Curve Detectors* (2020) there? | All ten are among the 20 most orientation-tuned of layer 3b's 480 neurons, and their preferred angles go round the circle. | Ours |
| **E0d** | What is curve neuron 3b:379 built from? | Knocking out its 10 strongest inputs from layer 3a removes 85% of its response on its top photos. Ten random 3a neurons remove 3.7% on average. 7 of the 10 inputs are themselves tuned to orientation. | Ours |
| **E1** | Is 4e:55 a "V detector"? | Its best drawn V (of 2,304) reaches 22% of its peak. That is weaker than its response to 16% of ordinary photos. Neuron 4e:798 reaches 82%. Rounding the V's point lowers 4e:55's response on exactly half of the drawings. Knocking out the 40 most V-selective neurons below it costs 14%, against 9% for 40 random neurons. | Liv Gorton's tests, run by us |
| **E1** | What does 4e:55 respond to? | Top 25 photos: 68% cats, 8% foxes, 8% cars (the first car is at rank 23). Bottom 25: 72% dogs, nearly all broad, jowly faces. Plain side-on cars reach 0.35-0.46 of its peak, above 95% of photos. | Ours |
| **E0c** | Are the car neuron's readers left/right pose neurons? | Flipping their top 100 photos keeps 97-98% of their response. A real right-facing-head neuron (4b:414) keeps 34%. | Liv Gorton's test, run by us |
| **E0b** | Does feature visualization "drift" from cars into dogs? | Two of the car neuron's four readers (4d:352, 4d:417) are car neurons that plain feature visualization draws as animals: steering the first steps along the route real car photos take gives cars (for 4d:417, only with the strongest steering). The other two (4d:277, 4d:241) stay animals however hard we steer. | Nick Cammarata's method, run by us |
| **E2** | How does a network squeeze 5 concepts into 2 neurons? | When the concepts are rare, all five get an arrow, 71-73 degrees apart (a pentagon). In 1,000 dimensions, 4,000 random arrows are still at least 80 degrees apart. | Anthropic's *Toy Models of Superposition* (2022), run by us |
| **SAE** | How much of layer 4e can a sparse autoencoder rebuild? | 71.9% of the variation across all 41,144 photos, from 8 active concepts (of 4,096) per photo. | Ours |
| **E4** | Can one written rule stand in for a random Qwen3-0.6B neuron? | **0 of 40** neurons explained (recovery >= 0.5). 2 partly explained, 38 not. Median recovery 0.02. A control that puts a neuron's own recorded values back recovers 1.000 on the four neurons checked. | Leo Gao's test, run by us (pre-registered) |

The full write-up, with every number and caveat, is in [docs/RESULTS.md](docs/RESULTS.md). E4's design was fixed before any neuron was looked at: see [docs/E4_PREREG.md](docs/E4_PREREG.md).

## Naming

- **"4e:55"** means channel 55 of InceptionV1's layer `mixed4e`, as in Distill's Circuits thread.
- **A neuron's value** on a photo is its strongest response anywhere in the photo: the spatial max of the pre-ReLU value. Pre-ReLU values let a neuron's negative side show.
- **"Share of peak"** is a value divided by the neuron's strongest value over all 41,144 photos.

## Setup

Python 3.11. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Lucent is installed from its `dev` branch, as in our runs, not from PyPI. It loads the original Lucid InceptionV1 weights (`inception5h`).

## Data

Nothing large is in the repository. By default the scripts look for data in `data/`; set `SUPERPOSITION_DATA` to use another folder.

**Open Images V6 validation set** (photos CC BY 2.0; the author of each photo is in `validation-images-with-rotation.csv`). Put these files in `data/openimages/`:

- https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
- https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
- https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv
- https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv
- https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv
- https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv

Then fetch the 41,144 upright photos at 384 px (about 20 minutes):

```bash
python code/fetch_openimages.py data/openimages
```

**C4 web text** (for E4): put `c4-validation.00000-of-00008.json.gz` from https://huggingface.co/datasets/allenai/c4 (`en/` folder) in `data/c4/`.

**Models** download on first use: InceptionV1 through Lucent, and `Qwen/Qwen3-0.6B` from Hugging Face.

## Running

Run the scripts from the repository root, in this order. Each script's docstring explains its method in detail.

**InceptionV1.** Every photo's values for every neuron in layers 3b to 5a are computed once, into `out/pool/pool.npz` (about 700 MB). The other scripts read from it.

```bash
python code/pool_acts.py data/openimages/val384 out/pool/pool.npz

python code/e0_curves.py out/e0/curves.npz                                                  # E0a
python code/e0_curve_knockout.py out/pool/pool.npz data/openimages out/e0/E0_curve_knockout.json   # E0d
python code/e0_mirror.py out/pool/pool.npz data/openimages out/e0/E0_mirror.json            # E0c
python code/e0_drift.py out/pool/pool.npz data/openimages out/e0                            # E0b
python code/e0_drift.py out/pool/pool.npz data/openimages out/e0 352,417,277,241 128 16     # E0b, stronger steering

python code/e1_stimuli.py out/e1/stimuli_acts.npz                                           # E1: the 2,304 drawn V's and O's
python code/e1_analyze.py out/pool/pool.npz out/e1/stimuli_acts.npz data/openimages out/e1  # E1: the 4e:55 battery
python code/e1_carviews.py out/pool/pool.npz data/openimages out/e1                         # car crops by shape
python code/e1b_car_sheets.py out/pool/pool.npz data/openimages out/e1                      # E1b: car sheets to sort by eye

python code/sae_layer4e.py out/pool/pool.npz data/openimages out/sae                        # sparse autoencoder (about 1 min)
python code/fv_sequences.py out/pool/pool.npz out/fv_seq seq                                # feature visualization from noise
```

**Toy models** (CPU, about 2 minutes):

```bash
python code/e2_toy.py out/e2
```

**Qwen3-0.6B** (E4; uses Apple's MPS if available):

```bash
python code/e4_collect.py data/c4/c4-validation.00000-of-00008.json.gz out/e4   # draw 40 neurons, record their values
python code/e4_contexts.py out/e4                 # example contexts the rule writer reads (explanation split only)
python code/e4_fit.py out/e4 27                   # rule writer's helper: check one rule on the explanation split
python code/e4_score.py out/e4                    # score every rule on the held-out split, Gao's way
python code/e4_oracle.py out/e4                   # control: put the neuron's own values back
python code/e4_worked_example.py out/e4 27        # one worked example of the test, for neuron n27
```

The 40 rules in `out/e4/rules/` were written by Claude, an AI model, after reading each neuron's contexts from the explanation split, as the pre-registration says. The rule writer never saw the held-out split, and the scoring is mechanical. `out/e4/rules/DESCRIPTIONS.md` has the rule writer's notes on each neuron.

## What is in `out/`

The small results: every JSON of numbers, the 40 rules, the E2 training snapshots, and our own feature-visualization pictures.

Left out, and regenerated by the scripts above:
- the large arrays: `pool.npz`, `stimuli_acts.npz` and E4's `acts.npz`, each over GitHub's file size limit;
- contact sheets made of Open Images photos (`e1_analyze.py`, `e1_carviews.py`, `e1b_car_sheets.py`, `make_board.py`), which [docs/RESULTS.md](docs/RESULTS.md) refers to by name;
- E4's example contexts, which are excerpts of web text.

## Sources

- Nick Cammarata, *Neural Networks Are Already Clean*, Goodfire, 15 Oct 2026. https://already-clean.goodfire.com/
- Leo Gao, *Neural networks are not already clean*, 17 Oct 2026. https://www.lesswrong.com/posts/Dkh7pXKjoNWkkxQDz/neural-networks-are-not-already-clean
- Liv Gorton, reply with tests on 4e:55, 20 Oct 2026. https://livgorton.com/already-clean-thoughts/
- Olah et al., *Zoom In: An Introduction to Circuits*, Distill, 2020. https://distill.pub/2020/circuits/zoom-in/
- Cammarata et al., *Curve Detectors*, Distill, 2020. https://distill.pub/2020/circuits/curve-detectors/
- Olah, Mordvintsev and Schubert, *Feature Visualization*, Distill, 2017. https://distill.pub/2017/feature-visualization/
- Elhage et al., *Toy Models of Superposition*, Anthropic, 2022. https://transformer-circuits.pub/2022/toy_model/index.html
- Bills et al., *Language models can explain neurons in language models*, OpenAI, 2023. https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html
- Lucent (PyTorch port of Lucid). https://github.com/greentfrapp/lucent
- Open Images V6 (annotations CC BY 4.0, photos CC BY 2.0), C4 (`allenai/c4`, ODC-BY), Qwen3-0.6B (Apache 2.0).

## License

The code is under the MIT License (see [LICENSE](LICENSE)). The datasets and models it downloads keep their own licenses.
