# E0 + E1 results (run 2026-10-24 on the 16 GB M1 Pro)

> Contact sheets named below (`A1`-`A6`, `D_reader_*`, `G_*`, `e1b_sheet*`) and the review board are made of Open Images photos and are not in this repository; `code/e1_analyze.py`, `code/e1_carviews.py`, `code/e1b_car_sheets.py` and `code/make_board.py` regenerate them.

Model: the original 2014 InceptionV1 (Lucid's inception5h via Lucent, GitHub `dev`). Photos: all 41,144 upright
Open Images V6 validation photos (CC BY 2.0, author per photo in `validation-images-with-rotation.csv`), 224 px centre
crops. Data: `data/openimages/` (not in the repository; see the README). Code: `code/`. Outputs: `out/`.
"Peak" below = the neuron's strongest response over all 41,144 photos (pre-ReLU spatial max).

## It is the right neuron
- 4e:55's 16 strongest inputs from layer 4d are Distill's weight-explorer list, in the same order; all top 30 match.
- Feature visualization with diversity (`out/fv/4e55_diversity_256.png`) gives cat and fox faces plus car chrome and headlights, as in the 2017 figure.

## E0a: curve sweep
- All ten curve detectors from *Curve Detectors* (2020) are among the 20 most orientation-tuned of 3b's 480 neurons; their preferred angles go round the circle (`out/e0/curves.npz`).

## E1: the 4e:55 battery
| Test | Result | Bears on |
|---|---|---|
| Top photos | Top 25: 68% cats, 12% other mammals (raccoon, red panda, mouse), 8% foxes, 8% dogs (papillon), 8% cars. First car at rank 23; from rank 100 on, cars dominate (Open Images has 15x more car photos than cat photos). It fires at the middle of the face (nose and muzzle) and at car grilles and headlights (`out/e1/A2_top48_where.jpg`) | Gorton: faces at the top, cars a little lower. Confirmed |
| Most negative photos | 72% of the bottom 25 are dogs, nearly all broad, jowly faces: mastiffs, pugs, French bulldogs, boxers, shar-peis, rottweilers; also two hippos (`A6`, board row 2). By label, pugs go most negative (median minimum -0.19 of peak, against -0.09 for a typical photo); clocks (-0.12), snails (-0.14) and balls (-0.10) go somewhat more negative than typical | NEW. Gorton noted jowly dogs; the essay's round objects are real but weaker |
| Drawn V's | Its favourite V points sideways ("<"), not down like a chin. 2,304 V's (sizes, angles, orientations, line and filled, both polarities). Best: 22% of its peak (Gorton ~25%), weaker than its response to 16% of ordinary photos. 448 of the layer's 832 neurons respond more strongly to V's relative to their own peak; 4e:798 reaches 82%, 4e:22 108%. Best O: 5% | Gorton. Confirmed |
| Rounding the V | On its 20 favourite V's, the response halves as the point is rounded to a U (0.19 -> 0.09 of peak); 4e:798 falls from 0.74 to 0.19 | Gorton. Confirmed: across all 2,304 drawings, rounding lowers it on exactly 50% (sharp 0.069, rounded 0.066); the halving on its 20 favourites is a selection effect (picked when sharp). Corrected 10-28 per the v4 factcheck |
| Knock out earlier V detectors | 40 strongest V-selective neurons in 4a-4d zeroed: 4e:55 drops 14% on its top photos; 40 random neurons drop it 9% on average (95th percentile 15%). 4e:798 drops 37% (random 10%) | Gorton. Confirmed |
| Pointed things "whatever the label" | Object crops, median response as a share of peak: cats 0.46, foxes 0.43, cars 0.33; daggers, tents, candles, Christmas trees, cocktails, wine glasses, trains 0.08-0.14; balls, clocks, pumpkins, snails, fruit 0.05-0.11. Whole photos: bullet trains 0.19 (80th percentile), billiard tables 0.14 (68th), flames 0.09 (44th, below a typical photo) | Against the essay's "whatever the label"; a weak pointed > round ordering exists |
| Side-on cars | Plain profiles (sedan, Porsche, Boxster, Golf, estate) fire it at 0.35-0.46 of peak, above 95% of photos; the strongest car responses are head-on fronts (0.62-0.70); rear views 0.18-0.38 (`G_*.jpg`). **E1b, sorted by eye with values hidden** (96 random unoccluded car photos, `out/e1/e1b_cars_labelled.json`): median share of peak by view: head-on 0.31 (n=12), front three-quarter 0.39 (41), side 0.36 (25), rear three-quarter 0.35 (10), straight rear 0.19 (4). Viewing angle barely matters except straight from behind | Against "side-on cars barely register" |
| Who reads it (layer 5a) | By knock-out: 5a:727, 377, 260, 547, 94. By weight: 5a:352 first. The essay's readers respond to their subjects: 5a:94 to flame photos (97th percentile; its top photos are candle flames and foxes), 5a:352 to coyotes (99th), 5a:317 to martinis (96th; also borzois, a piano, a rifle). 5a:377 (frog, gulls, small dogs) is a likely match for Gorton's "bullfrog" reader | Mixed: the essay's reader labels hold, and its readers are among several |

Reading: not a V detector (Gorton is right on the main test). It responds to a particular kind of face and to car fronts and profiles, and its negative side is a different kind of face (broad, jowly dogs). That is a distinction drawn among faces, closer to the essay's "verb" idea than to "cat OR car", but not the V-versus-O for any object the essay states. Whether "certain faces + cars" is one thing or two is still open.

## E0c: mirror test
- Share of response kept when the top 100 photos are flipped left to right: 4b:414 (a real right-facing-head neuron) 34%, lower than 99% of 128 random neurons. The car neuron's readers 4d:352, 417, 277, 241 keep 97-98%, more than most random neurons (median 94.5%). 4e:55 keeps 99%.
- Caveat: flipping tests left/right only; a "head-on" pose neuron would pass. Gorton's 3D turntable test covers viewpoint generally.

## E0b: feature-visualization drift
Unguided, and guided for the first steps toward the routes of the neuron's own car photos, dog photos, or other photos.
Scores are the final picture's response as a multiple of the neuron's strongest real photo.
| Reader | Unguided picture | Car-guided (64 steps, x1) | Scores free / car / dog / other |
|---|---|---|---|
| 4d:352 | terrier faces with some cars | mostly cars, one dog's nose left in the middle | 1.68 / 1.89 / 1.83 / 1.76 |
| 4d:417 | cats and dogs | cats and dogs | 2.10 / 2.24 / 2.22 / 2.22 |
| 4d:277 | hare- or insect-like creature | same creature | 1.94 / 2.00 / 1.72 / 1.75 |
| 4d:241 | shaggy dog faces | shaggy dog faces | 2.11 / 2.05 / 1.87 / 1.96 |
Stronger guidance (128 steps, x4 and x16; `E0_drift_g128_*.json`, `final_4d*_g128_*.png`):
- 4d:417 at x16: car-guided comes out as rows of cars, nothing else (`_zoom_car_guided_417_241.png`), and scores highest (2.34 vs 1.99 free).
- 4d:277: the same hare- or insect-like creature in every condition.
- 4d:241: shaggy dog faces in every condition, even at x16.
- Scores vary about +-0.15 between runs of the same condition (free: 2.10, 2.11, 1.99 for 417), so judge by the pictures, not small score gaps.

Reading: two of the four readers (352, 417) are car neurons that free feature visualization draws as animals, as the essay says. The other two (277, 241) stay animals however hard we steer, their real photos mix animals and vehicles (Gorton: 241 dogs 52-71%, vehicles 29-47%), and the flip test says they are not left/right pose neurons. The essay's count holds; its explanation of the second pair does not, on our tests and Gorton's.

## E2: toy models, `out/e2/`
- **E2a, 5 concepts in 2 neurons** (equal importance, best of 3 starts): every concept present (density 1.0) -> only 2 arrows; density 0.5-0.3 -> 4 arrows at 90 degrees (two opposite pairs); density 0.2 and rarer -> all 5 arrows, gaps 71-73 degrees (a pentagon). Training snapshots at density 0.05 for animation: `E2a_snapshots_density0.05.json`.
- **E2b, room in many dimensions** (4,000 random arrows): typical angle between two arrows 45 deg in 2-D, 77 in 10-D, 86 in 100-D, 88.8 in 1,000-D, 89.6 in 10,000-D. In 1,000-D the closest pair of 4,000 arrows is still 80 deg apart (|cos| 0.18).
- **E2c, what shares a neuron** (16 features, 4 or 6 hidden neurons): ring features that fire with their neighbours -> each neuron's response around the ring is a smooth wave: 76-88% of most neurons' tuning in one frequency (one broad lobe, or two opposite lobes), i.e. Fourier modes, as Gorton found in the squeezed layer. Features that never fire together -> neurons pick up scattered unrelated features (e.g. 4, 12, 14). The single-feature probe is meaningless for the ring case (training inputs are smooth bumps); use the orientation sweep.

## E4: 40 random Qwen3-0.6B neurons, scored Gao's way (`E4_PREREG.md`, `out/e4/`)
- 38,289 held-out tokens (docs 10,000-10,199). Rules: one per neuron, written as code by Claude from explanation-split examples (`out/e4/rules/`, `DESCRIPTIONS.md`).
- Result: **explained (recovery >= 0.5): 0 of 40. Partly (0.2-0.5): 2. Not explained: 38.** Median recovery 0.02, mean 0.06. Best: n27 (word-pieces ending in "a", any script) 0.40; n38 (French function words) 0.25.
- Many rules capture one side: e.g. n20 "masculine" positive side 0.77, n12 Australia/NZ negative side 0.46, n13 list second-member positive side 0.43.
- Rule writer's notes: 16/40 seemed to need a second unrelated description; 11/40 no clear pattern; 14/40 neither (4:13 has both flags; corrected 10-31, was '13/40 one description').
- **Control (oracle):** replacing the neuron with its own recorded values recovers 1.000 on n27, n28, n4, n1 (KL 1e-9 vs 1e-4), and the ablation KLs match the main run. The pipeline is aligned.
- Limits: 0.6B model (about 50x smaller than Qwen3-32B; Cammarata says bigger models have narrower neurons), web text only, rules as code are cruder than agent investigations. Not a measurement of Goodfire's 90%.

## E0d: what the curve neuron 3b:379 is built from (`out/e0/E0_curve_knockout.json`)
- Its 10 strongest inputs from layer 3a (attribution over its 192 strongest photos): knocking them out cuts its response by **85%**; 10 random 3a neurons cut it by 4% on average (largest of 20 draws 20%).
- 6 of the 10 inputs are orientation-tuned to curves facing about the same way (220-250 deg) as 3b:379 itself (230 deg): it is built from smaller curves. Contrast with 4e:55, whose loss from knocking out V detectors equals a random knockout (E1).
