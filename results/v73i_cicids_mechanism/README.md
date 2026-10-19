# v7.3i — does brittleness track CLASS OVERLAP? CIC-IDS2017 mechanism test — **H1 SUPPORTED**

Run 2026-08-20. `cicids_mechanism.py` → `cicids_mechanism.txt` / `.json`. 6.5 min.

**Both pre-registered criteria passed, the triviality guard cleared, and the effect is monotone
in every one of the six comparison cells. The robustness chapter and the plateau diagnosis are
one finding.**

## The test

Cover–Hart says brittleness comes from rows sitting near decision boundaries. `e_1NN` measures
that overlap **without reference to any attack**, so it is a clean independent variable and the
claim is falsifiable across datasets.

| | e_1NN | prediction |
|---|---|---|
| CIC-UNSW full release | 0.2084 | brittle — *measured* |
| **CIC-IDS2017** | **0.0026** (81× lower, reproduced in-run) | should be dramatically more robust |

## Result — monotone in all six cells

Retained fraction of each dataset's own clean value, identical budget machinery and matched
benign fpr 0.0168:

| mode | budget | **CIC-IDS2017** ret mF1 | ret gate | CIC-UNSW ret mF1 | ret gate |
|---|---|---|---|---|---|
| RANDOM | 0.002 | **0.9960** | 0.9996 | 0.8631 | 0.9558 |
| RANDOM | 0.020 | **0.9479** | 0.9976 | 0.7531 | 0.9208 |
| RANDOM | 0.200 | **0.8381** | 0.9169 | 0.4053 | 0.4939 |
| EVASION | 0.002 | **0.9210** | 0.9827 | 0.7967 | 0.7613 |
| EVASION | 0.020 | **0.8214** | 0.7784 | 0.5335 | 0.4761 |
| EVASION | 0.200 | **0.5232** | **0.5917** | 0.2143 | **0.0313** |

- **H1a PASS** — retained gate recall under white-box evasion at budget 0.20: **0.5917 vs 0.0313**,
  a 19× difference against a pre-registered threshold of 0.50.
- **H1b PASS** — retained macro-F1 under random noise at 0.20: **0.8381 vs 0.4053**, threshold 0.80.
- **Triviality guard CLEARED** — `moved` = 1.0000 at budget 0.20, so the greedy search genuinely
  ran and found descent directions. This is not a stalled attack being misread as robustness.

**Interpretation:** brittleness is a property of the **representation's class overlap**, not of the
architecture. Robustness comes from better features, not better models — the same conclusion every
other axis in this project reached independently (weighting, information and objectives paid;
capacity never did).

## ⚠️ Three things that must travel with this result

### 1. CIC-IDS2017 is *more robust*, not robust

At budget 0.20 it still loses **41% of gate recall** and **48% of macro-F1**. Low overlap buys a
large margin, not immunity. Write "more robust", never "robust" — the same discipline the RF gate
result required.

### 2. The proximate cause is DECISION MARGIN, and a critic will notice

The matched-fpr protocol put the CIC-IDS2017 gate at threshold **0.0002**, against CIC-UNSW's
**0.3280**. A row must be pushed below 0.0002 to evade here, versus below 0.328 there — so this
gate has vastly more probability margin to burn, and an obvious objection is *"you just used a
lower threshold."*

**The answer, and it strengthens rather than weakens the finding:** the threshold is not chosen, it
is *derived* by matching fpr — and the fact that 1.68% fpr is achievable at a threshold of 0.0002
is itself a direct measurement of separability. The causal chain is

> **low overlap → confident predictions → large decision margin → robustness**

so margin is the *mechanism by which* overlap produces robustness, not a competing explanation.
State the chain explicitly; do not let the threshold difference look like an accident of protocol.

### 3. The confound was pre-registered and still stands

CIC-IDS2017 differs from CIC-UNSW in more than overlap — different testbed, different class mix,
no protocol column (D1), no reject class (D2). **A large effect in the predicted direction is
SUGGESTIVE of the mechanism, not proof.** One confounded comparison cannot isolate overlap from
every other difference between two capture campaigns. Do not upgrade this to a controlled
experiment in the write-up.

The clean way to make it controlled, if ever wanted: vary overlap *within* one dataset (e.g. by
subsetting to family pairs of differing purity) and test whether robustness tracks it there.

## ⚠️ Clean levels are NOT comparable to `results/cicids2017_v6/`

This run's clean macro-F1 is **0.7762**, against the v6 run's 0.9835 on the same data. That is not
a regression — it is the matched-fpr protocol. Forcing a 1.68% benign fpr on a near-separable
dataset admits ~7,000 benign false positives which are then routed into families, and on classes
with a handful of test rows (Heartbleed 2, Infiltration 7) that destroys per-class precision and
therefore macro-F1.

**The handicap applies equally to the clean and perturbed arms, so the retained fractions — the
only quantity this run claims — are unaffected.** But never quote 0.7762 as a CIC-IDS2017
performance number; quote the v6 figure with its own caveats for that.

## What this does for the thesis

It merges two chapters into one argument:

- the **plateau diagnosis** said class overlap in the CICFlowMeter representation caps accuracy
  (ρ = 0.827; Cover–Hart bracket [0.213, 0.376])
- the **robustness chapter** said the model is brittle to perturbation
- **this run says they are the same fact.** Overlap sets both the accuracy ceiling *and* the
  robustness floor, through decision margin.

It also removes the single-dataset objection from the robustness claim — which mattered, because
"single-dataset evaluation" is one of the four gaps the literature review names.

## Method / deviations (stated, not silent)

D1 no protocol column in CIC-IDS2017 — IMPOSSIBLE tier is `Destination Port` + 22 service one-hots
derived from it. D2 no reject class (needs cached OOF that exists only for CIC-UNSW) — Stage B is a
plain 12-family classifier. D3 Web Attack variants collapsed to one class, inf/NaN rows dropped
(0.1%), read latin-1. D4 Stage B is a single XGB α=0.25 — justified because the v6 run found all
17 combiner variants within 0.0007 on this dataset. D5 `e_1NN` re-measured in this run (0.0026,
matching v6 exactly) so predictor and outcome come from one pipeline.

2,520,798 rows after dedupe (10.86% dupes), 13 classes, attack share 0.1689, 100 features,
train 2,016,638 / test 504,160. Budget in train-IQR units, COSTLY at half, integers rounded, all
derived columns recomputed after every move, only the 85,148 test attack rows perturbed.
