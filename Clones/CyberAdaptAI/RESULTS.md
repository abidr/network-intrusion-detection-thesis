# Reimplementation run log

Environment: `ai4beg` conda env, Python 3.12.8, sklearn 1.6.1, xgboost 3.3.0,
numpy 1.26.4, river 0.21.0, shap 0.46.0.

## Full run — CIC-UNSW-NB15-2024, default config

```
run_experiment.py --batch-size 2000 --out results/full_river_shap
```

447,915 rows -> 141,742 duplicates removed (31.6%) -> 306,173 rows -> 154 batches of 2,000,
sequential holdout 123 train / 31 test.

Two runs, differing only in the fourth pool member. `full_hybrid` predates the river install and
used the SGD fallback; `full_river_shap` is the complete framework as the paper specifies it.

| Metric | `full_river_shap` (real Hoeffding Tree) | `full_hybrid` (SGD fallback) |
|---|---|---|
| Accuracy | 0.8676 | 0.8639 |
| **Macro F1** | **0.5855** | 0.5509 |
| Macro precision | 0.7279 | 0.7356 |
| Macro recall | 0.5345 | 0.4977 |
| Weighted F1 | 0.8658 | 0.8610 |
| Latency / instance (Eq 16) | 0.296 ms | 0.051 ms |
| Drift events | 21 | 22 |

The genuine Hoeffding Tree is worth **+0.035 macro-F1** over the SGD substitute, almost all of
it recall (0.498 -> 0.535), concentrated in the rare classes: Worms 0.265 -> 0.367,
DoS 0.314 -> 0.388, Analysis 0.243 -> 0.292, Generic 0.754 -> 0.796. It costs **5.8x inference
latency** (river is pure-Python per-instance). Do not substitute it when quoting this baseline.

Per-class F1 (`full_river_shap`): Benign 0.970, Generic 0.796, Exploits 0.788, Fuzzers 0.707,
Backdoor 0.667, Recon 0.643, DoS 0.388, Worms 0.367, Analysis 0.292, Shellcode 0.238.

### SHAP (Eq 9) — executed

Backend `xgboost_native` (exact TreeSHAP via `pred_contribs=True`; `shap` 0.46 cannot load
XGBoost 3.3 multiclass models — see requirements.txt). **Eq 9 additivity residual 3.8e-06**,
i.e. `f(x) = phi_0 + sum_j phi_j` holds to float32 precision.

Top attributed features: `Bwd Packets/s`, `Fwd Seg Size Min`, `Fwd Packet Length Max`,
`Down/Up Ratio`, `Bwd Packet/Bulk Avg`.

### Comparison to this thesis — read the caveat first

**Not a like-for-like comparison.** CyberAdaptAI uses a *sequential holdout* (a contiguous tail
of the row order) where the thesis baselines use a *stratified* 80/20 split. The test segments
have different class compositions, so the gap below mixes method effect with protocol effect.
To compare properly, rerun the thesis models under sequential holdout, or this one under a
stratified split.

| Model | Macro F1 |
|---|---|
| CyberAdaptAI (this reimplementation, full framework) | 0.5855 |
| Thesis flat XGBoost baseline | 0.5807 |
| Thesis final (two-stage + alpha-diverse ensemble v3) | 0.6339 |

## Finding: the adaptive weights barely move — CONFIRMED with the real Hoeffding Tree

The central claim of the paper is that dynamic performance/diversity reweighting delivers the
gains. Measured over 31 test batches of the full framework, uniform = 0.2500:

| Member | Mean | Std | Min | Max | Range |
|---|---|---|---|---|---|
| random_forest | 0.2594 | 0.0031 | 0.2509 | 0.2642 | 0.0132 |
| xgboost | 0.2587 | 0.0028 | 0.2512 | 0.2634 | 0.0122 |
| extra_trees | 0.2586 | 0.0029 | 0.2508 | 0.2625 | 0.0117 |
| hoeffding_tree | 0.2233 | 0.0085 | 0.2102 | 0.2470 | 0.0368 |

Mean absolute deviation from uniform: **0.0133**. Max: **0.0398**. Mean within-batch spread
(strongest minus weakest member): **0.0363**.

**The four weights stay within ~1.3% of uniform for the entire stream.** Under the paper's own
hyperparameters (gamma = 0.8, alpha = 0.6) the ensemble is, to a very good approximation, a
plain unweighted majority vote. Whatever the framework achieves, adaptive weighting is not the
mechanism.

**This is not an artifact of the SGD substitution.** The obvious objection to the first run was
that three near-identical tree ensembles plus a linear SGD model gives the weighting nothing to
separate, and that a genuine Hoeffding Tree — the one truly online, structurally distinct
member — would break the collapse. It does not. With the real river Hoeffding Tree in the pool,
deviation from uniform goes *up* only marginally (0.0108 -> 0.0133) and stays negligible. The
Hoeffding Tree is the one member that separates at all (mean 0.2233, std 0.0085, ~3x the others)
and even it is only ~2.7 points below uniform. The collapse is a property of the weighting rule,
not of the pool.

This independently reproduces the thesis's own result in `two_stage_stageB_des.ipynb`, where
per-instance competence weights collapsed to 0.333 +/- 0.02. Two dynamic-weighting schemes, two
granularities (per-batch vs per-instance), two pool compositions, same collapse. That is a
substantive point for a related-work critique, and it strengthens the thesis's existing finding
rather than competing with it.

Mechanism: Eq 8's `gamma = 0.8` smoother pulls toward the mean of `(1 - E_m)`, and the four
base learners have similar batch accuracies, so the target is nearly identical across members.
Eq 3's diversity term is the only thing that could separate them, and it is scaled by
`(1 - alpha) = 0.4` on a disagreement measure that is itself near-uniform across a pool of
similar tree ensembles.

## Weight-rule ablation (40k subsample, 4 test batches)

Resolving README ambiguity 1 empirically — all three readings are within noise of each other:

| Rule | Macro F1 | Mean \|dev\| from uniform |
|---|---|---|
| `eq3` (Eq 3 verbatim) | 0.3650 | 0.0043 |
| `eq8` (Algorithm 1 verbatim) | 0.3658 | 0.0048 |
| `hybrid` (default) | 0.3650 | 0.0023 |

The ambiguity is real but immaterial: because the weights never separate, it does not matter
which update rule you pick. Worth stating explicitly rather than agonising over the reading.

## Drift detection

22 drift events over 31 test batches — roughly every 1.4 batches, all fired by ADWIN rather
than the accuracy-drop rule. At that rate the "detector" is close to firing continuously, which
suggests it is tracking batch-to-batch class-mix variation in the row ordering rather than
genuine concept drift. Consistent with the caveat that this dataset has no timestamp and the
"stream" is simulated from row order.

## Reproduction attempt on the ORIGINAL UNSW-NB15 — the paper's own dataset

`Resource/UNSW-NB15/`, pre-split form, 175,341 + 82,332 = 257,673 rows, 45 columns.
Default config, batch 2000, full framework (real river Hoeffding Tree + native TreeSHAP).

### Headline: the paper's numbers reproduce only with duplicate leakage AND no stream

Binary attack-vs-benign (the setting comparable to Table 4). 2x2 over the two pipeline choices
the paper leaves unspecified:

| Configuration | Accuracy | Precision | Recall | F1 | Weight dev. from uniform |
|---|---|---|---|---|---|
| dedup + sequential (leakage-safe, honest stream) | 0.8065 | 0.7917 | 0.6812 | 0.7323 | 0.0165 |
| no-dedup + sequential | 0.9019 | 0.9095 | 0.8570 | 0.8825 | 0.0118 |
| dedup + shuffled | 0.9142 | 0.9020 | 0.8850 | 0.8934 | 0.0015 |
| **no-dedup + shuffled** | **0.9479** | **0.9613** | **0.9567** | **0.9590** | **0.0012** |
| **Paper Table 4 (UNSW-NB15)** | **0.968** | **0.965** | **0.970** | **0.968** | not reported |

**The bottom row matches the paper to within ~0.01 on precision, recall and F1** (0.9613 vs
0.965, 0.9567 vs 0.970, 0.9590 vs 0.968); accuracy is still 0.02 short. Every other cell is far
off. Both factors matter and they compound:

- **Deduplication costs** 0.150 F1 in the sequential stream, 0.066 when shuffled. The dataset is
  **40.2% duplicate rows** (103,575 of 257,673) — higher than the 2024 re-extraction's 31.6%.
- **Shuffling gains** 0.161 F1 with dedup, 0.077 without.

### The contradiction this exposes

The paper cannot have both its accuracy and its drift narrative:

- If the stream is genuine (chronological mini-batches, §3.2), F1 is **0.73-0.88**, not 0.968.
- If F1 is 0.968, the rows must be shuffled and duplicated — in which case **there is no stream,
  no temporal structure, and concept drift is undefined by construction**. Tables 5/6's
  drift-detection-and-recovery case studies would then be describing artifacts.

### And in that same configuration, the adaptive weighting is provably inert

This is the sharpest result. Under shuffling the weights collapse to **essentially exactly
uniform**: mean deviation **0.0012**, max **0.0029** — i.e. all four members sit within 0.3% of
0.2500 for the entire stream.

So in **precisely the configuration that reproduces the paper's headline numbers, the mechanism
the paper credits for those numbers is doing nothing at all.** The ensemble is an unweighted
majority vote to three decimal places.

This also confirms the earlier prediction (made before running it): the easier and more IID the
data, the *tighter* the collapse, because Eq 8 drives every weight toward `(1 - E_m)` and the
four members' errors converge. Deviation across all runs to date:

| Setting | Mean dev. from uniform |
|---|---|
| CIC-UNSW-2024, multiclass, sequential | 0.0133 |
| UNSW-NB15, binary, sequential + dedup | 0.0165 |
| UNSW-NB15, binary, shuffled | 0.0015 |
| UNSW-NB15, binary, shuffled + no-dedup | **0.0012** |

### Multiclass (10-class), leakage-safe, sequential

Accuracy 0.7871, **macro-F1 0.4215**, macro P/R 0.4983/0.4120, weighted F1 0.7784, 10 drift
events, 0.157 ms/instance. Duplicates removed: 94,928 (36.8%).

Per-class F1: Normal 0.892, Recon 0.773, Generic 0.745, Exploits 0.741, Fuzzers 0.607,
Shellcode 0.333, DoS 0.072, Backdoor 0.032, Analysis 0.021, **Worms 0.000** (24 test rows).

The rare-class collapse is more severe than on the 2024 re-extraction, and the paper reports no
per-class numbers on its main datasets at all (Table 8, which promised them for the cross-dataset
setting, is a duplicate of Table 7 — see README critique).

TreeSHAP: additivity residual 4.8e-06. Top features `ct_dst_src_ltm`, `smean`, `sbytes`, `sttl`,
`ct_srv_dst`.

### Caveat on the sequential configuration

Concatenating the two official files puts all of `testing-set.csv` at the tail, and its class mix
differs sharply from `training-set.csv`: `Normal` is 30.9% of the first 80% of rows but **56.9%**
of the last 20%. So the leakage-safe sequential number (0.8065) is depressed by a genuine
train/test distribution shift, not only by honest evaluation. The file is *not* class-sorted
(99,170 attack_cat block changes), so this is a file-boundary artifact. `dedup + shuffled`
(0.9142 / F1 0.8934) is the fairer leakage-free estimate; treat 0.8065 as a lower bound.

## Not verified

- The paper's headline 98.1% / 96.8% on CIC-IDS2017 / UNSW-NB15. Neither dataset is in this
  repo, and neither this reimplementation nor the (unavailable) original has been run on them.
- The `shap.TreeExplainer` backend named in §4.1. It cannot load XGBoost 3.3 multiclass models
  at all under `shap` 0.46; values came from XGBoost's native TreeSHAP instead, which is the
  same algorithm. The two backends have not been cross-checked against each other here.
