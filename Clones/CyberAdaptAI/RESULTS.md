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

> **⚠️ Correction (2026-08-09).** This section was written on the assumption that Table 4 is a
> **binary** evaluation. It is not — **Fig. 4 is captioned "Multi-Class confusion matrices for
> CIC-IDS2017 and UNSW-NB15"**, so Table 4's 0.968 is a 10-class number. The 2x2 below is still
> valid and still the sharpest evidence about duplicate leakage and shuffling, but read it as
> *"binary is an upper bound on what the framework can do"*, not as *"this is Table 4's setting"*.
> The correct objection is the averaging scheme: the paper never states one ("macro" appears
> nowhere in 21 pages) and its `P ≈ R ≈ F1 ≈ Accuracy` implies **micro-averaging**, so Table 4
> must be compared against **weighted/micro F1**. See the multiclass subsection below, and the
> correction box in README.md.

Binary attack-vs-benign. 2x2 over the two pipeline choices the paper leaves unspecified:

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

### Multiclass (10-class), leakage-safe, sequential — THIS is the setting Table 4 reports

Accuracy 0.7871, **macro-F1 0.4215**, macro P/R 0.4983/0.4120, weighted F1 0.7784, 10 drift
events, 0.157 ms/instance. Duplicates removed: 94,928 (36.8%).

**Comparison against Table 4 (0.968), on the metric Table 4 actually uses.** Since the paper's
`P ≈ R ≈ F1 ≈ Accuracy` implies micro-averaging, the comparable columns are weighted F1 and
accuracy — *not* macro-F1:

| | duplicates | weighted F1 | accuracy | macro F1 |
|---|---|---|---|---|
| Paper Table 4 (UNSW) | not reported | — | **0.968** | not reported |
| this reimplementation, leakage-safe | 94,928 removed | 0.7784 | 0.7871 | 0.4215 |
| **thesis v6 port** (`final_model_v6_unsw.ipynb`) | 94,928 removed | **0.8380** | **0.8303** | **0.6387** |

The two independent pipelines agree exactly on the duplicate count (94,928), which is good
evidence both read the file correctly. The residual gap to 0.968 is duplicate leakage plus
shuffling, as the 2x2 above quantifies.

Per-class F1: Normal 0.892, Recon 0.773, Generic 0.745, Exploits 0.741, Fuzzers 0.607,
Shellcode 0.333, DoS 0.072, Backdoor 0.032, Analysis 0.021, **Worms 0.000** (24 test rows).

The rare-class collapse is more severe than on the 2024 re-extraction, and the paper reports no
per-class numbers on its main datasets at all (Table 8, which promised them for the cross-dataset
setting, is a duplicate of Table 7 — see README critique).

TreeSHAP: additivity residual 4.8e-06. Top features `ct_dst_src_ltm`, `smean`, `sbytes`, `sttl`,
`ct_srv_dst`.

### Multiclass + no-dedup + shuffled — the decisive cell (`results/unsw_repro_table4_multiclass`)

Run after Fig. 4 established that Table 4 is **multi-class**. This is the multiclass twin of the
`no-dedup + shuffled` cell that reproduced the paper in binary — i.e. Table 4's task with both
leakage assumptions granted.

| Configuration | task | dedup | shuffle | accuracy | weighted F1 | macro F1 |
|---|---|---|---|---|---|---|
| **Paper Table 4 (UNSW)** | ? | ? | ? | **0.968** | — | — |
| **Paper Fig. 4 (its own matrix)** | multi-class | — | — | **0.303** | — | ~0.15 |
| repro, binary + leakage | binary | no | yes | 0.9479 | 0.9590 | — |
| **repro, multiclass + leakage** | multi-class | no | yes | **0.8324** | 0.8099 | 0.5470 |
| repro, multiclass leakage-safe | multi-class | yes | no | 0.7871 | 0.7784 | 0.4215 |
| **thesis v6 port** | multi-class | yes | no | **0.8303** | **0.8380** | **0.6387** |

**Conclusion: Table 4's 0.968 is a binary-scale number.** Granting *both* leakage assumptions
lifts multiclass only +0.045 (0.7871 → 0.8324); the remaining 13.6 points to 0.968 are not
reachable by any dedup/shuffle setting, because the 9-way family split cannot score that high on
this data. Binary + the same two assumptions lands at 0.9479 / 0.9590, within ~0.01-0.02.

This **partially rehabilitates the original README assumption**, on firmer ground than it had:
not *"the paper describes a binary evaluation"* — Fig. 4 explicitly says multi-class — but
*"only a binary evaluation can produce that magnitude."* The paper's three pieces of evidence are
therefore mutually incompatible:

1. Fig. 4 caption: the evaluation is multi-class.
2. Table 4 magnitude: only achievable in binary, and only with leakage.
3. Fig. 4 contents: imply 0.303 accuracy, not 0.968.

Two further observations from this run:

- **Drift events = 0.** Shuffling makes the stream IID, so the configuration that best reproduces
  the headline eliminates the drift phenomenon Tables 5/6 analyse. Weight deviation from uniform
  is 0.0073.
- **Generic reaches F1 0.988 on 11,815 rows** — duplicate inflation, it being the most duplicated
  family in UNSW-NB15. Compare the leakage-safe run's 0.745.

**The thesis comparison worth quoting:** the leakage-safe v6 port (0.8303 / 0.8380) *matches or
beats* CyberAdaptAI's leakage-boosted multiclass run (0.8324 / 0.8099), and beats it by **+0.092
macro-F1**, with duplicates removed and no shuffling.

### Matching sweep — can any principled configuration hit Table 4's 0.968? (`results/match_*`)

Six cumulative configurations on top of the reproducing base (binary + `--no-dedup` + `--shuffle`),
each adding one assumption the paper leaves unspecified. Target: 0.968 / 0.965 / 0.970 / 0.968.

| config | accuracy | precision | recall | F1 | added assumption |
|---|---|---|---|---|---|
| **Paper Table 4** | **0.9680** | 0.9650 | 0.9700 | 0.9680 | — |
| base binary + leakage | 0.9479 | 0.9613 | 0.9567 | 0.9590 | `--no-dedup --shuffle` |
| A | 0.9476 | 0.9633 | 0.9541 | 0.9587 | `--no-hoeffding` (Fig. 6/7 plots only RF/XGB/ET) |
| B | 0.9476 | 0.9633 | 0.9541 | 0.9587 | `+ --retrain-scope cumulative` |
| C | 0.9472 | 0.9630 | 0.9538 | 0.9584 | `+ --scaler global` |
| **D** | **0.9807** | 0.9862 | 0.9835 | 0.9848 | **`+ --keep-id`** |
| E | 0.9807 | 0.9861 | 0.9836 | 0.9848 | `+ --batch-size 1500` |
| F | 0.9887 | 0.9852 | 0.9982 | 0.9917 | `+ --train-file-only` |

**Only one knob matters, and it is the row index.** Pool composition, retrain scope, scaler fitting
and batch size are collectively worth **0.0007** (0.9479 -> 0.9472, i.e. noise). Keeping the `id`
column — a row counter with no network semantics — is worth **+0.034 on its own** and **overshoots
the paper**.

**Table 4 is bracketed, not reproduced:**

```
    0.9479  (no id)   <    0.968  (paper)   <    0.9807  (id kept)
```

No principled configuration lands on 0.968. Hitting it exactly from here would mean tuning knobs
*toward the published value*, which is curve-fitting rather than reproduction, so the sweep stops
here. What can be stated is the bracket, plus the earlier finding that the multiclass ceiling is
0.8324 even with both leakage assumptions granted — so 0.968 is a binary-scale number regardless.

**`--keep-id` is worth reporting in its own right.** UNSW-NB15's `id` is a row index; any model that
gains 3.4 accuracy points from it is exploiting file ordering, not detecting intrusions. It is a
plausible unstated preprocessing omission in any paper that does not list its dropped columns — and
none of the ten reviewed papers publishes code that would let a reader check (see
`../../literature_artifact_audit.md`).

### The complete {dedup x order x id} cube, binary (`results/cube_*`)

All eight cells, default 4-member pool, batch 2000. Target = 0.968.

| dedup | order | `id` | accuracy | precision | recall | F1 | dev from 0.968 |
|---|---|---|---|---|---|---|---|
| ✓ | sequential | drop | **0.8065** | 0.7917 | 0.6812 | 0.7323 | 0.1615 |
| ✗ | sequential | drop | 0.9019 | 0.9095 | 0.8570 | 0.8825 | 0.0661 |
| ✓ | sequential | keep | 0.9016 | 0.9079 | 0.8581 | 0.8823 | 0.0664 |
| ✗ | sequential | keep | 0.9016 | 0.9079 | 0.8581 | 0.8823 | 0.0664 |
| ✓ | shuffled | drop | 0.9142 | 0.9020 | 0.8850 | 0.8934 | 0.0538 |
| ✗ | shuffled | drop | 0.9479 | 0.9613 | 0.9567 | 0.9590 | 0.0201 |
| ✓ | shuffled | keep | **0.9803** | 0.9862 | 0.9827 | 0.9845 | **0.0123** |
| ✗ | shuffled | keep | **0.9803** | 0.9862 | 0.9827 | 0.9845 | **0.0123** |

**Three findings.**

**1. `--keep-id` silently disables deduplication.** The dedup and no-dedup rows are *identical to four
decimals* whenever `id` is kept, in **both** ordering conditions. With a unique row index in the
feature matrix no two rows are duplicates, so the dedup pass removes nothing. Keeping an index column
does not merely add a leaky feature — **it reintroduces duplicate leakage through the back door.**

> **Generalisable reviewing rule:** a paper that reports "duplicate records were removed" *and* retains
> an `id`/index column has **not** deduplicated. The two statements are incompatible, and the second
> silently voids the first. Worth stating in the review chapter — it is checkable from a methods
> section without any code.

**2. Shuffling is a NECESSARY condition for the paper's number, and it is the one that breaks their
own narrative.** Every sequential cell is <= 0.9019; every cell within 2 points of 0.968 is shuffled.
This was tested directly: the `no-dedup + sequential + keep-id` cell was run specifically to see
whether the score could be reached *while preserving the stream* — it reaches only 0.9016. So there is
**no configuration that reproduces Table 4 and leaves concept drift well-defined.**

**3. Table 4 is bracketed but not reproducible.**

```
    0.9479  (shuffled, no id)   <    0.968  (paper)   <    0.9803  (shuffled, id kept)
```

Three necessary conditions for a number near 0.968, each measured: **binary** (multiclass caps at
0.8324 with all leakage granted), **shuffled** (all sequential cells <= 0.9019), and **duplicates
present** (directly, or indirectly via `id`). Two of the three contradict the paper's own method
section, which specifies temporally ordered mini-batches (§3.2) and duplicate removal (§3.2).

**Report the bracket and the three conditions, not a matched number.** Landing exactly on 0.968 from
here would require tuning alpha / gamma / batch size toward the published value, which is fitting to
the target, not reproduction. The one substantive untested variable is `--form full` (the 2.54M-row,
49-column release rather than the 257k pre-split); the paper does not say which it used, and those
files are not in `Resource/UNSW-NB15/`.

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
