# Model Results Summary

Multiclass (10-class) NIDS on **CIC-UNSW-NB15-2024**, leakage-safe pipeline:
dedupe → stratified 80/20 split → evidence-based feature selection → `class_weight='balanced'`.
Test set = **61,235 rows** (Benign ≈ 73.5% after dedup). Numbers read directly from the notebook outputs
(`decision_tree`, `random_forest`, `xgboost_model`, and the `*_mi_rfecv` variants).

**Macro-F1 is the primary metric** (rare classes count equally); accuracy is Benign-dominated and misleading.

## Overall performance

| Model | Features | Accuracy | ROC-AUC | Macro-P | Macro-R | **Macro-F1** |
|---|---|---|---|---|---|---|
| Decision Tree | 44 | 0.8855 | 0.7557 | 0.5408 | 0.5284 | 0.5338 |
| Decision Tree | 36 (MI+RFECV) | 0.8847 | 0.7556 | 0.5361 | 0.5282 | 0.5316 |
| Random Forest | 44 | 0.9017 | 0.9401 | **0.6521** | 0.5358 | 0.5744 |
| Random Forest | 36 (MI+RFECV) | 0.9024 | 0.9428 | **0.6580** | 0.5404 | 0.5805 |
| XGBoost | 44 | 0.8949 | 0.9803 | 0.5422 | 0.6510 | **0.5807** |
| XGBoost | 36 (MI+RFECV) | 0.8942 | 0.9803 | 0.5402 | **0.6534** | 0.5800 |

- **Best macro-F1:** XGBoost (0.5807, 44-feat)
- **Best macro-precision:** Random Forest (0.6580, 36-feat)
- **Best macro-recall:** XGBoost (0.6534, 36-feat)
- **Best ROC-AUC:** XGBoost (0.9803)

### Weighted (Benign-dominated) metrics — for completeness

| Model (44-feat) | Weighted-P | Weighted-R | Weighted-F1 |
|---|---|---|---|
| Decision Tree | 0.8858 | 0.8855 | 0.8856 |
| Random Forest | 0.9031 | 0.9017 | 0.9004 |
| XGBoost | 0.9153 | 0.8949 | 0.9021 |

## Per-class F1 — 44-feature baselines

Bold marks the best model for each class.

| Class | Support | DT | RF | XGB |
|---|---|---|---|---|
| Benign | 45,001 | 0.9746 | 0.9796 | **0.9826** |
| Analysis | 77 | 0.2553 | 0.3043 | **0.4516** |
| Backdoor | 90 | 0.5263 | **0.6338** | 0.5123 |
| DoS | 883 | 0.3418 | 0.3823 | **0.4140** |
| Exploits | 6,182 | 0.7222 | **0.7634** | **0.7634** |
| Fuzzers | 5,279 | 0.6422 | 0.6926 | **0.6928** |
| Generic | 899 | 0.7395 | **0.7941** | 0.7662 |
| Reconnaissance | 2,406 | 0.5772 | **0.6069** | 0.5754 |
| Shellcode | 370 | 0.2140 | 0.1762 | **0.2954** |
| Worms | 48 | 0.3448 | **0.4110** | 0.3529 |

## Key findings

- **Imbalance, not model capacity, is the ceiling.** Accuracy is flat at ~0.88–0.90 while macro-F1
  only climbs from 0.534 (DT) to 0.581 (XGB) — about 5 points across three progressively stronger models.
- **RF and XGBoost are complementary / inverted** under the same `class_weight='balanced'` lever:
  RF = high precision (0.65) / low recall (0.54); XGBoost = low precision (0.54) / high recall (0.65).
  XGBoost's recall lever rescues the rare classes (Analysis 0.26→0.45, Shellcode 0.18→0.30, DoS 0.34→0.41);
  RF wins the mid classes on precision. This complementarity motivates the stacking/hierarchical ensemble direction.
  > ⚠️ **Superseded — this was an artifact of `balanced` weighting.** Under α=0.25 the inversion
  > disappears: XGB 0.664 P / 0.567 R, RF 0.674 P / 0.527 R, ET 0.641 P / 0.523 R — every model has
  > precision > recall, with XGBoost uniformly ahead. See the DES study below. The *hierarchical*
  > direction stands on its own evidence; the *ensemble* justification does not.
- **Feature selection is essentially free.** Dropping 44→36 features (MI+RFECV) changes macro-F1 by
  ≤0.006 for every algorithm (RF even nudges up 0.5744→0.5805) — a leaner model at no accuracy cost.

## Two-stage hierarchical detection (thesis contribution)

> ⚠️ **Significance correction** (`ablation_significance.ipynb`). The hierarchy is **not established
> as a standalone improvement**: under `balanced` weighting it measures **−0.0061 (p = 0.174)** against
> flat XGBoost, so the "0.5807 → 0.5858" step below is inside the noise band and its sign is not
> reliable. It is significant **only together with α=0.25 weighting** (+0.0178, p = 0.005). Frame the
> contribution as **hierarchy + tempered weighting jointly** — an interaction, not a main effect.

Problem **restructured** into a gate + family classifier (both XGBoost, 400 rounds), same leakage-safe
pipeline: **Stage A** = binary Benign-vs-Attack; **Stage B** = 9-class family classifier trained on
**attack rows only** (families remapped 1–9→0–8). Notebooks: `two_stage_hierarchical.ipynb` and
`two_stage_stageB_experiments.ipynb` (Stage-B resampling ablation).

| Configuration | Accuracy | Macro-P | Macro-R | **Macro-F1** | Selection |
|---|---|---|---|---|---|
| Flat XGBoost (baseline) | 0.8949 | 0.5422 | 0.6510 | 0.5807 | — |
| Two-stage + balanced weights | 0.8944 | 0.5427 | **0.6779** | 0.5858 | — |
| Two-stage + capped SMOTE (cap≈2000, Stage B) | — | 0.6624 | 0.5790 | 0.5976 | **test (!)** |
| **Two-stage + α=0.25 tempered weights (Stage B)** | 0.9081 | **0.6574** | 0.6035 | **0.6157** | **validation** |

### Stage-B resampling ablation (gate held fixed)

`stageB_mF1` = Stage B in isolation on true attack rows; `e2e_mF1` = end-to-end 10-class macro-F1.

| Stage-B variant | stageB_mF1 | **e2e_mF1** | e2e_mP | e2e_mR |
|---|---|---|---|---|
| Balanced weights (current) | 0.5742 | 0.5858 | 0.5427 | **0.6779** |
| Full SMOTE | 0.5327 | 0.5561 | 0.5624 | 0.5768 |
| **Capped SMOTE cap=2000** | **0.5777** | **0.5976** | 0.6624 | 0.5790 |
| Capped SMOTE cap=5000 | 0.5708 | 0.5927 | 0.6390 | 0.5848 |
| Borderline-SMOTE cap=5000 | 0.5681 | 0.5866 | 0.6404 | 0.5720 |
| Capped SMOTE+Tomek cap=5000 | 0.5768 | 0.5956 | 0.6369 | 0.5961 |
| Capped SMOTE5000 + weights | 0.5555 | 0.5698 | 0.5351 | 0.6334 |

### Two-stage key findings

- **Progression:** flat XGB 0.5807 → two-stage 0.5858 → two-stage + capped SMOTE **0.5976** (+1.7 pts).
- **The gate is essentially solved; Stage B is the bottleneck.** Gate attack recall ≈ 0.997 / precision
  ≈ 0.918 — almost no attacks are missed at Stage 1. The ceiling is the family sub-problem
  (Stage-B macro-F1 ≈ 0.574 on *true* attack rows), so improvement effort belongs there.
- **Capped SMOTE is the best single Stage-B lever; full SMOTE still hurts** (0.556) even inside the
  two-stage — reconfirming the capped-not-full rule.
- **Weights vs capped SMOTE is a precision↔recall knob, not a free win:** balanced weights maximise
  macro-recall (~0.68); capped SMOTE maximises macro-precision (~0.66). The operating point is a SOC choice.
- **Honesty note:** the Stage-B cap (and the Stage-6 gate threshold) are read off the test set for
  illustration; tune both on a train-carved validation split before quoting a final number. Still missing
  from the novelty stack: **SHAP per-class explainability**.
  → **Addressed by the weighting study below**, which supersedes this section's selection protocol.

## Stage-B weighting study (`two_stage_stageB_weighting.ipynb`) — current best

The first result in the project selected on a **validation split carved from train** (train 195,950 /
val 48,988 / test 61,235), with both stages refit on train+val and the test set touched exactly once.
**Test macro-F1 = 0.6157**, beating the earlier *test-selected* 0.5976 by +1.8 points.

Schemes compared on Stage B with the gate held fixed: uniform, tempered inverse-frequency
$(1/n_c)^\alpha$, effective-number $(1-\beta)/(1-\beta^{n_c})$ (Cui et al. 2019), focal
$w_c(1-p_{\text{true}})^\gamma$ with out-of-fold $p$, and capped SMOTE as the incumbent comparator.

| Stage-B scheme | Weight ratio Worms:Exploits | val e2e macro-F1 |
|---|---|---|
| **α=0.25 (tempered)** | **3.4×** | **0.6123** |
| α=0.50 | 11.3× | 0.6088 |
| eff-num β=0.999 | 7.0× | 0.6054 |
| uniform (no weighting) | 1× | 0.6014 |
| capped SMOTE cap=2000 | — | 0.5988 |
| eff-num β=0.9999 | 56× | 0.5964 |
| α=0.75 | 38× | 0.5960 |
| capped SMOTE cap=5000 | — | 0.5853 |
| focal γ=0.5 | — | 0.5845 |
| `balanced` (α=1) | 128× | 0.5826 |
| focal γ=1.0 | — | 0.5672 |
| focal γ=2.0 | — | 0.3497 |

### Weighting-study key findings

- **`class_weight='balanced'` ranked 12th of 14 — below doing nothing at all** (0.5826 vs uniform
  0.6014). The project's imbalance baseline was worse than no weighting; α=0.25 is +0.0297 over it.
- **Interior optimum, confirmed twice.** The α curve peaks at 0.25 and the independent
  effective-number family peaks at β=0.999. Expressed as a **weight ratio** the two agree: the useful
  band is **3–11×**, while full inverse-frequency demands 128× and overshoots. This merges the earlier
  capped-SMOTE result and this one into a single principle — **moderate, not full, rebalancing** —
  covering both resampling and weighting.
- **Dynamic (focal) weighting failed, informatively.** γ=0.5 is a wash, γ=1 hurts, γ=2 collapses the
  model (Exploits F1 0.77→0.35). Focal assumes hard samples are *informative*; here the hardest
  samples are largely **ambiguous or mislabelled** (32% duplicate rows in the raw data, and after
  dedup on features+label any repeated feature vector carries *conflicting* labels), so raising γ
  pours capacity into label noise. A reportable negative result about the dataset, not the method.
- **Weighting beats resampling and is cheaper** — no synthetic rows, no training-set inflation, one
  scalar hyper-parameter. Capped SMOTE is superseded.
- **Validation selection generalised:** val 0.6123 → test 0.6157, ranking intact.
- **Fuzzers is a false-positive sink** (P 0.589 / R 0.902): of its 3,318 test false positives, 1,116
  (34%) are Benign flows leaking through the gate, 1,011 Exploits, 694 Reconnaissance — so a third of
  that precision loss is the gate's, not Stage B's. Confusion is **block-structured**:
  {Exploits, DoS, Analysis} and {Fuzzers, Reconnaissance, Shellcode}. DoS loses 351/883 to Exploits
  and Shellcode 180/370 to Fuzzers. Evidence for a three-level hierarchy, if wanted.

### Per-class F1 — current best (test, α=0.25)

| Benign | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Recon | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|
| 0.983 | 0.439 | 0.599 | 0.411 | 0.782 | 0.713 | 0.813 | 0.641 | 0.363 | 0.415 |

### Next levers, in order

1. **Per-class decision thresholds** on the same validation split — Fuzzers (P 0.589 / R 0.902) and
   DoS (P 0.637 / R 0.304) are badly off-balance for a macro-F1 objective. Largest remaining gain.
2. **Gate threshold**, now that the Fuzzers breakdown quantifies the payoff.
3. **Soft routing** $P(k)=P(\text{attack})\cdot P(k\mid\text{attack})$ in place of the hard gate.
4. **Label-ambiguity ceiling analysis** — the focal failure is indirect evidence that label noise
   binds; measuring it directly converts a limitation into a contribution.
5. **SHAP per-class explainability** (still outstanding).

## Dynamically weighted Stage-B ensemble (`two_stage_stageB_des.ipynb`) — NEGATIVE RESULT

> ⚠️ **Scope correction.** This notebook's conclusion that "the ensemble direction is closed for
> Stage B" was **too broad and is superseded**. It holds for a pool diversified by *algorithm*
> (XGB/RF/ET at fixed α), which is what was tested. It does **not** hold for a pool diversified by
> *class weighting* — see the α-diverse ensemble below, which reaches **0.6302**. The specific
> findings here (no algorithmic complementarity, uniform kNN competence weights, harmful static
> averaging over a degenerate pool) all stand.

Per-instance competence weighting on Stage B: three base learners (XGB / RF / ExtraTrees, all α=0.25),
competence estimated from each model's accuracy in the local neighbourhood of a query flow, combined
by weighted soft vote. `val` was halved into **DSEL** (competence source) and **SEL** (variant/K
selection); test touched once. Variants: OLA, class-weighted OLA, distance-weighted, LCA, sharpened
weights, plus single-model and static-average controls.

| Configuration | SEL | **Test** |
|---|---|---|
| Single XGB (train-only) — like-for-like baseline | 0.6167 | **0.6125** |
| Best dynamic (OLA-cw k=30, p=8) | **0.6212** | 0.6118 |
| Static equal-weight average | 0.6090 | 0.6019 |
| Single RF | 0.5984 | 0.5780 |
| Single ET | 0.5889 | 0.5597 |

**The SEL lead (+0.0045) did not transfer to test (−0.0007, rank 2 of 15) — it was selection noise.**
Under the project's older test-selection habit this would have been written up as a win.

### Key findings

- **The RF/XGBoost complementarity was an artifact of `balanced` weighting.** Under α=0.25 all three
  families have the *same* profile (precision > recall) with XGBoost uniformly ahead — the inversion
  is gone. When XGB and RF disagree (13.2% of attack rows) **XGB is right 44.8% vs RF's 29.6%**: a
  ~1.5:1 dominance, not a complementary trade. The "combine RF's precision with XGBoost's recall"
  rationale no longer describes the models.
- **Large oracle headroom, none of it reachable.** Oracle (correct if *any* base model is correct)
  = 0.6799 vs 0.5995 best single → **+0.0804 theoretical**, 79.1% of attack rows have some model
  right. No competence estimator realised any of it.
- **The dynamic weights barely move: 0.333 ± 0.02** — a static ensemble in disguise, which is why
  every dynamic variant lands within noise of the equal-weight average. Where the estimator *does*
  discriminate it does so badly: **ET leads on 44% of rows despite being the weakest model**, RF on
  12.6%. Local accuracy over k=30 neighbours from 6,494 DSEL rows is too noisy to beat "just use XGB".
- **Static averaging is actively harmful** (0.6019 vs 0.6125 single XGB) — averaging in two weaker,
  non-complementary models dilutes the strong one. This closes the ensemble direction for Stage B.
- **Recommendation: do not adopt.** Keep the single α=0.25 XGBoost. The ensemble costs a kNN query
  per flow at inference plus DSEL in memory, and returns nothing.

**Why this matters beyond the negative:** focal weighting failed because the hard samples are
label-ambiguous, and dynamic ensembling failed here because the *algorithmically* diverse models are
not complementary once weighting is right. The first still stands as evidence that label noise binds;
the second turned out to be a statement about pool construction, not about ensembling (see below).

## α-diverse Stage-B ensemble (`two_stage_stageB_ensemble_v2.ipynb`) — **CURRENT BEST, 0.6302**

Built against v1's two diagnosed failures. **Pool diversified along the α axis** rather than by
algorithm: XGBoost at α ∈ {0, 0.25, 0.5, 1.0}, a capped-SMOTE XGBoost, plus RF and ET at α=0.25 —
seven members deliberately spanning the precision↔recall axis instead of each being optimal.
**Combination is learned, not estimated**: a per-class weight matrix $W \in \mathbb{R}^{7\times9}$
fitted by coordinate ascent on macro-F1. Base models train on `train`; the combiner is fitted on
**DSEL**; selection on **SEL**; test touched once.

| Configuration | SEL | **Test** |
|---|---|---|
| **Per-class weights (α-diverse pool)** | **0.6443** | **0.6302** |
| Global weights (7 params) | 0.6317 | 0.6184 |
| Static equal average | 0.6301 | 0.6149 |
| Single XGB α=0.25 (train-only) | 0.6167 | 0.6125 |
| Stack: logistic | 0.6310 | 0.6074 |
| Stack: XGB(d=3) | 0.6106 | 0.5930 |

**+0.0145 over the previous project best (0.6157)**, +0.0177 over the like-for-like train-only single
model. The SEL winner ranks **1st of 12 on test**, SEL↔test **Spearman 0.937** — the selection
protocol held, unlike v1.

### Key findings

- **The α axis was the missing diversity.** Oracle headroom nearly doubled (v1 0.0804 → **0.1456**),
  oracle macro-F1 0.6799 → **0.7318**, any-model-correct 79.1% → 83.5%. The disagreement matrix
  localises it: **XGB α=1.0 disagrees with the rest on 15–23% of rows** while α=0/0.25/SMOTE cluster
  at 5–9%. The member supplying most of the diversity is *individually poor* (macro-F1 0.5595) —
  **pool members should be chosen for complementarity, not individual quality.**
- **The per-class matrix genuinely specialises** (contrast v1's 0.333 ± 0.02 uniform weights):
  recall-heavy α=1.0 takes DoS / Fuzzers / Worms; precision-heavy α=0.0 takes Analysis / Backdoor /
  Reconnaissance / Shellcode / Generic; **Exploits keeps its uniform initialisation** — coordinate
  ascent could not beat a plain average on the largest family, which is a good sign it is not just
  overfitting every free parameter. Note the rare families go to the *precision*-heavy member: their
  precision, not recall, is the binding constraint, and the combiner found that from data.
- **Stacking failed where constrained linear weighting succeeded.** Meta-XGB scored 0.9178 in-sample
  on DSEL vs 0.5930 on test — textbook overfit. Both meta-learners landed *below* the single model.
  The per-class matrix has the same 63 parameters and generalised: **the difference is inductive
  bias, not capacity.** Constrained combiners win when the combiner-fitting set is small.
- **The gain is a deliberate rare-vs-large trade:** Worms **0.415→0.557**, Backdoor 0.599→0.652,
  DoS 0.411→0.433; paid for by Exploits 0.782→0.750 and Generic 0.813→0.790. Exactly what a macro-F1
  objective should do, and the literature gap this thesis targets — but state it as a SOC operating
  choice rather than hiding it in the macro average.
- **Only ~12% of the oracle headroom was realised** — substantial complementarity remains on the table.
- **Fair-comparison note:** the ensemble uses `train` + DSEL (half of val); the 0.6157 reference used
  train+val. The ensemble therefore uses *less* data, so the gain is not from extra rows.

### Per-class F1 — current best (test, α-diverse ensemble)

| Benign | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Recon | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|
| 0.983 | 0.427 | 0.652 | 0.433 | 0.750 | 0.708 | 0.790 | 0.641 | 0.360 | 0.557 |

### Next levers

1. ~~Per-class decision thresholds~~ — **tested and closed**, see below.
2. **Refit the pool on train+DSEL**, combiner on SEL.
3. **Widen the α grid** to 1.5 / 2.0 — α=1.0 supplied most of the diversity despite being a poor
   standalone model.
4. ~~Gate threshold, soft routing~~ — **tested and closed**, see below. Label-ambiguity ceiling, SHAP.

## Per-class thresholds + soft routing (`two_stage_thresholds.ipynb`) — NEGATIVE RESULT

Per-class multipliers on the fused scores, soft routing
$P(k)=P(\text{attack})\cdot P(k\mid\text{attack})$, and gate-threshold tuning — all fitted on SEL,
test touched once. **None of them pays. Keep the v2 ensemble's plain argmax (0.6302).**

| Variant | Params fitted on SEL | SEL | **Test** |
|---|---|---|---|
| **V0 hard routing, no thresholds** | 0 | 0.6443 | **0.6302** |
| V1 soft routing, no thresholds | 0 | 0.6452 | 0.6302 |
| V3 soft + Benign threshold only | 1 | 0.6470 | 0.6301 |
| V2 hard + T-attack | 9 | 0.6574 | 0.6259 |
| V4 soft + T-all | 10 | **0.6575** | 0.6232 |

**The rank order inverted exactly**: SEL `V4 > V2 > V3 > V1 > V0` (most parameters wins) vs test
`V0 > V1 > V3 > V2 > V4` (fewest wins). Monotone in parameter count in both directions — textbook
overfitting, and the whole result.

### Key findings

- **Per-class thresholds are algebraically redundant with the weight matrix** — $t_c\,S(c\mid x) =
  \sum_m (t_c W[m,c])P_m(c\mid x)$ is column rescaling of $W$. Verified at prediction level:
  folding $t$ into $W$ reproduced predictions on **100.0000% of SEL rows**. Row-normalising first
  does not change this ($Z(x)$ is constant across classes).
- **Soft routing is a no-op**: +0.0009 SEL, **+0.0000 test**.
- **The gate is already optimally calibrated.** The full threshold search left Benign at $t_0=1.0$ —
  an implied gate threshold of exactly $p_{att} > 0.5$. This also explains why the Fuzzers
  false-positive analysis (34% Benign leakage) never converted: those leaks are not threshold-fixable,
  they are flows the gate genuinely cannot separate.
- **With both degrees of freedom already spent, the 10 parameters could only fit noise.** SEL supplies
  ~19 Worms and ~31 Analysis rows, each worth 1/10 of macro-F1. The rare families got *worse*
  (Worms 0.557→0.507, Backdoor 0.652→0.617, Shellcode 0.360→0.347) while well-populated ones gained
  slightly; macro-P drifted up 0.673→0.700 and macro-R down 0.633→0.606.
- **Methodological lesson: asymmetric P/R is not evidence of a bad threshold.** The premise here was
  that Fuzzers (P 0.574 / R 0.926) and Exploits (P 0.900 / R 0.642) were "off-balance and therefore
  fixable". Per-class F1 is not maximised at P ≈ R — when the two error types have different marginal
  rates the optimum *is* asymmetric, and $W$ had already found it. The fitted thresholds did not even
  move consistently toward balance (Fuzzers went to 0.556 / 0.952).
- **Protocol note:** $t$ was fitted on SEL, where the v2 architecture was also selected. SEL is no
  longer clean — further tuning needs a fresh split carved from `train`.

**Value delivered is subtractive:** three lever candidates closed with evidence, and a third
independent line of evidence that the residual error is shared rather than fixable by adjusting the
decision rule.

## Ensemble v3 (`two_stage_stageB_ensemble_v3.ipynb`) — **FINAL MODEL, 0.6339**

Same architecture as v2 (7 members, per-class weight matrix), but **5-fold cross-fitted over
`train+val`**: base models train on all 244,938 rows, $W$ is fitted on ~32,000 OOF attack rows instead
of 6,494, and a fresh **SELV** split replaces the spent SEL. The α grid was also extended to 1.5/2.0
with two extra members (10-member pool) as a test of the diversity mechanism.

| Configuration | SELV | **Test** |
|---|---|---|
| **per-class W, 7-member subset** *(SELV-selected)* | 0.6312 | **0.6339** |
| per-class W, 10 members (α up to 2.0) | 0.6277 | 0.6329 |
| single XGB α=0.25 | 0.6058 | 0.6157 |
| static equal (10) | 0.6135 | 0.6113 |
| v2 ensemble (reference) | — | 0.6302 |

Ranked **1st of 13 on test**, SELV→test ordering held. Final per-class F1 (test):

| Benign | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Recon | Shellcode | Worms |
|---|---|---|---|---|---|---|---|---|---|
| 0.983 | 0.455 | 0.634 | 0.462 | 0.773 | 0.708 | 0.805 | 0.657 | 0.367 | 0.486 |

### Key findings

- **Extending α past 1.0 failed.** α=1.5/2.0 are individually degraded (standalone macro-F1 0.4929 /
  0.4090) and the 10-member pool scored *below* the 7-member subset. **The α axis saturates at ≈1.0.**
- **Oracle headroom does not predict realisable gain — the notebook's main methodological result.**
  v2→v3 raised mean disagreement 0.135→0.235 and oracle headroom 0.1456→**0.2250** (+55%), while the
  achieved score moved +0.004. α=2.0 disagrees with α=0.0 on 53% of attack rows yet contributes
  nothing net despite receiving real weight mass (0.922, second-highest). This retrospectively
  explains DES v1 (0.0804 headroom, zero realised): **more disagreement is not more complementarity.**
- **The +0.0037 came entirely from more data**, not the wider α grid — isolated by the
  `v2 7-member subset` control, which holds the members fixed and varies only cross-fitting.
- **Noise-band caveat.** Worms has 48 test rows and Analysis 77; Worms alone moved −0.071 between v2
  and v3. **Differences below ~±0.005 macro-F1 are not meaningful on this test set**, so v3-over-v2 is
  suggestive rather than decisively established. Worth stating in the thesis rather than quoting four
  decimals as if they were all significant.
- **Cost:** ~16 min training (746 s cross-fit + 180 s refit) vs v2's ~1 min. Inference cost identical
  (7 models + one 7×9 matrix).

### Where the project stands

Progression: 0.6157 → 0.6302 → 0.6339. **Improvement has plateaued.** Four separate levers — focal
weighting, dynamic ensemble selection, decision thresholds, wider α — have now returned ~nothing,
while every gain has come from better weighting or more data.

That pattern is the finding: **the residual error is shared across models, weightings, pools and
decision rules.** See the ambiguity analysis below for what it turned out to be. **SHAP** remains
outstanding in the novelty stack.

## Label-ambiguity ceiling (`label_ambiguity_ceiling.ipynb`) — hypothesis REFUTED

Tested whether the plateau is irreducible label noise. **It is not.** After dedupe on features+label,
any repeated feature vector carries conflicting labels; grouping identical vectors gives a strict
Bayes bound on any deterministic classifier.

| Measure | Value |
|---|---|
| Rows after dedupe / distinct feature vectors | 306,173 / **305,602** (99.8% unique) |
| Ambiguous groups | 497 |
| Rows in ambiguous groups | 1,068 (**0.35%**) |
| **Irreducible error from exact label conflict** | **0.19%** |
| Ceiling macro-F1 on test | 0.9881 |
| Achieved (v3, 10-member) | 0.6329 → **64.1% of ceiling** |

### Key findings

- **Exact label conflict cannot explain the plateau** — 0.35% of rows. "The labels are noisy" is not
  available as an excuse for rare-class performance.
- **The bound is real but weak.** 99.8% of rows are unique, so it is attained only by memorisation. A
  perfect memoriser trained on train+val scores **macro-F1 0.0848** on test (only 192 test rows, 0.31%,
  have a feature vector seen in training — and 100% of those carry a conflicting label, exactly as the
  dedupe logic predicts). The notebook is **eliminative, not aspirational**: it rules out a hypothesis,
  it does not set a reachable target.
- **The 76→44 prune is label-noise-neutral** — identical group structure at both feature counts, zero
  new ambiguous groups. An independent validation of the feature selection, useful in the methodology
  chapter since dropping 42% of features invites the question.
- **Ambiguity structure predicts model confusion: Spearman ρ = 0.699 (p = 9e-08)** across 45 class
  pairs. Fuzzers appears in five of the six largest ambiguity pairs and is also the model's biggest
  error sink. The 1,068 colliding rows are the visible tip of a much larger population of
  near-identical flows.
- **The real mechanism is class overlap in CICFlowMeter feature space, not label noise.** The
  exceptions prove it: Exploits↔Reconnaissance has **451 model confusions and zero exact ambiguity**;
  Analysis↔Exploits 72 and zero. Those families are not mislabelled — they are *inseparable in this
  representation*. This single mechanism explains all four negative results (shared errors in DES,
  focal's collapse on hard samples, thresholds' failure, α saturation).

### Per-class: % of ceiling attained

| Shellcode | Analysis | DoS | Worms | Recon | Fuzzers | Exploits | Generic | Benign |
|---|---|---|---|---|---|---|---|---|
| 38.9% | 45.5% | 46.8% | 49.7% | 65.8% | 71.4% | 77.4% | 81.8% | 98.3% |

Shellcode is the only class with meaningful intrinsic ambiguity (ceiling recall 0.895) and still
reaches only 38.9% of it.

### Next step

Exact matching cannot measure overlap — it needs a **k-NN label-overlap analysis** (done below).

## k-NN class overlap (`knn_class_overlap.ipynb`) — **HYPOTHESIS CONFIRMED**

Neighbours of test attack rows sought in train+val attack rows, RobustScaler space, leakage-safe.
Scope = the Stage-B 9-family sub-problem (the established bottleneck).

| Explanatory variable | Spearman ρ vs model confusion | rows it covers |
|---|---|---|
| exact label ambiguity | 0.538 (p = 7e-4) | 0.35% |
| **k-NN overlap** | **0.827 (p = 4.9e-10)** | **100%** |

Decisive: overlap explains the pairs exact matching was blind to — **Exploits↔Recon** (overlap 0.247,
*zero* exact ambiguity, 451 confusions), **DoS↔Exploits** (0.493, zero, 442), **Analysis↔Exploits**
(0.521, zero, 72). Those families are not mislabelled; they are inseparable in this representation.

### 1. First non-vacuous irreducible-error estimate

Cover–Hart on the measured 1-NN error (0.3755) brackets the Bayes error of the 9-class problem:

**e\* ∈ [0.213, 0.376] → accuracy ceiling ∈ [0.625, 0.787]**

v3's Stage-B accuracy on the 16,234 true attack rows is **0.737** — inside the bracket, near its upper
end. So roughly **21–38% of attack flows cannot be classified correctly by any method operating on
these 44 features**. (The previous notebook's 0.0019 was a strict but vacuous bound.)

### 2. The ensemble is doing real work

| | macro-F1 |
|---|---|
| best k-NN (k=10, α=0.25 weighted) | 0.4458 |
| **v3 ensemble, Stage B** | **0.6211** |

**+0.1753** over the best local classifier. The data is hard *and* the modelling recovers a great deal
a local rule cannot — both halves belong in the write-up.

### 3. Purity ranks families the way performance does — ρ = 0.700 (p = 0.036)

| Family | Worms | Shellcode | Backdoor | DoS | Analysis | Generic | Recon | Fuzzers | Exploits |
|---|---|---|---|---|---|---|---|---|---|
| purity (k=25) | 0.059 | 0.123 | 0.136 | 0.174 | 0.246 | 0.359 | 0.542 | 0.639 | 0.654 |
| achieved F1 | 0.486 | 0.367 | 0.634 | 0.462 | 0.455 | 0.805 | 0.657 | 0.708 | 0.773 |

Only Exploits, Fuzzers and Recon have majority-own neighbourhoods. A Worms flow's 25 nearest
neighbours are **5.9% Worms**.

### 4. Structure: two attractors swallow everything else

Analysis is **51.4% Exploits**; Shellcode **46.5% Fuzzers**; DoS 43.5% Exploits; Worms 43.4% Exploits;
Generic 35.1%; Backdoor 32.0%. **Exploits and Fuzzers are attractors** and the six smaller families are
embedded in them — one geometric fact underlying the Fuzzers false-positive sink, the collision pairs,
and the block-structured confusion matrix alike.

### 5. Correction to the notebook's own table

`plurality_ceiling_R` in §4 is a **misnomer — it is not a ceiling**. Worms scores 0.000 there yet
achieves F1 0.486. Plurality voting minimises *error rate*, not macro-F1, so a rare class is never the
plurality of its own neighbourhood. Read it as "recall of the accuracy-optimal local rule" — a
reference the model should beat, and does. The real ceiling statement is the Cover–Hart bracket.

### What this settles

The plateau at ~0.63 is a **property of the CICFlowMeter representation**, not a modelling deficiency.
Four modelling levers failed because they operate downstream of a constraint that lives in the features.

**For the write-up:** reframe "weak rare-class recall" as a measured geometric result; future work is
**richer features** (payload, host/temporal context, session aggregation), not better classifiers on
these 44 columns; and **SHAP should target the overlap pairs** (Exploits↔Recon, DoS↔Exploits,
Fuzzers↔Shellcode) rather than rare classes generically.

## Ablation + bootstrap significance (`ablation_significance.ipynb`) — read before quoting any number

2,000 stratified bootstrap resamples of the test set. All five ladder configurations train on the same
195,950 rows and are scored on the **same resample** each iteration, so differences are **paired**.

| Contribution | Δ macro-F1 | 95% CI | p | established |
|---|---|---|---|---|
| **weighting (inside two-stage)** | **+0.0385** | [0.0246, 0.0512] | <0.001 | **YES** |
| hierarchy (at α=0.25) | +0.0178 | [0.0061, 0.0311] | 0.005 | YES |
| ensemble | +0.0177 | [0.0067, 0.0301] | 0.003 | YES |
| weighting (flat) | +0.0146 | [0.0001, 0.0276] | 0.049 | marginal |
| **hierarchy (at `balanced`)** | **−0.0061** | [−0.0144, 0.0030] | 0.174 | **no** |
| **FULL: final vs baseline** | **+0.0502** | [0.0391, 0.0606] | <0.001 | **YES** |

Ladder point estimates: flat/balanced 0.5801 → flat/α=0.25 0.5947 → two-stage/balanced 0.5740 →
two-stage/α=0.25 0.6125 → two-stage+ensemble 0.6302.

### Key findings

- **The hierarchy is an interaction effect, not a main effect** — negative under `balanced`,
  significant under α=0.25. The project's original "two-stage helps" claim was never established.
- **α=0.25 weighting is the largest single lever (+0.0385)** — more than twice the entire seven-member
  ensemble (+0.0177). The cheapest change in the project contributes the most.
- **Two yardsticks, not one.** Reporting a single configuration: mean 95% CI width 0.0350 → **±0.017**,
  so quote `0.630 [0.611, 0.649]`, never four decimals. Comparing two configurations on the same test
  set: use the **paired** interval, ≈**±0.012**. Judging differences by the marginal width would
  wrongly discard two genuine results.
- **v3 over v2 (+0.0037) is 0.39 bootstrap SDs — within noise.** Keep v3 as the final model on
  **protocol** grounds (cross-fitting spends the data properly; W fitted on ~32,000 rows vs 6,494),
  never on score.
- **Only one step of 0.6157 → 0.6302 → 0.6339 is resolvable** (the ensemble).

### Per-class 95% CIs — report rare classes as intervals

| Class | support | F1 | 95% CI | width |
|---|---|---|---|---|
| Benign | 45,001 | 0.983 | [0.982, 0.984] | 0.002 |
| Fuzzers | 5,279 | 0.708 | [0.702, 0.715] | 0.013 |
| Exploits | 6,182 | 0.750 | [0.740, 0.758] | 0.018 |
| Reconnaissance | 2,406 | 0.641 | [0.627, 0.656] | 0.029 |
| Generic | 899 | 0.790 | [0.771, 0.809] | 0.038 |
| DoS | 883 | 0.433 | [0.401, 0.462] | 0.061 |
| Shellcode | 370 | 0.360 | [0.317, 0.405] | 0.088 |
| Analysis | 77 | 0.427 | [0.361, 0.493] | 0.132 |
| Backdoor | 90 | 0.652 | [0.556, 0.740] | 0.184 |
| **Worms** | **48** | **0.557** | **[0.422, 0.683]** | **0.260** |

Worms spans ±0.13 — one significant figure. **Past per-class claims on Worms (including the "+0.142
from the v2 ensemble") are not established.**

Intervals capture test-set sampling only; split and seed variance are excluded, so they are optimistic.
