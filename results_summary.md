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

> ⚠️ **The "host/temporal context" recommendation is SUPERSEDED** by `final_model_v6_unsw.ipynb`.
> Given exactly those features (UNSW-NB15's 14 `ct_*` connection-history counters), the model assigns
> them **6.8% of importance**. The fields that actually matter on the same traffic are **service
> identity and TTL** — `svc_dns` (0.295), `ttl_diff` (0.226), `sttl`, `r_ttl` — neither of which
> CICFlowMeter emits. The representation claim in this section is **confirmed and strengthened**; only
> the prescription for *which* richer features changes.

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

---

## Dynamic Stage-B ensemble v4 (`two_stage_stageB_dynamic_v4.ipynb`) — two dynamic models, both within noise of v3

Per-instance competence weighting over the **α-diverse** pool, with competence estimated from
**cross-fitted OOF** predictions. Re-opens the question DES v1 closed, after v2/v3 showed v1's two
premises were wrong. Same prune, splits, seeds and 7-member pool as v3 — the combiner is the only
moving part.

| Model | Weights vary by | SELV | **Test macro-F1** | **Test acc** | vs v3 |
|---|---|---|---|---|---|
| **v4-hybrid** `HYBRID W × LCA k=30 p=1` | class **and** instance | 0.6299 | **0.6295** | 0.9082 | −0.0039 |
| **v4-dynamic** `DYN LCA k=30` | instance only | 0.6279 | **0.6262** | 0.9085 | −0.0072 |
| v3 static per-class `W` (baseline, refit here) | class only | 0.6269 | 0.6334 | 0.9086 | — |
| DES v1 (algorithm-diverse pool) | instance only | — | 0.6118 | 0.9049 | −0.0216 |

Report as `0.630 [0.612, 0.647]` and `0.626 [0.608, 0.644]`. **Both gaps to v3 are inside the ±0.012
paired band — state as "no measurable difference", not as a loss.** Both models were selected on SELV
*within their own combiner family*, so neither involves reading the test table.

### Key findings

- **Competence still cannot be estimated — and it is no longer attributable to the setup.** DES v1 had
  two excuses: a non-complementary pool and only 6,494 reference rows. v4 removes both (α-diverse pool,
  disagreement 0.137, oracle headroom 0.1579; **32,468** reference rows, 5×) and the failure is
  unchanged. Mean |deviation from uniform| = **0.0064** against a uniform of 0.1429 — **4.5% relative**,
  versus v1's 5.1%. Only XGB α=1.0, the one genuinely divergent member, shows elevated dispersion
  (std 0.0147 vs 0.0066–0.0101), which is the signal a working estimator should give — far too weak to
  change an argmax. **The binding constraint is the representation**, consistent with the Cover–Hart
  bracket: neighbourhoods in this space are too impure to say which model to trust locally.
- **The per-class prior is load-bearing; the per-instance term is not.** Every hybrid variant lands
  within 0.004 of v3; every pure-dynamic variant loses 0.007–0.014. v4-hybrid beats v4-dynamic by
  +0.0033 **because of `W`, not the competence estimator**. The hybrid was constructed so `p=0`
  recovers v3 exactly (verified: identical predictions on 100.0000% of rows), so its null result cannot
  be blamed on combiner form. Honest summary: the dynamic component is **inert but harmless**.
- **Oracle headroom fails to predict realisable gain for the third time**: v1 0.0804→0, v3 0.2250→
  +0.0037, **v4 0.1579→0**. At least one member is correct on 83.25% of SELV attack rows while the best
  single reaches 0.6012. **Do not justify an ensemble by its oracle bound.**
- **Scope correction confirmed.** DES v1's "the ensemble direction is closed for Stage B" was about
  *pool construction*, not ensembling — fixing the diversity axis and data volume recovers **+0.0144 to
  +0.0177** over v1. It converges to what the static combiner already achieved rather than exceeding it.
- **Selection resolution, not selection error.** Within the pure-dynamic family SELV transferred
  perfectly (`LCA k=30` best on both). Across families it did not — v4-hybrid was 1st on SELV, 6th of 23
  on test — but all six leaders sit inside the noise band. SELV cannot separate configurations 0.004
  apart. **Report intervals, not ranks.**
- **Deployment cost is the real differentiator:** both v4 models need a kNN query per flow plus a
  32,468-row reference set resident in memory; v3 needs one 7×9 matrix multiply. Meaningful operational
  cost for no measured gain.
- Per-class vs v3 (v4-hybrid): Backdoor +0.028, Exploits +0.011, Generic +0.009; Worms −0.048 (48 rows,
  ±0.13 — not established), Recon −0.014, DoS −0.013. Analysis again high-recall/low-precision
  (0.870/0.306).
- Cross-fitting took 308 s (7 members vs v3's 10); OOF cached to `cache_oof_v4.npz`, so combiner
  iteration is ~2 min.

---

## Decision-space dynamic ensemble v5 (`two_stage_stageB_dynamic_v5.ipynb`) — CORRECTS v4

Same pool, splits and protocol as v4; the **only** change is that competence is estimated by kNN in
**decision space** (the pool's 7x9=63 output profiles) rather than in RobustScaler feature space.
That is the standard DES formulation (KNORA, META-DES); v4 used the non-standard one.

| | FEAT (v4's space) | DEC (v5) |
|---|---|---|
| Neighbourhood purity k=10 | 0.5721 | **0.6698** |
| k=100 | 0.5147 | **0.6666** |
| Mean per-member competence AUC | 0.8193 | **0.8539** |
| Best variant (test) | 0.6334 | **0.6363** |

**v5 = `DYN OLA-cw k=100 p=8`, test macro-F1 0.6363 / accuracy 0.9089**, SELV-selected and ranked
**1st of 20 on test**. +0.0029 over v3 — inside the ±0.012 band, so not an established win, but the
first time a *purely dynamic* combiner topped the table.

### ⚠️ RETRACTION of a v4 claim

v4 concluded *"local-accuracy competence cannot be estimated finely enough"*. **That is wrong.**
Per-member AUC for predicting actual correctness is **0.82–0.85**, nowhere near the 0.50 that would
mean noise. The estimator works. Decomposing `w_m(x)` over 35,153 rows:

| Quantity | Value |
|---|---|
| Shared variance (row difficulty) | **98.1%** |
| Member-specific variance | **1.9%** |
| Mean pairwise correlation of `w_m` across members | **0.9818** |
| Median within-row spread (max−min) | **0.0000** |
| Competence-argmax picks a correct member | 0.7411 |
| Always using the single best member | 0.7417 |

**Competence is estimable, but what is estimable is how hard the row is, not who should handle it.**
All seven members' competences move together (r=0.98); on the median row they are identical. Each
`w_m` predicts its own member well (high AUC) while the *relative ordering* carries almost nothing —
which is why weights sit at uniform and why competence-based selection (0.7411) cannot beat always
using the strongest member (0.7417).

This explains DES v1, v4 and v5 with one measurement, and **predicts the failure of any scheme whose
weights derive from local accuracy — including META-DES-style learned competence — without running
them.** It also explains why the winner uses `p=8`: heavy sharpening is needed to amplify a residual
that is 1.9% of the variance.

---

## Stage-A gate weighting (`two_stage_gate_weighting.ipynb`) — NULL RESULT

The gate used `compute_sample_weight("balanced")` — exactly **α=1.0** — since the first two-stage
notebook. The Stage-B weighting study ranked α=1.0 **12th of 14, below no weighting at all**, and that
lesson was never carried across to Stage A. Swept α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1.0} × threshold
∈ {0.3 … 0.8}, Stage B frozen to the v5 winner.

| Config | macro-F1 | accuracy |
|---|---|---|
| SELV-selected α=0.00, t=0.40 | 0.6369 | 0.9100 |
| default `balanced`, t=0.50 | 0.6363 | 0.9089 |
| **difference** | **+0.0006** | **+0.0011** |

- **Weighting and thresholding are one degree of freedom.** Every α's best cell is within **0.0010** of
  every other, with the optimal threshold shifting monotonically to compensate (α=0→t=0.4,
  α=1.0→t=0.7). Gate F1 is flat across α (0.9562–0.9566): re-weighting slides the operating point
  along a fixed ROC curve, it does not reshape it.
- **This completes the earlier threshold finding.** `two_stage_thresholds.ipynb` had only tested
  thresholds *under* `balanced`; the conclusion now holds for the whole weighting family.
- **The oversight was real and cost nothing.** α=1.0 is the worst row in the grid, but the grid spans
  0.0010.
- **Why precision cannot be bought here:** dropping α to 0 removed 101 of 1,455 leaking Benign flows
  and cost 744 missed attacks. **The flows that leak are ones the gate is confident about.** 1,051 of
  the remaining leaks still go to Fuzzers — a representation problem, not an operating-point one.

---

## Feature engineering (`feature_engineering.ipynb`) — first positive result, with a metric caveat

> ⚠️ **Correction from `feature_engineering_v2.ipynb`.** This section's headline — *"the Cover–Hart
> ceiling MOVED"* — **must be restated as metric-relative.** A signed-log transform of the *same*
> columns adds **zero** information and leaves the deployed model unchanged (98.99% identical
> predictions), yet moves $e_{1NN}$ by **+0.0296** — **2.3× the size of the −0.0128 headline below.**
> $e_{1NN}$ is a property of *(features, metric)* jointly and does not bound a tree ensemble, which is
> invariant to the transform that moved it. It remains the right measurement for the *distance-based*
> parts of the project (k-NN overlap, purity, the v5/v6 DEC-space combiner). **The adoption decision
> below still stands** — it rested on the score (+0.0101) and purity, both metric-free. What does not
> stand is the round-1 argument that *"two independent instruments agreeing"* is strong evidence: in
> round 2 the same two instruments **disagree**, so they were never independent.

Answers Tier 1.1 (un-prune) and Tier 1.2 (derived features). Three label-free spaces, identical splits.

| Space | features | e_1NN | e* bracket | accuracy ceiling |
|---|---|---|---|---|
| BASE | 44 | 0.3757 | [0.2135, 0.3757] | [0.6243, 0.7865] |
| FULL | 67 | 0.3697 | [0.2094, 0.3694] | [0.6306, 0.7906] |
| **DERIVED** | **75** | **0.3629** | **[0.2053, 0.3632]** | **[0.6368, 0.7947]** |

BASE reproduces the published 0.3755 (got 0.3757), validating the replication before any claim.
**McNemar exact test on paired 1-NN errors: BASE vs DERIVED p = 9.3e-06** (net +207 rows corrected);
BASE vs FULL p = 0.039; FULL vs DERIVED p = 0.038. **All significant.**

31 engineered columns, all scale-free or per-unit: directional asymmetry, bytes-per-packet,
IAT coefficients of variation, packet-length dispersion, **flags as fractions of packets**, header
overhead, duration-normalised active/idle shares, TCP init-window ratio.

- **Purity rose for every family, most for the attractor victims**: Worms +0.058, Backdoor +0.024,
  Analysis +0.022, Recon +0.020, Fuzzers +0.019, Shellcode +0.015; overall 0.357→0.376. Exactly the
  families the k-NN overlap analysis found swallowed by Exploits/Fuzzers.
- **Score (single α=0.25, matched):** BASE 0.6157/0.9081 → DERIVED **0.6258/0.9117** (+0.0101 macro-F1,
  +0.0036 accuracy). BASE reproduces the published single-model figure exactly.
- **Gate precision 0.9175 → 0.9267 from features alone** — what gate re-weighting could not buy at any
  operating point. **The gate was never the problem; the representation was.**
- **Tier 1.1 answered:** FULL beats BASE (p=0.039), so "correlated is not redundant for trees" is
  correct and the prune does discard usable information. But DERIVED still beats FULL (p=0.038) while
  starting from the pruned set. **Keep the prune; spend the budget on derived features.**
- Engineered columns are 41.3% of features and carry **46.7% of importance**. `PSH Flag Count` is #1
  and `frac_PSH` is **#2** — the fraction carries information beyond the count.

---

## Final model v6 (`final_model_v6.ipynb`) — **BEST: macro-F1 0.6485 / accuracy 0.9123**

DERIVED features + decision-space dynamic combiner (`DYN OLA-cw k=30 p=8`). SELV-selected, ranked
**1st of 21 on test**.

| Model | Features | Combiner | macro-F1 | accuracy |
|---|---|---|---|---|
| **v6** | **DERIVED 75** | **DEC dynamic k=30 p=8** | **0.6485** | **0.9123** |
| v5 | BASE 44 | DEC dynamic k=100 p=8 | 0.6363 | 0.9089 |
| v3 | BASE 44 | static per-class W | 0.6334 | 0.9086 |
| single α=0.25 | DERIVED 75 | none | 0.6258 | 0.9117 |

- **The gains compounded, slightly super-additively.** Feature effect +0.0101, combiner effect +0.0029;
  additive predicts ~0.6464, observed **0.6485**. The expected overlap did not materialise — better
  features also improve the competence estimate.
- **The optimal neighbourhood shrank** k=100 → **k=30** (p unchanged at 8), consistent with purer
  neighbourhoods. The pre-registered guess that *p* would fall was wrong.
- **INTERACTION, not a feature effect** (`v6_significance.ipynb`): neither ingredient is established
  alone — features without ensembling +0.0100 (p=0.071), dynamic-vs-static on BASE +0.0029 (p=0.555)
  — but together +0.0152 (p=0.012). **Ensembling on DERIVED is strongly established** (+0.0231,
  p<0.001) while on BASE it was not: the combiner only pays once the features are pure enough.
  Dynamic-over-static remains unestablished (+0.0026, inside noise) — the effect is *ensembling*,
  not *dynamism*.
- **8 of 10 families improved**, and unlike earlier version bumps the large-support families all moved
  the right way: Exploits +0.005, Fuzzers +0.009, Generic +0.011, DoS +0.028.
- **⚠️ Worms accounts for 57% of the headline gain** (0.070 of the 0.122 summed delta) on **48 test
  rows** (±0.13). **Excluding Worms the gain is +0.0052 — inside the noise band.** See
  `v6_significance.ipynb` for the paired bootstrap.
- Gate precision 0.9195 → 0.9280 at unchanged recall, purely from features.

---

## Feature engineering round 2 (`feature_engineering_v2.ipynb`) — NULL on score, and it corrects round 1's instrument

38 further columns on top of `DERIVED` (`D2`, 113 features), targeted at the named confusion pairs,
plus **signed-log control arms** that add zero information so the metric effect can be separated from
the information effect. Same splits, same protocol.

| Space | features | $e_{1NN}$ | purity k=10 | single α=0.25 | at the fixed v6 combiner |
|---|---|---|---|---|---|
| `DERIVED` | 75 | 0.3632 | 0.3761 | **0.6258** | **0.6485 / 0.9123** |
| `D2` | 113 | **0.3517** | **0.3880** | 0.6185 | 0.6464 / **0.9135** |
| `DERIVED_log` | 75 | 0.3928 | 0.3421 | 0.6202 | — |
| `D2_log` | 113 | 0.3878 | 0.3457 | 0.6227 | — |

**`D2` is rejected.** −0.0073 single-model, −0.0021 at the v6 configuration — both inside ±0.012, so
*no measurable difference*, and it costs 38 columns. `DERIVED` reproduced v6 to four decimals
(0.6485 / 0.9123), so the comparison is sound. `D2` is very slightly better on **accuracy** at every
combiner setting and worse on macro-F1 (Worms 0.582 → 0.568) — the familiar large-vs-rare trade.

### Key findings

- **THE MAIN RESULT: the Cover–Hart bracket is metric-dependent, and the metric effect is larger than
  any information effect in the project.**

  | Change | Δ$e_{1NN}$ | information? | p |
  |---|---|---|---|
  | round 1 `BASE`→`DERIVED` (+31 cols) | −0.0128 | yes | 9.3e-06 |
  | round 2 `DERIVED`→`D2` (+38 cols) | −0.0115 | yes | 4.7e-07 |
  | **control `DERIVED`→`DERIVED_log`** | **+0.0296** | **no** | **4.8e-22** |

  Tree invariance verified, not assumed: 98.99% / 98.79% identical predictions vs the logged twins.
  **Quote the bracket as metric-relative.** It still correctly describes the geometry that k-NN
  overlap, purity and the DEC-space competence estimator operate in — but it does not bound a tree
  ensemble. See the correction banner on the round-1 section above.
- **Log-compression made the k-NN geometry *worse*** (purity 0.3761 → 0.3421, every family down). The
  heavy-tailed raw scale is more discriminative under RobustScaler than a compressed one — the
  opposite of the usual intuition.
- **Round-2 columns are heavily used yet add nothing: 43.9% of importance on 33.6% of columns**, with
  `handshake_bal` the single most important feature (0.1452). The reason is visible in the algebra:
  `handshake_bal` $=(S-F)/(S+F)=(r-1)/(r+1)$ is **strictly monotone in round 1's `r_syn_fin` $=S/F$**
  — an equivalent feature that is merely bounded. Same for `fwd_len_range_norm` / `bwd_len_range_norm`
  (#3, #5) vs round 1's span ratios. **Three of the top five features are bounded monotone rewrites of
  columns already present**, which is exactly why the bracket moved and the score did not. Independent
  arrival at the same mechanism as the log control.
- **The targeted pairs mostly did not move.** Recon→Exploits 0.1754→0.1731, DoS→Exploits 0.4071→0.4145
  (*worse*), Shellcode→Fuzzers 0.4619→0.4670 (*worse*); only Worms→Exploits 0.3896→**0.3438**. The
  design hypothesis — that a scan's lack of response separates Reconnaissance from Exploits — **failed
  on its own terms**. The overall purity gain came from the large families improving their own
  neighbourhoods, not from the intended targets.
- **Pre-registered mechanism test SUPPORTED.** v6 was explained as an interaction (the combiner pays
  only once neighbourhoods are pure). `D2` is purer, so it should gain *more* from ensembling despite
  a lower single-model score: observed **+0.0279 vs +0.0227**. Corroborates the mechanism in a second
  feature space — but the difference of differences is +0.005, inside noise, so it is corroboration,
  not an established effect.
- **Algebra on these 76 columns is exhausted.** Round 1 marginal (p=0.071), round 2 null. The other
  feature directions are **impossible on this file, not merely expensive**: `CIC-UNSW-Data.csv` has no
  IP, port or timestamp column (verified), so per-host/per-window aggregates cannot be built, and
  percentile flow statistics need PCAPs the project does not have. **Restate the plateau as a limit of
  what CICFlowMeter records** — a stronger claim than a limit of the 44-column pruned subset.

## v6 architecture on the original UNSW-NB15 (`final_model_v6_unsw.ipynb`) — the representation claim, tested from outside

The v6 pipeline re-run end-to-end on `Resource/UNSW-NB15/` (Moustafa & Slay 2015; 257,673 pooled rows
→ **162,745 after dedupe**, 36.8% duplicates). Same ten families, same names, **zero shared columns**.

**This is not a transfer test** (no shared feature space) and **not a fresh-traffic generalisation
test** (CIC-UNSW-NB15-2024 is a re-extraction of this same traffic). It is a controlled test of the
project's central claim: hold traffic and taxonomy fixed, swap the extractor.

**Result: macro-F1 0.6387 / accuracy 0.8303** (SELV-selected `STATIC: per-class W`, 2nd of 17 on test
by 0.0007). **Do not compare these levels to v6's 0.6485 / 0.9123** — different class balance (Benign
52.7% vs 73.5%), row counts and flow segmentation. The per-family, purity and bracket comparisons are
the meaningful ones.

| Space | n_ref | e_1NN | e\* bracket | accuracy ceiling |
|---|---|---|---|---|
| CIC DERIVED 75 | 64,936 | 0.3629 | [0.2053, 0.3632] | [0.6368, 0.7947] |
| **UNSW-NB15 83** | **61,618** | **0.3043** | **[0.1680, 0.3043]** | **[0.6957, 0.8320]** |

### Key findings

- **The claim survives: the plateau is representation-bound.** $e_{1NN}$ falls **0.0586** vs CIC
  DERIVED — **4.6× the entire round-1 feature gain**, and more than rounds 1 and 2 combined. The
  comparison is **conservative**: UNSW has *fewer* attack rows (61,618 vs 64,936), and a smaller
  reference set can only raise the 1-NN error.
- **THE SURPRISE — the missing ingredient is NOT host/temporal context.** `knn_class_overlap.ipynb`
  named host/window context as the way forward and `feature_engineering_v2.ipynb` proved it
  unbuildable on the CIC file. UNSW has 14 such `ct_*` columns and they carry **only 6.8% of
  importance**. The top five features are `svc_dns` (0.295), `ttl_diff` (0.226), `svc_http`, `sttl`,
  `r_ttl` — **all built from service identity and TTL, two fields CICFlowMeter does not emit at all.**
  This also explains why two rounds of algebra returned nothing: no ratio of packet counts and
  durations can reconstruct *"this flow is DNS"* or *"TTL is 254 one way and 62 the other"*.
  **Name those two fields in future work, not "richer features" generically.**
- **Performance mostly tracks purity — 6 of 9 families move consistently**, in both directions.
  Shellcode 0.388→**0.650** (purity +0.214; its neighbourhood is 15.5% Fuzzers here vs 46.5% on CIC),
  Reconnaissance 0.640→**0.807**, Generic 0.828→0.924, Exploits 0.791→0.849. Exceptions: Fuzzers and
  DoS (small purity gains that did not convert) and Worms (34 rows, meaningless either way).
- **The gain is NOT uniform across the weak families — state this plainly.** Of CIC's four weakest
  (Shellcode 0.388, Analysis 0.446, DoS 0.468, Worms 0.582), only **Shellcode clearly improves**;
  Analysis and DoS get *worse*. The representation change trades one set of confusions for another
  rather than lifting everything.
- **Two families collapse.** **Backdoor 0.635 → 0.183** — precision 0.974, recall **0.101**; Analysis
  is the mirror (0.163 / 0.779). A known Analysis/Backdoor/DoS labelling pathology in UNSW-NB15, not
  something the pipeline introduced. **Worms +0.124 is on 34 test rows, contradicts the purity story
  (purity fell), and must not be quoted.**
- **The gate inverts.** CIC gate 0.9961 recall / 0.9280 precision; UNSW **0.9176 / 0.9209** — it
  misses 8% of attacks. **The two representations are good at different halves**: CICFlowMeter's
  timing/size statistics separate attack-from-benign, UNSW's service/TTL separate family-from-family.
  The strongest system would be the union of both field sets.
- **v4's finding replicated on an independent representation:** the static per-class `W` beats every
  pure-dynamic combiner by **−0.031**, and the top three variants all carry `W`. Ensembling gain over
  the single α=0.25 model is **+0.045** here vs +0.023 on CIC. Note SELV picked a *static* combiner
  here and a *dynamic* one on CIC — the protocol adapts rather than inheriting a fixed answer.
- **Still does not close the single-dataset gap** — that needs genuinely different traffic
  (CIC-IDS2017). The UNSW pre-split files are also a curated subset, so their class balance is itself
  an artifact of how that subset was drawn.

## Dedupe ablation on UNSW-NB15 (`results/unsw_dedupe_ablation/`) — the 10–15 point claim does NOT replicate

The v6-on-UNSW pipeline re-run with the dedupe step toggled off, everything else held fixed (same
seeds, splits, prune rule, pool, combiner grid, SELV protocol). Motivated by
`literature_artifact_audit.md`, which **at the time** asserted deduplication moves reported accuracy by
10–15 points. That assertion has since been **scoped to CyberAdaptAI's binary + sequential-stream
setting** — this ablation is what forced the correction, and the audit now carries it.

| | DEDUPE | NO-DEDUPE | Δ |
|---|---|---|---|
| rows used | 162,745 | 257,673 | +94,928 |
| features after prune | 83 | 80 | −3 |
| **macro-F1** | **0.6378** | **0.6403** | **+0.0025** |
| **accuracy** | **0.8299** | **0.8228** | **−0.0071** |
| e_1NN | 0.3044 | 0.2435 | −0.0609 |
| gate precision | 0.9242 | 0.9794 | +0.0552 |
| test rows with an exact train twin | 0.00% | **42.70%** | — |

- **⚠️ The 10–15 point claim does not hold for this pipeline on this dataset.** macro-F1 moves +0.0025
  (inside the ±0.012 paired band) and accuracy moves *down* 0.7 points, despite **42.7% of test rows
  having an exact features+label twin in train**. Before re-quoting the 10–15 figure, trace it to its
  source — most likely the CyberAdaptAI **binary** repro, a different task where memorising a
  duplicated flow pays off cleanly.
- **THE DECISIVE DIAGNOSTIC — leaked rows score *worse* than clean ones.** Within the no-dedupe arm:
  accuracy **0.8064** on the 22,007 leaked rows vs **0.8350** on the 29,528 clean ones. Same model,
  same test set, only the partition differs, so this comparison carries no confounds. **The model is
  not memorising.**
- **MECHANISM: duplication acts as class re-weighting, not memorisation**, because duplicates are
  wildly uneven across families — Generic 95.8% leaked, DoS 76.0%, Exploits 46.0%, Shellcode 4.6%,
  Worms 2.9%. Generic (95.8% duplicated) does inflate, 0.914→**0.990**. But that is cancelled inside
  the known DoS/Exploits/Analysis labelling triangle: DoS gets ~3× the effective weight, so DoS
  recall 0.327→**0.844** while Exploits recall 0.827→**0.562**. Per-class DoS +0.123 and Generic
  +0.076 against Exploits −0.167 and Analysis −0.104. **Net ≈ zero on the macro average.**
- **Where leakage DOES contaminate: the distance-based diagnostics.** e_1NN 0.3044→**0.2435**, a
  −0.061 move — larger than the entire round-1 feature-engineering gain — and pure artifact, since the
  test row's own twin sits in the reference set. Gate precision similarly inflates 0.924→0.979.
  **Dedupe is essential for the Cover–Hart bracket and every k-NN/purity/DES measurement, and nearly
  irrelevant for the classifier score.** This is the sharper and more defensible version of the claim.
- **⚠️ The two levels are not directly comparable.** The prune keeps 3 more columns under dedupe (83
  vs 80 — duplication inflates correlations past 0.95), and the test sets are different distributions,
  not just different sizes (Generic 1,520 → 11,774 test rows, 7.7×). Only the leaked-vs-clean split is
  confound-free.
- Reproduction note: the dedupe arm gives 0.6378/0.8299 vs the notebook's published 0.6387/0.8303, and
  SELV picked `HYBRID W × OLA-cw k=30 p=8` where the notebook picked `STATIC: per-class W`. Those two
  are 0.0026 apart — noise-band resolution failure, not a discrepancy. XGBoost `hist` with `n_jobs=-1`
  is not bit-deterministic.

## v6 on CIC-IDS2017 (`results/cicids2017_v6/`) — 0.9835, and why that is a RED FLAG not a win

v6 re-run on `Resource/CIC-IDS2017/` (2,830,743 pooled flows → **2,520,798** after dropping 2,867
inf/NaN rows and 10.86% exact duplicates). The three `Web Attack` variants collapsed into one class,
DoS variants kept separate → **13 classes**. All 31 round-1 derived features map onto CICFlowMeter v3
via a six-name rename (verified present, no silent zero-fill) → **78 features** (v6 had 75).

**Result: macro-F1 0.9835 / accuracy 0.9991.** Excluding Heartbleed (2 test rows) and Infiltration
(7): 0.9805.

| | CIC-IDS2017 | CIC-UNSW-2024 | UNSW-NB15 |
|---|---|---|---|
| macro-F1 | **0.9835** | 0.6485 | 0.6387 |
| accuracy | **0.9991** | 0.9123 | 0.8303 |
| e_1NN | **0.0026** | 0.3629 | 0.3043 |
| gate recall / precision | 0.9996 / 0.9955 | 0.9961 / 0.9280 | 0.9176 / 0.9209 |
| exact-duplicate rate | **10.86%** | 32% | 36.8% |

- **⚠️ TREAT THIS NUMBER WITH SUSPICION — it is the near-100% pattern CLAUDE.md warns about.**
  **e_1NN = 0.0026**: 1-NN alone separates the 12 attack families with 99.74% accuracy, *after*
  deduplication, vs ~0.30–0.36 error on both other datasets. Per-class F1 is 1.000 for DDoS,
  FTP-Patator, SSH-Patator, Heartbleed and Infiltration; only **Bot (0.820)** sits below 0.99.
- **⚠️ UNTESTED HYPOTHESIS:** that exact dedupe (only 10.86% here) leaves tool-generated *near*-
  duplicate bursts intact, so every test row still has a near-twin in train. Consistent with the
  evidence but **not verified** — the near-duplicate distance check was not run. Do not present the
  mechanism as established.
- **THE ARCHITECTURE BECOMES IRRELEVANT — the most informative finding here.** All 17 combiner
  variants land within **0.0007** on test (best `DYN OLA-cw k=30 p=3` 0.9835, worst `single XGB α=1.0`
  0.9828). α-weighting, the hierarchy, the 7-member ensemble and the dynamic combiner are jointly
  worth nothing, against **+0.0502** for the full stack on CIC-UNSW. Even equal-average and plain
  ExtraTrees match the winner. **The architecture's value is conditional on the problem being hard** —
  the flip side of the v6 interaction result, and worth stating as a contribution in its own right.
- **This does NOT close the single-dataset-evaluation gap.** CIC-IDS2017 as processed cannot test the
  "plateau is representation-bound" claim, because there is no plateau to explain. Making it a genuine
  generalisation test needs a **chronological split** (requires `GeneratedLabelledFlows.zip` for
  timestamps) so a burst's siblings cannot appear in training.
- **THE ONE SOLID POSITIVE — the engineered features replicate on a third extractor.** Derived columns
  are 39.7% of the feature space but carry **59.0% of importance**, and the top three are all derived
  (`hdr_per_pkt_fwd`, `frac_PSH`, `cv_bwd_len`). **`frac_PSH` lands top-2 on both CIC-UNSW and
  CIC-IDS2017 independently** (CIC-UNSW: `PSH Flag Count` #1, `frac_PSH` #2). This survives whatever
  the headline score turns out to mean.
- `Destination Port` ranks 13th (0.0321) — mild evidence against the service-identity hypothesis from
  the UNSW run, though on a dataset this easy nothing discriminates much.
- **Do not compare the level to 0.6485/0.6387** — different taxonomy (13 classes vs 10), different
  class balance (BENIGN 83.1%), different traffic. Note also that 0.9991 accuracy vs the bracket's
  [0.9974, 0.9987] is **not** a ceiling violation: the bracket is Stage-B 12-family error on attack
  rows, the accuracy is 13-class including an 83%-BENIGN majority.
- Deviations from the notebook protocol, all forced by 10× scale and recorded in the JSON: combiner/DES
  reference capped at 40,000 stratified attack rows (v6 used 32,468, v6-unsw 30,809, so the regime
  matches); inf/NaN rows dropped not imputed; SMOTE `k_neighbors` clamped (Heartbleed has 11 rows);
  float32. Runtime 30.4 min total, 23 min of which is the cross-fit.

### CIC-IDS2017 dataset facts (`results/cicids2017_audit/`)

Measured, not assumed — the equivalent of CLAUDE.md's "verified data facts" for this file.

- **2,830,743 rows × 79 columns** (78 features + `Label`) across 8 day-files; all 8 agree on the column
  list after stripping, so `concat` is safe.
- **Not clean, unlike CIC-UNSW-2024**: `Flow Bytes/s` 1,358 NaN + 1,509 inf, `Flow Packets/s` 2,867
  inf (0.10% of rows). These break `RobustScaler` and every distance computation.
- **65 of 79 column names carry leading whitespace**; naming is CICFlowMeter v3
  (`Fwd Avg Bytes/Bulk`) vs the 2024 file's `Fwd Bytes/Bulk Avg`.
- **`Fwd Header Length` is duplicated** — pandas silently renames the copy to `Fwd Header Length.1`,
  so a duplicate-*name* check finds nothing; the correlation prune catches it at r=1.000.
- **The three `Web Attack` labels contain a non-ASCII en-dash** — read with `encoding="latin-1"` or the
  classes silently fragment.
- **Duplicates 10.89% on features+label, 10.92% on features only** → only **698 rows** carry
  conflicting labels (0.02%). Contrast UNSW-NB15, where label ambiguity drove the whole dedupe result.
- **Extreme imbalance, 206,645:1**: BENIGN 80.30% down to Heartbleed 11 rows, SQL Injection 21,
  Infiltration 36. **Heartbleed and Infiltration cannot support a stratified split** (~2 and ~7 test
  rows) — report them, but never in a headline macro average.
- Prune: **78 → 70 → 47 features** (CIC-UNSW: 76 → 67 → 44), so the two representations are comparable
  in width. All four `Subflow *` columns are r=1.000 duplicates of the corresponding `Total *`.

### v7.1d — per-region W (`results/v71d_region_w/`) — NULL, but it brackets the granularity axis

The combiner ladder had two endpoints and nothing between them: `K=1` global per-class `W`
(established) and `K=N` per-instance competence (inert). This fills the gap. KMeans over the 70-dim
**decision space** (v5: decision space beats feature space) gives K regions; each gets its own 7×10
`W` fitted by the same coordinate ascent under the same **end-to-end objective** as v7.1b. Shrinkage
`W_k ← (1−λ)·W_glob + λ·W_k` and a 500-row minimum region size guard the small-sample overfitting v2
warned about. `K=1` reproduces `STATIC W_e2e` on **100.0000%** of rows (asserted), so any delta is
attributable to regions alone. Same prep/splits/caches as v7/v7.1/v7.1b; SELV selects, test once.

**SELV-selected `REGION K=2 × OLA-cw k=30 p=3`: test macro-F1 0.6907 / acc 0.9859, rank 4 of 14.
Paired vs v7.1b +0.0018, 95% CI [−0.0034, +0.0076], p ≈ 0.550 — NOT established. Keep v7.1b.**

- **Regional specialisation is REAL — this is not the v4/v5 null.** Distinct locally-best members per
  region: 2 of 2, 2 of 3, 3 of 5, **4 of 8** (`XGB a=0.0`, `a=0.5`, `a=1.0`, `smote2000`). The fitted
  matrices genuinely diverge: mean `|W_k − W_glob|` 0.109–0.165, mean pairwise `|W_a − W_b|`
  0.20–0.31 — against v4's dynamic weights sitting within **4.5% of uniform**.
- **The regions are semantically meaningful and unsupervised.** K=8 recovers the project's own
  confusion structure with no labels: Exploits 90% (n=9,589), Recon 90% (4,149), Generic 90% (1,670),
  Shellcode 73% (766) — and the two **highest-spread** regions are exactly the two long-standing
  problems, the Benign 59%/Fuzzers 40% sink (spread 0.069) and the Exploits 42%/DoS 19%/Recon 15%
  triangle (0.070).
- **THE MECHANISM — disagreement is concentrated where it is useless.** Across the 8 regions,
  best-member accuracy and best-worst spread are anti-correlated at **Spearman ρ = −0.857 (p = 0.007)**.
  Where members disagree most, all of them are wrong (acc 0.521, 0.593); where they are accurate they
  agree almost perfectly (acc 0.898 at spread 0.009). Local selection can only act where selection
  cannot pay. This one fact explains v4, v5 and v7.1d together. ⚠️ **It is a symptom, not the binding
  constraint — see v7.1e, which removes it and still gets no dynamic gain.**
- **Classic in-sample → held-out shrinkage:** COMB +0.0076 → SELV +0.0009 → test +0.0018. Same
  winner's-curse pattern the v7.2 two-stage protocol was built to catch.
- **⚠️ Do not read the shrinkage ordering as evidence.** On SELV λ=0.5 beat λ=1.0 at **every** K
  (monotone in the constraint); on test the ordering **reverses** at K=2, 3 and 8. Both orderings sit
  inside the noise band, so neither is informative.
- **Total SELV↔test inversion, and it is resolution failure not selection error**: `REGION K=8 λ=1.0`
  was *last* on SELV (0.6643) and *first* on test (0.6924); v7.1b was 1st on SELV and 10th on test.
  All 14 variants span **0.0043** — about a third of the ±0.012 paired band.
- The nominal +0.0018 is rare-class noise: Worms +0.0339 (**49** test rows) and Backdoor +0.0309
  (**90**), paid by Fuzzers −0.0181 (5,287) and Exploits −0.0124 (6,185). Macro precision rises
  0.7152 → 0.7241 while recall falls — the precision/recall knob again, not a free win.
- Runtime 6.1 min. Same-run v7.1b reference 0.6888 vs 0.6874 previously (±0.0014 refit jitter).

### v7.1e — disjoint feature-view pool (`results/v71e_view_pool/`) — the mechanism prediction lands, the payoff does not

v7.1d's ρ = −0.86 was traced to pool construction: all 7 members see the **same 99 features** and
differ only in class weighting, which moves decision boundaries but not information, so every member
is blind on the same rows. This tests the implied fix — members trained on **disjoint feature views**
should fail on *different* rows, producing member-specific competence rather than shared row
difficulty. The 99 columns are partitioned (disjoint and exhaustive, asserted in code) into
`SVC` 25, `TIME` 25, `SIZE` 39, `FLAG` 10, plus an `ALL` 99 full-feature control member. Three pools
are scored under one protocol: **ALPHA** (the current 7, class-weight diversity), **VIEW** (5 members,
information diversity), **UNION** (12). Go/no-go criteria were pre-registered and printed *before* the
combiner grid.

**Diagnostic — the hypothesis is CONFIRMED:**

| measure | ALPHA | **VIEW** | UNION |
|---|---|---|---|
| shared variance in competence weights | 93.4% | **51.2%** | 54.9% |
| median within-row weight spread | 0.0000 | **0.2956** | 0.3296 |
| mean pairwise weight correlation | 0.928 | **0.465** | 0.664 |
| region acc-vs-spread Spearman ρ | **−0.857** (p=0.007) | **+0.071** (p=0.867) | +0.429 |
| mean pairwise disagreement | 0.107 | **0.366** | 0.252 |

**Score — SELV-selected `UNION HYBRID W × OLA-cw p=1`: test macro-F1 0.6955 / acc 0.9868, rank 3 of
21. Paired vs v7.1b +0.0066, 95% CI [−0.0017, +0.0144], p ≈ 0.105 — NOT established (point-positive).**

- **The anti-correlation is annihilated (−0.857 → +0.071) and dynamism still does not pay.** The cell
  v7.1d said had to fill, filled — VIEW regions now include high-spread **and** high-accuracy cells
  (best-acc 0.949 / spread 0.396; 0.909 / 0.704), which did not exist under ALPHA. **⚠️ This REFUTES
  the v7.1d mechanism as the binding constraint.** Competence became estimable and dispersed exactly
  as designed, and the payoff never arrived.
- **THE REAL MECHANISM — competence optimises the wrong objective.** Every pure-dynamic variant, in
  every pool, has the **highest accuracy and highest macro-precision and lowest macro-recall** in the
  table (UNION `DYN OLA-cw p=8`: mP 0.7912 / mR 0.6417 / acc 0.9881, vs `UNION STATIC W` 0.7219 /
  0.7008 / 0.9868). Competence *is* local accuracy, so competence weighting minimises **error rate**
  while the objective is macro-F1; it buys accuracy already saturated at 98.7% and pays in rare-class
  recall. Same trap as `knn_class_overlap.ipynb`'s "plurality voting minimises error rate, not
  macro-F1", and the mirror image of v7.1b's lesson: **fit the combiner on the deployment objective.**
- **Dynamic-vs-static is flat or negative in all three pools** — ALPHA +0.0007 / VIEW −0.0025 /
  UNION −0.0003 for the hybrid, and **−0.0119 / −0.0123 / −0.0149** for pure-dynamic. The best test
  variant overall is `UNION STATIC W` (0.6976).
- **The gain is the POOL, not the dynamism:** ALPHA STATIC 0.6881 → UNION STATIC 0.6976 = **+0.0095**;
  UNION STATIC → UNION HYBRID p=1 = **−0.0021**; net +0.0066.
- **SELV transferred at the pool level and not below it**: all three UNION variants ranked 1-2-3 on
  both SELV and test, shuffling only among themselves.
- **v2's rule replicates a third time — pick members for complementarity, not quality.** `V:ALL` is the
  single best member (mF1 0.6325) while `V:FLAG` (0.2279) and `V:TIME` (0.3296) are the worst in the
  project, and adding them helps. `VIEW EQUAL-AVG` is catastrophic (0.6345 test, 0.6005 SELV): the
  weak views only pay **with** learned per-class weights.
- **VIEW alone does not beat ALPHA** (static 0.6845 vs 0.6881). The union of strong-but-redundant and
  weak-but-complementary members is what moves.
- Per-class, 6 of 10 improve and the large families are flat-to-positive (Exploits +0.0007,
  Fuzzers +0.0082, Recon +0.0032), but **Backdoor +0.0354 (90 rows) and Analysis +0.0217 (77 rows)
  carry most of the gain** — the usual rare-class caveat.
- **⚠️ ALPHA's shared variance here is 93.4%, not v5's 98.1%** — different setting (10-class reject
  problem on the full release). Same conclusion, different number; do not quote them as identical.
- Deployment cost: 12 members per flow vs 7, for an unestablished +0.0066. **Headline stays v7.1b.**
- Runtime 16.1 min (7 min cross-fit); view OOF cached to `cache_oof_v71e_views.npz`.

**Joint conclusion of v7.1d + v7.1e — dynamic combination is closed, properly.** Three combiner
families (per-instance, per-region, hybrid) across two pool constructions (class-weight diversity,
information diversity), with the suspected confounder eliminated by construction. The negative is now
mechanistic rather than empirical: *we built a pool where competence is demonstrably estimable and
member-specific, and dynamic weighting still did not pay, because local accuracy optimises a different
objective than macro-F1.*

### v7.3 — adversarial evasion under an attacker-controllable budget (`results/v73_adv_budget/`)

The standard adversarial-NIDS experiment perturbs the feature vector freely, as if it were an image.
A flow record is not a picture: it is a receipt of packets that had to traverse a network and still
work as an attack. Two constraints follow. **(1) Some columns cannot move at all** — if the attack
exploits RPC, the traffic goes to port 111; setting `svc_rpc = 0` describes a different attack, not a
stealthier one. Those columns carry **70.4% of importance**. **(2) The vector must stay internally
consistent** — so the attacker sets PRIMITIVES only and every derivable column (CICFlowMeter rates and
ratios plus all 31 engineered features) is **recomputed**. Buckets: FREE 49 raw cols (timing, padding,
size shape), COSTLY 18 (packet/byte/flag counts, half budget), IMPOSSIBLE 25 (svc/proto/dst port).
Budget in units of each column's train IQR; greedy per-feature sign search against the gate's
`p(attack)`; only the 16,298 test attack rows are perturbed; model fixed. Clean run reproduces v7.1b
(0.6888 / 0.9860).

| budget (IQR) | clean | **random** | realisable adversarial | unconstrained |
|---|---|---|---|---|
| 0.002 | 0.6888 | **0.5592** | 0.4758 | 0.3489 |
| 0.020 | 0.6888 | **0.4826** | 0.3379 | 0.2522 |
| 0.200 | 0.6888 | **0.2815** | 0.1280 | 0.1089 |

- **THE HEADLINE IS THE CONTROL: undirected random noise does most of the damage.** At 0.002 IQR
  (realised L2 0.008) *uniform noise with no model access* costs 0.13 macro-F1 and drops attack recall
  0.8994 → 0.7737. Adversarial search roughly doubles that (gap 0.08–0.18), so the model is attackable
  too — but the primary property is **brittleness**, which is far broader than evasion: it implies
  degradation under ordinary measurement noise, a different flow timeout, or another CICFlowMeter
  build. Consistent with Cover–Hart: rows sit near boundaries, so small displacements flip them.
- **⚠️ THE ORIGINAL HYPOTHESIS IS REFUTED.** The prediction was that the curve would stay flat because
  `svc` is 70.4% of importance and outside attacker control. It does not — attack recall halves at a
  realisable budget of **0.020 IQR** (L2 0.071 across ~16 columns). **Feature importance does not
  predict adversarial robustness**: a model can lean 70% on unforgeable evidence and still be evaded
  through the remaining 30%, because the attacker needs only *some* direction the boundary is close in.
- **THE METHODOLOGICAL RESULT SURVIVES, with a number**: the realisability gap is **+0.127 macro-F1 at
  0.002**, 0.06–0.12 across the realistic regime, decaying to 0.019 at 0.20. **The standard
  unconstrained experiment overstates the damage by 0.10–0.13** by editing evidence no attacker owns
  and emitting records whose rates contradict their own primitives.
- Constraint ordering is monotone (timing/size-only 0.5656 > free+costly 0.4758 at 0.002), evidence the
  bucketing is meaningful rather than arbitrary.
- **Three bugs were found and fixed mid-run; the first version's numbers are void**: budget accumulated
  across greedy passes instead of being clipped (stated budgets wrong by up to 2×); no random control;
  and an L∞ budget across 67 columns was described as if it were small. Realised L∞/L2/n_moved now
  print per row. Runtime 9.6 min.

### v7.3b — is the brittleness real, and is it the model or the representation? (`results/v73b_brittle_diag/`)

Two pre-registered checks before attempting any fix. **PART 1**: v7.3 perturbed integer-valued columns
continuously; **43 of 67** raw columns are integers, and sub-integer moves are not realisable flows.
**PART 2**: seven model families spanning the capacity range, identical data, identical noise — the
same eliminative move that killed the label-noise hypothesis.

- **PART 1 — the claim SURVIVES.** Integer rounding moves recall **+0.0085** at budget 0.002 and stays
  within ±0.011 across the sweep. Sub-integer moves were not driving it.
- **REFINEMENT worth carrying:** the *gate* only falls 0.9893 → 0.9339 at 0.002, while end-to-end
  attack recall falls 0.8994 → 0.7737 and macro-F1 falls 0.13. **Most of the damage is Stage B
  scrambling which family, not the gate missing the attack.**
- **PART 2 — MODEL-bound, not representation-bound.** At matched benign fpr = 0.0168:

| family | clean recall | retained @0.02 | retained @0.20 |
|---|---|---|---|
| **RF 100** | **0.9758** | **1.009** | **0.928** |
| DT d6 | 0.8723 | 0.970 | 0.868 |
| XGB d8/50 | 0.9845 | 0.889 | 0.544 |
| **XGB d8/400 (deployed gate)** | **0.9838** | **0.835** | **0.461** |
| XGB d3/400 | 0.9847 | 0.769 | 0.354 |
| 15-NN | 0.9224 | 0.842 | 0.685 |
| LogReg | 0.0511 | *degenerate* | *degenerate* |

- **⚠️ MATCHING THE OPERATING POINT INVERTED THE ANSWER.** Compared at the default 0.5 cut, RF looked
  *worst* (retained 0.392) and `XGB d8/50` best. RF's probabilities are compressed (matched threshold
  0.0900), so the naive comparison measured threshold placement, not robustness. **Never compare
  robustness across families without matching fpr first.**
- **Depth is not the lever** — `d3/400` is *worse* than `d8/400`. It is the averaging-of-independent-trees
  inductive bias, not capacity.
- **LogReg is degenerate, not robust**: clean recall 0.0511 at 1.68% fpr — a linear boundary cannot do
  this task at that operating point, so its retention figures (2.108 at 0.20) are arithmetic on a broken
  model. The script's verdict line quotes it as the low end of the span; the real span across usable
  families is 0.769–1.009. **Smoothness alone is not the answer.**
- **Retention above 1.0 is real**: RF's recall *rises* slightly under small noise — the signature of a
  model whose decision regions are wide relative to the perturbation.
- Scope: binary gate task, 400k stratified subsample (`d8/400` scores 0.9838 here vs 0.9893 deployed).
  Runtime 3.1 min.

### v7.3c — RF vs XGB end-to-end (`results/v73c_rf_gate/`) — the GATE is the lever, not Stage B

Four arms isolate the two stages. Both gates trained on the same 800k subsample and thresholded to the
same benign fpr. **Pre-registered prediction: arm C (Stage-B swap) would help more than arm B (gate
swap), since v7.3b showed most damage is family scrambling. REFUTED.**

| budget | A: XGB gate | **B: RF gate** | C: RF Stage B | D: both |
|---|---|---|---|---|
| 0.000 | 0.6865 | 0.6827 | 0.6617 | 0.6563 |
| 0.020 | 0.4797 | **0.5072** | 0.4619 | 0.4855 |
| 0.200 | 0.2433 | **0.2996** | 0.2377 | 0.2953 |

- **Gate recall tells the whole story**: XGB 0.9862 → **0.4686** at budget 0.20; RF 0.9823 → **0.9688**.
  The RF gate loses 1.4% of its detections where XGB loses more than half.
- **Swapping Stage B buys nothing** (arm C −0.0056 vs baseline at 0.20) *and* costs −0.0248 clean.
  Arm D is no better than B. The entire benefit is the gate.
- **THE RESIDUAL PROBLEM:** even with the RF gate, end-to-end macro-F1 still falls 0.6827 → 0.2996 while
  attack recall holds at 0.79. **Detection can be made robust; family attribution cannot, by either
  model family.** In SOC terms: you still get the alert, you just cannot trust the label on it.
- **Confound, stated:** the reject set, Stage-B pool, `W_e2e` and competence reference all derive from
  the cached **XGB** OOF gate. Arm B is scored on a gate-positive population its reject class never saw,
  so the confound works **against** RF and the result is conservative. Runtime 4.3 min.

### v7.3d — paired bootstrap for the gate swap (`results/v73d_rf_gate_sig/`) — ESTABLISHED both ways

2,000 stratified resamples, both arms on the same resample each iteration. Predictions cached to
`rf_gate_sig_preds.npz`.

| budget | XGB | RF | delta | 95% CI | p | verdict |
|---|---|---|---|---|---|---|
| 0.000 | 0.6865 | 0.6827 | **−0.0038** | [−0.0064, −0.0013] | 0.005 | **worse, established** |
| 0.002 | 0.6089 | 0.6178 | +0.0089 | [+0.0048, +0.0131] | 0.001 | ESTABLISHED |
| 0.010 | 0.5575 | 0.5698 | +0.0123 | [+0.0074, +0.0179] | <0.001 | ESTABLISHED |
| 0.020 | 0.4797 | 0.5072 | +0.0275 | [+0.0202, +0.0366] | <0.001 | ESTABLISHED |
| 0.050 | 0.3621 | 0.4121 | +0.0500 | [+0.0417, +0.0607] | <0.001 | ESTABLISHED |
| 0.100 | 0.2978 | 0.3498 | +0.0520 | [+0.0465, +0.0587] | <0.001 | ESTABLISHED |
| 0.200 | 0.2433 | 0.2996 | **+0.0563** | [+0.0516, +0.0619] | <0.001 | ESTABLISHED |

- **NOT a free win — a cheap, well-characterised trade.** Pay **0.0038 macro-F1 clean** (a real,
  established loss, p=0.005); gain **+0.009 to +0.056** under noise, established at every budget down
  to 0.002 IQR. Report as an operating choice, the same shape as weights-vs-capped-SMOTE and the reject
  class's sink inversion: clean in-distribution traffic → keep XGBoost; measurement drift or an
  adversary → take RF.
- **⚠️ THE ±0.012 PAIRED BAND IS NOT A CONSTANT — see the correction in CLAUDE.md.** The measured paired
  CI here is ≈**±0.0025**, five times tighter, because the two arms share the Stage-B pool, reject set,
  `W_e2e` and every benign prediction and differ *only* in the gate. Applying the remembered ±0.012
  would have hidden a real effect.
- **Textbook illustration of the two-yardsticks rule:** the marginal intervals overlap almost entirely
  (XGB 0.6865 [0.6699, 0.7027] vs RF 0.6827 [0.6657, 0.6987]) while the paired difference is
  significant at p=0.005. Same data, opposite conclusions, depending on the yardstick.
- Runtime 5.9 min.

### TON-IoT dataset audit (`results/toniot_audit/`) — gate FAILS on `src_ip`; the data is too easy

`Train_Test_Network.csv`, 211,043 rows × 44 cols (file verified complete; note this is **not** the
461k-row version commonly cited). Author-balanced: normal 50,000 (23.7%), eight families at exactly
20,000, `mitm` 1,043 — **attack share 76.31%**, an artificial base rate. Duplicates **50.07%**, the
highest in the project, with 1,781 conflicting labels. Prune 52 → 46 after dropping 15 columns that are
>99% missing (all `ssl_*`, `http_*`, `weird_*` — the application-layer richness is absent in this file).

- **LEAKAGE GATE FAILS on `src_ip`**: out-of-sample recall **0.9998 at fpr 0.0402**, in-sample 0.9999.
  In-sample ≈ out-of-sample is the signature of a genuine fingerprint, exactly like `sttl`. Any TON-IoT
  result including `src_ip` measures the testbed.
- **The out-of-sample guard earned its keep**: `src_port` scores rec_in 0.9720 → **OOS 0.2797** (26,628
  values, none with ≥20 support). Scored in-sample it would have been flagged SUSPECT — a false alarm on
  an innocuous column. `dst_ip` hit recall 0.9949 at fpr 0.7733 and was correctly not flagged: at a 76%
  attack share, predicting "attack" for nearly everything is free recall.
- **TOO EASY to test the architecture**: attack-only **e_1NN 0.0306**, bracket [0.0155, 0.0306],
  accuracy ceiling [0.969, 0.985]. That is the CIC-IDS2017 regime (0.0026), where all 17 combiner
  variants landed within 0.0007 — not the CIC-UNSW regime (0.3629). Eight of nine families have k=10
  purity ≥ 0.91; only `mitm` (0.6212, 1,043 rows) is below.
- **PREDICTION CONFIRMED**: service identity is worth only **−0.0028 e_1NN** here versus **−0.153** on
  CIC-UNSW. The +0.0982 svc gain was recovering an omission specific to CICFlowMeter; Zeek already ships
  `service` and `dst_port`, so there is nothing to recover.
- **Usable two ways**: (1) as a *pre-registered prediction test* — write down in advance that all
  combiner variants will tie within ~0.005, then run it; a predicted null on a fourth dataset is far
  stronger than three post-hoc ones. (2) the `src_ip` finding stands alone in the literature-audit
  chapter, next to TTL: two datasets, two measured endpoint-identity label proxies, same failure mode.
- A bug was found and fixed during development: the fingerprint rule was originally scored in-sample,
  which flagged `ts` and `src_port` as SUSPECT purely because nearly every value is unique to one row.
  Validated on a synthetic file with a planted `src_ip` leak. See `results/toniot_audit/README.md`.

### Literature positioning (web search, 2026-08-18)

Six searches; two sources could not be retrieved directly (PDF parse failure, and a 403).

- **TTL as a "hidden label" is independently confirmed.** The Sarhan / Layeghy / Portmann feature-analysis
  line reports that `sttl`, `dttl` and `ct_state_ttl` should be **removed** from UNSW-NB15 for
  "unrealistically high predictive power, similar to what could be called a *hidden label*."
  **⚠️ Quote comes from search snippets, not a verified PDF — get the exact citation before it goes in
  the thesis**, since it is load-bearing for the literature-audit chapter. Our contribution beyond it:
  we *measured the mechanism* (benign 97% `sttl ∈ {31,29}` at 0.0000 attack share; the rule
  `sttl ∈ {62,254}` matches the entire 99-feature learned gate) and withdrew our own prior
  recommendation on the strength of it.
- **"The exporter matters more than the classifier" is published** — [Flow Exporter Impact on
  Intelligent Intrusion Detection Systems](https://arxiv.org/pdf/2412.14021) (2024) compares
  CICFlowMeter, Zeek, Argus and nfstream. Independent confirmation of the representation claim. We added
  the quantified ceiling (Cover–Hart bracket), one pipeline across three extractors, and then *cashed the
  diagnosis in* for +0.0982.
- **Tempered class weighting is corroborated outside NIDS**: the imbalanced-learning literature
  independently reports full inverse-frequency weighting as detrimental with the optimum far below it.
  α=0.25 with `balanced` ranking 12/14 is the NIDS instance of a general result.
- **No published multiclass macro-F1 baseline was found on CIC-UNSW-NB15-2024.** The under-explored
  premise holds — though six searches is not a systematic review.
- Closest related work to chase: [Zoghi & Serpen 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242)
  on UNSW-NB15 *data overlap and imbalance* — the nearest published thing to the k-NN overlap analysis.
  Returned 403; get it through the university library and position against it explicitly.

## What is left to try

See **`NEXT_STEPS.md`** — the forward-looking counterpart to this file. Ranked roadmap, the evidence
behind the ranking, and the list of levers already closed so they are not re-opened.

### v7.3e — targeted white-box attack on the RF gate (`results/v73e_rf_whitebox/`) — H1 SUPPORTED, but it scopes v7.3c/d

The one flank the v7.3 series left open. RF's smoother decision surface is the property credited for
its noise robustness — and smoothness is also what makes a surface easier to descend deliberately. So:
is RF actually more robust, or was it merely never the target? Five arms at matched benign fpr 0.0168,
same attacker-controllable budget, greedy per-feature sign search (2 passes), pre-registration printed
before any result. 8.3 min.

Retained fraction of clean **gate recall** at budget 0.20 IQR:

| Arm | Retained | Reading |
|---|---|---|
| `XGB<-XGB` white-box on XGB | **0.0313** | the deployed gate is destroyed |
| `RF <-RF` white-box on RF | **0.2143** | better, still loses 79% |
| `RF <-XGB` transfer from XGB | **0.8916** | what v7.3c-style evidence shows |

- **Pre-registered H1 SUPPORTED**: `retained(RF<-RF) - retained(XGB<-XGB) = +0.1830` against a +0.10
  threshold. **RF's advantage is a property of the model, not of the attack being aimed elsewhere** —
  the v7.3c/d recommendation stands. End-to-end macro-F1 agrees (RF 0.2146 vs XGB 0.1446).
- **⚠️ SURROGATE INFLATION +0.6773 — the reusable methodological result.** Aim the attack at RF and
  four-fifths of its apparent robustness evaporates (0.8916 -> 0.2143). **Every v7.3c/d number was
  measured under noise or an XGB-aimed attack, so all of them overstate RF's protection against a
  real adversary.** Any cross-model robustness claim from a surrogate or from noise must report this
  gap. It is the robustness analogue of the matched-fpr rule from v7.3b.
- **⚠️ THE DETECTION/ATTRIBUTION LADDER INVERTS WITH THE THREAT MODEL.** Under undirected noise RF
  retains 0.9931 gate recall vs 0.4834 macro-F1 (**attribution fails first**); under targeted
  white-box it retains 0.2143 vs 0.3144 (**detection fails first**). Mechanism: the attack optimises
  against the gate, so the gate is what breaks; noise has no target and damages the harder
  sub-problem. **Which layer fails first is a property of what the adversary optimises, not of the
  system alone.** The `novelty.md` headline must always carry its threat model.
- **State RF as "more robust", never "robust".** It is near-immune to undirected noise (retains
  0.9931) and clearly not to a targeted attack (0.2143).
- **A targeted attack is far stronger than noise at equal budget** for both families (XGB gate recall
  0.4871 random vs 0.0309 white-box). v7.3's "adversarial ~ 2x random" was measured on a full-data
  gate at a 0.5 cut; at matched fpr the multiple is much larger.
- **The pre-registered stall trap did NOT fire** (`moved` = 0.998 on RF<-RF at every budget), so the
  RF number is a genuine descent of RF's surface, not a greedy search stalling on piecewise-constant
  probabilities. The check was written in advance precisely so a stall could not be reported as
  robustness.
- **⚠️ `rf_whitebox.txt` prints a reproduction WARNING (max deviation 0.0718). It is a FALSE ALARM
  from a check that was mis-specified**: v7.3c `perturb()` applies `budget*IQR` to every column
  uniformly, while v7.3e's random arm applies the attacker structure (COSTLY at half budget) and is
  therefore strictly weaker noise. `verify_repro.py` re-ran it under the uniform model and reproduced
  v7.3c arm A at **0.0000 deviation across all seven budgets**. Prep, gate and threshold are exact;
  the greedy arms never used the v7.3c noise model. The warning is left in the log as the honest
  record, with the resolution beside it in that directory's README.
- **Confound carried over unchanged** from v7.3c: reject set, Stage-B pool, `W_e2e` and the competence
  reference all derive from the cached XGB OOF gate, so `RF<-RF` is scored on a population its reject
  class never saw. Works **against** RF, so the RF result is conservative.

### v7.3f — the misattribution threat model (`results/v73f_misattribution/`) — H1 REFUTED, threat model survives

Every adversarial-NIDS attack in the literature optimises `P(benign | x_hat)` -- Tafreshian & Zhang
(TrustCom 2024) state that fitness literally; the surveys describe the field the same way. **That
objective is inexpressible in a binary detector**, so nobody asks the other question: what if the
attacker does not hide, but makes you misidentify what they are? Stage A and Stage B being separately
trained components is what makes the question implementable and measurable. Three objectives, one
budget machinery, mutually exclusive outcomes (undetected / misattributed / correct), pre-registered.
15.8 min.

| Objective | success metric | b50 |
|---|---|---|
| EVASION | final prediction is Benign | **0.0145 IQR** |
| MISATTRIBUTION | detected, wrong family | **0.0545 IQR** |
| ABSTENTION | gate fires, Stage B rejects | >0.20 (never reached) |
| RANDOM control | misattributed | >0.20 (never reached) |

- **PRE-REGISTERED H1 REFUTED, in the opposite direction.** Predicted price ratio > 2; measured
  **0.27**. **Misattribution costs ~3.8x MORE than evasion** -- the cheapest attack on this NIDS is
  still hiding from it, not lying to it. The inference from "attribution collapses under undirected
  noise" (v7.3c/e) to "attribution is cheap to corrupt deliberately" was **wrong**: fragility under
  undirected perturbation does not imply cheapness under direction. Third instance in this project of
  a plausible mechanism failing to predict a realisable effect.
  - ⚠️ `b50` is mildly biased against misattribution (~28% of rows abstain, capping achievable
    misattribution at ~0.72, so its 0.50 target is ~69% of still-attributable rows). Far too small to
    flip a 0.27-vs-2 verdict; quote the ratio as approximate.
- **BOTH TRIVIALITY GUARDS PASSED, so the threat model survives its own refutation.** (a) The 2x2 is
  not diagonal -- the EVASION attack drives misattribution *down* to 0.0038 from a 0.1122 baseline,
  because you cannot mislabel a row you have deleted from the alert stream; these are genuinely
  different attacks. (b) **Direction buys a great deal**: directed misattribution reaches 0.6311 at
  budget 0.20 where random noise plateaus at 0.1497 (**delta +0.481**, growing monotonically from
  +0.069). The claim is "attribution is cheap to TARGET", not merely "attribution is fragile".
- **WHAT MISATTRIBUTION BUYS IS STEALTH, NOT CHEAPNESS -- the reportable finding.** At budget 0.20
  evasion drops gate recall to **0.0309** (97% of attack flows stop alerting -- any volume monitor
  sees that), while misattribution holds gate recall at **0.9969** with **63% of labels wrong**.
  Alert volume looks normal. It is also the most efficient attack on the *metric* at every budget
  (macro-F1 0.4712 at budget 0.002 vs evasion 0.5494, random 0.6035) and the worst on alert volume.
  **An evasion-only literature is structurally unable to see this attack.**
- **UNPLANNED FINDING: the reject class is an attack surface.** v7.1's reject class was a *defence*
  (benign-as-attack false alarms 8,370 -> 2,093, +0.0242 macro-F1). Under attack it inverts:
  `abstained` rises 0.0879 -> 0.2238 at budget **0.002** and plateaus near 0.29. In the misattribution
  arm `undetected` ~ `abstained` at every budget (0.2571 vs 0.2541 at 0.20), so **essentially every
  missed detection there is a reject-class abstention, not a gate miss** -- the gate never fails
  (0.997) and Stage B files the row as a false alarm. The attacker was never told to do this; told
  only to corrupt the label while staying detected, it **found the reject class as the cheapest
  route**. The dedicated ABSTAIN arm confirms the channel is directly targetable and cheap (0.2258 at
  budget 0.002) but bounded (never 50%). **Report the +0.0242 gain and this surface together, never
  the gain alone** -- same shape as the v7.1 Fuzzers operating-choice finding, extended to the
  adversarial case.
- **Scope**: attacker is white-box on ONE component (gate, or Stage-B member `XGB a=0.25`); deployed
  system is the full 7-member dynamic combiner. v7.3e measured surrogate inflation at +0.6773, so
  these attacks are **understated** -- conservative for the stealth and reject-surface findings, and
  the true misattribution price is likely below 0.0545, though not enough to claim H1 would flip.
  Symmetric across arms (each objective attacks exactly one model white-box), so the price comparison
  is fair. Gate, splits, pool, `W_e2e`, threshold and budget machinery identical to v7.3c/e; same
  XGB-OOF-derived confound, unchanged.

### v7.3g — controllability-aware feature allocation (`results/v73g_controllability_aware/`) — REFUTED

The natural defence to build from the v7.3 series: allocate feature access per stage by THREAT
EXPOSURE rather than accuracy -- restrict the gate to the 25 uncontrollable svc/proto/port features
(the easy stage can afford to be robust), keep Stage B at full width (the hard stage cannot). 2x2 over
gate x Stage-B feature access; Stage B a single XGB a=0.25 in every arm so feature access is the only
moving part. Pre-registered. 21.7 min. **Refuted, and the failure is more informative than the claim.**

| Test | Result |
|---|---|
| P1 svc-only gate recall constant under frozen svc | **PASS**, spread 0.00e+00 -- partition airtight, run valid |
| P2a hardening defends evasion (gap > 0.30) | PASS +0.3044 -- **but misleading, see below** |
| P2b hardening does NOT defend misattribution (abs < 0.10) | **FAIL** +0.4109 |
| P3 clean cost < 0.02 macro-F1 | **FAIL** -0.3354 |
| P4 non-tautology under service retargeting | **catastrophic collapse** |

- **THE PREMISE WAS WRONG, and it was a reasoning error worth recording.** The svc-only gate has clean
  recall **0.4603 vs the full gate's 0.9862** -- it misses 54% of attacks before any adversary appears.
  "svc lifted gate precision 0.9267 -> 0.9411" describes svc **added to** the full feature set at
  unchanged recall, and svc's 70.4% importance was measured on the **Stage-B family** problem. **Two
  different importance claims, on two different sub-problems, were conflated into a design premise
  neither supports.** Attack-vs-benign needs the behavioural features; "which service" does not tell
  you "is this malicious". The pre-registered cost test (P3) is what caught it.
- Clean macro-F1 across the 2x2: A full/full **0.6718** | B svcGate/full 0.3365 | C svcGate/svcB
  0.2977 | D full/svcB 0.4362. Not level-comparable to the 0.687 headline (single-model Stage B).
- **WHY P2a's PASS IS MISLEADING.** Arm B's evasion rate barely moves under attack (0.6650 clean ->
  0.6659 at budget 0.20) -- but only because it is **already failing at budget zero**. A pre-emptively
  broken detector is not a robust detector.
- **MEASUREMENT LESSON, generally applicable: `misattributed` is upper-bounded by gate recall** -- a
  row that never passes the gate cannot be mislabelled. Arm B's low misattribution (0.2305 vs A's
  0.6414) is an artifact of its broken gate, not protection. **Never compare attribution metrics
  across arms with different gate recall without normalising by rows that reached Stage B.**
- **P4 IS THE REAL FINDING: restriction CONCENTRATES risk, it does not remove it.** With svc frozen an
  svc-only gate is unattackable by construction (a tautology, which is why the MOBILE arm exists).
  Allow the attacker to retarget the service -- switch dst port, svc block recomputed coherently --
  and arm B collapses **totally at the SMALLEST budget tested**: undetected 0.6659 -> **1.0000**, gate
  recall **0.0000**, 99.98% of rows retargeted, at 0.002 IQR. The full-feature gate by contrast
  degrades **gracefully** across the same grid (0.9862 -> 0.7508 -> ... -> 0.0309). **A model
  concentrated on uncontrollable evidence has a single point of failure; a model spread across many
  features degrades continuously.**
- **Mirror image of v7.3's refuted hypothesis.** There, a model leaning 70% on unforgeable evidence
  was still evaded through the other 30%. Here, a model leaning 100% on unforgeable evidence is
  destroyed the moment that evidence becomes forgeable. **Feature restriction is not a robustness
  mechanism in either direction.**
- **What survives**: nothing recommendable -- 13th pre-registered null. Keep the generalisable claim
  (*robustness through feature restriction concentrates risk*), the measurement lesson, and the
  conflation error as a methodological note.

### v7.3h — controllability-aware ADVERSARIAL TRAINING (`results/v73h_adv_training/`) — real defence, wrong threat model

The correction to v7.3g: same controllability knowledge applied by **augmentation** (as Tafreshian &
Zhang, TrustCom 2024 do) rather than **restriction** (which v7.3g refuted). Four arms: A baseline,
B noise-augmented, C evasion-adversarial gate, D + misattribution-adversarial Stage B. Stage B a
single XGB a=0.25 throughout so augmentation is the only moving part. 24.8 min.

> **The script's printed verdict ("NOISE AUGMENTATION SUFFICIENT") is SUPERSEDED by the confound
> check in the same run.** Quote the split conclusion below, not the log line.

- **P1 FAILED by -0.4924, in the opposite direction.** Retained gate recall at budget 0.20 under
  white-box evasion: **A baseline 0.0313 | B noise-aug 0.8160 | C adversarial 0.3236**. **Plain
  random-noise augmentation is 26x the baseline and 2.5x better than adversarial training**, at a
  clean cost indistinguishable from zero (paired bootstrap **-0.0051 [-0.0139, +0.0031], p=0.234**;
  C is +0.0013 [+0.0000, +0.0028], a trivial *improvement*). This is the pre-registered
  positive-practical branch: **the adversarial machinery is unnecessary and worse than the cheap
  alternative** -- the question the source paper cannot ask, having no noise-control arm.
- **P3 FAILED by +0.2171, also backwards.** Misattribution-adversarial training of Stage B made
  misattribution **worse** (0.8566 vs C's 0.6395 at budget 0.20) -- training on boundary-perturbed
  rows blurs class boundaries rather than sharpening them. Both arms sit at gate recall 1.0000, so
  v7.3g's "misattributed is capped by gate recall" caveat does not rescue D. **The last surviving
  fragment of v7.3g's stage-asymmetry idea is now closed too.**
- **THE CONFOUND CHECK FIRES AND SPLITS THE FINDING.** Augmentation added only perturbed ATTACK rows
  labelled attack=1, so a gate can satisfy it by learning "perturbed => attack". Thresholds were
  matched on CLEAN benign, hiding that. Perturbing BENIGN rows at fixed threshold: **fpr 0.0168 ->
  0.1293 for arm B (10.7x) while the baseline stays flat at 0.0121**; arm C inflates 7.9x. **This is
  the project's own matched-fpr rule (v7.3b) applied to drift rather than across model families, and
  it invalidates the naive reading of P1** -- the augmented gates have moved their operating point,
  not simply become robust.
- **CORRECTED CONCLUSION -- the split is the finding.** Against an **adversary** (perturbs only their
  own flows) noise augmentation is a real, cheap, effective defence, because benign traffic is
  unperturbed and the clean-matched threshold stays correct. Against **measurement drift** (perturbs
  everything) it is **not** a defence: 13% fpr at the real 97.5% benign base rate is an unusable
  alert volume. **The robustness chapter is motivated primarily by drift, so this defence works for
  the scenario that motivated it least -- state that irony rather than hiding it.**
- **Reusable method**: whenever augmentation uses one class only, verify the model has not learned
  "perturbed => that class" by perturbing the OTHER class and re-measuring the operating point.
- **Untested, required before any drift claim**: re-match fpr *under the drift condition* and
  re-measure. Only that separates a genuine robustness gain from an operating-point shift.
- **Scope**: attacks are white-box on one component and Stage B is a single model; v7.3e's +0.6773
  surrogate inflation makes attack strengths understated -- conservative for an attack claim but
  **GENEROUS for a defence claim**, which is what arm B is. One round of adversarial training only.
  Arm A clean macro-F1 0.6718, not level-comparable to the 0.687 headline.

### v7.3i — does brittleness track CLASS OVERLAP? (`results/v73i_cicids_mechanism/`) — H1 SUPPORTED

Cover-Hart says brittleness comes from rows sitting near decision boundaries. `e_1NN` measures that
overlap **without reference to any attack**, making the claim falsifiable across datasets:
CIC-UNSW e_1NN 0.2084 (brittle, measured) vs CIC-IDS2017 e_1NN **0.0026** -- 81x lower, reproduced
in-run. Same budget machinery, same matched benign fpr 0.0168. 6.5 min.

Retained fraction of each dataset's own clean value:

| mode | budget | CIC-IDS2017 ret mF1 | ret gate | CIC-UNSW ret mF1 | ret gate |
|---|---|---|---|---|---|
| RANDOM | 0.002 | **0.9960** | 0.9996 | 0.8631 | 0.9558 |
| RANDOM | 0.200 | **0.8381** | 0.9169 | 0.4053 | 0.4939 |
| EVASION | 0.020 | **0.8214** | 0.7784 | 0.5335 | 0.4761 |
| EVASION | 0.200 | **0.5232** | **0.5917** | 0.2143 | **0.0313** |

- **H1a PASS** (retained gate recall under evasion @0.20: **0.5917 vs 0.0313**, 19x, threshold 0.50);
  **H1b PASS** (retained macro-F1 under noise @0.20: **0.8381 vs 0.4053**, threshold 0.80).
  **Monotone in all six cells.** Triviality guard cleared -- `moved` = 1.0000, so the search
  genuinely ran and this is not a stalled attack misread as robustness.
- **INTERPRETATION: brittleness is a property of the REPRESENTATION's class overlap, not the
  architecture.** Robustness comes from better features, not better models -- the same conclusion
  every other axis of this project reached independently.
- **⭐ THIS MERGES TWO CHAPTERS INTO ONE ARGUMENT.** The plateau diagnosis said overlap caps accuracy
  (rho 0.827, Cover-Hart bracket [0.213, 0.376]); the robustness chapter said the model is brittle.
  **They are the same fact**: overlap sets both the accuracy ceiling and the robustness floor,
  through decision margin. It also removes the single-dataset objection from the robustness claim.
- **⚠️ "MORE robust", never "robust"**: at budget 0.20 CIC-IDS2017 still loses 41% of gate recall and
  48% of macro-F1. Low overlap buys margin, not immunity.
- **⚠️ THE PROXIMATE CAUSE IS DECISION MARGIN, and a critic will notice.** Matched fpr put this gate
  at threshold **0.0002** vs CIC-UNSW's **0.3280**, so a row must be pushed far further to evade.
  The objection "you just used a lower threshold" is answered by noting the threshold is *derived*
  by matching fpr, and that 1.68% fpr being achievable at 0.0002 is itself a measurement of
  separability. The chain is **low overlap -> confident predictions -> large margin -> robustness**;
  margin is the mechanism by which overlap acts, not a competing explanation. State the chain.
- **⚠️ CONFOUND STANDS, as pre-registered**: CIC-IDS2017 differs in more than overlap (testbed, class
  mix, no protocol column, no reject class). **Suggestive of the mechanism, NOT proof.** The
  controlled version would vary overlap *within* one dataset and test whether robustness tracks it.
- **⚠️ Clean levels NOT comparable to `results/cicids2017_v6/`**: this run's clean macro-F1 is 0.7762
  vs v6's 0.9835, because forcing 1.68% fpr on a near-separable dataset admits ~7,000 benign FPs that
  are routed into families, destroying precision on classes with a handful of test rows (Heartbleed
  2, Infiltration 7). The handicap applies to clean and perturbed arms alike, so the retained
  fractions -- the only quantity claimed -- are unaffected. Never quote 0.7762 as a performance number.
