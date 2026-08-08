# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An undergraduate thesis project (AIUB) on **Network Intrusion Detection (NIDS)** using machine learning. The work is at an early stage: the dataset and literature review are in place, but the modeling pipeline has not been built yet (`test.ipynb` is a placeholder). The intended direction, drawn from the literature review, is a **feature-selection-driven ensemble classifier** with explicit **class-imbalance handling** evaluated on a modern (2024) dataset.

## Repository layout

- `Resource/` — the working dataset (see below).
- `Papers/` — 21 reviewed papers (`Papers/Abid/`, `Papers/Ovi/`) plus the extraction workbook.
  - `Papers/Data_Extraction_Template_updated).xlsx` — the literature synthesis. **Sheet1** = full extraction grid (17 cols A–Q: aim, methodology, findings, limitations, etc.) for 21 papers. **Sheet2** = condensed Title / Dataset / Algorithm / Findings table. Sheet3 is empty.
- `test.ipynb` — placeholder notebook ("Hello World"), not a real pipeline.

## Dataset (`Resource/`) — read before any modeling

**CIC-UNSW-NB15-2024**: the UNSW-NB15 attack taxonomy re-extracted with CICFlowMeter-style flow features. Two parallel files, row-aligned by position (no shared key — never sort one without the other):
- `CIC-UNSW-Data.csv` — 447,915 rows × **76 numeric features** (all `int64`/`float64`).
- `CIC-UNSW-Label.csv` — single `Label` column, values **0–9** (Benign + 9 UNSW attack families). The 0–9 → name mapping is **not** in the files; it was confirmed by the project owner (attack families are alphabetical, Benign = 0):

  | 0 Benign | 1 Analysis | 2 Backdoor | 3 DoS | 4 Exploits | 5 Fuzzers | 6 Generic | 7 Reconnaissance | 8 Shellcode | 9 Worms |
  |---|---|---|---|---|---|---|---|---|---|

### Verified data facts (these drive every design decision)
- **Clean**: 0 NaNs, 0 infs, all columns numeric. No imputation needed.
- **32% duplicate rows (142,313).** This is the single biggest correctness trap. Dedupe and **split before any resampling/scaling**, or duplicates leak across train/test and produce fake ~100% scores — the same leakage the extraction flagged in Papers 4, 6, and 10.
- **Severe class imbalance**: class 0 (Benign) = 80%; rare classes are tiny (class 9 Worms = 246 rows / 0.055%, class 1 Analysis = 385 / 0.086%, class 2 Backdoor = 452 / 0.101%). Plain accuracy is meaningless. Use a **stratified** split and report **macro-F1 + per-class precision/recall + confusion matrix**.
- **9 zero-variance columns to drop**: `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `URG Flag Count`, `CWR Flag Count`, `ECE Flag Count`, `Fwd Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg`.

### Mandatory pipeline ordering (to stay leakage-safe)
unsupervised prune (drop zero-variance + |r|≥0.95 redundant) → **dedupe** → **stratified** train/test split → **supervised feature selection (MI + RFECV) fit on train only** → fit scaler (RobustScaler; features are heavy-tailed) on train only → resample (SMOTE variants / class weights) on **train only** → train → evaluate. Never fit a scaler, run SMOTE, **or run supervised feature selection** before the split. (The variance/correlation prune is label-free, so it may precede the split; MI/RFECV use `y` and must not.)

## Project Workflow (notebook structure)

Notebooks follow **The 7 Stages of Machine Learning**. Every model notebook (e.g. `decision_tree.ipynb`) uses these as top-level markdown sections, numbered to match, with numbered sub-steps inside each stage (`### 2.1`, `### 2.2`, …):

1. **Data Collection** _(Sources & Quality)_ — load `CIC-UNSW-Data.csv` + `CIC-UNSW-Label.csv`; confirm row alignment.
2. **Data Preparation** _(Clean & Transform)_ — the bulk of the work, broken into evidence-first sub-steps: 2.1 check duplicates → 2.2 drop duplicates → 2.3 class distribution → 2.4 variance check → 2.5 correlation → 2.6 drop uninformative/redundant features (76 → 44, unsupervised) → 2.7 stratified train/test split → (scale on train only). Steps 2.4–2.6 are label-free and precede the split.
   - **Feature-selection variant** (suffix `_mi_rfecv`, e.g. `decision_tree_mi_rfecv.ipynb`): the baseline notebook (`decision_tree.ipynb`) stops at 44 features; the variant adds **2.8 supervised feature selection (MI ranking + RFECV, train-only → 44 → 36)** → **2.9 apply selected set** between the split and Stage 3. 2.8 uses `y`, so it runs **after** the split on train only. Keep both so the variant is an honest ablation against the 44-feature baseline.
3. **Model Selection** _(Algorithm Choice)_ — instantiate the classifier (current track: classic ML baselines, starting with Decision Tree).
4. **Model Training** _(Learning Process)_ — fit on train; use `class_weight='balanced'` as the imbalance baseline.
5. **Model Evaluation** _(Test Performance)_ — report Accuracy, **ROC-AUC** (multiclass One-vs-Rest, macro), and both **weighted ("normal") and macro Precision / Recall / F1-score**, plus the per-class `classification_report` and confusion matrix (never bare accuracy). Macro-F1 stays the primary metric (rare classes count equally); weighted shows Benign-dominated overall performance. ROC-AUC uses `predict_proba`.
6. **Model Improvement** _(Fine-tune & Optimize)_ — hyperparameter tuning, resampling (SMOTE variants), feature selection.
7. **Prediction** _(Deployment & Inference)_ — inference on new flows.

**Evidence-first rule:** never hard-code a data decision. Show it first (variance table, correlation heatmap), then act on what the evidence reveals. Dead/redundant columns are *surfaced* in 2.4/2.5 and only dropped in 2.6.

## Baseline results (numbers to beat)

Multiclass (10-class), leakage-safe pipeline: dedupe → evidence-based feature selection (76 → **44 features**) → stratified 80/20 split → `class_weight='balanced'`. Test set = 61,235 rows (Benign ≈ 73.5% after dedup). **Macro-F1 is the primary metric**; accuracy is misleading here.

| Model | Accuracy | Macro-F1 | Macro precision | Macro recall |
|---|---|---|---|---|
| Decision Tree | 0.8855 | 0.5338 | 0.5408 | 0.5284 |
| Random Forest (200 trees) | 0.9017 | 0.5744 | **0.6521** | 0.5358 |
| XGBoost (400 rounds, balanced sample weights) | 0.8949 | **0.5807** | 0.5422 | **0.6510** |

**Key findings:**
- **Imbalance, not model capacity, is the bottleneck.** DT→RF→XGB lifts macro-F1 only ~5 points (0.534→0.581); ~0.58 looks like the single-model ceiling with balanced weighting alone. Motivates resampling (SMOTE) as the next lever.
- **RF and XGBoost have complementary, inverted error profiles** (same "balanced" lever, opposite behaviour): RF = high precision (0.65) / low recall (0.54), conservative and misses attacks; XGBoost = low precision (0.54) / high recall (0.65), catches rare attacks (Shellcode recall 0.14→0.50, Analysis 0.27→0.64, DoS 0.27→0.44) at the cost of more false alarms. ~~This complementarity is a data-driven justification for the stacking ensemble.~~ **⚠️ SUPERSEDED: this inversion is an artifact of `balanced` weighting.** Under α=0.25 every model has precision > recall (XGB 0.664/0.567, RF 0.674/0.527, ET 0.641/0.523) with XGBoost uniformly ahead, and on disagreements XGB is right 44.8% vs RF 29.6%. **Do not use complementarity to justify an ensemble** — see the DES negative result below. The *hierarchical* direction rests on separate evidence and is unaffected.
- **Full SMOTE balancing UNDERPERFORMED class weighting** (XGBoost, Stage 6): macro-F1 0.5807 (weights) → **0.5088** (SMOTE), with both precision and recall dropping. Cause: balancing every class to the majority inflates the rarest classes ~950× (Worms ~190 real → ~180K synthetic), and interpolating from so few points injects noise — Worms precision collapsed to 0.06. Confirms the SMOTE-noise limitation from review Papers 2 & 5. **Takeaway: prefer class weighting; if resampling, use capped/moderate oversampling, not full balance.**

### Two-stage hierarchical detection (the thesis contribution)

> ⚠️ **Significance correction (see `ablation_significance.ipynb`).** The two-stage hierarchy is **NOT established as a standalone improvement**. Under `balanced` weighting it measures **−0.0061 (p=0.174)** vs flat — i.e. the "flat 0.5807 → two-stage 0.5858" step below is **inside the noise band** and its sign is not reliable. The hierarchy is significant **only in combination with α=0.25 weighting** (+0.0178, p=0.005). **Frame the contribution as "hierarchy + tempered weighting" jointly — an interaction, not a main effect.** Never claim two-stage helps on its own.

Same leakage-safe pipeline, but the problem is **restructured** into a gate + family classifier (both XGBoost, 400 rounds). Stage A = binary Benign-vs-Attack; Stage B = 9-class family classifier trained on **attack rows only** (families remapped 1–9→0–8). Notebooks: `two_stage_hierarchical.ipynb` (main) and `two_stage_stageB_experiments.ipynb` (Stage-B resampling ablation).

| Configuration | Accuracy | Macro-F1 | Macro precision | Macro recall | Selected on |
|---|---|---|---|---|---|
| Flat XGBoost (baseline) | 0.8949 | 0.5807 | 0.5422 | 0.6510 | — |
| Two-stage + balanced weights | 0.8944 | 0.5858 | 0.5427 | **0.6779** | — |
| Two-stage + capped SMOTE (cap≈2000, Stage B) | — | 0.5976 | 0.6624 | 0.5790 | test (!) |
| Two-stage + α=0.25 tempered weights (Stage B) | 0.9081 | 0.6157 | 0.6574 | 0.6035 | validation |
| Two-stage + α-diverse ensemble (Stage B) | 0.9030 | 0.6302 | 0.6733 | 0.6330 | SEL split |
| **Two-stage + cross-fitted α-diverse ensemble (v3, FINAL)** | 0.9073 | **0.6339** | **0.6701** | **0.6475** | **SELV split** |

**Key findings:**
- **The gate is essentially solved; Stage B is the bottleneck.** Gate attack recall ≈ 0.997 / precision ≈ 0.918 — almost no attacks are missed at Stage 1. The ceiling is the family sub-problem (Stage-B macro-F1 ≈ 0.574 on *true* attack rows), so improvement effort belongs there, not on the gate.
- **Capped SMOTE on Stage B is the best single lever: macro-F1 0.5807 → 0.5858 → 0.5976** (+1.7 pts vs flat/weighted). **Full SMOTE still hurts** even inside the two-stage (e2e macro-F1 ≈ 0.556), reconfirming the capped-not-full rule above.
- **Weights vs capped SMOTE is a precision↔recall knob, not a free win:** balanced weights maximise macro-recall (~0.68); capped SMOTE maximises macro-precision (~0.66). The operating point is a SOC choice.
- **Gotcha:** with the sklearn `XGBClassifier` wrapper, let it infer the objective — passing `num_class` manually makes the binary gate's `.predict` return a 2-D array.
- **Honesty note:** the Stage-B cap and the Stage-6 gate threshold in the rows above are read off the test set; that is fixed in the weighting study below, which is the number to quote. Still missing from the novelty stack: **SHAP per-class explainability**.

### Stage-B weighting study — current best (`two_stage_stageB_weighting.ipynb`)

First result selected on a **validation split carved from train** (195,950 / 48,988 / 61,235), both stages refit on train+val, test touched once. **Test macro-F1 = 0.6157.** Compares uniform, tempered inverse-frequency `(1/n_c)^α`, effective-number `(1-β)/(1-β^n_c)`, focal `w_c(1-p_true)^γ` (out-of-fold `p`), and capped SMOTE.

- **`class_weight='balanced'` is a bad default here — it ranked 12th of 14, below no weighting at all** (0.5826 vs uniform 0.6014 on val). Use **α≈0.25** instead: +0.0297 macro-F1 over `balanced`.
- **Interior optimum, confirmed by two independent families.** α peaks at 0.25, effective-number peaks at β=0.999; expressed as a Worms:Exploits **weight ratio** both agree the useful band is **3–11×**, where `balanced` demands 128×. This generalises the earlier capped-SMOTE finding into one principle: **moderate, not full, rebalancing**, for resampling *and* weighting.
- **Focal (dynamic) weighting failed:** γ=0.5 a wash, γ=1 worse, γ=2 collapses the model (0.3497). Focal assumes hard samples are informative, but here they are largely **label-ambiguous** — after dedup on features+label, any repeated feature vector carries *conflicting* labels. Keep as a reportable negative result.
- **Weighting beats resampling** (0.6123 vs capped SMOTE 0.5988) and is cheaper. Capped SMOTE is superseded.
- **Fuzzers is the false-positive sink** (P 0.589 / R 0.902); 34% of its 3,318 test FPs are Benign flows leaking through the gate. Confusion is block-structured: {Exploits, DoS, Analysis} and {Fuzzers, Reconnaissance, Shellcode}.
- **Next levers:** per-class decision thresholds (largest remaining gain) → gate threshold → soft routing `P(k)=P(attack)·P(k|attack)` → label-ambiguity ceiling analysis → SHAP.

### Dynamically weighted Stage-B ensemble — NEGATIVE RESULT (`two_stage_stageB_des.ipynb`)

> ⚠️ **Scope correction:** this notebook's "the ensemble direction is closed for Stage B" was too broad. It holds for a pool diversified by *algorithm*; it does **not** hold for a pool diversified by *class weighting* — see the α-diverse ensemble below (0.6302). The specific v1 findings still stand.

Per-instance competence weighting (XGB / RF / ExtraTrees, all α=0.25; competence from local kNN accuracy over a DSEL split; OLA / class-weighted OLA / LCA / distance-weighted / sharpened variants). `val` halved into DSEL + SEL, test touched once.

- **Do not adopt.** Best dynamic variant led on SEL (+0.0045 over single XGB) but **did not transfer to test** (−0.0007, rank 2 of 15). Keep the single α=0.25 XGBoost.
- **The RF/XGB complementarity was an artifact of `balanced` weighting** (see the correction above). No diversity left to exploit.
- **Oracle headroom is large but unreachable:** oracle 0.6799 vs best single 0.5995 (+0.0804), yet the dynamic weights sit at **0.333 ± 0.02** — a static ensemble in disguise. Where the estimator discriminates it picks **ET (the weakest model) on 44% of rows**.
- **Static equal-weight averaging is actively harmful** (0.6019 vs 0.6125) — averaging in weaker, non-complementary models dilutes the strong one.
- **Pattern:** focal failed (label-ambiguous hard samples) — that one still stands as evidence label noise binds. DES failed because *algorithmically* diverse models aren't complementary; that turned out to be about pool construction, not ensembling.

### α-diverse Stage-B ensemble — CURRENT BEST 0.6302 (`two_stage_stageB_ensemble_v2.ipynb`)

Pool diversified along the **α axis**, not by algorithm: XGB at α ∈ {0, 0.25, 0.5, 1.0} + capped-SMOTE XGB + RF/ET at α=0.25. Combination **learned**, not estimated: per-class weight matrix `W (7×9)` by coordinate ascent on macro-F1. Bases on `train`, combiner on **DSEL**, selection on **SEL**, test once. SEL winner ranks 1st of 12 on test; SEL↔test **Spearman 0.937**.

- **Diversity must come from the weighting axis.** Oracle headroom nearly doubled vs v1 (0.0804 → **0.1456**). **XGB α=1.0 disagrees with the rest on 15–23% of rows** while α=0/0.25/SMOTE cluster at 5–9% — and it is individually one of the *worst* members (0.5595). **Pick pool members for complementarity, not individual quality.**
- **Per-class weights specialise:** recall-heavy α=1.0 → DoS/Fuzzers/Worms; precision-heavy α=0.0 → Analysis/Backdoor/Recon/Shellcode/Generic; **Exploits keeps uniform 1/7** (ascent couldn't beat a plain average there — evidence it isn't just overfitting). Rare families go to the *precision*-heavy member because their precision, not recall, is the binding constraint.
- **Stacking overfits, constrained weighting doesn't.** Meta-XGB 0.9178 in-sample on DSEL vs 0.5930 test; both meta-learners fell below the single model. Same 63 parameters as the per-class matrix — **the difference is inductive bias, not capacity**. Prefer constrained combiners when the combiner-fitting set is small.
- **The gain is a rare-vs-large trade:** Worms 0.415→**0.557**, Backdoor 0.599→0.652, DoS 0.411→0.433; paid by Exploits 0.782→0.750, Generic 0.813→0.790. Report as a SOC operating choice, not hidden in the macro average.
- Fair comparison: ensemble uses `train`+DSEL, *less* data than the 0.6157 single (train+val).
- **Next:** refit pool on train+DSEL → widen α grid to 1.5/2.0 → ambiguity ceiling → SHAP. (Per-class thresholds, soft routing and gate-threshold tuning are all **tested and closed** — see below.)

### Per-class thresholds + soft routing — NEGATIVE RESULT (`two_stage_thresholds.ipynb`)

**Keep the v2 ensemble's plain argmax (0.6302).** Thresholds fitted on SEL, test once. Rank order **inverted exactly**: SEL `V4 > V2 > V3 > V1 > V0` (10 params → 0), test `V0 > V1 > V3 > V2 > V4` (0 params → 10). Monotone in parameter count both ways — pure overfitting.

- **Per-class thresholds are algebraically redundant with `W`**: `t_c·S(c|x) = Σ_m (t_c·W[m,c])·P_m(c|x)` is column rescaling. Row-normalising doesn't help (`Z(x)` is constant across classes). **Verified: folding `t` into `W` reproduced predictions on 100.0000% of rows.** Don't re-attempt this.
- **Soft routing is a no-op** (+0.0000 on test). **The gate is already optimally calibrated** — the threshold search left Benign at `t_0=1.0`, i.e. `p_att > 0.5` exactly. This is why the Fuzzers Benign-leakage finding never converted into a gain: those flows are not threshold-separable.
- **Rare classes got worse** (Worms 0.557→0.507, Backdoor 0.652→0.617) — SEL has ~19 Worms rows, each worth 1/10 of macro-F1, so the search chased noise.
- **Methodological lesson: asymmetric P/R is NOT evidence of a bad threshold.** Per-class F1 is not maximised at P ≈ R; when the error types have different marginal rates the optimum is genuinely asymmetric. Do not read a lopsided P/R pair as a fixable defect.
- **SEL is now spent** (used for both architecture selection and `t`) — further tuning needs a fresh split from `train`.

### Ensemble v3 — FINAL MODEL, 0.6339 (`two_stage_stageB_ensemble_v3.ipynb`)

Same architecture as v2 (7 members, per-class `W`), but **5-fold cross-fitted over `train+val`**: base models get all 244,938 rows, `W` is fitted on ~32k OOF attack rows instead of 6,494, and a fresh SELV split replaces the spent SEL. Ranked 1st of 13 on test; SELV→test order held.

- **Extending α past 1.0 failed.** α=1.5/2.0 are individually degraded (0.4929 / 0.4090) and the 10-member pool scored *below* the 7-member subset (0.6329 vs 0.6339). **The α axis saturates at 1.0 — do not extend it further.**
- **KEY METHODOLOGICAL FINDING: oracle headroom does not predict realisable gain.** v2→v3 raised disagreement 0.135→0.235 and oracle headroom 0.1456→0.2250 (+55%) while the achieved score moved +0.004. This also retro-explains DES v1 (0.0804 headroom, 0 realised). **Do not justify a pool by its oracle bound.**
- **The +0.0037 came entirely from more data**, not the wider α grid — isolated by the `v2 7-member subset` control (identical members, only cross-fitting differs).
- **Noise-band caveat:** Worms has 48 test rows, Analysis 77. Worms alone moved −0.071 between v2 and v3. **Differences below ~±0.005 in macro-F1 are not meaningful** on this test set — v3-over-v2 is suggestive, not decisive.
- Cost: ~16 min training (746s cross-fit + 180s refit) vs v2's ~1 min. Inference cost identical.
- **Improvement has plateaued:** 0.6157 → 0.6302 → 0.6339. Four levers (focal, DES, thresholds, wider α) returned ~nothing; every gain came from better weighting or more data. See the ambiguity analysis below for why. **SHAP** still outstanding.

### Dynamic Stage-B ensemble v4 (`two_stage_stageB_dynamic_v4.ipynb`) — the dynamic-ensemble contribution

Per-instance competence weighting over the **α-diverse** pool with **cross-fitted** competence — i.e.
DES v1 re-run after v2/v3 showed both of v1's premises were wrong. Same prune/splits/seeds/pool as v3;
combiner is the only moving part. **Two models are reported**, each selected on SELV within its own
combiner family:

| Model | Weights vary by | Test macro-F1 | Test acc | vs v3 |
|---|---|---|---|---|
| **v4-hybrid** `HYBRID W × LCA k=30 p=1` | class **and** instance | **0.6295** | 0.9082 | −0.0039 |
| **v4-dynamic** `DYN LCA k=30` | instance only | **0.6262** | 0.9085 | −0.0072 |
| v3 static (baseline, refit here) | class only | 0.6334 | 0.9086 | — |

Quote as `0.630 [0.612, 0.647]` and `0.626 [0.608, 0.644]`. **Both gaps are inside the ±0.012 paired
band — report as "no measurable difference", never as a loss.**

- **Competence still cannot be estimated, and the setup is no longer to blame.** v4 gave the estimator
  the α-diverse pool *and* 32,468 reference rows (5× v1's 6,494); weights still sit within **4.5% of
  uniform** (mean |dev| 0.0064 vs uniform 0.1429) — v1 was 5.1%. Only XGB α=1.0 disperses at all
  (std 0.0147). **The binding constraint is the representation**, matching the Cover–Hart bracket.
- **The per-class prior is load-bearing; the per-instance term is not.** Hybrids land within 0.004 of
  v3; pure-dynamic loses 0.007–0.014. The hybrid was built so `p=0` recovers v3 exactly (verified on
  100.0000% of rows), so the null cannot be blamed on combiner form. The dynamic component is **inert
  but harmless** — state this before an examiner asks.
- **Third failure of oracle headroom to predict gain**: v1 0.0804→0, v3 0.2250→+0.0037, v4 0.1579→0.
- **Scope correction confirmed:** v1's "ensemble direction is closed" was about *pool construction* —
  v4 recovers **+0.0144 to +0.0177** over v1.
- **SELV/test rank changes inside the noise band are resolution failure, not selection error.** Within
  the pure-dynamic family SELV transferred perfectly; across families it did not (1st → 6th of 23), but
  all six leaders are within noise.
- Deployment: v4 needs a kNN query per flow + 32,468-row reference set in memory; v3 needs one 7×9
  matmul. **v3 remains the recommended deployed combiner.**
- OOF cached to `cache_oof_v4.npz` (cross-fit 308 s); combiner iteration ~2 min.

### Label-ambiguity ceiling — hypothesis REFUTED (`label_ambiguity_ceiling.ipynb`)

Tested whether the plateau is irreducible label noise. **It is not.** Only **497 ambiguous groups / 1,068 rows (0.35%)**; Bayes-optimal accuracy 0.9981, ceiling macro-F1 0.9881 → v3 attains **64.1% of ceiling**.

- **Do not claim "noisy labels" as the reason for weak rare-class performance** — the evidence is against it. No rare class is near its exact-collision ceiling (Shellcode 38.9%, Analysis 45.5%, DoS 46.8%, Worms 49.7%).
- **The bound is strict but weak:** 99.8% of rows are unique, so it's attained only by memorisation — a perfect memoriser scores macro-F1 **0.0848** on test. Treat this notebook as **eliminative, not aspirational**.
- **The 76→44 prune creates ZERO new ambiguity** (identical group structure at both feature counts) — independent validation of the feature selection, worth citing in the methodology chapter.
- **Ambiguity predicts model confusion: Spearman ρ = 0.699 (p=9e-08).** Fuzzers is in 5 of the 6 largest ambiguity pairs and is the model's biggest error sink.
- **REAL MECHANISM: class overlap in CICFlowMeter feature space, not label noise.** Exploits↔Recon has 451 model confusions and *zero* exact ambiguity; Analysis↔Exploits 72 and zero. These families aren't mislabelled — they're inseparable in this representation. One mechanism explains all four negative results.
- **Next:** k-NN label-overlap analysis — done, see below.

### k-NN class overlap — HYPOTHESIS CONFIRMED (`knn_class_overlap.ipynb`)

**This settles why the project plateaued at ~0.63.** Neighbours of test attack rows in train+val attack rows, RobustScaler space, leakage-safe; Stage-B 9-family scope.

- **Overlap explains model confusion at ρ = 0.827 (p=4.9e-10)** vs exact ambiguity's 0.538 — and explains the pairs exact matching couldn't: Exploits↔Recon (451 confusions, **zero** exact ambiguity), DoS↔Exploits (442, zero), Analysis↔Exploits (72, zero).
- **IRREDUCIBLE ERROR (Cover–Hart on 1-NN error 0.3755): e\* ∈ [0.213, 0.376]** → accuracy ceiling [0.625, 0.787]. v3's Stage-B accuracy is **0.737**, inside the bracket near its top. **~21–38% of attack flows are unclassifiable on these 44 features by any method.** Quote this, not the exact-match 0.0019 (vacuous).
- **The ensemble is NOT trivial:** best k-NN macro-F1 0.4458 vs v3 Stage-B 0.6211 (**+0.175**). The data is hard *and* the modelling recovers a lot.
- **Purity ρ = 0.700 with achieved F1.** Worms purity 0.059, Shellcode 0.123, Backdoor 0.136, DoS 0.174, Analysis 0.246 — vs Exploits 0.654, Fuzzers 0.639, Recon 0.542. Only those three have majority-own neighbourhoods.
- **Exploits and Fuzzers are attractors:** Analysis is 51.4% Exploits, Shellcode 46.5% Fuzzers, DoS 43.5% Exploits, Worms 43.4% Exploits. One geometric fact explains the Fuzzers FP sink, the collision pairs, and the block-structured confusion matrix.
- **Notebook mislabel:** `plurality_ceiling_R` is NOT a ceiling (Worms scores 0.000 there yet achieves F1 0.486) — plurality voting minimises error rate, not macro-F1. Read it as "recall of the accuracy-optimal local rule".
- **Framing for the thesis:** the plateau is a property of the CICFlowMeter representation, not a modelling deficiency. Future work = **richer features** (payload, host/temporal context, session aggregation), not better classifiers. **SHAP should target overlap pairs** (Exploits↔Recon, DoS↔Exploits, Fuzzers↔Shellcode), not rare classes generically.

### Ablation + bootstrap significance (`ablation_significance.ipynb`) — READ BEFORE QUOTING ANY NUMBER

2,000 stratified bootstrap resamples of the test set; all five ladder configs trained on the same 195,950 rows and scored on the same resample each iteration (**paired** differences).

| Contribution | Δ macro-F1 | 95% CI | p | established |
|---|---|---|---|---|
| **weighting (inside two-stage)** | **+0.0385** | [0.0246, 0.0512] | <0.001 | **YES** |
| hierarchy (at α=0.25) | +0.0178 | [0.0061, 0.0311] | 0.005 | YES |
| ensemble | +0.0177 | [0.0067, 0.0301] | 0.003 | YES |
| weighting (flat) | +0.0146 | [0.0001, 0.0276] | 0.049 | marginal |
| **hierarchy (at `balanced`)** | **−0.0061** | [−0.0144, 0.0030] | 0.174 | **no** |
| **FULL: final vs baseline** | **+0.0502** | [0.0391, 0.0606] | <0.001 | **YES** |

- **α=0.25 weighting is the single largest lever (+0.0385) — more than twice the entire 7-member ensemble (+0.0177).** Consistent with the whole project record: gains come from weighting or data, never model capacity.
- **TWO YARDSTICKS, don't mix them.** Reporting one config: mean 95% CI width 0.0350 → **±0.017**, quote as `0.630 [0.611, 0.649]`, never 4 decimals. Comparing two configs on the same test set: use the **paired** interval, ≈**±0.012**. Using the marginal width to judge differences would wrongly discard two real results.
- **v3 vs v2 (+0.0037) is 0.39 bootstrap SDs — WITHIN NOISE.** Keep v3 as final on **protocol** grounds (cross-fitting uses train+val properly, W fitted on ~32k rows not 6,494). Never justify it by score.
- **Per-class CIs — report rare classes as intervals:** Worms 0.557 **[0.422, 0.683]** (±0.13!), Backdoor 0.652 [0.556, 0.740], Analysis 0.427 [0.361, 0.493], Shellcode 0.360 [0.317, 0.405], DoS 0.433 [0.401, 0.462]. **Past per-class claims on Worms (incl. the "+0.142 from the v2 ensemble") are not established.**
- Intervals capture test-set sampling only — split and seed variance excluded, so they are optimistic.

## Environment & commands

- Windows; Python 3.12.8 via miniconda (`C:\Users\night\miniconda3`). Primary shell is PowerShell; a Bash tool is also available. **The ML stack (`sklearn`, `xgboost`, `imblearn`) lives in the `ai4beg` conda env, not base** — for scripts run outside the notebook kernel use `C:\Users\night\miniconda3\envs\ai4beg\python.exe` (base `python` raises `ModuleNotFoundError: sklearn`).
- Terminal stdout is **cp1252** — printing Unicode (e.g. `±`, `–`, accented author names) raises `UnicodeEncodeError`. When dumping text, write to a UTF-8 file (`io.open(path, "w", encoding="utf-8")`) and read it back rather than printing to the console.
- Reading the Excel workbook requires `openpyxl` (installed). Use `data_only=True` to get computed values. Close the file in Excel first — a `~$...` lock file in `Papers/` means it is open and writes will fail/conflict.

## Conventions for this project

- Frame results around the **literature gaps** the review identified: weak recall on rare attack classes, unaddressed imbalance, single-dataset evaluation, and missing explainability (XAI). The thesis contribution leans on the under-explored 2024 dataset plus proper imbalance handling.
- Be skeptical of near-100% results — for this data they almost always indicate duplicate leakage, not a good model.
- The extraction workbook has a known error to be careful with: in **Sheet2, the SL 20 row** carries the title of the Opt-Forest paper but the Dataset/Algorithm/Findings of HAEnID (paper 21); the real Opt-Forest data (UNSW-NB15, GA-optimized forest, 99.45% acc) is missing there.
