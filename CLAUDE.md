# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An undergraduate thesis project (AIUB) on **Network Intrusion Detection (NIDS)** using machine learning. The work is at an early stage: the dataset and literature review are in place, but the modeling pipeline has not been built yet (`test.ipynb` is a placeholder). The intended direction, drawn from the literature review, is a **feature-selection-driven ensemble classifier** with explicit **class-imbalance handling** evaluated on a modern (2024) dataset.

## Repository layout

- `Resource/` — the datasets. Three are now present:
  - `CIC-UNSW-Data.csv` + `CIC-UNSW-Label.csv` — **the primary dataset** (see below).
  - `Resource/UNSW-NB15/` — the original Moustafa & Slay 2015 files; same traffic, different
    extractor. Used by `final_model_v6_unsw.ipynb`. See its `PUT_DATASET_HERE.md`.
  - `Resource/CIC-IDS2017/` — the 8 `MachineLearningCSV` day-files, 2.83M flows, genuinely
    different traffic. **Not clean** (inf/NaN present) and **not a plateau test** — see the
    results section below and `PUT_DATASET_HERE.md` for its many gotchas.
- `results/` — output artifacts for experiments run as scripts rather than notebooks. Each
  subdirectory holds its own runner `.py` plus the `.txt` log and `.json` results, so it is
  reproducible from the repo alone.
- `Papers/` — the reviewed papers plus the extraction workbook. **11 PDFs = 10 unique papers**
  (`Network-based intrusion detection.pdf` and `Network-based_intrusion_detection_using_deep_learn.pdf`
  are the same Farhan et al. paper, preprint vs published — do not double-count). The
  `Papers/Abid/` and `Papers/Ovi/` subdirectories referenced by earlier versions of this file no
  longer exist. **Not one of the 10 releases its code** — see `literature_artifact_audit.md`.
  - `Papers/Data_Extraction_Template_updated).xlsx` — the literature synthesis. **Sheet1** = full extraction grid (17 cols A–Q: aim, methodology, findings, limitations, etc.) for 21 papers. **Sheet2** = condensed Title / Dataset / Algorithm / Findings table. Sheet3 is empty.
- `test.ipynb` — placeholder notebook ("Hello World"), not a real pipeline.

## Dataset (`Resource/`) — read before any modeling

**CIC-UNSW-NB15-2024**: the UNSW-NB15 attack taxonomy re-extracted with CICFlowMeter-style flow features. Two parallel files, row-aligned by position (no shared key — never sort one without the other):
- `CIC-UNSW-Data.csv` — 447,915 rows × **76 numeric features** (all `int64`/`float64`).
- `CIC-UNSW-Label.csv` — single `Label` column, values **0–9** (Benign + 9 UNSW attack families). The 0–9 → name mapping (attack families alphabetical, Benign = 0) is **file-verified** against the string labels in `CICFlowMeter_out.csv` — see `results/cic_unsw_svc/`:

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

### Decision-space dynamic ensemble v5 (`two_stage_stageB_dynamic_v5.ipynb`) — RETRACTS a v4 claim

Same pool/splits/protocol as v4; only the competence **metric space** changes: kNN over the pool's
7×9=63 **output profiles** (the standard DES formulation) instead of RobustScaler feature space.

- **v5 = `DYN OLA-cw k=100 p=8`, test macro-F1 0.6363 / accuracy 0.9089**, ranked 1st of 20 on test.
  +0.0029 over v3 — inside ±0.012, so not established, but the first purely-dynamic combiner to top
  the table.
- Decision space beats feature space on every measure: purity k=10 **0.670 vs 0.572** (and flat in k,
  where FEAT degrades 0.572→0.515), mean competence AUC **0.854 vs 0.819**.
- **⚠️ RETRACTION.** v4 said *"competence cannot be estimated"*. **False** — per-member AUC is
  **0.82–0.85**. The real mechanism: decomposing `w_m(x)` gives **98.1% shared row-difficulty variance
  vs 1.9% member-specific**, pairwise correlation **0.982**, median within-row spread **0.0000**, and
  competence-argmax selection (0.7411) cannot beat always using the best member (0.7417).
  **Competence is estimable, but what is estimable is how hard the row is, not who should handle it.**
  This predicts the failure of any local-accuracy-derived weighting (incl. learned/META-DES) without
  running it.

### Stage-A gate weighting (`two_stage_gate_weighting.ipynb`) — NULL RESULT

`compute_sample_weight("balanced")` is exactly **α=1.0**, the point Stage B ranked 12th of 14; it was
never revisited for Stage A. Swept α × threshold with Stage B frozen.

- **+0.0006 macro-F1, +0.0011 accuracy.** Null.
- **Weighting and thresholding are one degree of freedom**: every α's best cell within **0.0010**, with
  the optimal threshold shifting to compensate (α=0→t=0.4 … α=1→t=0.7). Gate F1 flat (0.9562–0.9566).
  This completes `two_stage_thresholds.ipynb`, which had only tested thresholds under `balanced`.
- **Leaking Benign flows are ones the gate is confident about** — not threshold-separable. 1,051 of
  1,354 remaining leaks still go to Fuzzers: a representation problem.

### Feature engineering (`feature_engineering.ipynb`) — **FIRST POSITIVE RESULT**

31 engineered columns (scale-free ratios, per-packet normalisations, IAT/length coefficients of
variation, **flags as fractions**, header overhead, duration-normalised activity, window ratio).

| Space | features | e_1NN | e* bracket |
|---|---|---|---|
| BASE | 44 | 0.3757 | [0.2135, 0.3757] |
| FULL | 67 | 0.3697 | [0.2094, 0.3694] |
| **DERIVED** | **75** | **0.3629** | **[0.2053, 0.3632]** |

- **The Cover–Hart ceiling MOVED** — first time in the project. **McNemar BASE vs DERIVED p = 9.3e-06**
  (net +207 rows); BASE vs FULL p=0.039; FULL vs DERIVED p=0.038.
  > ⚠️ **Restate as metric-relative** (`feature_engineering_v2.ipynb`). A signed-log transform of the
  > same columns adds **zero** information and leaves the model's predictions 98.99% unchanged, yet
  > moves e_1NN **+0.0296 — 2.3× this headline**. e_1NN is a property of *(features, metric)* jointly
  > and does **not** bound a tree ensemble. It stays valid for the distance-based parts (k-NN overlap,
  > purity, DEC-space competence). Adoption still stands on the score and purity. **Drop the "two
  > independent instruments agree" argument** — in round 2 they disagree.
- Purity rose for every family, most for the attractor victims: Worms +0.058, Backdoor +0.024,
  Analysis +0.022, Shellcode +0.015.
- Single α=0.25: BASE 0.6157/0.9081 → DERIVED **0.6258/0.9117**.
- **Gate precision 0.9175 → 0.9267 from features alone** — what gate re-weighting could not buy.
- **Tier 1.1:** FULL beats BASE (p=0.039) so the prune does discard information, but DERIVED beats FULL
  (p=0.038) from the pruned base. **Keep the prune; spend on derived features.**
- Engineered = 41.3% of columns, **46.7% of importance**; `PSH Flag Count` #1, `frac_PSH` **#2**.

### Final model v6 (`final_model_v6.ipynb`) — **CURRENT BEST 0.6485 / 0.9123**

DERIVED features + decision-space dynamic combiner (`DYN OLA-cw k=30 p=8`). SELV-selected, 1st of 21
on test. See `v6_significance.ipynb` for the paired bootstrap.

- **Gains compounded slightly super-additively**: feature +0.0101, combiner +0.0029, additive predicts
  0.6464, observed **0.6485**.
- Optimal neighbourhood shrank k=100 → **k=30** (p unchanged at 8) — purer neighbourhoods need fewer
  neighbours.
- **INTERACTION, not a feature effect** (`v6_significance.ipynb`): neither ingredient is established
  alone — features without ensembling +0.0100 (p=0.071), dynamic-vs-static on BASE +0.0029 (p=0.555)
  — but together +0.0152 (p=0.012). **Ensembling on DERIVED is strongly established** (+0.0231,
  p<0.001) while on BASE it was not: the combiner only pays once the features are pure enough.
  Dynamic-over-static remains unestablished (+0.0026, inside noise) — the effect is *ensembling*,
  not *dynamism*.
- 8 of 10 families improved, and unlike earlier bumps the large-support families all moved the right
  way (Exploits +0.005, Fuzzers +0.009, Generic +0.011, DoS +0.028).
- **⚠️ Worms is 57% of the headline gain** (48 test rows, ±0.13). **Excluding Worms: +0.0052, inside
  noise.** Quote as `0.649 [0.631, 0.666]`.
- **Independently replicated** in `feature_engineering_v2.ipynb` (0.6485 / 0.9123, four decimals).

### Feature engineering round 2 (`feature_engineering_v2.ipynb`) — NULL on score; corrects round 1's instrument

`D2` = DERIVED + 38 columns targeted at the confusion pairs, plus **signed-log control arms** that add
zero information so metric effects separate from information effects.

| Space | feat | e_1NN | purity | single α=0.25 | at v6 combiner |
|---|---|---|---|---|---|
| DERIVED | 75 | 0.3632 | 0.3761 | **0.6258** | **0.6485 / 0.9123** |
| D2 | 113 | **0.3517** | **0.3880** | 0.6185 | 0.6464 / 0.9135 |
| DERIVED_log | 75 | 0.3928 | 0.3421 | 0.6202 | — |

- **`D2` REJECTED**: −0.0073 single, −0.0021 at the v6 config — inside ±0.012, so *no measurable
  difference*, for 38 extra columns. Keep DERIVED.
- **MAIN RESULT: the bracket is metric-dependent.** Zero-information signed-log moves e_1NN **+0.0296
  (p=4.8e-22)** vs round 1's −0.0128 and round 2's −0.0115 — **2.3× any real feature effect**. Tree
  invariance verified (98.99% identical predictions). **Quote the bracket as metric-relative; it does
  not bound a tree ensemble.**
- **Log-compression made the geometry WORSE** (purity 0.3761→0.3421, every family) — heavy-tailed raw
  scale beats a compressed one under RobustScaler. Counter-intuitive; worth stating.
- **43.9% of importance on 33.6% of columns, and it still adds nothing.** `handshake_bal` is the #1
  feature overall — but `(S−F)/(S+F) = (r−1)/(r+1)` is **strictly monotone in round 1's `r_syn_fin`
  = S/F**. 3 of the top 5 are bounded monotone rewrites of existing columns. **That is why the bracket
  moved and the score did not.**
- **The targeted pairs mostly did NOT move** — DoS→Exploits and Shellcode→Fuzzers got *worse*; only
  Worms→Exploits improved (0.390→0.344). The scan-has-no-response hypothesis failed on its own terms.
- **Pre-registered mechanism test SUPPORTED**: ensembling gain +0.0279 on the purer D2 vs +0.0227 on
  DERIVED, as the v6 interaction predicts (difference itself inside noise — corroboration, not proof).
- **Algebra on these 76 columns is exhausted.** Other feature directions are **impossible on this
  file**: no IP/port/timestamp column (verified), no PCAPs. **Restate the plateau as a limit of what
  CICFlowMeter records.**

### v6 on the original UNSW-NB15 (`final_model_v6_unsw.ipynb`) — representation claim CONFIRMED from outside

v6 pipeline re-run on `Resource/UNSW-NB15/` (162,745 rows after 36.8% dedupe). Same 10 families,
**zero shared columns**. **Not** a transfer test (no shared space) and **not** a fresh-traffic test
(same traffic, different extractor). **macro-F1 0.6387 / acc 0.8303** — *do not compare levels* to
v6's 0.6485/0.9123 (Benign 52.7% vs 73.5%, different row counts and flow segmentation).

- **e_1NN 0.3043 vs CIC DERIVED 0.3629 — a −0.0586 move, 4.6× the entire round-1 feature gain.**
  Bracket [0.168, 0.304], accuracy ceiling [0.696, 0.832]. **Conservative**: UNSW has *fewer* attack
  rows (61,618 vs 64,936), which can only raise 1-NN error.
- **⚠️ THE SURPRISE: host/temporal context is NOT the missing ingredient.** The 14 `ct_*` columns
  carry **6.8% of importance**. Top five: `svc_dns` 0.295, `ttl_diff` 0.226, `svc_http`, `sttl`,
  `r_ttl` — **all from service identity and TTL, which CICFlowMeter does not emit.** Supersedes
  `knn_class_overlap.ipynb`'s "host/temporal context" recommendation. Explains why two rounds of
  algebra failed: no ratio of counts/durations reconstructs "this flow is DNS".
- **Performance mostly tracks purity (6 of 9 families, both directions)**: Shellcode 0.388→**0.650**
  (neighbourhood 15.5% Fuzzers vs 46.5% on CIC), Recon 0.640→**0.807**, Generic 0.828→0.924, Exploits
  0.791→0.849. Exceptions: Fuzzers, DoS, Worms.
- **⚠️ NOT a rising tide.** Of CIC's four weakest (Shellcode .388, Analysis .446, DoS .468, Worms
  .582) only **Shellcode clearly improves**; Analysis and DoS get *worse*. The representation trades
  one set of confusions for another.
- **Backdoor COLLAPSES 0.635→0.183** (P 0.974 / R 0.101); Analysis mirrors it (0.163/0.779) — a known
  UNSW-NB15 Analysis/Backdoor/DoS labelling pathology. **Worms +0.124 is on 34 rows, contradicts the
  purity story, do not quote.**
- **The gate INVERTS**: CIC 0.9961 recall vs UNSW **0.9176** — misses 8% of attacks. **CICFlowMeter is
  better at attack-vs-benign; UNSW is better at family-vs-family.** Best system = union of both field
  sets.
- **v4 replicated on an independent representation**: static per-class `W` beats every pure-dynamic
  combiner by −0.031; ensembling gain +0.045 (vs +0.023 on CIC). SELV picked *static* here, *dynamic*
  on CIC — the protocol adapts.
- **Does NOT close the single-dataset gap** — same traffic. That still needs CIC-IDS2017.

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
- **IRREDUCIBLE ERROR (Cover–Hart on 1-NN error 0.3755): e\* ∈ [0.213, 0.376]** → accuracy ceiling [0.625, 0.787]. v3's Stage-B accuracy is **0.737**, inside the bracket near its top. **~21–38% of attack flows are unclassifiable on these 44 features by any method.** Quote this, not the exact-match 0.0019 (vacuous). ⚠️ **Add "under the RobustScaler Euclidean metric"** — `feature_engineering_v2.ipynb` shows a zero-information rescaling moves this bracket by more than any feature change did, so "by any method" overstates it. The bracket bounds *distance-based* methods in that geometry.
- **The ensemble is NOT trivial:** best k-NN macro-F1 0.4458 vs v3 Stage-B 0.6211 (**+0.175**). The data is hard *and* the modelling recovers a lot.
- **Purity ρ = 0.700 with achieved F1.** Worms purity 0.059, Shellcode 0.123, Backdoor 0.136, DoS 0.174, Analysis 0.246 — vs Exploits 0.654, Fuzzers 0.639, Recon 0.542. Only those three have majority-own neighbourhoods.
- **Exploits and Fuzzers are attractors:** Analysis is 51.4% Exploits, Shellcode 46.5% Fuzzers, DoS 43.5% Exploits, Worms 43.4% Exploits. One geometric fact explains the Fuzzers FP sink, the collision pairs, and the block-structured confusion matrix.
- **Notebook mislabel:** `plurality_ceiling_R` is NOT a ceiling (Worms scores 0.000 there yet achieves F1 0.486) — plurality voting minimises error rate, not macro-F1. Read it as "recall of the accuracy-optimal local rule".
- **Framing for the thesis:** the plateau is a property of the CICFlowMeter representation, not a modelling deficiency. Future work = **richer features** (payload, host/temporal context, session aggregation), not better classifiers. **SHAP should target overlap pairs** (Exploits↔Recon, DoS↔Exploits, Fuzzers↔Shellcode), not rare classes generically.
  > ⚠️ **The "host/temporal context" half is SUPERSEDED** by `final_model_v6_unsw.ipynb`: given those features, the model barely uses them (6.8% of importance). **Ask for service/port identity and TTL instead** — they carry the top five importances on the same traffic. The representation claim itself is confirmed and strengthened.

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

### Dedupe ablation on UNSW-NB15 (`results/unsw_dedupe_ablation/`) — the 10–15 point claim does NOT replicate

v6-on-UNSW re-run with dedupe toggled off; seeds, splits, prune rule, pool, combiner grid and SELV
protocol all held fixed.

| | DEDUPE | NO-DEDUPE | Δ |
|---|---|---|---|
| macro-F1 | 0.6378 | 0.6403 | **+0.0025** |
| accuracy | 0.8299 | 0.8228 | **−0.0071** |
| e_1NN | 0.3044 | 0.2435 | −0.0609 |
| gate precision | 0.9242 | 0.9794 | +0.0552 |
| test rows with exact train twin | 0.00% | **42.70%** | — |

- **No inflation, despite 42.7% leakage.** Both deltas are inside the ±0.012 paired band and
  accuracy moves the *wrong* way. See the scoped dedupe convention above before quoting any figure.
- **DECISIVE, CONFOUND-FREE DIAGNOSTIC: leaked rows score WORSE than clean ones** — 0.8064 (n=22,007)
  vs 0.8350 (n=29,528), same model and same test set. **The model is not memorising.**
- **MECHANISM: duplication re-weights classes, it does not memorise.** Duplicate rates are wildly
  uneven (Generic 95.8%, DoS 76.0%, Exploits 46.0%, Worms 2.9%). Generic does inflate 0.914→0.990,
  but that cancels inside the DoS/Exploits/Analysis triangle: DoS recall 0.327→0.844 while Exploits
  recall 0.827→0.562. Net ≈ zero on macro.
- **⚠️ Leakage DOES destroy the distance-based diagnostics**: e_1NN moves −0.061, larger than the whole
  round-1 feature gain, because the test row's own twin sits in the reference set. **Dedupe is
  essential for the Cover–Hart bracket, k-NN purity and DES competence — and nearly irrelevant for the
  classifier score.**
- **⚠️ Levels are confounded**: the prune keeps 83 vs 80 columns (duplication inflates correlations) and
  the test sets are different distributions (Generic 1,520 → 11,774 rows). Only the leaked-vs-clean
  split is clean.

### v6 on CIC-IDS2017 (`results/cicids2017_v6/`) — 0.9835, and that is a RED FLAG

2,830,743 flows → 2,520,798 after dropping inf/NaN (0.10%) and 10.86% duplicates. Web Attack variants
collapsed to one class, DoS variants kept separate → **13 classes**; all 31 derived features map onto
CICFlowMeter v3 (verified, no silent zero-fill) → 78 features. **macro-F1 0.9835 / accuracy 0.9991**
(0.9805 excluding Heartbleed's 2 and Infiltration's 7 test rows). Runtime 30 min.

| | CIC-IDS2017 | CIC-UNSW-2024 | UNSW-NB15 |
|---|---|---|---|
| macro-F1 | **0.9835** | 0.6485 | 0.6387 |
| e_1NN | **0.0026** | 0.3629 | 0.3043 |
| exact-duplicate rate | 10.86% | 32% | 36.8% |

- **⚠️ DO NOT REPORT THIS AS A RESULT WITHOUT THE CAVEAT.** e_1NN = 0.0026 means 1-NN alone separates
  the 12 families with 99.74% accuracy *after* dedupe. Only **Bot (0.820)** is below 0.99.
- **⚠️ UNVERIFIED:** the hypothesis that tool-generated *near*-duplicate bursts survive exact dedupe.
  The near-duplicate distance check was **not run**. Do not state the mechanism as established.
- **THE ARCHITECTURE BECOMES IRRELEVANT — the real finding.** All 17 combiner variants land within
  **0.0007** on test; equal-average and plain ExtraTrees match the winner. Against +0.0502 for the full
  stack on CIC-UNSW. **The architecture's value is conditional on the problem being hard** — the flip
  side of the v6 interaction result, and a contribution worth stating.
- **Does NOT close the single-dataset gap** — no plateau to explain. A genuine generalisation test
  needs a **chronological split**, which requires `GeneratedLabelledFlows.zip` for timestamps.
- **THE SOLID POSITIVE — engineered features replicate on a third extractor**: derived = 39.7% of
  columns, **59.0% of importance**, top three all derived. **`frac_PSH` is top-2 on both CIC-UNSW and
  CIC-IDS2017 independently.** Survives whatever the headline turns out to mean.
- Deviations, all scale-forced and recorded in the JSON: DES reference capped at 40,000 (v6 used
  32,468, so the regime matches); inf/NaN rows dropped; SMOTE `k_neighbors` clamped; float32.
- **Do not compare levels** to 0.6485/0.6387 (13 classes vs 10, BENIGN 83.1%). And 0.9991 accuracy vs
  the bracket's [0.9974, 0.9987] is **not** a ceiling violation — the bracket is Stage-B 12-family
  error on attack rows, the accuracy is 13-class including the BENIGN majority.

### CIC-IDS2017 verified data facts (`results/cicids2017_audit/`)

- **2,830,743 rows × 79 cols** (78 features + `Label`), 8 day-files, column lists identical after strip.
- **NOT clean, unlike CIC-UNSW-2024**: `Flow Bytes/s` 1,358 NaN + 1,509 inf, `Flow Packets/s` 2,867 inf.
- **65 of 79 names carry leading whitespace**; CICFlowMeter **v3** naming (`Fwd Avg Bytes/Bulk`).
- **`Fwd Header Length` is duplicated** — pandas renames it `Fwd Header Length.1`, so a duplicate-*name*
  check finds nothing; the correlation prune catches it at r=1.000.
- **`Web Attack` labels contain a non-ASCII en-dash** — read with `encoding="latin-1"`.
- **Duplicates 10.89% (features+label) vs 10.92% (features only)** → only **698 rows** carry conflicting
  labels, unlike UNSW-NB15 where ambiguity drove the whole dedupe result.
- **Imbalance 206,645:1.** Heartbleed 11 rows, SQL Injection 21, Infiltration 36. **Heartbleed and
  Infiltration cannot support a stratified split** — never put them in a headline macro average.
- Prune **78 → 70 → 47** (CIC-UNSW: 76 → 67 → 44), so widths are comparable. All four `Subflow *`
  columns are r=1.000 duplicates of the corresponding `Total *`.

### Service-identity features (`results/cic_unsw_svc/`) — **LARGEST GAIN IN THE PROJECT: 0.6478 → 0.7460**

The missing identity columns were recovered from the official release's `CICFlowMeter_out.csv`
(`Resource/`, 3,540,241 rows: `Flow ID, Src IP, Src Port, Dst IP, Dst Port, Protocol, Timestamp` +
the same 76 features + string labels). **`Data.csv` is a verbatim subset but NOT row-aligned**
(diverges at row 27): all 447,915 rows were found by exact feature-vector match, every attack row
kept (per-family counts match exactly), Benign downsampled ~10× (3,450,658 → 358,332). Port/protocol
attached per row by feature-vector-hash consensus: port used only when all matches agree (94.1% of
deduped rows; protocol 100%), else `svc_ambiguous`. 26 svc columns (service one-hots, port-range
bins, proto, raw port) → **101 features**. Two arms, identical splits (stratify on `y` only), full v6
protocol, SELV-selected, test once.

- **Paired delta +0.0982, 95% CI [+0.0885, +0.1081], p<0.001** — 2× the entire previous stack
  (+0.0502). Quote as **0.746 [0.727, 0.763] / acc 0.9486**. First result to break the ~0.65 plateau.
- **The UNSW-extractor prediction is CONFIRMED on the CIC side**: given service identity, the same
  families recover to almost the same levels — Shellcode 0.376→**0.658** (UNSW: 0.650), Recon
  0.642→**0.833** (0.807), Fuzzers 0.723→**0.899**. The overlap attractors dissolve: svc splits
  Shellcode (ephemeral ports) from Fuzzers (BGP 179 / SMB 445) and Recon (RPC 111) from Exploits.
- **svc = 25.7% of columns, 70.4% of importance**; `svc_rpc` #1 (0.372), `svc_bgp` #2. Every family
  improved; biggest: Shellcode +0.28, Recon +0.19, Fuzzers +0.18, DoS +0.10.
- **e_1NN 0.3632 → 0.2098** (−0.153, 12× the round-1 feature gain; metric-relative as always).
  Gate precision 0.9267 → 0.9411 at unchanged recall — svc buys what gate re-weighting could not.
- **⚠️ Control-arm caveat:** the DERIVED control reproduced 0.6478/0.9127, not the published
  0.6485/0.9123 — ±0.0007 run-to-run jitter (XGBoost `hist` + threading), same SELV winner, same
  rank 1/21. The paired delta is measured against the same-run control, so it is unaffected; when
  quoting the *gain*, use +0.0982 from this run, not 0.7460−0.6485.
- **⚠️ SELV winner ranked 4/21 on test** (top variants within 0.003 — resolution failure, not
  selection error). Quote the SELV-selected 0.7460, never the test-max 0.7472.
- **⚠️ Honesty caveats:** (1) port consensus is computed over the full 3.54M file including test
  rows — label-free and deterministic (same class as the pre-split prune), but state it; (2) IXIA
  PerfectStorm targets attacks at specific services, so service identity may be partly a testbed
  artifact — the single-dataset limitation stands; a cross-network claim still needs CIC-IDS2017
  chronological or cross-dataset evidence. (3) 2.2% of rows have feature vectors carrying >1 label
  in the full file — slightly larger label-ambiguity estimate than the within-subset 0.35%.
- The 0–9 label mapping is now **file-verified** (all 447,915 rows consistent with alphabetical
  Benign=0…Worms=9), no longer just owner-confirmed.

### v7 — full release at the real base rate (`final_model_v7_full.ipynb`) — svc gain SURVIVES; ensembling does not

The v6+svc architecture on the full `CICFlowMeter_out.csv` (3,540,241 flows, **Benign 97.5%** vs the
curated file's 80%), Dst Port/Protocol as native features (no consensus join). Dedupe on
features+port+proto+label: → 2,606,803 rows (26.4% dupes); prune gives 43 (not 44); 99 features;
test = 521,361 rows. **macro-F1 0.6601 [0.6430, 0.6765] / accuracy 0.9799.** **NOT level-comparable**
to 0.6485/0.7460 — different base rate; this is the realism stress test.

- **The service-identity gain survives where it lives: attack-vs-attack.** Recon 0.817, Generic
  0.884, DoS 0.568, Backdoor 0.671 — at or near curated-svc levels; attack recalls held or rose
  (Exploits 0.907, Fuzzers 0.943, Shellcode 0.764). e_1NN 0.2084 ≈ curated 0.2098 — the attack
  geometry is unchanged, as predicted (same attack rows).
- **What the base rate destroys is precision on the benign-adjacent sink families**: Fuzzers
  0.899→0.598 (P 0.438 — **6,327 of the 8,370 leaked Benign flows land in Fuzzers**), Analysis
  0.452→0.169 (280 benign FPs vs 37 TPs), Shellcode 0.658→0.515. Same sink, 12× the volume.
- **The gate genuinely improved from 10× more Benign**: test precision **0.6583 vs the 0.535
  fixed-rate expectation** (fpr 2.25%→1.66%) at recall 0.9893. Report both numbers — the gap
  is the honest measure of what more benign data buys.
- **Ensembling adds NOTHING at full scale**: best single (XGB α=0.25, 0.6603) ≈ SELV-selected
  ensemble (0.6601); all 15 combiner variants span 0.0038; SELV winner ranked 10/21 (pure
  resolution failure). Third demonstration that **the architecture's value is conditional on the
  problem being hard** (CIC-IDS2017 was the second).
- **SOC translation**: ~1 in 3 Stage-B alerts is a false alarm at the real base rate, vs 1 in 16
  on the curated file. This is the number a deployment section should quote.
- svc importance replicates at scale: 72.4% (curated 70.4%), `svc_rpc` #1 both times.
- Splits are its own (2.6M rows); OOF cached in `cache_oof_v7_full.npz`; runtime ~15 min.

### v7.1 — Stage-B reject class (`final_model_v7_1_reject.ipynb`) — ESTABLISHED +0.0242; the sink inverts

Stage B extended to 10 classes: 9 families + "Benign-that-fooled-the-gate", trained on the 34,101
OOF gate FPs (1.69% of benign train+val). Same splits as v7 (asserted); gate-proba and 10-class OOF
cached (`cache_oof_v71_*`). SELV selected across {baseline, reject} × 21 variants × gate threshold.

- **Paired +0.0242 [+0.0142, +0.0345], p<0.001** (reject vs baseline, both t=0.5). Final:
  **0.684 [0.667, 0.700] / acc 0.9875**. Benign-as-attack false alarms **8,370 → 2,093 (4×)**.
- **The gate threshold is STILL a null lever, now at the real base rate too**: SELV picked t=0.7
  but test 0.6842 vs 0.6843 at t=0.5 — identical. For the *baseline* arm t did matter on SELV
  (0.6449→0.6570), i.e. the threshold's job is done better by the reject class; with it, t is
  redundant again. Quote reject@0.5.
- **Shellcode is the big winner: 0.515 → 0.725** (P 0.39→0.86, its ephemeral-port FP flood gone).
  DoS +0.033, Exploits +0.028, Fuzzers +0.024, Recon +0.008, Benign 0.9915→0.9953.
- **⚠️ THE FUZZERS SINK INVERTS, NOT DISSOLVES.** Fuzzers P 0.44→0.68 but R 0.94→0.58 — the reject
  class swallows 2,003 real Fuzzers (of 2,272 swallowed attacks total). Net +0.024 only. The
  Fuzzers↔Benign boundary is genuinely inseparable in this representation; the reject class lets you
  *choose which side pays* (FP flood vs FN swallow) — a SOC operating choice, same status as the
  weights-vs-SMOTE knob. State it that way, not as "Fuzzers fixed".
- **⚠️ Analysis got worse: 0.169 → 0.134** (R 0.48→0.31; 52 of its 77 test rows go to Benign).
  Analysis flows look like benign leaks. Rare-class casualty, wide CI — report as interval.
- **Per-class W is load-bearing again in the 10-class problem**: the reject arm's SELV winner is
  HYBRID W × OLA-cw (all 5 hybrids beat all pure-DYN variants there), unlike v7 where everything
  tied. Consistent with the project rule: the combiner pays only when the problem is hard.
- Reject accounting @final: 7,562 gate-positives sent back to Benign — 5,290 correct, 2,272 real
  attacks. Print lives in the notebook; quote both numbers together.

### v7.1b — W refitted under the END-TO-END objective (`results/v71_w_e2e/`) — +0.0032, established

Same 63+7 parameters, same ascent; only the objective changes from Stage-B 10-class macro-F1 (reject
= 1/10 of the score) to end-to-end macro-F1 over the COMB half (reject→Benign, gate-negatives as
fixed background). **Test 0.6874 / 0.9866; paired +0.0032 [+0.0019, +0.0047], p<0.001** vs the
same-run v7.1 reference (which reproduced 0.6843 exactly).

- **The gain is exactly the predicted mechanism**: Fuzzers recall 0.576→**0.731** (F1 0.622→0.641),
  attacks swallowed by reject 2,457→1,712, paid with benign-as-attack 2,140→3,302. Everything else
  ±0.004.
- **The reject column of W did not move** — the ascent rebalanced the *family* columns instead. The
  operating point lives in the whole matrix, not in one column.
- **Full-release ladder: 0.6601 → 0.6843 (reject) → 0.6874 (e2e objective).** Quote the final as
  `0.687 [0.671, 0.704]`.
- Thesis point: **fit the combiner on the deployment objective, not the sub-problem objective** —
  a clean, general lesson that costs nothing to apply.

### v7.2 — hyperparameter sweep (`results/v72_hparam/`) — NOT ESTABLISHED; keep the defaults

The one conventional lever never previously tested. Pre-registered two-stage protocol: 40 random
XGB configs + default, each a single 10-class reject-arm model trained on COMB attack+reject rows,
scored by e2e macro-F1 on SELV (gate fixed via cached probas); promote to a full pipeline run only
if best-vs-default > +0.005 on SELV. Test touched once, at the very end.

- **Probe cleared the bar (+0.0079**, C17 = depth 12, mcw 5, subsample 1.0, colsample 0.9, lr/rounds
  unchanged), **but the full pipeline did not confirm: paired +0.0054 [−0.0037, +0.0136], p ≈ 0.23.**
  Test 0.6942 (tuned) vs 0.6888 (same-run v7.1b reference). **Not established — keep the default
  parameters in the headline model.** Point-positive-but-inside-noise; same category as v3-over-v2.
- The probe→pipeline shrinkage (0.0079 → 0.0054, CI through zero) is the winner's-curse pattern the
  two-stage protocol was built to catch. The single tuned XGB ranks *last* on test (0.6721) — the
  probe's gain does not even survive as a single model at full refit scale.
- Sweep shape: most of the 40 configs score *below* default; the top handful cluster at +0.004–0.008.
  Depth/regularisation moves are absorbed by the weighting+ensemble machinery around them.
- **Completes the capacity story: every capacity-side lever is now tested** (algorithms, ensembles,
  dynamic selection, thresholds, and hyperparameters) — none established. Gains in this project came
  from weighting, information, and objectives only. Quote this sweep when an examiner asks "did you
  tune?".
- Same-run v7.1b reference reproduced at 0.6888 vs 0.6874 previously — ±0.0014 refit jitter; the
  headline quote `0.687 [0.671, 0.704]` stands.

### v7.1c — reject-set enrichment (`results/v71c_reject_enrich/`) — NULL; the lever list is now EMPTY

Reject class trained on all near-boundary benign (OOF gate p > 0.2) instead of only the p > 0.5
FPs; inference, ref set, W protocol, selection all pinned. **Paired +0.0009 [−0.0082, +0.0096],
p ≈ 0.82** (SELV-selected 0.6898 vs same-run v7.1b 0.6888).

- **Why it was null, mechanistically**: the enrichment added only **+5,240 rows** — the gate is
  well-calibrated, so very little benign mass sits in p ∈ (0.2, 0.5]. There was no denser picture
  of the boundary to learn. Per-class deltas all ≤ ±0.010; the FP/swallow accounting barely moved
  (3,809→3,865 / 1,465→1,436). **Boundary estimation is saturated** — completes the reject-class
  story.
- SELV winner ranked 3rd of 4 on test with the static variant nominally top (0.6957) — inside-noise
  rank shuffle; quote the SELV-selected number, as always.
- **Status: every in-scope lever is now tested.** Established: α-weighting, DERIVED features,
  service identity, hierarchy+reject, e2e objective. Null/closed: full SMOTE, focal, DES v1,
  algorithm-diverse pools, thresholds, soft routing, gate weighting (×2), wider α, D2 algebra,
  TTL (label proxy), hyperparameters, reject enrichment. Remaining score paths require payload
  (PCAPs) — future work. Headline stands: **0.687 [0.671, 0.704] / acc 0.9866** (v7.1b).

### TTL join (`results/ttl_join/`) — join SUCCEEDED; TTL is a measured LABEL PROXY — do NOT add it

`sttl`/`dttl` from the original UNSW-NB15 full flow files (`UNSW-NB15_1–4.csv`) joined onto all
3,540,241 CFM rows by 5-tuple (both orientations) + Argus time window; empirical clock offset
+14400 s (Atlantic time, 98.4% of samples within ±2 s). **Coverage: 93.1% direct + 4.6% reversed
+ 2.3% ip-pair-modal fallback = 0.00% unmatched.** Runtime 36 s (`cache_ttl_raw.npz`).

- **THE FINGERPRINT CHECK FAILED TTL, CLEANLY.** Benign is 97% sttl ∈ {31, 29} — and those values
  are **0.0000 attack** across 3.35M rows. Every attack family is ≥98% sttl ∈ {62, 254}. The rule
  `sttl ∈ {62,254}` alone recovers ~100% attack recall at fpr ≈ 1.6% — matching the entire
  99-feature learned gate. TTL here encodes *which machine generated the traffic* (benign
  simulation hosts vs attack generator, one router hop), and machine ↔ label by testbed
  construction. **Adding it is equivalent to adding the source IP, which v7 excluded on purpose.**
- **Retro-reinterpretation of `final_model_v6_unsw.ipynb`:** the top importances of
  `ttl_diff`/`sttl`/`r_ttl` on the UNSW extractor were substantially fingerprinting, not
  behaviour. The service-identity half of that finding (svc_dns/svc_http) stands — svc gains were
  validated by ablation on OUR side (+0.098) and services are what attacks *target*, not who sends
  them. The "ask for TTL" half of the prescription is **withdrawn**: on this testbed it cannot be
  distinguished from endpoint memorisation. This also explains sttl's notorious dominance in the
  UNSW-NB15 literature — papers using it inherit the artifact.
- **Optional quantification run (not done):** adding TTL and measuring the paired delta would
  *quantify the artifact's worth*, publishable as "what sttl-using papers gain from topology" —
  frame as artifact measurement, never as model improvement.
- The join methodology itself is a reusable contribution: same-traffic/two-extractor field
  transfer at 100% coverage, with the fingerprint audit as a pre-registered go/no-go gate.

## Environment & commands

- Windows; Python 3.12.8 via miniconda (`C:\Users\night\miniconda3`). Primary shell is PowerShell; a Bash tool is also available. **The ML stack (`sklearn`, `xgboost`, `imblearn`) lives in the `ai4beg` conda env, not base** — for scripts run outside the notebook kernel use `C:\Users\night\miniconda3\envs\ai4beg\python.exe` (base `python` raises `ModuleNotFoundError: sklearn`).
- Terminal stdout is **cp1252** — printing Unicode (e.g. `±`, `–`, accented author names) raises `UnicodeEncodeError`. When dumping text, write to a UTF-8 file (`io.open(path, "w", encoding="utf-8")`) and read it back rather than printing to the console.
- Reading the Excel workbook requires `openpyxl` (installed). Use `data_only=True` to get computed values. Close the file in Excel first — a `~$...` lock file in `Papers/` means it is open and writes will fail/conflict.

## Conventions for this project

- Frame results around the **literature gaps** the review identified: weak recall on rare attack classes, unaddressed imbalance, single-dataset evaluation, and missing explainability (XAI). The thesis contribution leans on the under-explored 2024 dataset plus proper imbalance handling.
- **A fifth gap is now evidenced: claimed-but-unavailable artifacts** (`literature_artifact_audit.md`). **0 of 10 reviewed papers release code.** The most common "availability" statement points at a public benchmark *dataset*, which is not the pipeline. Without the code a reader cannot tell which protocol a published number was produced under — and the protocol is worth more than the model. CyberAdaptAI is the worked example, not an outlier.
- **⚠️ SCOPE THE DEDUPE CLAIM — it is protocol-specific, not universal.** Earlier wording here said deduplication moves reported accuracy by "10–15 points on both datasets". **That is measured only in CyberAdaptAI's binary + sequential-stream setting** (`Clones/CyberAdaptAI/RESULTS.md`: dedup+sequential 0.8065 acc / 0.7323 F1 → no-dedup+sequential 0.9019 / 0.8825; shuffled, the gap shrinks to +0.034 acc / +0.066 F1). Under the **v6 multiclass batch pipeline with a stratified random split it does not replicate: +0.0025 macro-F1 and −0.0071 accuracy** despite 42.7% of test rows having an exact train twin (`results/unsw_dedupe_ablation/`). **Always state task, split protocol and learner when quoting a dedupe effect.**
- Be skeptical of near-100% results — for this data they almost always indicate duplicate leakage, not a good model. **Worked example: v6 scores 0.9835 macro-F1 / 0.9991 accuracy on CIC-IDS2017 with e_1NN = 0.0026, and every architectural component is worth 0.0007** (`results/cicids2017_v6/`). A near-perfect score is a prompt to interrogate the dataset, not to report the number.
- The extraction workbook has a known error to be careful with: in **Sheet2, the SL 20 row** carries the title of the Opt-Forest paper but the Dataset/Algorithm/Findings of HAEnID (paper 21); the real Opt-Forest data (UNSW-NB15, GA-optimized forest, 99.45% acc) is missing there.
