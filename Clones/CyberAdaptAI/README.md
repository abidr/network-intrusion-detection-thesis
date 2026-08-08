# CyberAdaptAI / AdaptEnsembleNet — reimplementation

> **This is NOT the authors' code.** No official implementation exists (see *Provenance*).
> Everything here was written from the paper text: Algorithm 1, Equations 1-16, and the
> hyperparameters listed in §4.1. Every module cites the equation or section it implements.

**Paper.** Nagamani Uddamari, P. Sammulal. *CyberAdaptAI: A Dynamic Ensemble Learning
Framework for Real-Time Cyberattack Detection Using AdaptEnsembleNet.* SSRG International
Journal of Electronics and Communication Engineering, 12(9), 11-31, Sept 2025.
DOI `10.14445/23488549/IJECE-V12I9P102`. Local copy:
`../../Papers/CyberAdaptAI_A_Dynamic_Ensemble_Learning.pdf`

---

## Provenance — why there is nothing to clone

§4.1 states:

> "For reproducibility, all source code, hyperparameter configurations, and batch simulation
> scripts are modularized in Python files and documented in a Git-based repository."

That sentence is the paper's **entire** reproducibility provision. It contains no URL, no
repository name, no DOI, no archive link, and the paper has no data/code availability section.
Searches for `CyberAdaptAI` and `AdaptEnsembleNet` (GitHub, web) return no matching repository.

**Treat the reproducibility claim as unfulfilled.** That is itself a citable observation for a
literature-review chapter: it is the same "claimed but unavailable artifact" gap the review
already flags around single-dataset evaluation and missing explainability.

## Claimed results (from the paper — reproduced here for reference only, NOT verified)

| Dataset | Accuracy |
|---|---|
| CIC-IDS2017 | 98.1% |
| UNSW-NB15 | 96.8% |

Both are near-ceiling numbers on datasets with known duplicate-leakage issues, and the paper
does not report a deduplication step (§3.2 says "eliminating duplicate records" but gives no
counts, and dedup is applied *before* the stream split). See *Critique* below.

---

## Layout

```
cyberadaptai/
  config.py         All paper hyperparameters, each tagged with its source line in §4.1
  preprocessing.py  Eq 1 (min-max), label encoding, cleaning (§3.2)
  stream.py         Mini-batch stream simulation, 1500-2000 rows/batch (§3.2, §4.1)
  base_learners.py  RF / XGBoost / ExtraTrees / Hoeffding Tree pool (§4.1)
  diversity.py      Eq 4 (Q-statistic) + the disagreement measure (§3.3)
  drift.py          ADWIN via Eq 6-7, + the "8% over 3 batches" trigger (§3.4, §4.1)
  ensemble.py       AdaptEnsembleNet: Eq 2 weighted vote, Eq 3 + Eq 8 weight updates (§3.3)
  metrics.py        Eq 12-16 incl. per-batch latency (§3.7)
  explain.py        SHAP TreeExplainer (Eq 9), softmax confidence (Eq 10-11) (§3.5)
  runner.py         Algorithm 1, the main streaming loop
adapters/
  cic_unsw_2024.py  Loads THIS thesis's Resource/ dataset into the stream format
  unsw_nb15.py      Loads the ORIGINAL UNSW-NB15 -- the paper's own dataset (Table 4)
run_experiment.py   CLI entry point
```

## Install / run

The ML stack lives in the `ai4beg` conda env, not base:

```bash
C:/Users/night/miniconda3/envs/ai4beg/python.exe -m pip install -r requirements.txt
C:/Users/night/miniconda3/envs/ai4beg/python.exe run_experiment.py --help
```

Run against this thesis's dataset:

```bash
C:/Users/night/miniconda3/envs/ai4beg/python.exe run_experiment.py \
    --dataset cic-unsw-2024 --batch-size 2000 --out results/
```

Reproduce the paper's Table 4 on its own dataset (needs UNSW-NB15 in
`Resource/UNSW-NB15/` — see the README there):

```bash
# binary attack-vs-benign: the setting comparable to Table 4
run_experiment.py --dataset unsw-nb15 --task binary --out results/unsw_binary
# 10-class: comparable to the thesis baselines
run_experiment.py --dataset unsw-nb15 --task multiclass --out results/unsw_multiclass
```

**Why binary is the default for `unsw-nb15`.** Table 4 reports a single Precision/Recall/F1
triple, Eq 13-15 are stated in bare TP/FP/FN form, and §4.6 says *"recasting the problem as a
multi-class one results in numerous misclassifications for minority attack types"* — implying
the headline evaluation was not multiclass. In binary mode the run prints
`Precision/Recall/F1 (attack)`, which is the directly comparable triple. Multiclass mode prints
macro-F1 as usual. The paper never states its averaging scheme, so **run both and report both**.

`river` (Hoeffding Tree) is **optional**: without it the Hoeffding Tree falls back to an SGD
online learner — a substitution the paper itself sanctions in §3.4, *"Hoeffding Trees, adaptive
SGD-based models"* — and the run reports the fallback rather than swapping silently.

`shap` is **not required at all**. TreeSHAP values come from XGBoost natively
(`pred_contribs=True`), which is the same exact algorithm `shap.TreeExplainer` dispatches to
for XGBoost models. The library call from §4.1 is kept as a fallback backend.

**Version pins matter here** (`requirements.txt`): current `river`/`shap` releases pull
`numpy>=2.4`, which would upgrade the numeric stack the thesis notebooks depend on. The pinned
`river==0.21.0` / `shap==0.46.0` install into `ai4beg` with no other changes. Note that
`shap<=0.46` cannot load XGBoost ≥3.x multiclass models at all (`base_score` became a per-class
vector and shap still casts it with `float()`), which is the other reason the native backend is
the default.

---

## Where the paper is ambiguous, and what I chose

These are decisions the paper forced. Each is a config switch so you can flip it.

### 1. Two contradictory weight-update rules (`WeightRule`)

The paper gives **both**, and never reconciles them:

- **Eq 3**: `w_m = alpha * A_m + (1 - alpha) * D_m` — accuracy/diversity blend, `alpha = 0.6`.
  Has no temporal smoothing and **no diversity-free path**.
- **Eq 8**: `w_m^{t+1} = gamma * w_m^t + (1 - gamma) * (1 - E_m^t)` — EWMA smoothing,
  `gamma = 0.8`. Has **no diversity term at all**.

Algorithm 1 line 9 uses **only Eq 8**, which would make the entire diversity apparatus
(Eq 4, the `alpha = 0.6` tuning) dead code. That cannot be the intent.

**Default = `hybrid`**: form the Eq-3 target `alpha*A + (1-alpha)*D`, then smooth it with the
Eq-8 `gamma`. Modes `eq3` and `eq8` reproduce each rule literally. **Report which you used** —
they do not give the same model.

### 2. The Q-statistic has the wrong sign for its stated purpose (`DiversityMeasure`)

Eq 4 defines Yule's Q. Q is **+1 when two classifiers agree** and negative when they disagree.
But Eq 3 adds `D_m` with a positive coefficient and the text says *"If the average disagreement
score is high, it indicates that the classifier brings new decision boundaries"* — so the term
must **increase** with disagreement. Plugging Q in raw rewards *redundancy*.

**Default = `disagreement`**: `Dis(h_m,h_n) = (N10 + N01) / N`, which matches the prose.
`q_statistic` (raw Eq 4) and `neg_q` (sign-corrected) are available. This is a genuine defect in
the paper, not a transcription slip on my part — worth a sentence in a related-work critique.

### 3. Min-max scaling is fitted on... something unstated (`ScalerFit`)

Eq 1 uses `min(x_j)` / `max(x_j)` with no scope. In a streaming setting there are three
possibilities and the paper picks none: global (leaks the future), first-batch-only, or
running. **Default = `first_batch`** (leakage-safe). `global` reproduces the most likely
intended reading and is what you'd need to match their numbers.

### 4. "Retrain underperforming classifiers" is undefined (`RetrainPolicy`)

Algorithm 1 line 11 says "reset or retrain underperforming classifiers". Undefined: which
classifiers, on what data, reset vs. refit. **Default**: refit any learner whose batch accuracy
is below the pool mean, on the current ADWIN window.

### 5. No `random_state` anywhere

None of the four base learners is seeded in §4.1. All are seeded here (`config.SEED = 42`).

---

## Critique — read before using this as a baseline

Relevant to this thesis, since the two projects overlap heavily (dynamic ensemble weighting +
SHAP + imbalance):

- **Accuracy is the headline metric on severely imbalanced data.** §4.1 reports accuracy,
  precision, recall, F1 and AUC but the abstract leads with 98.1% / 96.8% accuracy. Neither
  macro-averaging nor per-class rare-attack performance is foregrounded. This thesis's own
  finding — that accuracy is meaningless at 80% Benign — applies directly.
- **The streaming premise is not established.** Both CIC-IDS2017 and UNSW-NB15 are batch
  datasets. §3.2 *simulates* a stream by chunking rows in order. Whether real concept drift
  exists in that ordering is never demonstrated, so "recovers quickly after drift events" may
  be measuring artifacts of row ordering.
- **Duplicate handling is unquantified.** "Eliminating duplicate records" (§3.2) with no counts,
  on datasets where duplicates are the classic leakage trap.
- **Table 8 is a byte-for-byte duplicate of Table 7.** Table 8 is captioned "Class-wise
  F1-Scores for Cross-Dataset evaluation (Train: CIC-IDS2017, Test: UNSW-NB15)" but its contents
  are Table 7's SHAP feature-importance grid — same five features, same values, same
  Instance 1/2/3 columns. **The cross-dataset F1 scores appear nowhere in the paper**, even
  though the surrounding prose discusses them ("generalized well for regular and DoS traffic,
  poorly for Shellcode and Analysis") and cross-dataset generalization is one of the three
  headline novelty claims.
- **Table 7 is almost certainly not SHAP output.** Its columns each sum to exactly 1.00, values
  come in 0.05 increments, and only five features are listed. Real SHAP values sum to
  `f(x) - E[f(x)]`, are frequently negative, and would span the full feature set. The caption
  hedges it as "SHAP-**Like** analysis." For contrast, this reimplementation's native TreeSHAP
  gives an Eq-9 additivity residual of 3.8e-06 across 40+ features (see RESULTS.md).
- **The metric averaging scheme is never stated**, and Eq 13-15 are written in binary TP/FP/FN
  form. Table 4's 0.968 UNSW-NB15 accuracy is routine for binary attack-vs-benign but would sit
  well above the usual 0.75-0.87 range for 10-class. Treat Table 4 as binary unless proven
  otherwise — it changes what any comparison means.
- **Dynamic weighting is the same lever this thesis already tested and rejected.**
  `two_stage_stageB_des.ipynb` found per-instance competence weights collapsed to
  0.333 +/- 0.02 (a static ensemble in disguise) and did not transfer to test. CyberAdaptAI's
  weights are per-*batch* rather than per-instance, which is a weaker adaptation, so expect the
  same flattening. **This reimplementation logs the weight trajectory (`weights.csv`) precisely
  so you can check whether the weights actually move.** That plot is the honest test of the
  paper's central claim, and it is a result either way.
