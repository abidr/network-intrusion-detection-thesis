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
# binary attack-vs-benign
run_experiment.py --dataset unsw-nb15 --task binary --out results/unsw_binary
# 10-class: what Table 4 actually reports, and comparable to the thesis baselines
run_experiment.py --dataset unsw-nb15 --task multiclass --out results/unsw_multiclass
```

> ### ⚠️ CORRECTION (2026-08-09): Table 4 is MULTI-CLASS, not binary
>
> An earlier version of this README argued Table 4 was a binary evaluation and made
> `--task binary` the comparison of record. **That was wrong, and the paper settles it directly:
> Fig. 4 is captioned "Multi-Class confusion matrices for CIC-IDS2017 and UNSW-NB15",** and the
> surrounding text describes "strong diagonal values ... across categories". Table 4's 0.968 is a
> 10-class number.
>
> The original inference was not unreasonable — Table 4 reports a single Precision/Recall/F1
> triple and Eq 13-15 are stated in bare TP/FP/FN form — but the §4.6 sentence it leaned on
> (*"recasting the problem as a multi-class one results in numerous misclassifications for
> minority attack types"*) is about the **cross-dataset** experiment in §4.6, not about Table 4.
>
> **What the real objection is.** The paper never states its averaging scheme — **the word
> "macro" does not appear anywhere in the 21 pages.** Its reported values are
> Accuracy 0.968, P 0.965, R 0.970, F1 0.968: `P ≈ R ≈ F1 ≈ Accuracy` is the arithmetic
> signature of **micro-averaging**, where micro-P = micro-R = micro-F1 = accuracy identically.
> So Table 4 should be read as accuracy-equivalent, and compared against **weighted/micro F1**,
> never against macro-F1.
>
> Concretely, the right comparison for Table 4's 0.968 on UNSW-NB15 is:
>
> | | task | duplicates | weighted F1 | accuracy | macro F1 |
> |---|---|---|---|---|---|
> | this reimplementation, leakage-safe | 10-class | 94,928 removed | 0.7784 | 0.7871 | 0.4215 |
> | thesis v6 port (`final_model_v6_unsw.ipynb`) | 10-class | 94,928 removed | **0.8380** | 0.8303 | 0.6387 |
>
> **Follow-up (same day): Table 4's *magnitude* is binary even though its *task* is labelled
> multi-class.** Running the multiclass twin of the cell that reproduced the paper in binary
> (`--task multiclass --no-dedup --shuffle`, `results/unsw_repro_table4_multiclass`) reaches only
> **accuracy 0.8324** — granting both leakage assumptions buys multiclass just +0.045 over the
> leakage-safe 0.7871, leaving 13.6 points unexplained. Binary under the same assumptions hits
> 0.9479 / F1 0.9590. So the original "binary" reading was wrong about what the paper *says* and
> right about what the number *is*. The paper's three claims cannot be reconciled:
> Fig. 4 caption (multi-class), Table 4 magnitude (binary-only), Fig. 4 contents (0.303).
>
> **Run both tasks and report both**, and quote Table 4 as a binary-scale figure with that
> inconsistency stated.

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
- **The metric averaging scheme is never stated — "macro" appears nowhere in the 21 pages**, and
  Eq 13-15 are written in bare TP/FP/FN form. Table 4 is **multi-class** (Fig. 4: "Multi-Class
  confusion matrices"), and its `P ≈ R ≈ F1 ≈ Accuracy` (0.965 / 0.970 / 0.968 / 0.968) is the
  signature of **micro-averaging**, under which micro-F1 equals accuracy identically. **Compare
  Table 4 against weighted/micro F1, never macro-F1** — it changes what any comparison means.
  See the correction box in *Running* above.
- **⚠️ Fig. 4's confusion matrices contradict Table 4 by a factor of three.** Counted directly from
  the published figure (extracted image, page 12):

  | panel | classes shown | total samples | diagonal | **implied accuracy** | **Table 4 claims** |
  |---|---|---|---|---|---|
  | (b) UNSW-NB15 | **6** of 10 | **300** | 91 | **0.303** | 0.968 |
  | (a) CIC-IDS2017 | **5** of ~15 | **300** | 82 | **0.273** | 0.981 |

  UNSW row sums: Normal 161, Analysis 49, Backdoor 28, DoS 38, Reconnaissance 16, Shellcode 8.
  Exploits, Fuzzers, Generic and Worms do not appear at all. Per-class recall from the panel:
  Normal 0.435, Analysis 0.224, Backdoor 0.071, DoS 0.184, Recon 0.062, **Shellcode 0.000** —
  macro-F1 ≈ 0.15.

  The surrounding text says *"Strong diagonal values indicate high true positive rates across
  categories, while the minimal off-diagonal entries indicate little misclassification."* The
  figure's own numbers show the reverse: the Backdoor row sends 18 of 28 to Normal and gets 2
  right; Reconnaissance gets 1 of 16.

  Both panels totalling **exactly 300** and omitting most classes is the signature of an
  illustrative or synthetic figure rather than output of the evaluation Table 4 describes.
  **Consequence: the paper's headline numbers cannot be reconciled with its own reported
  confusion matrices, so "failing to reproduce Table 4" is not a reimplementation defect.**
- **Fig. 6/7 plots only three ensemble members** — Random Forest, XGBoost, Extra Trees — with no
  Hoeffding Tree, contradicting the four-member pool of §3.4 and §4.1. Its weights also swing
  0.18 → 0.50 across batches, where this reimplementation measures them flat to within 1.3% of
  uniform under the paper's own hyperparameters (see RESULTS.md).
- **Table 8 is a copy-paste of Table 7.** It is captioned "Class-wise F1-Scores for Cross-Dataset
  evaluation" but its contents are byte-identical to Table 7's feature-importance values — rows
  `Flow Duration` / `Fwd Packet Length` / `Protocol` / `Src Bytes` / `Dst Bytes`, columns
  "Instance 1/2/3". They are **features, not classes**. So the *only* place the paper promised
  per-class numbers, it printed the wrong table, and **rare-class performance is unverifiable
  from the published text** — which is exactly where a 10-class macro-F1 would be decided. For
  contrast, this reimplementation's leakage-safe 10-class run gets Worms **0.000**,
  Analysis 0.021, Backdoor 0.032.
- **Table 3's batch-wise accuracies are monotone for all four models across all ten batches** —
  RF 0.905→0.919, XGBoost 0.915→0.927, Static Ensemble 0.930→0.939, CyberAdaptAI 0.958→0.968,
  every step +0.001 or +0.002 and never once decreasing. Real per-batch accuracy fluctuates with
  batch composition; a strictly increasing sequence in every model at every step is worth
  flagging when citing it.
- **Dynamic weighting is the same lever this thesis already tested and rejected.**
  `two_stage_stageB_des.ipynb` found per-instance competence weights collapsed to
  0.333 +/- 0.02 (a static ensemble in disguise) and did not transfer to test. CyberAdaptAI's
  weights are per-*batch* rather than per-instance, which is a weaker adaptation, so expect the
  same flattening. **This reimplementation logs the weight trajectory (`weights.csv`) precisely
  so you can check whether the weights actually move.** That plot is the honest test of the
  paper's central claim, and it is a result either way.
