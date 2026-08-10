# Results — short form

One row per model. **Macro** precision/recall/F1 treat all 10 classes equally — **macro-F1 is the
project's primary metric**. **Weighted ("normal")** precision/recall/F1 are support-weighted and
Benign-dominated; they are reported for completeness and always look flattering — never quote them
without the macro numbers. Values are copied from each notebook's `classification_report` (3–4
decimals as printed; `—` = the notebook does not print that report). Full context and caveats:
`results_summary.md` and `CLAUDE.md`.

> **Reporting convention** (`ablation_significance.ipynb`): a single config carries a marginal 95% CI
> of ≈ ±0.017 macro-F1 — quote e.g. `0.687 [0.671, 0.704]`, never four bare decimals. Differences
> between two configs on the same test set use the paired band ≈ ±0.012.

---

## 1. Curated CIC-UNSW-2024 (447,915 rows, Benign 80%; test = 61,235)

### Flat baselines

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| Decision Tree (44 feat) | `decision_tree.ipynb` | 0.8855 | 0.5408 | 0.5284 | 0.5338 | 0.8858 | 0.8855 | 0.8856 |
| Decision Tree (36 feat) | `decision_tree_mi_rfecv.ipynb` | 0.8847 | 0.5361 | 0.5282 | 0.5316 | 0.8851 | 0.8847 | 0.8849 |
| Random Forest (44 feat) | `random_forest.ipynb` | 0.9017 | 0.6521 | 0.5358 | 0.5744 | 0.9031 | 0.9017 | 0.9004 |
| Random Forest (36 feat) | `random_forest_mi_rfecv.ipynb` | 0.9024 | 0.6580 | 0.5404 | 0.5805 | 0.9037 | 0.9024 | 0.9012 |
| XGBoost (44 feat) | `xgboost_model.ipynb` | 0.8949 | 0.5422 | 0.6510 | 0.5807 | 0.9153 | 0.8949 | 0.9021 |
| XGBoost (36 feat) | `xgboost_mi_rfecv.ipynb` | 0.8942 | 0.5402 | 0.6534 | 0.5800 | 0.9151 | 0.8942 | 0.9016 |
| XGBoost + full SMOTE | `xgboost_model.ipynb` (Stage 6) | — | — | — | 0.5088 | — | — | — |

### Two-stage hierarchy (gate + family classifier) — BASE 44 features

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| Two-stage + balanced weights | `two_stage_hierarchical.ipynb` | 0.8944 | 0.5427 | 0.6779 | 0.5858 | 0.9162 | 0.8944 | 0.9020 |
| Two-stage + capped SMOTE (test-selected ⚠) | `two_stage_stageB_experiments.ipynb` | — | 0.6624 | 0.5790 | 0.5976 | — | — | — |
| Two-stage + α=0.25 weights | `two_stage_stageB_weighting.ipynb` | 0.9081 | 0.6574 | 0.6035 | 0.6157 | 0.920 | 0.908 | 0.910 |
| α-diverse ensemble v2 (per-class W) | `two_stage_stageB_ensemble_v2.ipynb` | 0.9030 | 0.6733 | 0.6330 | 0.6302 | 0.922 | 0.903 | 0.906 |
| Cross-fitted ensemble v3 | `two_stage_stageB_ensemble_v3.ipynb` | 0.9073 | 0.6701 | 0.6475 | 0.6339 | 0.925 | 0.906 | 0.910 |
| v4-hybrid (W × per-instance LCA) | `two_stage_stageB_dynamic_v4.ipynb` | 0.9082 | 0.688 | 0.639 | 0.6295 | 0.925 | 0.908 | 0.911 |
| v4-dynamic (per-instance only) | `two_stage_stageB_dynamic_v4.ipynb` | 0.9085 | — | — | 0.6262 | — | — | — |
| v5 (decision-space dynamic) | `two_stage_stageB_dynamic_v5.ipynb` | 0.9089 | 0.673 | 0.638 | 0.6363 | 0.922 | 0.909 | 0.911 |

### DERIVED features (44 + 31 engineered = 75)

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| Single XGB α=0.25, DERIVED | `feature_engineering.ipynb` | 0.9117 | — | — | 0.6258 | — | — | — |
| **v6 (DERIVED + dynamic combiner)** | `final_model_v6.ipynb` | **0.9123** | 0.693 | 0.641 | **0.6485** | 0.924 | 0.912 | 0.914 |
| v6 replication (control arm) | `results/cic_unsw_svc/` | 0.9127 | 0.6937 | 0.6380 | 0.6478 | 0.924 | 0.913 | 0.914 |

### Service-identity features (DERIVED + 26 svc = 101)

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| **v6 + svc (DERIVED_SVC)** | `results/cic_unsw_svc/` | **0.9486** | 0.7706 | 0.7522 | **0.7460** | 0.952 | 0.949 | 0.949 |

Paired gain over the same-run control: **+0.0982 [+0.0885, +0.1081], p < 0.001** — the largest
single result in the project. Quote as `0.746 [0.727, 0.763]`.

---

## 2. Full release CICFlowMeter_out.csv (3,540,241 rows, Benign 97.5%; test = 521,361)

> ⚠ **Not level-comparable with Section 1** — same traffic, 12× the Benign base rate. This is the
> realism stress test.

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| v7 (svc features, native ports) | `final_model_v7_full.ipynb` | 0.9799 | 0.6568 | 0.7329 | 0.6601 | 0.988 | 0.980 | 0.983 |
| v7.1 (+ Stage-B reject class) | `final_model_v7_1_reject.ipynb` | 0.9875 | 0.7164 | 0.6744 | 0.6843 | 0.987 | 0.988 | 0.987 |
| **v7.1b (+ end-to-end W objective)** | `final_model_v7_1_reject.ipynb` §7 / `results/v71_w_e2e/` | **0.9866** | **0.7111** | **0.6874** | **0.6874** | 0.988 | 0.987 | 0.987 |

Ladder: 0.6601 → 0.6843 (paired +0.0242, p < 0.001) → 0.6874 (paired +0.0032, p < 0.001).
Final quote: **`0.687 [0.671, 0.704]`**. v7.1b's accuracy is *lower* than v7.1's by design — it
trades benign false alarms for Fuzzers recall, which macro-F1 rewards and accuracy penalises.
Note how the weighted columns sit at ~0.98–0.99 for all three rows while macro-F1 spans
0.66–0.69 — the weighted metrics are Benign; the macro numbers are the model.

---

## 3. Other datasets (external checks — never compare levels to Sections 1–2)

| Model | Notebook | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 |
|---|---|---|---|---|---|---|---|---|
| v6 on original UNSW-NB15 | `final_model_v6_unsw.ipynb` | 0.8303 | 0.724 | 0.671 | 0.6387 | 0.860 | 0.830 | 0.838 |
| v6 on CIC-IDS2017 (13-class) ⚠ | `results/cicids2017_v6/` | 0.9991 | 0.977 | 0.992 | 0.9835 | 0.999 | 0.999 | 0.999 |

⚠ CIC-IDS2017: e_1NN = 0.0026 — the dataset is nearly 1-NN-separable after dedupe; the number is a
property of the dataset, not the model (`CLAUDE.md`, "0.9835 is a RED FLAG"). UNSW-NB15: different
extractor and Benign share (52.7%); used to confirm the representation claim, not as a benchmark.

---

*TTL features were evaluated and rejected as a testbed label proxy (`results/ttl_join/`) — no model
above uses them.*
