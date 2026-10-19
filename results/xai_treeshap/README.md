# XAI — verifiable TreeSHAP of the deployed v7.1b pipeline — 6 of 7 pre-registered predictions SUPPORTED

Run 2026-10-20. `xai_treeshap.py` → `xai_treeshap.txt` / `.json`; per-row values in `xai_phi.npz`
(git-ignored). 6.0 min. Env: shap 0.51.0, xgboost 3.3.0, numpy 2.2.6.

**Framing (`novelty.md` §3.2): SHAP is not the contribution — CyberAdaptAI already has it. The
contribution is explanations that carry their own verification**, plus what they reveal about the
mechanisms the rest of the project measured.

## What is explained

The **deployed system itself**, not a surrogate. Refit reproduces v7.1b exactly: **test macro-F1
0.6874 / acc 0.9866** (W_e2e COMB 0.6906, identical to the v7.1b log).

| level | what | space | how verified |
|---|---|---|---|
| 1 | every XGBoost model (gate + 5 XGB Stage-B members), per column | log-odds | additivity + 2nd implementation |
| 2 | the **fused 7-member score** incl. RF/ET, per feature group {SVC 25, TIME 25, SIZE 39, FLAG 10} | probability | exact (all 16 coalitions), efficiency + decision reproduction |

Level 2 uses Shapley linearity: `S_c(x) = Σ_m coef_mc(x)·P_m(c|x)` ⇒ `φ_g(S_c) = Σ_m coef_mc(x)·φ_g(P_m(c))`.
**Scope:** `coef` holds the per-instance competence routing fixed at its value for `x`. Interventional
background = 40 Stage-B training rows, 4 per head class. Explain set = 2,643 gate-positive test rows:
1,603 stratified (≤200/class) + confusion-enriched rows for the overlap pairs.

## Verification (P1) — SUPPORTED, all three parts

- **Additivity** on all 6 XGBoost models: max |Σφ + bias − margin| **≤ 2.0e-05** (float32 over 400 trees).
- **Two independent TreeSHAP implementations agree to 0.0**: shap 0.51 `TreeExplainer` vs xgboost
  `pred_contribs`, gate and XGB α=0.25.
- **Group Shapley efficiency 5.5e-09**; the full coalition reproduces the deployed Stage-B decision on
  **100.0000%** of rows. (Algebraically guaranteed — a code-correctness check, not evidence.)

Contrast for the thesis: CyberAdaptAI's Table 7 cannot be SHAP output (columns sum to exactly 1.00);
here every number has a published residual.

## ⭐ Finding 1 — gain importance misstates where the evidence is, at both stages

| share of … | SVC | TIME | SIZE | FLAG |
|---|---|---|---|---|
| columns | 0.253 | 0.253 | 0.394 | 0.101 |
| **gate SHAP** | 0.116 | 0.161 | **0.708** | 0.016 |
| gate *gain* | 0.004 | 0.003 | **0.989** | 0.004 |
| **Stage-B ensemble SHAP** | **0.426** | 0.135 | 0.355 | 0.084 |
| XGB α=0.25 *gain* | **0.640** | 0.039 | 0.249 | 0.072 |

- **P2 SUPPORTED:** SVC leads Stage B; SIZE, not SVC, leads the gate — consistent with v7.3g (service
  identity does not say "malicious").
- **⚠️ The project's "svc = 70.4% of importance" is a gain figure.** On the deployed ensemble, by
  SHAP, svc is **42.6%** — still the largest group, but SIZE carries 35.5%. Gain says 98.9% SIZE at the
  gate; SHAP says 70.8%, with TIME 16.1% and SVC 11.6% (21.2% on attack rows). Column-level
  Spearman(SHAP, gain) is only **0.58** (gate) and **0.42** (α=0.25), top-10 overlap 5/10 each.
  **Quote SHAP shares, not gain shares, and restate the 70.4% as gain-based.**
- `dst_port_num` is the #1 SHAP column for Stage B and #3 for the gate; `frac_PSH` is top-5 by gain
  but not by SHAP.
- Per family the reliance differs sharply: Fuzzers 0.668 / Recon 0.581 / Shellcode 0.529 SVC vs
  Generic 0.593 / Backdoor 0.481 / Exploits 0.449 SIZE, and **Analysis 0.416 TIME** — the only family
  led by timing.

## ⭐ Finding 2 — the overlap pairs, mechanistically (P3 SUPPORTED, 4 of 6 non-SVC)

Contrast `D_g = φ_g(S_wrong) − φ_g(S_true)` on confused rows; controls are correctly classified rows of
the same true class (negative there, as they should be).

| confusion | n | driver | the group's push (mean D) | rows where it is top driver |
|---|---|---|---|---|
| Recon → Exploits | 216 | **SVC** | +0.017 | 52% |
| Exploits → Recon | 146 | SIZE | +0.018 | 58% |
| DoS → Exploits | 216 | SIZE | +0.029 | 48% |
| Exploits → DoS | 177 | SIZE | +0.027 | 39% |
| **Benign → Fuzzers** (the sink) | 262 | **SVC** | **+0.056** | 62% |
| **Fuzzers → REJECT** (swallowed) | 234 | **TIME** | +0.032 | **71%** |

- **The Exploits/DoS/Recon triangle is a SIZE problem** — byte/length profiles overlap.
- **The Fuzzers sink is explained in both directions, by different groups.** Benign leaks land in
  Fuzzers because they sit on **Fuzzers' services** (SVC, 3× any other group); real Fuzzers get
  swallowed by the reject class because of **timing** (TIME top on 71% of rows). This is the v7.1
  "the sink inverts, not dissolves" result given a mechanism: the two error directions run through
  different evidence, which is why no single lever fixed both.
- **Shellcode↔Fuzzers is no longer a pair: 0 confusions** in either direction in the deployed model
  (service identity dissolved it, as predicted in `cic_unsw_svc`). Analysis→Exploits: 2 rows. Neither
  evaluable — that absence is itself a result.
- ⚠️ P3 rests on 6 pairs, and SVC drives the two with the largest volume-weighted impact. Report the
  table, not just the 4/6 count.

## Finding 3 — explanations are faithful (P4 SUPPORTED, by a wide margin)

Deleting the top-k SHAP columns (to the train median) vs k random columns:

| | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| gate ratio | 23.3× | 18.0× | **11.4×** | 6.3× |
| XGB α=0.25 ratio | 32.0× | 19.4× | **12.2×** | 6.8× |

Threshold was 2×. Caveat: median replacement is off-manifold (derived columns are not recomputed);
the random-k arm has the same defect, so the ratio is the fair quantity.

## Finding 4 — explanations are brittle too (P5 REFUTED)

Under v7.3 random noise at **0.002 IQR**, on rows whose decision did **not** flip:

| | flip rate | median Spearman | median top-5 Jaccard |
|---|---|---|---|
| gate | 0.051 | **0.842** (pre-reg ≥ 0.90) | 0.667 (≥ 0.60 ✓) |
| XGB α=0.25 | 0.188 | **0.870** (≥ 0.90) | 0.667 (≥ 0.60 ✓) |

At 0.020 IQR: Spearman 0.736 / 0.769. Flipped rows: 0.59 / 0.70.

- **The prediction is unchanged but the explanation moved.** Typically 4 of the top-5 columns survive
  (Jaccard 0.667 = 4 shared of 6), while the full ranking degrades. This extends the v7.3 brittleness
  result from predictions to explanations: **an analyst reading a SHAP plot for an alert is reading
  something a 0.002-IQR measurement wobble can reorder.**
- ⚠️ **Partly an instrument effect — state it.** Spearman over all 99 columns includes many
  near-zero contributions whose ranks are noise. The top-5 Jaccard, which passed, is the more
  operationally relevant number. **Write "the top of the explanation is mostly stable; the full
  ranking is not"**, never "SHAP is unstable".

## Pre-registered verdicts

| | |
|---|---|
| P1a additivity < 1e-3 | SUPPORTED (2.0e-05) |
| P1b two implementations agree < 1e-4 | SUPPORTED (0.0) |
| P1c group efficiency + 100% reproduction | SUPPORTED |
| P2 SVC leads Stage B, not the gate | SUPPORTED |
| P3 non-SVC driver in majority of pairs | SUPPORTED (4/6) |
| P4 top-5 deletion ≥ 2× random | SUPPORTED (11–12×) |
| P5 stable where predictions hold | **REFUTED** (Spearman 0.84/0.87; Jaccard passes) |

## Not done / follow-ups

- RF and ET are explained only at group level (column TreeSHAP on 300 fully grown trees is too slow).
- SHAP under the v7.3f **misattribution** attack (does the explanation show what the attacker moved?)
  — the natural next robustness link; needs the greedy crafting loop.
- Figures: `../../figures/xai_fig1–5_*.png`, generated by `figures/make_xai_figures.py`.
