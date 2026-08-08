# Next Steps — where the remaining improvement is

Written 2026-08-07, after `final_model.ipynb` was consolidated (test macro-F1 **0.6339**, or
`0.634 [0.617, 0.651]` at the reporting yardstick).

Companion to `results_summary.md`, which records what has been **done**. This file records what is
**left**, ranked by expected value, with the evidence for each ranking. Read `results_summary.md`
§"Ablation + bootstrap significance" before acting on anything here.

---

## The constraint everything below is measured against

Two facts bound the remaining work:

1. **Irreducible error.** The k-NN overlap analysis gave a Cover–Hart bracket of
   **e\* ∈ [0.213, 0.376]** for the 9-family sub-problem → accuracy ceiling [0.625, 0.787]. The final
   model's Stage-B accuracy is **0.737**, inside that bracket near its top.
2. **Resolution.** The paired bootstrap noise band is **±0.012** macro-F1. Any lever that returns less
   than that cannot be claimed, no matter how clean the idea.

**Six** modelling levers (focal weighting, dynamic ensemble selection, per-class thresholds, wider α
grid, per-instance competence over an α-diverse pool, per-class × per-instance hybrid) have now
returned ~nothing, because all six operate *downstream* of a constraint that lives in the features.

The v4 result is the sharpest evidence for this. It gave the competence estimator everything v1 lacked
— the right diversity axis and 5× the reference data — and the per-instance weights **still** sat
within 4.5% of uniform. Local neighbourhoods in the 44-feature space are too impure to say which model
to trust where. That is a statement about the representation, not about combiners.

**The loophole:** the Cover–Hart bracket was measured **in the 44-feature RobustScaler space**. It
bounds *this representation*, not the problem. Changing the representation is the only lever that moves
the bound rather than approaching it. That is why Tier 1 is what it is.

---

## Tier 1 — can actually move macro-F1

### 1.1 Un-prune the correlation drop — do this first (~30 min)

The unsupervised prune took 67 non-constant features → 44 by dropping |r| ≥ 0.95 pairs. **Correlated is
not redundant for tree ensembles** when the redundancy is only near-linear: a split on the "duplicate"
column can still cut the space differently.

- Re-run the α=0.25 Stage-B on all 67 non-constant columns, everything else identical.
- Re-measure the **1-NN error in the 67-D space** — this is the real point of the experiment. If the
  bracket tightens, the ceiling is representation-bound and Tier 1.2 is justified. If it does not move,
  the prune is independently validated a second time (the label-ambiguity notebook already showed the
  prune creates zero new ambiguity) and that is a clean methodology result for the write-up.
- Cheap, decisive either way, and it calibrates how much to invest in 1.2.

### 1.2 Derived features — highest expected value

CICFlowMeter emits raw counts and durations; trees cannot express `a/b` without deep, sample-hungry
splits. Candidates, all constructible from the existing 44:

- forward/backward **ratios** (byte ratio, packet ratio, segment-size ratio) — the asymmetry, not the
  magnitudes
- **bytes per packet**, per direction
- **duration-normalised rates** where only the raw totals are present
- **IAT coefficient of variation** (std/mean) — burstiness independent of scale
- **flag counts as fractions** of total packets rather than absolute counts

Rationale: if Analysis is 51.4% Exploits *by neighbourhood* and Shellcode 46.5% Fuzzers, a feature that
separates those geometrically is worth more than any classifier change. Feed 1.3 (SHAP) into this — SHAP
on a confused pair names the columns whose interaction is doing the separating work, which is where the
ratio should be built.

Evaluate on the **same** leakage-safe protocol; a new feature set does not license a new split.

### 1.3 Block-structured Stage B — the one untested architectural idea

The k-NN analysis found the confusion matrix is block-structured. Route **block first, family within
block second**:

- {Exploits, DoS, Analysis}
- {Fuzzers, Reconnaissance, Shellcode}
- {Generic, Backdoor, Worms}

This is the direct extension of the project's only significant architectural win — and it must carry
**α-tempered weighting at every level**, because the ablation showed the hierarchy is an *interaction*
effect (+0.0178 at α=0.25, −0.0061 under `balanced`, n.s.). A block hierarchy under `balanced` would
likely reproduce the negative result.

**Risk, stated up front:** if the blocks are internally inseparable this relocates the error rather than
removing it — the block gate will be accurate and the within-block classifiers will absorb the full
confusion. That outcome is still reportable (it localises the difficulty), but do not expect a gain
merely because the two-stage hierarchy worked at the Benign/Attack boundary; the gate worked there
because it was nearly *solved* (attack recall ≈ 0.997), and no block boundary looks that clean.

---

## Tier 2 — cheap, low expected gain, closes a hole

### 2.1 Hyperparameter search — never done

`n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9` have been
hand-fixed since the first notebook. There is **no `GridSearchCV` / `RandomizedSearchCV` / Optuna
anywhere in the repository**.

The project record predicts this returns little ("gains come from weighting or data, never model
capacity" — DT→RF→XGB moved macro-F1 only ~5 points). But the current state cannot support the sentence
*"we searched the capacity axis and it is not the bottleneck"*, and that is a hole an examiner will find.
Run it once, report the negative result, move on.

**Constraint:** `SEL` is spent (used for both architecture selection and thresholds). Any new tuning
needs a fresh split carved from `train`, or cross-fitting as in v3.

---

## Tier 3 — improves the thesis rather than the number

This is where the time is best spent. The score is near a measured floor; the *contribution* is not.

### 3.1 SHAP — the last outstanding item in the novelty stack

Missing XAI is one of the four literature gaps the review named, and it is the only element of the
planned contribution not yet delivered. Target it at the **overlap pairs** — Exploits↔Reconnaissance
(451 confusions, zero exact ambiguity), DoS↔Exploits (442, zero), Fuzzers↔Shellcode — **not** at rare
classes generically. Two reasons: those pairs are where the error actually is, and the per-class
intervals on the rare families (Worms ±0.13) cannot support a per-class explanatory claim anyway.

Feeds directly into 1.2.

### 3.2 Multi-seed / repeated-split variance

Current confidence intervals capture **test-set sampling only** — `results_summary.md` already flags
them as optimistic. Five seeds of the entire split→train→evaluate pipeline converts that caveat into a
measurement, and is the single cheapest credibility gain available.

### 3.3 Cross-dataset evaluation

"Single-dataset evaluation" is another named gap from the review. Original UNSW-NB15 or CIC-IDS2017.
Will not improve the number on this test set; addresses the gap directly.

---

## Do not re-open

Closed with measurements, recorded in `results_summary.md`:

| Lever | Verdict |
|---|---|
| `class_weight='balanced'` | ranked 12th of 14, below no weighting; use α ≈ 0.25 |
| full SMOTE | macro-F1 0.5807 → 0.5088 |
| capped SMOTE | superseded by weighting (0.6123 vs 0.5988) and cheaper |
| focal / dynamic weighting | γ=2 collapses the model (0.3497) |
| DES / algorithm-diverse pool | did not transfer to test (−0.0007) |
| per-class thresholds | algebraically redundant with `W`; verified on 100.0000% of rows |
| soft routing | +0.0000 |
| gate-threshold tuning | gate is already optimally calibrated (`t_0 = 1.0`) |
| α > 1.0 | axis saturates; 10-member pool scored below the 7-member subset |
| "labels are noisy" | refuted — only 0.35% ambiguous rows |
| **per-instance competence (v4)** | **weights sit within 4.5% of uniform even with the α-diverse pool and 5× reference data — both of v1's excuses eliminated** |
| **per-class × per-instance hybrid (v4)** | **ties v3 (−0.0039, inside noise); the per-class matrix does the work, the dynamic term is inert** |

Three standing methodological rules from those failures: **do not justify a pool by its oracle bound**
(headroom rose +55% v2→v3 for +0.004 realised; v4 had 0.1579 and realised 0), **asymmetric per-class
P/R is not evidence of a bad threshold**, and **do not read a SELV/test rank change as a wrong pick
when the candidates sit inside the ±0.012 band** — that is resolution failure, not selection error.

> **Note on v4 and the thesis framing.** v4 is retained as the **dynamic-ensemble contribution**
> (`v4-hybrid` 0.630, `v4-dynamic` 0.626) even though neither beats v3 — see `results_summary.md`. It is
> closed as a *lever for improving the score*, not as a deliverable. Do not re-run it hoping for a
> different number.

---

## Recommended order

_Updated 2026-08-08, after v4._

1. **3.1 (SHAP)** — now first. It is the **last outstanding item in the novelty stack** and is required
   regardless of what it finds, which is no longer true of anything score-related: six levers have
   returned nothing and the score sits at a measured floor. v4 also makes it more informative — there
   are now seven members whose per-class weights specialise, so SHAP can explain *why* α=0.0 owns
   Shellcode while α=1.0 owns DoS.
2. **1.1** (67-feature ablation) — 30-minute diagnostic; decides whether 1.2 is worth funding. Decisive
   either way: if the Cover–Hart bracket tightens, the ceiling is representation-bound; if it does not
   move, the prune is independently validated a second time.
3. **1.2** (derived features) — driven by what 3.1 surfaces. The only lever that moves the bound rather
   than approaching it.
4. Then 1.3 / 2.1 / 3.2 as time allows.

SHAP moved ahead of 1.1 because the thesis now needs *contribution*, not score. The dynamic-ensemble
contribution is delivered (v4); XAI is the one named literature gap still open.
