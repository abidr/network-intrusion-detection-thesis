# v7.3j — is v7.3 brittleness a `repair()` artifact? — MOSTLY NO; 19% of the attribution damage is

Run 2026-10-20. `repair_confound.py` → `repair_confound.txt` / `.json`. 1.3 min. Found while building
the XAI notebook's noise-stability step.

## The confound

Every v7.3 script compares a **budget-0 arm that uses the original flows** against **budget > 0 arms
that pass through `repair()`**. `repair()` recomputes CICFlowMeter's dependent columns (rates,
Down/Up Ratio, Packet Length Mean) from the counts — and on *unperturbed* flows it disagrees with
CICFlowMeter's own values (first 200k rows: `Down/Up Ratio` > 50% off on 18.7% of rows; `Packet
Length Mean` > 1% off on 43.7%). Five repaired columns are model inputs: `Down/Up Ratio`, `Flow
Packets/s`, `Bwd Packets/s`, `Packet Length Mean`, `Flow Bytes/s`. So every noise arm differed from
its clean arm by noise **plus** repair.

## Result (v7.3f setup: 800k-row XGB gate at matched benign fpr 0.0168; XGB α=0.25 9-family model)

| arm, 16,298 test attack rows | gate recall | family acc | family macro-F1 |
|---|---|---|---|
| ORIGINAL (= every v7.3 clean arm) | 0.9862 | 0.8746 | 0.7495 |
| **REPAIR ONLY** (budget 0 through repair) | **0.9861** | 0.8743 | **0.7311** |
| NOISE+REPAIR @0.002 (= the v7.3 arm) | 0.9418 | 0.8451 | 0.6516 |
| NOISE ONLY @0.002 (no repair) | 0.9384 | 0.8264 | 0.6258 |

- **Detection brittleness is untouched by the confound: repair alone explains 0% of the gate-recall
  drop** (0.0001 of 0.0444).
- **Attribution brittleness is ~19% overstated**: repair alone costs 0.018 of the 0.098 family
  macro-F1 drop at 0.002 IQR. The noise effect proper is ~0.080.
- Repair is not what makes the model look fragile — **noise without repair hurts *more*** (0.626 vs
  0.652), because unrepaired noise leaves physically inconsistent flows.
- Interpretation of the 19%: the model has learned CICFlowMeter's own way of computing those
  columns (e.g. integer-truncated Down/Up Ratio); a faithful recomputation is itself a small
  distribution shift — a mild instance of the extractor-version sensitivity v7.3 warned about.

## What to change

- **v7.3 claims stand for detection.** For attribution, quote the noise-only share or add the caveat:
  "≈19% of the macro-F1 drop at 0.002 IQR is the recomputation of dependent columns, not the noise".
- Future perturbation runs should use **REPAIR ONLY as the clean reference** (as
  `final_model_v7_1_xai.ipynb` §5.10 now does), so the comparison isolates noise.
- Measured at 0.002 IQR only, where the share is largest; at larger budgets the noise term dominates
  and the fixed repair offset shrinks as a fraction.
