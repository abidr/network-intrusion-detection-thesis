# Next Steps — after the service-identity / v7 campaign

Updated **2026-08-09**, end of session. Previous version (2026-08-07) predates everything below;
its Tier-1/Tier-2 items are now all resolved. Companion to `results_summary.md` / `CLAUDE.md`
(what has been **done**) and `results_short.md` (the metrics table).

---

## What changed on 2026-08-09 (one screen)

1. **`CICFlowMeter_out.csv` obtained** (full official release, 3.54M rows) — carries the Flow ID /
   IPs / **ports** / protocol / **timestamps** that `Data.csv` stripped. The old "host/temporal
   context is impossible on this file" claim is **obsolete**; `Data.csv` verified as a verbatim
   subset (all attacks kept, Benign ~10× downsampled); the 0–9 label mapping is now file-verified.
2. **Service identity broke the plateau** (`results/cic_unsw_svc/`): DERIVED+svc **0.7460** on the
   curated file, paired **+0.0982, p<0.001** — the largest gain in the project. The UNSW-extractor
   prediction transferred almost quantitatively (Shellcode 0.658 vs predicted 0.650).
3. **v7 generation on the full release** (Benign 97.5%, not level-comparable to the curated file):
   v7 0.6601 → v7.1 reject class 0.6843 (+0.0242, p<0.001) → v7.1b e2e-objective W **0.6874**
   (+0.0032, p<0.001). **Headline: `0.687 [0.671, 0.704]` / acc 0.9866.**
4. **TTL ruled out as a feature** (`results/ttl_join/`): join succeeded at 100% coverage, but sttl
   is a measured **attacker-host label proxy** (benign 97% sttl∈{31,29} at 0.0000 attack share;
   attacks ≥98% sttl∈{62,254}). The UNSW "ask for TTL" recommendation is withdrawn.
5. **Hyperparameters null** (`results/v72_hparam/`): 40-config sweep, probe cleared the bar but the
   pipeline did not confirm (+0.0054 [−0.0037, +0.0136], p≈0.23). Defaults kept.
6. **Reject-set enrichment null** (`results/v71c_reject_enrich/`): +0.0009, p≈0.82 — only 5,240
   benign rows live in p∈(0.2,0.5]; boundary estimation is saturated.

**The in-scope lever list is now EMPTY.** Established: α=0.25 weighting, DERIVED features, service
identity, hierarchy + reject class, e2e-objective W. Everything else: tested null or measured
artifact. The only remaining score path is **payload from PCAPs** → future-work chapter.

---

## Current headline numbers (two settings — never compare across)

| Setting | Model | macro-F1 | Accuracy |
|---|---|---|---|
| Curated file (Benign 80%) | v6+svc, `results/cic_unsw_svc/` | **0.746 [0.727, 0.763]** | 0.9486 |
| Full release (Benign 97.5%) | v7.1b, `final_model_v7_1_reject.ipynb` §7 | **0.687 [0.671, 0.704]** | 0.9866 |

Reporting rules unchanged: marginal CI ≈ ±0.017 for one config, paired ≈ ±0.012 for comparisons,
quote SELV-selected numbers never test-max, rare classes as intervals (Worms ±0.13, Analysis wide).

---

## Open items, ranked

### 1. Chronological split — NOW POSSIBLE, run it first

Timestamps are sitting unused in `CICFlowMeter_out.csv` (offset vs epoch verified: +14400 s,
`results/ttl_join/`). This is the evaluation-realism gap **all 10 reviewed papers** share.
Two cuts, both on the v7.1b architecture:
- **Session split**: train on the 22 Jan 2015 capture, test on 17 Feb — genuine cross-campaign.
- Fallback/secondary: time-ordered 80/20 within the whole capture.

Design constraints decided in advance: no stratification is possible — print the per-family test
support table *before* any metric and floor tiny families out of the headline macro; reject set,
combiner reference and all fitting from training-period rows only; levels are NOT comparable to the
random split — report the paired gap of same-pipeline-both-ways. **Informative in both directions**
(hold = certifies the headline; drop = measures the realism gap). Existing caches do not apply —
full re-run, ~30–45 min.

### 2. SHAP — the last outstanding novelty-stack item

Missing XAI is a named literature gap and the only undelivered element of the planned contribution.
Targets are sharper now than when this was first listed:
- **What does the reject class key on**, and why does it swallow Fuzzers specifically?
- **Why is `svc_rpc` worth 0.372–0.418 of importance** (Recon/port-111 concentration) — and what
  does that say about testbed service-targeting (feeds the honesty section)?
- The surviving overlap pairs: Exploits↔Recon (515 confusions), DoS↔Exploits — the ones service
  identity did **not** fix because those attacks target the same services.
Not rare classes generically (their CIs cannot support per-class claims).

### 3. Multi-seed variance — ⏳ UNFINISHED (1 of 10 runs done)

`multiseed_variance.ipynb` is built and validated (seed 42/BASE reproduces 0.6157/0.6334/0.6363 to
four decimals); needs **~2 h of compute**, pins all 16 cores. Resume:

```
python -m jupyter nbconvert --to notebook --execute --inplace multiseed_variance.ipynb
```

Progress: `python -c "import pandas as pd; print(pd.read_csv('cache_multiseed.csv').to_string())"`.
Note its grid covers the **curated-file v6-era ladder** (BASE/DERIVED). That still matters — the
0.746 svc headline sits on that pipeline — but consider whether one v7.1b seed-repeat is worth
adding (~35 min/seed) so the full-release headline also gets an across-seed check.

### 4. Optional: TTL artifact quantification (half a day)

One paired run with sttl added would measure *how many macro-F1 points the testbed topology is
worth* — i.e., what sttl-using UNSW-NB15 papers gain from the artifact. Frame strictly as an audit
measurement (feeds the literature-audit chapter), never as a model improvement. Skip if time is
short; the fingerprint tables in `results/ttl_join/` already carry the qualitative claim.

### 5. Writing — the thesis is no longer score-limited

The arc is complete and unusually defensible: five established gains, twelve pre-registered nulls,
one measured artifact, a diagnosed plateau with an externally-confirmed missing ingredient that was
then cashed in (+0.098). Key framings already worked out (see CLAUDE.md sections): moderate-not-full
rebalancing; architecture value is conditional on problem hardness (three demonstrations); fit the
combiner on the deployment objective; the Fuzzers↔Benign boundary as a SOC operating choice;
near-100% literature numbers vs the dedupe/leakage/TTL mechanics this project measured.

---

## Do not re-open (extended)

Everything from the 2026-08-07 table still stands (balanced weighting, full/capped SMOTE, focal,
DES/algorithm pools, per-class thresholds, soft routing, gate threshold+weighting, α>1.0, label
noise, per-instance competence incl. META-DES-class, hybrid, D2 algebra, bracket-chasing
transforms). Added this session:

| Lever | Verdict |
|---|---|
| TTL / sttl features | label proxy — equivalent to source IP; `results/ttl_join/` |
| XGB hyperparameters | 40-config sweep; probe +0.0079 did not survive the pipeline (p≈0.23) |
| reject-set enrichment | +0.0009, p≈0.82; only 5,240 rows in p∈(0.2,0.5] — estimation saturated |
| gate threshold at real base rate | redundant once the reject class exists (v7.1: t=0.7 ≡ t=0.5) |
| ensembling at full scale / easy regimes | v7: all 21 variants within 0.004 of single XGB; also CIC-IDS2017. Value is conditional on hardness |
| block-structured Stage B (old Tier 1.3) | **obsolete before testing** — svc features dissolved the block geometry it was designed around (Recon 0.83, Shellcode 0.72); the surviving confusions are within-service, which a block gate cannot separate either |

Standing methodological rules: don't justify a pool by its oracle bound; asymmetric P/R is not a
bad threshold; SELV/test rank shuffles inside ±0.012 are resolution failure, not selection error;
**probe gains must survive the full pipeline before being believed** (v7.2's lesson).

---

## Recommended order

1. **Chronological split** (item 1) — one run, biggest remaining credibility item.
2. **SHAP** (item 2) — required regardless of what it finds.
3. **Multi-seed** (item 3) — pure execution, ~2 h unattended.
4. TTL quantification (item 4) only if time allows.
5. **Write.** The contribution is delivered; the chapters are waiting.
