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

## Addendum 2026-08-18 — combiner axis closed; one open candidate

Two runs added after the above (`results/v71d_region_w/`, `results/v71e_view_pool/`; full write-ups in
`results_summary.md`). Neither changes the headline.

1. **Per-region W (v7.1d)** — the missing rung between global per-class `W` and per-instance
   competence. **+0.0018 [−0.0034, +0.0076], p ≈ 0.550.** Regional specialisation is genuinely there
   (4 distinct locally-best members at K=8; regions recover the Fuzzers sink and the Exploits/DoS/Recon
   triangle unsupervised) and it fits in-sample (+0.0076 on COMB) but does not transfer. **Granularity
   beyond per-class is now closed at both ends.**
2. **Disjoint feature-view pool (v7.1e)** — built to remove v7.1d's suspected mechanism. It did:
   shared competence variance 93.4% → **51.2%**, median within-row weight spread 0.0000 → **0.2956**,
   region acc-vs-spread ρ **−0.857 → +0.071**. **Dynamism still did not pay** (hybrid −0.0003,
   pure-dynamic −0.0149 on the UNION pool). **Real mechanism: competence is local *accuracy*, so it
   minimises error rate while the objective is macro-F1** — every pure-dynamic variant has the highest
   accuracy and macro-precision and the lowest macro-recall in the table.
3. **ONE OPEN CANDIDATE — the UNION pool (12 members).** Test **0.6955 [0.6788, 0.7112]**, paired
   **+0.0066 [−0.0017, +0.0144], p ≈ 0.105.** Point-positive, not established. The gain is the *pool*
   (ALPHA static 0.6881 → UNION static 0.6976 = +0.0095), not the combiner. Costs 12 members at
   inference instead of 7. **SELV has now carried selection for v7.1, v7.1b, v7.1c, v7.2, v7.1d and
   v7.1e** — it is worn, so do not settle this with another selection pass on it. The clean routes are
   the multi-seed run (item 3 below) or leaving it as a reported ablation.
4. **TON-IoT audit is built and validated, waiting on the data** (`results/toniot_audit/`). Run it
   *before* any modelling — it is a gate, not an experiment, and fits nothing. Drop the network CSV in
   `Resource/TON-IoT/` and run it; ~minutes. It answers (a) does any single column solve
   attack-vs-benign alone (the `sttl` failure mode — TON-IoT ships `src_ip`/`dst_ip`/`ts`), and
   (b) what is `e_1NN`, i.e. can the architecture matter at all. The fingerprint rule is scored
   **out of sample** with a ≥20-row support floor; validated on a synthetic file with a planted
   `src_ip` leak, where it flags `src_ip` (recall 1.000) and correctly declines to flag `ts` and
   `src_port` (in-sample 0.95, out-of-sample 0.00 — cardinality, not fingerprint). **TON-IoT is
   genuinely different traffic, so it is the right target for the single-dataset gap** — but the
   chronological split (item 1) is cheaper and uses data already in hand.

---

## Addendum 2026-08-19 — robustness opened as a new axis; TON-IoT audited

Full write-ups in `results_summary.md`; condensed in `CLAUDE.md`. **The headline model and score are
unchanged: v7.1b, `0.687 [0.671, 0.704]` / acc 0.9866.**

### What was established

1. **The model is BRITTLE** (`results/v73_adv_budget/`). Undirected *random* noise at 0.002 IQR — no
   model access at all — costs **0.13 macro-F1**. Adversarial search only doubles it. This is broader
   than evasion: it implies degradation under measurement noise or a different CICFlowMeter build.
2. **Brittleness is MODEL-bound, not representation-bound** (`v73b_brittle_diag/`). Seven families at
   matched fpr: RF retains **0.93** of clean recall at budget 0.20 where the deployed XGB gate retains
   **0.46**. Survives the integer-rounding validity check (+0.0085).
3. **The gate is the lever, not Stage B** (`v73c_rf_gate/`). Gate recall under noise: XGB 0.9862 →
   0.4686, RF 0.9823 → 0.9688. Swapping Stage B buys nothing and costs clean performance.
4. **The RF gate swap is established both ways** (`v73d_rf_gate_sig/`): clean **−0.0038
   [−0.0064, −0.0013], p=0.005**; **+0.0089 → +0.0563** under noise, established at all six budgets.
   A cheap trade, not a free win — report as an operating choice.
5. **The standard adversarial experiment overstates damage by 0.10–0.13 macro-F1** by editing evidence
   no attacker owns. That is the publishable methodological result from this series.

### Two corrections that change how future numbers are reported

- **⚠️ ±0.012 is not a constant.** Measured paired CI for the RF-vs-XGB gate comparison is ≈**±0.0025**,
  because the arms share everything except the gate. **Always compute the paired interval for the
  specific comparison.** Quoting ±0.012 from memory would have hidden a real effect.
- **⚠️ Never compare robustness across model families without matching fpr first.** At the default 0.5
  cut the answer inverted completely (RF looked worst at 0.392 retained; matched, it is best at 1.009).

### Open items from this session

- ✅ **DONE 2026-08-19 — RF under a *targeted* white-box attack** (`results/v73e_rf_whitebox/`,
  8.3 min + 1.1 min verification). Pre-registered **H1 supported (+0.1830)**: RF retains 0.2143 of
  clean gate recall at budget 0.20 vs XGB's 0.0313, so the recommendation stands — **but RF loses 79%
  of its detection, so it is "more robust", not "robust".** Two findings came out of it that matter
  more than the verdict: **surrogate inflation +0.6773** (an XGB-crafted attack leaves RF at 0.8916 —
  four-fifths of v7.3c/d's apparent RF advantage is the attack having been aimed elsewhere), and **the
  detection/attribution ladder INVERTS with the threat model** (noise breaks attribution first,
  targeted attack breaks detection first, because the attack optimises against the gate). Both are
  now in `novelty.md` §0.
- **Confound not removed:** reject set / pool / `W_e2e` / competence reference all derive from the
  cached **XGB** OOF gate, so arm B is scored on a population its reject class never saw. Works against
  RF, so current results are conservative; a full rebuild around RF is ~30 min of cross-fitting.
- **Family attribution stays fragile** under noise for *both* model families. Unsolved; the honest
  framing is "you get the alert, you cannot trust the label."
- **`final_model_v7_1_reject_stages.ipynb` is not standalone.** Cell 32 hard-asserts
  `cache_oof_v7_full.npz` (from `final_model_v7_full.ipynb`) for the §6.1 baseline arm. Relaxing that
  assert into a conditional — skip the baseline arm if absent — was started and **not finished**.
  Also worth adding: a summary-first cell after cell 0 showing the ladder `0.6601 → +0.0242 → +0.0032`
  with p-values, so the contribution is visible before the 40% of code that is worth ~0.0007 at scale.
- **TON-IoT is audited and usable, but not as a benchmark** (`results/toniot_audit/`). `src_ip` fails
  the leakage gate; attack-only e_1NN 0.0306 puts it in the CIC-IDS2017 "too easy" regime. Best use is
  a **pre-registered prediction test** — write down in advance that all combiner variants tie within
  ~0.005, then run it. The `src_ip` finding stands alone for the literature-audit chapter.
- **Citation to verify:** the Sarhan / Layeghy / Portmann TTL "hidden label" quote comes from search
  snippets, not a retrieved PDF. It is load-bearing — get the exact reference. Also chase
  [Zoghi & Serpen 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.8242) on UNSW-NB15 overlap
  (403 on the open web) through the university library and position against it explicitly.

### Novelty directions ranked (from the 2026-08-18 literature search)

1. **Protocol price list** — reverse each methodological choice one at a time (no dedupe, curated file,
   include `sttl`, select on test, report accuracy, binary) and measure what each shortcut is worth.
   Produces a waterfall from 0.687 to ~0.99. **Nobody has priced the shortcuts**; the pieces already
   exist in `results/unsw_dedupe_ablation/`, `results/ttl_join/` and the curated-vs-full comparison.
   This converts the low headline from a liability into the finding.
2. **Quantify the TTL artifact** (half a day) — one paired run with `sttl` added. Others *assert* it is
   a hidden label; nobody has published what it is worth. Frame strictly as artifact measurement.
3. **Chronological split** (one run, ~45 min) — still the biggest single credibility item; timestamps
   and the +14400 s offset are already verified.
4. **SHAP** — the last undelivered *planned* item, but **not a novelty** (CyberAdaptAI has it; see
   item 2 below). Run it as analysis. Targets: what the reject class keys on, why `svc_rpc`
   carries 0.372 importance, and the overlap pairs service identity did *not* fix.
5. **Open-set / leave-one-family-out** (~2h, 9 runs) — no reviewed paper evaluates zero-day
   generalisation, and the reject class is already the abstention machinery.
6. **Statistical power analysis** (an afternoon) — how large a test set is needed to resolve +0.005?
   Makes every four-decimal claim in the related work legible as a measurement error. The ±0.0025
   finding above is the seed for it.

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

### 2. SHAP — a diagnostic instrument, NOT a novelty claim

> ⚠️ **CORRECTED 2026-08-19.** This item previously read *"the last outstanding novelty-stack item —
> missing XAI is a named literature gap"*. **That is wrong.** CyberAdaptAI already applies SHAP to a
> dynamic NIDS ensemble (§3.5, Eq 9, Table 7); the corpus scan gives **1 of 10 implementing XAI, 2
> naming it as future work only**. **"We add explainability" is not available as a contribution.**
> What *is* original is that CyberAdaptAI's SHAP table fails SHAP's defining property while our
> reimplementation executes the same equation at additivity residual 3.8e-06 — the gap is
> **unverifiable** explainability, not missing explainability. See `novelty.md` §3.2.

Still worth running, as analysis rather than contribution — it goes in the analysis chapter.
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
| per-region W (any K) | +0.0018, p≈0.55; fits in-sample, does not transfer; `results/v71d_region_w/` |
| dynamic / competence weighting, ANY pool | closed mechanistically: competence = local accuracy = wrong objective; `results/v71e_view_pool/` |
| per-instance competence estimators (OLA/LCA/META-DES-class) | closed — the objective mismatch predicts failure without running them |

Standing methodological rules: don't justify a pool by its oracle bound; asymmetric P/R is not a
bad threshold; SELV/test rank shuffles inside ±0.012 are resolution failure, not selection error;
**probe gains must survive the full pipeline before being believed** (v7.2's lesson).

---

## Recommended order

1. **Chronological split** (item 1) — one run, biggest remaining credibility item.
2. **SHAP** (item 2) — required regardless of what it finds, but as *analysis*, not novelty.
3. **Multi-seed** (item 3) — pure execution, ~2 h unattended.
4. TTL quantification (item 4) only if time allows.
5. **Write.** The contribution is delivered; the chapters are waiting.
