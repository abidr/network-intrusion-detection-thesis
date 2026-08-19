# v7.3f — the misattribution threat model: what does it cost to make a NIDS *lie*?

Run 2026-08-19. `misattribution.py` → `misattribution.txt` / `.json`. 15.8 min.

**Pre-registered H1 was REFUTED, in the opposite direction: misattribution costs ~3.8× MORE
than evasion, not 2× less. But both triviality guards passed, and the run surfaced an
unplanned finding — the reject class is an attack surface.**

## The pre-registered test

`b50` = budget at which an objective first reaches 50% success (interpolated).

| Objective | success metric | b50 |
|---|---|---|
| **EVASION** | final prediction is Benign | **0.0145 IQR** |
| **MISATTRIBUTION** | detected, wrong family | **0.0545 IQR** |
| ABSTENTION | gate fires, Stage B rejects | >0.20 (never reached) |
| RANDOM control | misattributed | >0.20 (never reached) |

**Price ratio `b50(EVASION)/b50(MISATTRIB)` = 0.27**, against an H1 threshold of >2.
**H0: the cheapest attack on this NIDS is still hiding from it, not lying to it.**

The hypothesis was that attribution, having collapsed under undirected noise while detection
held (v7.3c/e), would be cheaper to corrupt deliberately. **That inference was wrong.** Fragility
under undirected noise did not imply cheapness under direction — a reusable lesson, and the
third time in this project that a plausible mechanism failed to predict a realisable effect.

> ⚠️ **One caveat that softens but does not rescue H1.** `b50` is mildly biased against
> misattribution: in the MISATTRIB arm ~28% of rows end up abstained, capping achievable
> misattribution at ~0.72, so its 0.50 target is ~69% of still-attributable rows against
> evasion's clean 50%. The result (0.27 vs a threshold of 2) is far too large for this to
> flip the verdict, but quote the ratio as approximate.

## Both triviality guards PASSED — the threat model is real even though H1 failed

**(a) The 2×2 is not diagonal, and the off-diagonal is informative.** At budget 0.20 the
EVASION attack drives misattribution *down* to 0.0038 (from a 0.1122 clean baseline) — you
cannot mislabel a row you have deleted from the alert stream. The two objectives are genuinely
different attacks, not one attack with two names.

**(b) Direction buys a great deal.** Directed misattribution vs random noise at equal budget:

| budget | directed | random | delta |
|---|---|---|---|
| 0.002 | 0.1887 | 0.1196 | +0.069 |
| 0.020 | 0.3188 | 0.1389 | +0.180 |
| 0.200 | **0.6311** | 0.1497 | **+0.481** |

Random noise plateaus near 0.15 while the directed attack reaches **0.63**. So this is
**"attribution is cheap to target"**, not merely "attribution is fragile" — the guard that could
have reduced the finding to a restatement of v7.3c did not fire.

## What misattribution buys that evasion cannot: stealth

| | EVASION @0.20 | MISATTRIB @0.20 |
|---|---|---|
| gate recall | **0.0309** | **0.9969** |
| undetected | 0.9697 | 0.2571 |
| misattributed | 0.0038 | **0.6311** |
| macro-F1 | 0.1446 | 0.1348 |

The evasion attack is loud in the only way that matters: **97% of attack flows stop generating
alerts**, which any volume monitor sees. The misattribution attack keeps the gate firing at
**99.7%** — alert volume is *normal* — while **63% of the labels are wrong**. Nothing in the
alert count reveals it.

**Per unit of budget it is also the more efficient way to damage the reported metric.** At
budget 0.002 MISATTRIB puts macro-F1 at **0.4712** vs EVASION's 0.5494 and RANDOM's 0.6035,
while holding gate recall at 0.9955. It is the best attack on the *metric* at every budget, and
the worst attack on *alert volume* — which is exactly why an evasion-only literature cannot see it.

## ⭐ UNPLANNED FINDING: the reject class is an attack surface

The v7.1 reject class was added as a **defence** — it cut benign-as-attack false alarms 4×
(8,370 → 2,093) for +0.0242 macro-F1. Under attack it inverts.

In the MISATTRIB arm, `abstained` rises **0.0879 → 0.2238 at budget 0.002** and plateaus near
0.29. Crucially, `undetected` ≈ `abstained` in that arm at every budget (0.2571 vs 0.2541 at
0.20), meaning **essentially every "undetected" outcome there is a reject-class abstention, not a
gate miss** — the gate never fails (0.997), Stage B files the row as a false alarm.

The attacker was never told to do this. It was told only to minimise `P(true family)` while
staying gate-positive, and **it discovered the reject class as the cheapest route**. The
dedicated ABSTAIN arm confirms the channel is directly targetable and cheap — 0.2258 abstention
at budget **0.002** — though bounded (never reaches 50%, b50 > 0.20).

**Thesis framing: the reject class is a genuine defensive gain that creates a new attack surface.
Report both together, never the +0.0242 alone.** This is the same shape as the v7.1 Fuzzers
finding — the reject class lets you choose which side pays — now extended to the adversarial case.

## Honest scope

- **Attacker is white-box on ONE component** (the gate, or Stage B's strongest single member
  `XGB α=0.25`); the deployed system is the full 7-member dynamic combiner. v7.3e measured
  surrogate inflation at **+0.6773**, so these attacks are **understated**. That is conservative
  for the stealth and reject-surface findings, and it means the true misattribution price is
  likely *lower* than 0.0545 — but not enough is known to claim H1 would flip.
- Symmetric across arms (every objective attacks exactly one model white-box), so the **price
  comparison is fair** even though the absolute levels are understated.
- Outcomes are mutually exclusive and exhaustive (`undetected` / `misattributed` / `correct`),
  so arms are directly comparable rather than each grading itself on its own scale.
- Gate, splits, pool, `W_e2e`, threshold and budget machinery identical to v7.3c/e. Same
  XGB-OOF-derived confound as those runs, unchanged.

## Method

Greedy per-feature sign search, 2 passes, offsets clipped to the cap. Budget in train-IQR units;
COSTLY at half; IMPOSSIBLE (25 svc/proto/dst-port cols) frozen; integers rounded (round → repair
→ round); CICFlowMeter identities and all 31 derived columns recomputed after every move. Only
the 16,298 test attack rows move. Gate matched to benign fpr 0.0168 (threshold 0.3280).
Misattribution enforces "stay detected" as a hard penalty (+10 whenever `p_gate ≤ thr`), so the
search can never buy a wrong label by sacrificing the alert.
