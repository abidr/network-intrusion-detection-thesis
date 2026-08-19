# v7.3g — controllability-aware stage-asymmetric feature allocation — **REFUTED**

Run 2026-08-19. `ctrl_aware.py` → `ctrl_aware.txt` / `.json`. 21.7 min.

**The design claim is refuted, and the failure is more informative than the claim would have
been: restricting a model to uncontrollable features does not make it robust, it makes it
catastrophically brittle to the single action that moves those features.**

## The claim under test

> Allocate feature access per stage by threat exposure, not accuracy. The easy stage
> (detection) can afford to be robust; the hard stage (attribution) cannot.

Concretely: restrict the **gate** to the 25 uncontrollable svc/proto/port features, keep Stage B
at full width. Rationale was that the gate is "essentially solved" (recall ≈0.997) and svc lifted
gate precision 0.9267 → 0.9411, so restricting it should be nearly free.

## Verdict: P2 FAILED, P3 FAILED

| Test | Result |
|---|---|
| **P1** svc-only gate recall constant under frozen svc | **PASS** — spread `0.00e+00`; partition airtight, run valid |
| **P2a** hardening defends evasion (gap > 0.30) | PASS +0.3044 — **but see "why P2a is misleading"** |
| **P2b** hardening does NOT defend misattribution (\|gap\| < 0.10) | **FAIL** +0.4109 |
| **P3** clean cost < 0.02 macro-F1 | **FAIL** −0.3354 |
| **P4** non-tautology under service retargeting | **catastrophic — see below** |

## ⚠️ The premise was wrong, and it was a reasoning error worth recording

**The svc-only gate is bad: clean recall 0.4603 vs the full gate's 0.9862.** It misses 54% of
attacks before any adversary shows up.

The supporting evidence had been misread. "svc lifted gate precision 0.9267 → 0.9411" is a
statement about svc **added to** the full feature set, at unchanged recall. It is not evidence
that svc **alone** can do the gate's job. Likewise svc's 70.4% importance was measured on the
**Stage-B family** problem, not on attack-vs-benign. **Two different importance claims were
conflated into a premise neither supports.** Attack-vs-benign genuinely needs the behavioural
features; "which service" does not tell you "is this malicious".

Clean cost across the 2×2 — the trade is nowhere near free:

| Arm | gate / Stage-B features | clean macro-F1 | clean gate recall |
|---|---|---|---|
| **A** | full 99 / full 99 | **0.6718** | 0.9862 |
| B | svc 25 / full 99 | 0.3365 | 0.4603 |
| C | svc 25 / svc 25 | 0.2977 | 0.4603 |
| D | full 99 / svc 25 | 0.4362 | 0.9862 |

## Why P2a "PASS" is misleading — and the measurement lesson

Arm B's evasion rate barely moves under attack (0.6650 clean → 0.6659 at budget 0.20). That
looks like a defence. **It is not: arm B is already failing at budget zero.** The attack cannot
make it much worse because two-thirds of attack flows are already undetected before the attacker
does anything. A pre-emptively broken detector is not a robust detector.

**MEASUREMENT LESSON, generally applicable: `misattributed` is upper-bounded by gate recall.**
A row that never passes the gate cannot be mislabelled. So arm B's low misattribution (0.2305 vs
arm A's 0.6414) is an artifact of its broken gate, not evidence of protection — and comparing
misattribution rates across arms with different gate recall is confounded. **Always normalise
attribution metrics by the number of rows that actually reached Stage B**, or state gate recall
alongside. This applies retroactively to any cross-arm attribution comparison.

## ⭐ P4 — the real finding: restriction concentrates risk, it does not remove it

With svc **FROZEN**, an svc-only gate is unattackable by construction — a tautology, which is
exactly why the MOBILE arm exists. Allow the attacker to **retarget the service** (switch dst
port, whole svc block recomputed coherently):

| | B FROZEN | **B MOBILE** |
|---|---|---|
| undetected @ budget 0.002 | 0.6659 | **1.0000** |
| gate recall | 0.4603 | **0.0000** |
| rows retargeted | — | **0.9998** |

**Total collapse at the smallest budget tested.** Not gradual — 100% evasion, zero gate recall,
at 0.002 IQR, because once the attacker pays the service-switch cost the gate has *nothing else
to look at*.

Contrast the full-feature gate, which degrades **gracefully** over the same grid (0.9862 → 0.7508
→ 0.5714 → … → 0.0309). **A model concentrated on uncontrollable evidence has a single point of
failure; a model spread across many features degrades continuously.**

This is the mirror image of v7.3's refuted hypothesis. There, a model leaning 70% on unforgeable
evidence was still evaded *through the other 30%*. Here, a model leaning 100% on unforgeable
evidence is destroyed the moment that evidence becomes forgeable. **Feature restriction is not a
robustness mechanism in either direction.**

## What survives

- **Nothing recommendable.** Do not present controllability-aware feature allocation as an
  architecture contribution. It is the 13th pre-registered null in the project, with a mechanism.
- **The generalisable claim**: *robustness through feature restriction concentrates risk rather
  than removing it* — worth one paragraph in the robustness chapter as the counterpart to the
  importance-≠-robustness result.
- The measurement lesson above (attribution metrics are capped by gate recall).
- The reasoning error is itself reportable: **two different feature-importance claims, measured
  on two different sub-problems, were conflated into a design premise.** Cheap to make, and the
  pre-registered cost test (P3) is what caught it.

## Honest scope

- **This defence rested entirely on svc, which may be partly a testbed artifact** (IXIA
  PerfectStorm targets attacks at specific services — `novelty.md` §6.3). Since the approach is
  refuted on its own terms, the caveat is moot here, but it would have been load-bearing had P3
  passed. Recorded because the *next* svc-dependent idea will face it too.
- Stage B is a single XGB α=0.25 in every arm (v7: the 7-member ensemble is worth ~0.0038 at full
  scale), so feature access is the only moving part. Arm A's clean macro-F1 is therefore 0.6718,
  not the 0.687 headline — **not level-comparable to v7.1b**.
- Reject-class definition still derives from the cached full-feature XGB OOF gate, as in
  v7.3c/e/f. For arms B and C the deployed gate differs from the one that defined the reject set;
  this works against those arms and is another reason not to read their numbers as a fair ceiling.
- Same prep, splits, budget machinery and matched-fpr protocol as v7.3c/e/f. The svc gate matched
  to fpr 0.0152 (its probability granularity prevents hitting 0.0168 exactly).
