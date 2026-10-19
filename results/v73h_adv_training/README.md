# v7.3h — controllability-aware adversarial training — the defence is REAL but only against an ADVERSARY

Run 2026-08-19. `adv_training.py` → `adv_training.txt` / `.json`. 24.8 min.

> ⚠️ **THE SCRIPT'S PRINTED VERDICT ("NOISE AUGMENTATION SUFFICIENT") IS SUPERSEDED.** That line
> was computed from the P1/P2/P3 logic only; the confound check added to the same run **fires**, and
> it changes the conclusion. Read §2 before quoting anything from the log.

The correction to v7.3g. Same controllability knowledge, applied by **augmentation** (the way
Tafreshian & Zhang, TrustCom 2024 apply it) rather than **restriction** (which v7.3g refuted).

| Arm | Gate | Stage B |
|---|---|---|
| A | baseline | baseline |
| **B** | **noise-augmented** | noise-augmented |
| C | evasion-adversarial | baseline |
| D | evasion-adversarial | misattribution-adversarial |

## 1. Both pre-registered hypotheses failed, and both failed informatively

**P1 FAILED by −0.4924, in the opposite direction.** Retained gate recall at budget 0.20 under
white-box evasion:

| Arm | retained |
|---|---|
| A baseline | **0.0313** |
| **B noise-augmented** | **0.8160** |
| C adversarially-augmented | 0.3236 |

**Plain random-noise augmentation is 26× the baseline and 2.5× better than adversarial training.**
Clean cost is negligible: paired bootstrap **−0.0051 [−0.0139, +0.0031], p=0.234** — not
distinguishable from zero. (C: +0.0013 [+0.0000, +0.0028], p=0.043 — a *trivially small
improvement*, not a cost.)

**P3 FAILED by +0.2171, also in the wrong direction.** Misattribution-adversarial training of
Stage B made misattribution **worse**: 0.8566 vs arm C's 0.6395 at budget 0.20. Training on
boundary-perturbed rows appears to blur class boundaries rather than sharpen them. **Drop the
stage-specific augmentation story.** (Both arms sit at gate recall 1.0000 here, so v7.3g's
"`misattributed` is capped by gate recall" caveat does not rescue D — the comparison is fair.)

## 2. ⭐ THE CONFOUND CHECK FIRES — and it splits the finding in two

The augmentation added **only perturbed attack rows, labelled `attack=1`**. A gate can satisfy that
by learning **"perturbed ⇒ attack"** rather than anything about intrusions. Thresholds were matched
on *clean* benign traffic, so the shortcut is invisible there. Perturbing **benign** rows at the
same budgets, thresholds fixed:

| budget | A baseline | **B noise-aug** | C adv-gate |
|---|---|---|---|
| 0.000 | 0.0168 | 0.0168 | 0.0168 |
| 0.020 | 0.0148 | 0.0889 | 0.0699 |
| 0.200 | **0.0121** | **0.1293** | 0.0951 |

**B's false-positive rate explodes 10.7× while the baseline's stays flat.** The shortcut is real,
and it is present in the adversarial arm too (C: 7.9× vs baseline at the same budget).

**This is the project's own matched-FPR rule (v7.3b) applied to drift instead of across model
families** — and it invalidates the naive reading of P1. The augmented gates are not simply "more
robust"; at a fixed threshold they have **moved their operating point**, firing more on everything.

### The corrected conclusion — the split is the finding

| Threat model | Is noise augmentation a defence? | Why |
|---|---|---|
| **Adversary** (perturbs only their own flows) | **YES — real, cheap, effective** | benign traffic is unperturbed, so the clean-matched threshold remains the correct operating point. Retained gate recall 0.8160 vs 0.0313, clean cost indistinguishable from zero |
| **Measurement drift / new exporter build** (perturbs everything) | **NO** | fpr 0.0168 → 0.1293. At the real 97.5% benign base rate that is an unusable alert volume — the "robustness" is the shortcut misfiring |

**⚠️ Note the irony, and state it:** the robustness chapter is motivated primarily by *drift* —
undirected noise costing 0.13 macro-F1 with no attacker involved. **This defence works for the
scenario that motivated it least.**

**Untested and required before any drift claim:** re-match FPR *under the drift condition* and
re-measure. Only that isolates a genuine robustness gain from an operating-point shift.

## 3. What to report

1. **Adversarial training is not necessary and is worse than the cheap alternative** — the
   pre-registered P1 branch. This is the question the source paper structurally cannot ask,
   because it has **no noise-control arm**. Both arms nonetheless share the shortcut, so the
   comparison between them is fair while the comparison to baseline is not.
2. **Noise augmentation is a genuine, near-free adversary defence** (26× retained gate recall, clean
   cost p=0.234) — scoped explicitly to an adversary who perturbs only their own traffic.
3. **It is not a drift defence**, and the check that revealed this is a reusable method: *whenever
   augmentation uses one class only, verify the model has not learned "perturbed ⇒ that class" by
   perturbing the other class and re-measuring the operating point.*
4. **Stage-specific adversarial augmentation backfires** (P3) — the last surviving fragment of
   v7.3g's stage-asymmetry idea is now also closed.

## 4. Honest scope

- **Attacks are white-box on ONE component** and Stage B here is a single XGB α=0.25.
  v7.3e measured surrogate inflation at **+0.6773**, so attack strengths are understated —
  **conservative for an attack claim but GENEROUS for a defence claim.** A defence that only works
  against understated attacks is not established. This applies directly to the arm-B result.
- One round of adversarial training only; iterated adversarial training is untested.
- Augmentation used 30,000 train attack rows split across six budgets, generated against the
  **baseline** models (standard one-round protocol).
- Stage B is a single model in every arm (v7: the 7-member ensemble is worth ~0.0038 at full scale),
  so training augmentation is the only moving part. Arm A's clean macro-F1 is 0.6718 — **not
  level-comparable to the 0.687 headline.**
- Reject-class definition still derives from the cached full-feature XGB OOF gate, as in v7.3c–g.
- An earlier execution of this script was interrupted during the bootstrap; the re-run added the
  confound check. All arm results reproduced identically between the two executions.
