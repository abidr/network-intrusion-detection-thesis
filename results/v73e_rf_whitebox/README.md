# v7.3e — does the RF gate's robustness survive a TARGETED white-box attack?

Run 2026-08-19. `rf_whitebox.py` → `rf_whitebox.txt` / `.json`; `verify_repro.py` →
`verify_repro.txt`. Total 8.3 min + 1.1 min verification.

**Answer: yes, but only in the relative sense, and the absolute picture is much worse than
v7.3c implied. The pre-registered H1 passes (+0.1830 > +0.10) while RF still loses 79% of its
detection to a targeted attacker.**

## The pre-registered test

Retained fraction of clean **gate recall** at budget 0.20 IQR, matched benign fpr 0.0168:

| Arm | Retained | Reading |
|---|---|---|
| `XGB<-XGB` white-box on XGB | **0.0313** | the deployed gate is destroyed |
| `RF <-RF` white-box on RF | **0.2143** | better, but still loses 79% |
| `RF <-XGB` transfer from XGB | **0.8916** | what v7.3c-style evidence would have shown |

- **H1 statistic** `retained(RF<-RF) − retained(XGB<-XGB) = +0.1830`, threshold +0.10 →
  **H1 SUPPORTED**. RF's advantage is a property of the model, not of the attack having been
  aimed elsewhere. The v7.3c/d recommendation stands.
- **⚠️ SURROGATE INFLATION = +0.6773.** `RF<-XGB` retains 0.8916 where `RF<-RF` retains 0.2143.
  **Four-fifths of RF's apparent robustness disappears once the attacker actually aims at it.**
  Any RF-vs-XGB robustness claim measured with an XGB-crafted or noise perturbation — which is
  every number in v7.3c and v7.3d — **overstates RF's protection against a real adversary by
  ~0.68 of retained recall.**

## ⚠️ The finding that changes the framing: the decomposition is THREAT-MODEL-DEPENDENT

`novelty.md` §0 states "robust detection, fragile attribution" from the v7.3c noise result
(gate recall retains 99%, macro-F1 retains 44%). **Under a gate-targeted white-box attack the
ladder inverts:**

| Threat model | retained gate recall | retained macro-F1 | which layer fails first |
|---|---|---|---|
| Undirected noise (RF, v7.3c/e) | **0.9931** | 0.4834 | **attribution** |
| Targeted white-box (RF) | **0.2143** | 0.3144 | **detection** |
| Targeted white-box (XGB) | **0.0313** | 0.2106 | **detection** |

**Mechanism, and it is not mysterious:** the attack optimises against the *gate*, so the gate is
what it breaks. Undirected noise has no target, so it damages the harder sub-problem — family
attribution — first. **The decomposition is real in both cases, but which layer fails first is a
property of what the adversary optimises, not of the system alone.**

Consequences, both of which must be applied before quoting §0:

1. **"Robust detection, fragile attribution" must be scoped to the non-adversarial setting**
   — measurement noise, exporter drift, a different flow timeout. That is the setting the
   result was measured in, and it remains the more *operationally common* one.
2. **Against a deliberate gate-targeted adversary, neither layer is robust.** Detection is
   destroyed outright for XGB (0.0313) and badly degraded for RF (0.2143). The honest SOC line
   in this setting is *"you may not get the alert at all"*, not "you get the alert but not the
   label".

## Supporting observations

- **The pre-registered stall trap did NOT fire.** `moved` = 0.998 on `RF<-RF` at every budget,
  so the greedy search really did descend RF's surface; the RF result is not an artifact of a
  search stalling on piecewise-constant probabilities. This check was written in advance
  precisely so a stalled search could not be reported as robustness.
- **A targeted attack is far stronger than noise at equal budget**, for both families — XGB gate
  recall 0.4871 (random) vs 0.0309 (white-box); RF 0.9755 vs 0.2105. v7.3's "adversarial ≈ 2×
  random" was measured on a full-data gate at a 0.5 cut; at matched fpr the multiple is much larger.
- **RF is near-immune to undirected noise** (retains 0.9931 of gate recall at 0.20) and clearly
  not to a targeted attack. Stating RF as "robust" without naming the threat model is misleading.
- End-to-end macro-F1 at budget 0.20: `RF<-RF` 0.2146 vs `XGB<-XGB` 0.1446 — RF still ahead
  end-to-end, consistent with v7.3d.

## Reproduction check — failed, then diagnosed

`rf_whitebox.txt` prints **"WARNING: does not reproduce"** (max deviation 0.0718 vs v7.3c arm A).
**That warning is a false alarm and the check was mis-specified by me**, not a prep divergence:

- v7.3c `perturb()` applies `budget * IQR` to **every** column uniformly.
- v7.3e `craft(mode="random")` applies the **attacker structure** — FREE at `budget*IQR`,
  COSTLY at `budget*0.5*IQR` — so it is strictly *weaker* noise and must damage less.

`verify_repro.py` re-ran the arm under the uniform model: **deviation 0.0000 at all seven
budgets** (`verify_repro.txt`). Prep, gate and threshold reproduce exactly. The greedy arms never
used the v7.3c noise model, so they were never affected.

**Do not delete the warning from the log** — it is the honest record, and the resolution belongs
next to it rather than in place of it.

## Confound carried over from v7.3c, unchanged

The reject set, Stage-B pool, `W_e2e` and the competence reference all derive from the cached
**XGB** OOF gate, so arm `RF<-RF` is scored on a population its reject class never saw. This works
**against** RF, so the RF result is conservative. A rebuild around RF is still untested.

## Method

Greedy per-feature sign search, 2 passes, cumulative offset clipped to the cap so passes refine
direction without exceeding budget. Budget in train-IQR units; COSTLY at half; **IMPOSSIBLE
(25 svc/proto/dst-port columns) frozen**; integer columns rounded (round → repair → round); all
CICFlowMeter identities and 31 derived columns recomputed after every move. Only the 16,298 test
attack rows are perturbed. Both gates trained on the same 800,000-row stratified subsample and
thresholded to the same benign fpr 0.0168 — v7.3b showed comparing families at their default 0.5
cut measures threshold placement, not robustness.
