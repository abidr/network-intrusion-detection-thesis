# Robustness of a flow-based NIDS — consolidated findings

_Nine pre-registered runs, `results/v73*`, 2026-08-19/20. Chapter-ready synthesis of
`v73_adv_budget`, `v73b_brittle_diag`, `v73c_rf_gate`, `v73d_rf_gate_sig`, `v73e_rf_whitebox`,
`v73f_misattribution`, `v73g_controllability_aware`, `v73h_adv_training`, `v73i_cicids_mechanism`. Companion to
`novelty.md` (what is new) and `results_summary.md` (per-run detail)._

---

## 0. The one-page version

The deployed model — v7.1b, macro-F1 `0.687 [0.671, 0.704]` — is **brittle**, and the brittleness
is not primarily an adversarial problem.

1. **Undirected random noise at 0.002 IQR, with no model access at all, costs 0.13 macro-F1.**
   That is measurement noise, a different flow timeout, or another CICFlowMeter build — not an
   attacker.
   **And the cause is now identified: class overlap.** On CIC-IDS2017, where 1-NN error is 81×
   lower, the same budget costs a fraction as much — monotone across all six comparison cells
   (§3.9). **Overlap sets both the accuracy ceiling and the robustness floor**, which welds this
   chapter to the project's plateau diagnosis.
2. **Under noise, detection survives and attribution fails.** Gate recall retains 99%, macro-F1
   retains 44%. You get the alert; you cannot trust the label on it.
3. **Under a targeted attack the ladder inverts** — detection fails *first*, because the attack
   optimises against the gate. **Which layer fails first is a property of what the adversary
   optimises, not of the system.**
4. **Brittleness has BOTH a representation component and a model component — and they operate at
   different levels.** *Across* representations, overlap sets the floor (§3.9). *Within* a fixed
   representation, the model family still matters: at matched false-positive rate a Random Forest
   gate retains 0.93 of clean recall where the deployed XGBoost gate retains 0.46, for a real but
   small clean cost of −0.0038 macro-F1 (p=0.005).
   > ⚠️ **This supersedes the earlier wording "model-bound, not representation-bound"** (§3.2),
   > which was written before the cross-dataset test existed and is now too strong. The reconciled
   > claim: **overlap determines how much robustness is available; the model determines how much of
   > it you collect.**
5. **A new threat model — misattribution — is invisible to the field's binary framing.** It costs
   ~3.8× more than evasion but is far stealthier: it holds gate recall at 0.9969 while corrupting
   63% of labels, so alert *volume* looks normal.
6. **The obvious defence fails, informatively.** Restricting the model to attacker-uncontrollable
   features *concentrates* risk rather than removing it.
7. **One defence does work — plain noise augmentation, 26x retained gate recall for no clean cost —
   but only against an *adversary*. Under drift it is a "perturbed => attack" shortcut whose
   false-positive rate explodes 10.7x.**

**Five of the nine runs refuted at least one of their own pre-registered predictions.** That is the chapter's
credibility, not a weakness — the ones that passed are believable because the ones that failed
were allowed to.

---

## 1. Why this chapter exists

Robustness is **not one of the four gaps the literature review named**, and that is the point.

**Verified by scan + read-in-context over all 10 reviewed papers: none runs a robustness or
adversarial experiment on its own proposed model.** The terms appear — but every `adversarial`
hit in CyberAdaptAI is a related-work citation or reference entry, and the one paper with
substantial hits is a *survey* whose job is to review others' work. Most uses of "robustness" are
the informal sense (*"robust performance"*), not a measurement.

So this is a **sixth gap the review did not notice**, and it should be introduced as such rather
than retrofitted onto the original four.

> ⚠️ **This is the third time in this project that a keyword count and a reading disagree** — after
> the code-availability audit (§3.1 of `novelty.md`) and the SHAP audit (§3.2). State the method,
> never just the count.

---

## 2. The instrument: an attacker-controllable budget

The standard evasion experiment perturbs the feature vector freely, letting the attacker edit
evidence they do not own. Every run in this chapter instead partitions the 99 features by **who
controls them**:

| Tier | Cols | Examples | Budget |
|---|---|---|---|
| FREE | 57 | timing, padding, size shape | full |
| COSTLY | 10 | packet/byte/flag counts | half |
| **IMPOSSIBLE** | 25 | svc / proto / dst port | **zero** — changing them abandons the attack |

Every derivable column — CICFlowMeter rates and ratios plus all 31 engineered features — is
**recomputed** after each move, so the perturbed record stays physically consistent. Budgets are
in units of each column's training IQR; integer columns are rounded; only test attack rows move.

### ⚠️ Honest positioning — do not oversell the instrument

**Constraining the attacker is a crowded, published area.** Verified 2026-08-19:

| Element | Status |
|---|---|
| Partition features by attacker controllability | **TAKEN** — Perturb-ability Score, A2PM, the SoK, Tafreshian & Zhang (TrustCom 2024) |
| Recompute dependent features | **TAKEN** — Tafreshian & Zhang do this too |
| "Unconstrained overstates vulnerability" | **TAKEN** — published as *80.3% of examples invalid* and severity dropping 52.49–99.84% |
| Multiclass per-family robustness | **TAKEN** — *Categorical Robustness Assessment* |

Our realisability gap (**+0.127 macro-F1 at 0.002 IQR, decaying to +0.019 at 0.20**) is therefore
a **replication in a new unit** — the same phenomenon expressed as *damage in the deployment
metric on a multiclass task*, where the literature reports validity rates and perturbation
magnitudes on binary detection. Present it as corroboration plus a more decision-useful unit.
**Never as "nobody has measured this".**

**What is genuinely open**, verified against the nearest multiclass paper (which has the data and
does not do it):

- separating **detection** from **attribution** under perturbation
- **random noise as a same-axis, same-budget control arm**
- **matched-FPR** cross-model robustness comparison
- **surrogate-inflation** measurement
- the **misattribution objective**, and **abstention machinery as an attack surface**

---

## 3. The nine findings

### 3.1 The model is brittle — and the control is the headline (`v73_adv_budget`)

| budget (IQR) | random noise | adversarial | gap |
|---|---|---|---|
| 0.000 | 0.6888 | 0.6888 | — |
| 0.002 | **0.5592** | 0.4758 | 0.083 |
| 0.020 | 0.4826 | 0.3379 | 0.145 |
| 0.200 | 0.2815 | 0.1280 | 0.154 |

**Undirected random noise at 0.002 IQR — L2 0.008, no model access — costs 0.13 macro-F1** and
drops attack recall 0.8994 → 0.7737. Adversarial search roughly doubles the damage.

**Reframe from evasion to fragility.** The same degradation is expected from measurement noise, a
different flow timeout, or another CICFlowMeter build. This is far broader than security, and it
is consistent with the Cover–Hart result: rows sit near decision boundaries.

> ⚠️ **Pre-registered hypothesis REFUTED (1 of 5 runs with a refuted prediction):** feature importance does *not* predict
> robustness. The prediction was a flat curve, because `svc` is 70.4% of importance and outside
> attacker control. Attack recall still halves at 0.020 IQR. **A model can lean 70% on unforgeable
> evidence and still be evaded through the other 30%.**

### 3.2 The model family matters, at a fixed representation (`v73b_brittle_diag`)

> ⚠️ **This section was originally titled "brittleness is model-bound, not representation-bound".
> §3.9 supersedes that**: across representations, overlap sets the floor. Read this section as
> *within a fixed representation, the model still matters* — the two are complementary, not rival.

Seven model families at **matched benign fpr 0.0168**, budget 0.20:

| Model | retained clean recall |
|---|---|
| **RF 100** | **0.93** |
| XGB d8/400 (deployed) | 0.46 |
| XGB d3/400 | worse than d8 |

**Depth is not the lever** — the shallower model is *less* robust, so it is the
averaging-of-independent-trees bias, not capacity. Integer rounding moves recall only **+0.0085**,
so the claim survives that validity check.

> ⚠️ **METHODOLOGICAL RULE — never compare robustness across model families without matching FPR
> first.** At the default 0.5 cut RF looked **worst** (retained 0.392); RF's probabilities are
> compressed, and its matched threshold is 0.0900. **Matching inverted the answer completely.**
> The same check exposes LogReg as *degenerate* (clean recall 0.0511) rather than robust —
> smoothness alone is not the answer.

### 3.3 The gate is the lever, not Stage B (`v73c_rf_gate`)

Four arms isolating the two stages. Gate recall under noise:

| | clean | budget 0.20 |
|---|---|---|
| XGB gate | 0.9862 | **0.4686** |
| RF gate | 0.9823 | **0.9688** |

Swapping **Stage B** buys nothing (−0.0056) and costs −0.0248 clean.

> ⚠️ **Pre-registered hypothesis REFUTED (2 of 5):** the prediction was that Stage B would matter
> more, since v7.3b showed most of the damage was families scrambling rather than the gate
> failing. Wrong — fixing the gate is what pays.

**⚠️ The residual problem, unsolved:** even with the RF gate, macro-F1 still falls 0.6827 → 0.2996
while attack recall holds at 0.79. **Detection can be made robust; family attribution cannot, by
either model family.**

### 3.4 The gate swap is established in both directions (`v73d_rf_gate_sig`)

2,000 paired bootstrap resamples:

| budget | XGB | RF | paired Δ | 95% CI | p |
|---|---|---|---|---|---|
| 0 (clean) | 0.6865 | 0.6827 | **−0.0038** | [−0.0064, −0.0013] | **0.005** |
| 0.02 | 0.4797 | 0.5072 | +0.0275 | [+0.0202, +0.0366] | <0.001 |
| 0.20 | 0.2433 | 0.2996 | **+0.0563** | [+0.0516, +0.0619] | <0.001 |

**An operating choice, never a better model:** a real 0.0038 clean loss bought against +0.009…
+0.056 under input noise. Clean in-distribution traffic → keep XGBoost. Measurement drift or an
adversary → take RF.

> ⚠️ **±0.012 IS NOT A CONSTANT.** The measured paired CI here is **≈±0.0025**, five times tighter,
> because the two arms share the Stage-B pool, reject set, `W_e2e` and every benign prediction and
> differ *only* in the gate. **Always compute the paired interval for the specific comparison.**
> It cuts both ways: quoting ±0.012 from memory would have hidden a real effect. This run is also
> the cleanest illustration of the two-yardstick rule — marginal intervals overlap almost entirely
> (0.6865 [0.6699, 0.7027] vs 0.6827 [0.6657, 0.6987]) while the paired difference is significant.

### 3.5 Targeted attack: RF survives, but only relatively (`v73e_rf_whitebox`)

Retained clean **gate recall** at budget 0.20, matched fpr:

| Arm | Retained |
|---|---|
| `XGB←XGB` white-box | **0.0313** |
| `RF←RF` white-box | **0.2143** |
| `RF←XGB` transfer | **0.8916** |

**Pre-registered H1 SUPPORTED**: +0.1830 against a +0.10 threshold. RF's advantage is a property
of the model, not of the attack having been aimed elsewhere. **But RF still loses 79% of its
detection — write "more robust", never "robust", and always name the threat model.**

> ⚠️ **SURROGATE INFLATION = +0.6773.** Aim the attack at RF and four-fifths of its apparent
> robustness evaporates (0.8916 → 0.2143). **Every v7.3c/d number was measured under noise or an
> XGB-aimed attack, so all of them overstate RF's protection against a real adversary.** Any
> cross-model robustness claim from a surrogate or from noise must report this gap — it is the
> robustness analogue of the matched-FPR rule.

**The pre-registered stall trap did not fire** (`moved` = 0.998 at every budget), so the RF number
is a genuine descent of RF's piecewise-constant surface, not a search failing. Writing that check
in advance is what makes the number quotable.

### 3.6 ⭐ The misattribution threat model (`v73f_misattribution`)

Every adversarial-NIDS attack in the literature optimises `P(benign|x̂)` — Tafreshian & Zhang state
that fitness literally. **That objective is inexpressible in a binary detector**, which is why the
field never asks: *what if the attacker does not hide, but makes you misidentify what they are?*

Stage A and Stage B being **separately trained components** is what makes the question both
implementable and measurable.

| | EVASION @0.20 | **MISATTRIBUTION @0.20** |
|---|---|---|
| gate recall | 0.0309 | **0.9969** |
| labels wrong | 0.0038 | **0.6311** |
| macro-F1 | 0.1446 | 0.1348 |

> ⚠️ **Pre-registered hypothesis REFUTED (3 of 5):** predicted price ratio > 2; **measured 0.27**.
> Misattribution costs **~3.8× more** than evasion (b50 0.0545 vs 0.0145 IQR). **The cheapest
> attack on this NIDS is still hiding from it, not lying to it.** The inference from "attribution
> collapses under undirected noise" to "attribution is cheap to corrupt deliberately" was wrong —
> **fragility under undirected perturbation does not imply cheapness under direction.**

**The threat model survives its own refutation, because both triviality guards passed:**

- **The 2×2 is not diagonal** — the evasion attack drives misattribution *down* to 0.0038, because
  you cannot mislabel a row you have deleted from the alert stream. Genuinely different attacks.
- **Direction buys a great deal** — directed misattribution reaches 0.6311 where random noise at
  the same budget plateaus at 0.1497 (**Δ +0.481**). The claim is *"attribution is cheap to
  target"*, not merely *"attribution is fragile"*.

**The finding is STEALTH, not cheapness.** Evasion is loud in the only way that matters: 97% of
attack flows stop alerting, which any volume monitor sees. Misattribution keeps alert volume
**normal** while 63% of labels are wrong, and it is the most efficient attack on macro-F1 at every
budget. **An evasion-only literature is structurally unable to see it.**

#### ⭐ Unplanned: the reject class is an attack surface

v7.1's reject class was a **defence** — benign-as-attack false alarms 8,370 → 2,093, +0.0242
macro-F1. Under attack it inverts: `abstained` rises 0.0879 → **0.2238 at budget 0.002**, and in
the misattribution arm `undetected` ≈ `abstained` at every budget (0.2571 vs 0.2541 at 0.20), so
**essentially every missed detection is a reject-class abstention, not a gate miss.** The gate
never fails (0.997); Stage B files the attack as a false alarm.

**The attacker was never told to do this.** Told only to corrupt the label while staying detected,
**it found the reject class as the cheapest route.**

**Always report the +0.0242 gain together with this surface, never alone** — the same shape as the
v7.1 Fuzzers operating-choice finding, extended to the adversarial case.

### 3.7 The obvious defence fails, informatively (`v73g_controllability_aware`)

The natural architecture to build from §3.3: restrict the **gate** to the 25 uncontrollable
features (the easy stage can afford to be robust), keep Stage B full.

**Refuted. Clean cost −0.3354 macro-F1; asymmetry not confirmed.**

| Arm | gate / Stage-B features | clean macro-F1 | clean gate recall |
|---|---|---|---|
| A | full 99 / full 99 | **0.6718** | 0.9862 |
| B | svc 25 / full 99 | 0.3365 | **0.4603** |
| C | svc 25 / svc 25 | 0.2977 | 0.4603 |
| D | full 99 / svc 25 | 0.4362 | 0.9862 |

**The premise was a conflation, and it is worth recording as a methodological note.** "svc lifted
gate precision 0.9267 → 0.9411" describes svc *added to* the full feature set at unchanged recall;
svc's 70.4% importance was measured on the **Stage-B family** problem. **Two importance claims
from two different sub-problems were merged into a premise neither supports.** Knowing *which
service* does not tell you *whether it is malicious*. The pre-registered cost test caught it.

#### ⭐ The real finding: restriction concentrates risk

With svc frozen, an svc-only gate is unattackable **by construction** — a tautology, which is why
the mobility arm exists. Let the attacker **retarget the service** (switch dst port, whole svc
block recomputed coherently):

| | frozen | **mobile** |
|---|---|---|
| undetected @ 0.002 IQR | 0.6659 | **1.0000** |
| gate recall | 0.4603 | **0.0000** |
| rows retargeted | — | 0.9998 |

**Total collapse at the smallest budget tested** — where the full-feature gate degrades
*gracefully* (0.9862 → 0.7508 → … → 0.0309).

> **Robustness through feature restriction concentrates risk rather than removing it.** Mirror
> image of §3.1: a model leaning **70%** on unforgeable evidence is still evaded through the other
> 30%; a model leaning **100%** on it is destroyed the moment that evidence becomes forgeable.
> **Feature restriction is not a robustness mechanism in either direction.**

> ⚠️ **Measurement lesson, applies retroactively: `misattributed` is upper-bounded by gate
> recall.** A row that never passes the gate cannot be mislabelled, so arm B's low misattribution
> is an artifact of its broken gate, not protection. Never compare attribution metrics across arms
> with different gate recall without normalising by rows that reached Stage B.

### 3.8 A defence that works — for the wrong threat model (`v73h_adv_training`)

v7.3g failed by applying controllability knowledge as **restriction**. This run applies it as
**augmentation**, the way the closest paper in the literature (Tafreshian & Zhang, TrustCom 2024)
does — keep every feature, add perturbed training examples.

**Retained gate recall at budget 0.20 under white-box evasion:**

| Arm | retained | clean cost (paired bootstrap) |
|---|---|---|
| A baseline | 0.0313 | — |
| **B noise-augmented** | **0.8160** | −0.0051 [−0.0139, +0.0031], p=0.234 |
| C adversarially-augmented | 0.3236 | +0.0013 [+0.0000, +0.0028] |

> ⚠️ **Pre-registered P1 FAILED by −0.4924, in the opposite direction — and this is the positive
> branch.** **Plain random-noise augmentation is 26× the baseline and 2.5× better than adversarial
> training**, at a clean cost indistinguishable from zero. **The adversarial machinery is
> unnecessary and worse than the cheap alternative.** This is the question the source paper
> structurally cannot ask, because it has **no noise-control arm**.

> ⚠️ **Pre-registered P3 also FAILED, backwards:** misattribution-adversarial training of Stage B
> made misattribution *worse* (0.8566 vs 0.6395). Boundary-perturbed training rows blur class
> boundaries rather than sharpening them. **The last surviving fragment of v7.3g's stage-asymmetry
> idea is now closed.**

#### ⭐ The confound check splits the finding in two

Augmentation added **only perturbed attack rows, labelled `attack=1`** — so a gate can satisfy it by
learning **"perturbed ⇒ attack"**. Thresholds were matched on *clean* benign, hiding that. Perturb
**benign** rows at fixed threshold:

| budget | A baseline | **B noise-aug** | C adv-gate |
|---|---|---|---|
| 0.020 | 0.0148 | 0.0889 | 0.0699 |
| 0.200 | **0.0121** | **0.1293** | 0.0951 |

**Arm B's false-positive rate explodes 10.7× while the baseline's stays flat.** This is §3.2's
matched-FPR rule applied to *drift* rather than across model families, and it invalidates the naive
reading of P1: the augmented gates have **moved their operating point**, not simply become robust.

| Threat model | Is it a defence? | Why |
|---|---|---|
| **Adversary** (perturbs only their own flows) | **YES — real, cheap, effective** | benign traffic is unperturbed, so the clean-matched threshold stays correct |
| **Drift** (perturbs everything) | **NO** | 13% FPR at a 97.5% benign base rate is an unusable alert volume |

> ⚠️ **State the irony rather than hiding it: this chapter is motivated primarily by *drift* (§3.1),
> and the one defence that works is an *adversary* defence.**

**Reusable method:** whenever augmentation uses one class only, verify the model has not learned
"perturbed ⇒ that class" by perturbing the **other** class and re-measuring the operating point.
**Untested and required before any drift claim:** re-match FPR *under the drift condition*.

### 3.9 ⭐ Brittleness tracks CLASS OVERLAP — the mechanism test (`v73i_cicids_mechanism`)

Everything above explains brittleness by Cover–Hart: rows sit near decision boundaries. That is not
a story but a **falsifiable prediction across datasets**, because `e_1NN` measures overlap *without
reference to any attack*.

| | e_1NN | prediction |
|---|---|---|
| CIC-UNSW full release | 0.2084 | brittle — measured throughout §3.1–§3.8 |
| **CIC-IDS2017** | **0.0026** (81× lower) | should be dramatically more robust |

Retained fraction of each dataset's own clean value, identical budget machinery, matched fpr:

| mode | budget | **CIC-IDS2017** ret mF1 | ret gate | CIC-UNSW ret mF1 | ret gate |
|---|---|---|---|---|---|
| RANDOM | 0.002 | **0.9960** | 0.9996 | 0.8631 | 0.9558 |
| RANDOM | 0.200 | **0.8381** | 0.9169 | 0.4053 | 0.4939 |
| EVASION | 0.020 | **0.8214** | 0.7784 | 0.5335 | 0.4761 |
| EVASION | 0.200 | **0.5232** | **0.5917** | 0.2143 | **0.0313** |

**Both pre-registered criteria PASS** (H1a 0.5917 > 0.50; H1b 0.8381 > 0.80), the effect is
**monotone in all six cells**, and the triviality guard cleared (`moved` = 1.0000 — the search
genuinely ran).

> ⭐ **THIS MERGES TWO CHAPTERS INTO ONE ARGUMENT.** The plateau diagnosis said class overlap in the
> CICFlowMeter representation caps accuracy (ρ = 0.827, Cover–Hart bracket [0.213, 0.376]). This
> chapter said the model is brittle. **They are the same fact.** Overlap sets both the accuracy
> ceiling *and* the robustness floor. **Robustness comes from better features, not better models** —
> independently the same conclusion every other axis of this project reached.

**Three qualifications that must travel with it:**

1. **"More robust", never "robust"** — CIC-IDS2017 still loses 41% of gate recall and 48% of
   macro-F1 at budget 0.20. Low overlap buys margin, not immunity.
2. **The proximate cause is DECISION MARGIN, and a critic will spot it.** Matched fpr put this gate
   at threshold **0.0002** vs CIC-UNSW's **0.3280**, so a row must be driven much further to evade.
   The objection *"you just used a lower threshold"* is answered by the fact that the threshold is
   **derived** by matching fpr — and 1.68% fpr being achievable at 0.0002 is itself a measurement of
   separability. The chain is **low overlap → confident predictions → large margin → robustness**;
   margin is *how* overlap acts, not a rival explanation. State the chain explicitly.
3. **The confound stands, as pre-registered.** CIC-IDS2017 differs in more than overlap — testbed,
   class mix, no protocol column, no reject class. **Suggestive of the mechanism, not proof.** The
   controlled version varies overlap *within* one dataset.

⚠️ Clean levels here (macro-F1 0.7762) are **not** comparable to `results/cicids2017_v6/` (0.9835):
forcing 1.68% fpr on a near-separable dataset admits ~7,000 benign false positives that are routed
into families, wrecking precision on classes with a handful of test rows. The handicap applies to
clean and perturbed arms alike, so the retained fractions are unaffected — but never quote 0.7762
as a performance number.

---

## 4. Synthesis — which claim holds under which threat model

**The single most important table in the chapter.** Nothing here should be quoted without its row.

All figures are **retained fractions of that arm's own clean value** at budget 0.20 IQR, so rows are
comparable despite different baselines. Raw values in parentheses.

| Threat model | gate recall retained | macro-F1 retained | fails first | mitigation |
|---|---|---|---|---|
| **Undirected noise / drift** (RF gate) | **0.9931** (0.9755) | 0.4834 (0.3300) | **attribution** | RF gate, +0.009…+0.056 |
| **Targeted at the gate** (RF / XGB) | **0.2143** / **0.0313** | 0.3143 / 0.2106 | **detection** | RF helps *relatively* only |
| **Targeted at the label** | **1.0108** (0.9969) | **0.1964** (0.1348) | **attribution, silently** | **none known** |
| Service retargeting vs a restricted gate | **0.0000** (0.0000) | 0.2918 (0.0982) | total collapse | do not restrict |

The second row's macro-F1 uses each arm's own clean value (RF 0.6827, XGB 0.6865); the third uses the
v7.3f full-pool clean 0.6865; the fourth uses arm B's already-degraded clean 0.3365, which is why its
macro-F1 column looks deceptively survivable — **read that row's gate recall, not its macro-F1.**

**Note the third row's gate recall exceeds 1.0.** That is not an error: the misattribution attack
makes rows *more* confidently detected, because staying above the gate threshold is a hard constraint
of its objective. It is the numerical signature of an attack that hides inside normal alert volume.

Read across the rows and the chapter's thesis appears: **there is no single "robustness" property
of this system.** Each threat model breaks a different layer, and the mitigation for one is
irrelevant or harmful to another.

### The operational translation

- **Clean in-distribution traffic** → keep XGBoost. RF costs a real, established 0.0038.
- **Measurement drift, a new exporter build, a changed flow timeout** → take the RF gate. This is
  the *most likely* scenario in practice and the cheapest to defend.
- **A deliberate adversary** → you may not get the alert at all (XGB retains 3% of gate recall).
  Neither model family is adequate.
- **A sophisticated adversary** → the alert fires normally and **the label is wrong**, with alert
  volume undisturbed. Playbook routing, analyst priority and containment all key off the family
  label. **This is the scenario nothing in the literature measures and nothing here defends.**

---

## 5. Methodological contributions

Independent of what they were used to measure:

0. **`e_1NN` as an a-priori predictor of robustness.** Overlap is measurable before any attack is
   run, so a practitioner can tell whether an architecture's robustness results will even be
   meaningful on their data — and a reviewer can tell whether a published robustness claim was made
   on a dataset where it was easy (§3.9).
1. **Random noise as a same-axis, same-budget control arm.** It is what separates *"the model is
   attackable"* from *"the model is merely brittle"* — and here the control turned out to be the
   headline.
2. **Matched-FPR before any cross-family robustness comparison** (§3.2). Omitting it inverted the
   answer completely.
3. **Surrogate-inflation measurement** (§3.5). +0.6773 — the robustness analogue of the
   matched-FPR rule.
4. **Attribution metrics are capped by gate recall** (§3.7). Applies retroactively to any cross-arm
   comparison.
5. **Pre-registration with triviality guards and void conditions.** The v7.3f guards (is the 2×2
   diagonal? does direction beat noise?) and the v7.3g P1 sanity assertion (`spread < 1e-9`, else
   the run is void) are what let a null be reported as a finding instead of a failure.
6. **Five of eight runs refuted at least one pre-registered prediction.** Report this explicitly — it is the evidence
   that the two supported hypotheses were not fitted after the fact.

---

## 6. Limitations — state these with the results, not after them

1. **⚠️ Everything rests on `svc`, which may be partly a testbed artifact.** IXIA PerfectStorm
   targets attacks at specific services (`novelty.md` §6.3). The IMPOSSIBLE tier *is* svc, so the
   entire budget model inherits this. §3.7 shows what happens when that assumption is relaxed.
2. **One confound runs through the whole series.** The reject set, Stage-B pool, `W_e2e` and the
   competence reference all derive from the cached **XGB** OOF gate, so RF arms are scored on a
   population their reject class never saw. It works **against** RF, so RF results are
   conservative — but conservative is weaker than clean. A rebuild around RF is ~30 min and
   **untested**.
3. **Attacks are white-box on one component**, while the deployed system is a 7-member dynamic
   combiner. Given the +0.6773 surrogate inflation, all attack strengths here are **understated**.
4. **Single dataset, single testbed.** No robustness result has been replicated on CIC-IDS2017,
   UNSW-NB15 or TON-IoT.
5. **The `b50` price metric is mildly biased against misattribution** (~28% of rows abstain,
   capping achievable misattribution at ~0.72). Far too small to flip a 0.27-vs-2 verdict, but
   quote the ratio as approximate.
6. **No defence is offered for misattribution.** The chapter identifies a threat model and prices
   it; it does not solve it. Say so plainly.
7. **The one working defence (SS3.8) is scoped to an adversary and is UNTESTED under re-matched
   FPR in the drift condition.** Until that is run, it must not be presented as a drift defence.

---

## 7. How to present this

**Lead with the control, not the attack.** *"Undirected random noise, with no model access, costs
0.13 macro-F1"* is stronger, broader, and harder to dismiss than any evasion number — and it
reframes the chapter from security to measurement validity.

**Then the decomposition, with its threat model attached**: under noise you keep the alert and
lose the label; under a targeted attack you lose the alert first. Never state either half without
naming the adversary.

**Then the misattribution model** as the chapter's novel contribution, presented honestly: the
price hypothesis was refuted, the *stealth* finding stands, and the reject class — a defence added
earlier in the thesis — turned out to be an attack surface the optimiser discovered on its own.

**Close on §4's table.** The defensible thesis is not "our model is robust" or "our model is
fragile" — it is that **robustness is not a scalar property**, and a NIDS evaluation that reports
one number under one threat model has not measured it.

### Framings to avoid

- ❌ "We constrain the attacker realistically" — crowded and published (§2).
- ❌ "We show unconstrained attacks overstate damage" — published; ours is a new *unit* (§2).
- ❌ "RF is robust" — it loses 79% of detection to a targeted attack (§3.5).
- ❌ "Feature importance tells you what to protect" — refuted twice, in both directions (§3.1, §3.7).
- ❌ Any robustness number without its threat model and its FPR.
