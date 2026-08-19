# Novelty — what this thesis contributes

_Compiled 2026-08-19. Companion to `results_summary.md` (what was run), `results_short.md` (the
metrics), `NEXT_STEPS.md` (what is left) and `literature_artifact_audit.md` (the corpus audit)._

Every claim below is stated with its **status** and its **evidence path**. Three statuses are used:

| Status | Meaning |
|---|---|
| **ESTABLISHED** | Paired bootstrap CI excludes zero at the stated p |
| **MEASURED** | A quantity or diagnostic, not a comparison — no CI applies |
| **NULL / CLOSED** | Pre-registered test returned nothing; reported as a negative result |

**Reporting rules that apply to every number here:** quote SELV-selected numbers, never test-max;
marginal CI for a single config ≈ ±0.017; the paired CI must be **recomputed per comparison** (it
ranges from ±0.0025 to ±0.012 depending on how much the two arms share); rare classes as intervals
(Worms ±0.13).

---

## 0. THE HEADLINE CONTRIBUTION — robust detection, fragile attribution

> **Under realistic input degradation, a flow-based NIDS keeps detecting and stops explaining.
> You still get the alert. You can no longer trust the label on it.**

> ⚠️ **SCOPED 2026-08-19 by `results/v73e_rf_whitebox/`. This holds for the NON-ADVERSARIAL
> setting** — measurement noise, exporter drift, a different flow timeout — **which is the
> setting it was measured in and the more operationally common one. It does NOT hold against a
> deliberate gate-targeted adversary, where the ladder INVERTS and detection fails first.**
> Never state the headline without naming the threat model. See "the ladder inverts" below.

`results/v73c_rf_gate/` (arm B) — the same model, the same rows, three questions asked separately:

| Question the system answers | Metric | Clean | Budget 0.20 IQR | Retained |
|---|---|---|---|---|
| **Did we see it at all?** | Stage-A gate recall | 0.9823 | **0.9688** | **99%** |
| **Did we call it an attack?** | end-to-end attack recall | 0.8983 | **0.7900** | 88% |
| **Did we say which attack?** | end-to-end macro-F1 | 0.6827 | **0.2996** | **44%** |

**That monotone ladder is the finding.** Detection is essentially intact while family attribution has
more than halved — and the two have never been separated in the literature (see the positioning table
below). It is not an artifact of one weak model: the ladder is measured on the *hardened* RF-gate
configuration, the best detector we have. On the deployed XGB gate the collapse is worse in every row
(gate recall 0.9862 → **0.4686**, macro-F1 0.6865 → 0.2433).

### ⚠️ The ladder INVERTS under a targeted attack — and that is a second finding, not a defeat

`results/v73e_rf_whitebox/`, pre-registered, run 2026-08-19. Retained fraction of clean value at
budget 0.20 IQR:

| Threat model | gate recall | macro-F1 | fails first |
|---|---|---|---|
| Undirected noise (RF) | **0.9931** | 0.4834 | **attribution** |
| Targeted white-box (RF) | **0.2143** | 0.3144 | **detection** |
| Targeted white-box (XGB) | **0.0313** | 0.2106 | **detection** |

**Mechanism, and it is not mysterious:** the attack optimises against the *gate*, so the gate is
what breaks. Undirected noise has no target, so it damages the harder sub-problem — attribution —
first. **Which layer fails first is a property of what the adversary optimises, not of the system
alone.** That is a sharper and more defensible claim than the original one, and it subsumes it.

Against a deliberate adversary the honest SOC line is **"you may not get the alert at all"**
(XGB retains 3% of gate recall), not "you get the alert but not the label".

### The misattribution threat model — a NEW attacker objective the field cannot express

`results/v73f_misattribution/`, pre-registered, 2026-08-19. Every adversarial-NIDS attack in the
literature optimises `P(benign|x̂)` — Tafreshian & Zhang (TrustCom 2024) state that fitness
literally, and the surveys describe the field the same way. **That objective is inexpressible in a
binary detector**, which is why nobody asks the other question: *what if the attacker does not hide,
but makes you misidentify what they are?* Stage A and Stage B being separately trained components is
what makes the question both implementable and measurable.

| | EVASION @0.20 | **MISATTRIBUTION @0.20** |
|---|---|---|
| gate recall | 0.0309 | **0.9969** |
| misattributed | 0.0038 | **0.6311** |
| macro-F1 | 0.1446 | 0.1348 |

- **⚠️ THE PRE-REGISTERED HYPOTHESIS WAS REFUTED.** Predicted `b50(EVASION)/b50(MISATTRIB) > 2`;
  measured **0.27** — misattribution costs **~3.8× MORE** than evasion (b50 0.0545 vs 0.0145 IQR).
  **The cheapest attack on this NIDS is still hiding from it, not lying to it.** The inference from
  "attribution collapses under undirected noise" to "attribution is cheap to corrupt deliberately"
  was wrong — the **third** time in this project a plausible mechanism failed to predict a
  realisable effect.
- **But the threat model survives its own refutation, because both triviality guards passed.**
  (a) The 2×2 is not diagonal: the evasion attack drives misattribution *down* to 0.0038 — you
  cannot mislabel a row you have deleted from the alert stream, so these are genuinely different
  attacks. (b) **Direction buys a great deal**: directed misattribution reaches 0.6311 where random
  noise at the same budget plateaus at 0.1497 (**delta +0.481**). The claim is *"attribution is
  cheap to target"*, not merely *"attribution is fragile"*.
- **What misattribution buys is STEALTH, not cheapness — and that is the reportable finding.**
  Evasion is loud in the only way that matters: 97% of attack flows stop alerting, which any volume
  monitor sees. Misattribution holds gate recall at **99.7%** — alert volume looks *normal* — while
  63% of labels are wrong. It is also the most efficient attack on the *metric* at every budget
  (macro-F1 0.4712 at budget 0.002 vs evasion's 0.5494), and the worst on *alert volume*. **An
  evasion-only literature is structurally unable to see it.**
- ⭐ **UNPLANNED: the reject class is an attack surface.** The v7.1 reject class was a *defence*
  (false alarms 8,370 → 2,093, +0.0242). Under attack it inverts: `abstained` rises 0.0879 → 0.2238
  at budget **0.002**, and in the misattribution arm `undetected` ≈ `abstained` at every budget —
  **essentially every missed detection there is a reject-class abstention, not a gate miss.** The
  attacker was never told to do this; told only to corrupt the label while staying detected, **it
  found the reject class as the cheapest route.** Report the +0.0242 and this together, never the
  gain alone — same shape as the v7.1 Fuzzers operating-choice finding, extended to the adversarial
  case.

### Why this matters operationally

A SOC does not act on "something bad happened". Playbook routing, analyst priority and containment all
key off **which family** it is. This result says the layer the field measures (detection) is the layer
that survives, and the layer operations actually consumes (attribution) is the one that fails first.

### The attacker-controllable budget is the *method*, not the claim

The budget is what makes the collapse **believable** rather than an artifact. The attacker sets
primitives only — **FREE** (49 raw timing/padding/size columns), **COSTLY** (18 packet/byte/flag
counts, half budget), **IMPOSSIBLE** (25 svc/proto/dst-port columns — changing them abandons the
attack) — and every derivable column (CICFlowMeter rates/ratios + all 31 engineered features) is
**recomputed** so the record stays physically consistent. Without that, we would be perturbing
evidence no attacker owns and the collapse would be fake.

### Honest positioning — what is taken, and what is open

⚠️ **Constraining the attacker is NOT novel** and must not be claimed as such. Verified against the
literature 2026-08-19:

| Element | Status |
|---|---|
| Constrain attacker-controllable features | **TAKEN** — Perturb-ability Score, A2PM, the SoK |
| "Unconstrained overstates vulnerability" | **TAKEN** — published as **80.3% of examples invalid** and severity dropping 52.49–99.84%; also "2× perturbation for the same success rate" |
| Multiclass per-family robustness | **TAKEN** — *Categorical Robustness Assessment* (Recon 78% F1 degradation vs brute force 0.04%) |
| Gaussian-noise robustness | **TAKEN**, but reported *separately* from adversarial evaluation |
| **Splitting detection from attribution** | ⭐ **OPEN** — *Categorical Robustness* is multiclass and has the data, and **explicitly does not** separate benign-vs-attack survival from family labelling |
| **Controllability partition + budget + recompute** | ⭐ **OPEN** — the same paper perturbs all 30 features uniformly in normalised space and **concedes this as a limitation** |
| **Random noise as a same-axis, same-budget control arm** | ⭐ **OPEN** |
| **Matched-FPR cross-model robustness comparison** | ⭐ **OPEN** — and it *inverts the conclusion* (§1.8) |
| **Surrogate-inflation measurement** (attack aimed at model A, faced by model B) | ⭐ **OPEN** — +0.6773 of RF's apparent robustness; the robustness analogue of the matched-FPR rule |
| **Threat-model dependence of which layer fails first** | ⭐ **OPEN** — noise breaks attribution, targeted attack breaks detection |
| **The misattribution objective** (stay detected, corrupt the label) | ⭐ **OPEN** — inexpressible in binary; the field optimises `P(benign)` only. ⚠️ *targeted/source-target misclassification exists in the general adversarial-ML taxonomy*, so claim "not run on NIDS as a threat model", never "unheard of" |
| **Abstention/reject machinery as an attack surface** | ⭐ **OPEN** — a defence that inverts under attack |

**So the contribution is the decomposition, supported by the budget — not the budget itself.**

### Four supporting results, all already measured

1. **You do not need an attacker.** Undirected *random* noise at 0.002 IQR — no model access — costs
   **0.13 macro-F1** (0.6888 → 0.5592) and drops attack recall 0.8994 → 0.7737. Adversarial search
   roughly doubles the damage (0.4758 at the same budget; the adversarial−random gap peaks at 0.1845).
   **This reframes the finding from evasion to fragility** — the same degradation is expected from
   measurement noise, a different flow timeout, or another CICFlowMeter build.
2. **We located the failure: the gate, not Stage B** — a *pre-registered prediction we got wrong*.
   Gate recall under noise XGB 0.9862 → 0.4686 vs RF 0.9823 → 0.9688; swapping Stage B buys −0.0056
   and costs −0.0248 clean.
3. **Detection is fixable *against noise*; attribution is not — and neither is fixable against a
   targeted attacker.** Under noise the RF gate rescues detection almost completely while macro-F1
   **still** collapses to 0.2996, in both model families. **v7.3e then tested the adversarial case
   (pre-registered) and RF's advantage SURVIVED but shrank to a relative one:** at budget 0.20 the
   RF gate retains 0.2143 of clean gate recall vs XGB's 0.0313 — H1 passes (+0.1830 > +0.10), so the
   RF recommendation stands, **but RF still loses 79% of its detection.** State RF as "more robust",
   never as "robust", and always name the threat model.
   - ⚠️ **SURROGATE INFLATION +0.6773 — the reusable warning.** An attack crafted on XGB and faced by
     RF leaves RF retaining **0.8916**; aimed at RF directly, **0.2143**. **Four-fifths of RF's
     apparent robustness is an artifact of the attack having been aimed elsewhere.** Every v7.3c/d
     number was measured under noise or an XGB-aimed attack, so all of them overstate RF's protection
     against a real adversary. **Any cross-model robustness claim measured with a surrogate or with
     noise must report this gap** — it is the robustness analogue of §1.8's matched-FPR rule.
   - The pre-registered **stall trap did not fire** (`moved` = 0.998 on RF←RF at every budget), so the
     RF result is a genuine descent of RF's surface, not a greedy search failing on piecewise-constant
     probabilities. Writing that check in advance is what makes the number quotable.
4. **Feature importance does not predict robustness** — a second refuted pre-registration. `svc` is
   70.4% of importance and *outside attacker control*, which predicted a flat curve; attack recall
   still halves at 0.020 IQR. **A model can lean 70% on unforgeable evidence and still be evaded
   through the other 30%.**

### What must be run before this is defensible

| Gap | Why a reviewer will ask | Cost |
|---|---|---|
| ~~RF under a *targeted* white-box attack~~ | ✅ **DONE 2026-08-19** (`results/v73e_rf_whitebox/`, 8.3 min). Pre-registered H1 **supported**, +0.1830. Recommendation stands, scoped: RF is *more* robust, not robust. Surfaced surrogate inflation (+0.6773) and the threat-model inversion above | done |
| **Rebuild reject set / pool / `W_e2e` around the RF gate** | Removes the one confound. It works *against* RF today, so results are conservative — but conservative is weaker than clean | ~30 min |
| **Report the unconstrained arm beside the constrained one in macro-F1** | Makes explicit that we are *replicating* a published overstatement finding in a new unit (0.1269 at 0.002 IQR, decaying to 0.0191 at 0.20), not claiming it | already measured |

### The rest of the contribution, in one paragraph

The reviewed literature reports 99%+ accuracy on this family of datasets and releases no code. This
project reports **0.687 macro-F1** and can say exactly why every point of the difference exists. Beyond
the headline above: **every capacity-side lever was tested and none was established**, so there is no
better-classifier claim. What remains is (a) a diagnosis of *why* flow-based NIDS plateaus, carried
from hypothesis to external confirmation to a **+0.0982 cash-in** (§2.1); (b) **measurement protocols**
that make the difference between 0.687 and 0.99 legible instead of mysterious (§1); and (c) **twelve
pre-registered negative results** mapping the dead ends of the design space (§4).

---

## 1. Methodological contributions

These generalise beyond this dataset and are the strongest publishable material.

### 1.1 The attacker-controllable budget — the **instrument** behind §0, not a claim of its own

`results/v73_adv_budget/`

> ⚠️ **Scope this carefully.** Constraining the attacker is a **crowded, published area** (see the
> positioning table in §0). This subsection documents *how* our budget is built, because the headline
> decomposition depends on it being right. **Do not present the budget itself as the novelty.**

The standard evasion experiment perturbs the feature vector freely. That lets the attacker edit
evidence they do not own. This work partitions the 99 columns by **who controls them**:

| Class | Cols | Examples | Budget |
|---|---|---|---|
| FREE | 49 | timing, padding, size shape | full |
| COSTLY | 18 | packet/byte/flag counts | half |
| **IMPOSSIBLE** | 25 | svc / proto / dst port | **zero** — changing them abandons the attack |

Every derivable column (CICFlowMeter rates/ratios + all 31 engineered features) is **recomputed** so
the perturbed record stays physically consistent.

- **Result: the unconstrained experiment overstates damage by 0.10–0.13 macro-F1** (gap +0.127 at
  budget 0.002 IQR, decaying to 0.019 at 0.20).
- ⚠️ **This is a REPLICATION IN A NEW UNIT, not a new finding.** The overstatement itself is published
  — reported as **80.3% of unconstrained adversarial examples being invalid** and attack severity
  dropping **52.49–99.84%** once domain constraints are enforced, and elsewhere as "2× perturbation
  needed for the same success rate". Those are *validity rates* and *perturbation magnitudes*, both on
  **binary** detection. Ours is the same phenomenon expressed as **damage in the deployment metric
  (macro-F1) on a multiclass task**. Present it that way — as corroboration plus a more decision-useful
  unit — never as "nobody has measured this".
- **Two design points that are genuinely less common** and are worth defending on their own: the
  **recomputation** of every derived column after perturbation (the nearest multiclass paper perturbs
  all features uniformly in normalised space and concedes this as a limitation), and the **three-tier**
  budget in which one tier is *zero* because acting on it abandons the attack.

### 1.2 Two yardsticks for significance, and the paired CI is not a constant — **MEASURED**

`ablation_significance.ipynb`, corrected by `results/v73d_rf_gate_sig/`

- Reporting **one** config: marginal 95% CI width ≈ 0.035 → quote `0.687 [0.671, 0.704]`, never four
  decimals.
- Comparing **two** configs on the same test set: use the **paired** interval.
- **⚠️ The correction that makes this a contribution:** the paired band is **not a fixed ±0.012**. It
  scales with how much the two arms share. When two arms share the Stage-B pool, reject set, `W_e2e`
  and every benign prediction, and differ *only* in the gate, the measured paired CI is **≈±0.0025** —
  five times tighter, and a −0.0038 difference is **established at p=0.005**.
- **It cuts both ways.** Quoting ±0.012 from memory there would have *hidden a real effect*; using the
  marginal width to judge differences would have wrongly discarded two real results.
- That run is also the cleanest illustration: marginal intervals overlap almost entirely
  (0.6865 [0.6699, 0.7027] vs 0.6827 [0.6657, 0.6987]) while the paired difference is significant.

### 1.3 Oracle headroom does not predict realisable gain — **MEASURED, four instances**

A pool's oracle bound is the standard justification for building an ensemble. It is worthless here:

| Run | Oracle headroom | Realised gain |
|---|---|---|
| DES v1 | 0.0804 | 0 |
| Ensemble v3 | 0.2250 (+55% over v2) | +0.0037 |
| Dynamic v4 | 0.1579 | 0 |

**Standing rule produced: never justify a pool by its oracle bound.**

### 1.4 Fit the combiner on the deployment objective, not the sub-problem objective — **ESTABLISHED**

`results/v71_w_e2e/` — **+0.0032 [+0.0019, +0.0047], p<0.001**

Identical parameter count (63+7), identical ascent; only the objective changes from Stage-B 10-class
macro-F1 (where the reject class is 1/10 of the score) to **end-to-end** macro-F1. The gain arrives by
exactly the predicted mechanism (Fuzzers recall 0.576→0.731). A general lesson that costs nothing to
apply, and one of the few free wins in the project.

### 1.5 The irreducible-error bracket is **metric-relative** — **MEASURED**

`knn_class_overlap.ipynb`, corrected by `feature_engineering_v2.ipynb`

Cover–Hart on 1-NN error gives **e\* ∈ [0.213, 0.376]** → accuracy ceiling [0.625, 0.787], the first
non-vacuous irreducible-error estimate for this dataset (the exact-collision figure, 0.0019, is
vacuous). The **correction is the contribution**:

- A **zero-information** signed-log transform of the same columns moves e_1NN **+0.0296 (p=4.8e-22)** —
  **2.3× larger than any real feature effect measured in the project** — while leaving a tree
  ensemble's predictions **98.99% unchanged**.
- Therefore `e_1NN` is a property of **(features, metric) jointly** and does **not** bound a tree
  ensemble. It remains valid for the distance-based parts (k-NN overlap, purity, DES competence).
- **Consequence for the literature:** any "ceiling" or "Bayes error" claim derived from k-NN in a
  scaled feature space is reporting the scaler as much as the data.
- Second-order finding, counter-intuitive and worth stating: log-compression made the geometry
  **worse** (purity 0.3761→0.3421, every family). Heavy-tailed raw scale beats a compressed one under
  RobustScaler.

### 1.6 Pre-registered fingerprint audit as a go/no-go gate on new features and datasets — **MEASURED**

`results/ttl_join/`, `results/toniot_audit/`

A reusable audit that asks, *before* modelling: **does any single column solve attack-vs-benign
alone?** Two design points make it work:

1. **Score the rule out of sample**, with a ≥20-row support floor. On TON-IoT, `src_port` scores
   in-sample recall 0.9720 → **out-of-sample 0.2797**. Scored in-sample it would have been a false
   alarm. A genuine fingerprint has in-sample ≈ out-of-sample; a large gap means high cardinality.
2. **Validate on a synthetic file with a planted leak.** The audit flags the planted `src_ip`
   (recall 1.000) and correctly declines to flag `ts` and `src_port`.

Both live datasets **failed** it — see §2.6 (TTL) and §2.9 (TON-IoT `src_ip`).

### 1.7 The probe → pipeline protocol, and the winner's curse it catches — **MEASURED**

`results/v72_hparam/`

Pre-registered two-stage design: a cheap probe over 40 configs, with a promotion bar (+0.005 on SELV)
that must be cleared before a full pipeline run; test touched once at the very end.

- The probe **cleared the bar** (+0.0079) and the pipeline **did not confirm**
  (+0.0054 [−0.0037, +0.0136], p≈0.23). The tuned single model ranks **last** on test.
- **Standing rule produced: probe gains must survive the full pipeline before being believed.** This
  is the exact failure mode that produces unreproducible published tuning results.

### 1.8 Never compare robustness across model families without matching FPR first — **MEASURED**

`results/v73b_brittle_diag/`

At the default 0.5 cut, RF looked **worst** (retained 0.392 of clean recall). Matched to benign fpr
0.0168 — RF's probabilities are compressed, matched threshold 0.0900 — RF is **best** (retained ~1.0).
**Matching inverted the answer completely.** The same check exposes LogReg as degenerate (clean recall
0.0511) rather than robust: smoothness alone is not the answer.

### 1.9 Same-traffic / two-extractor field transfer at 100% coverage — **MEASURED**

`results/ttl_join/`

`sttl`/`dttl` joined from the original UNSW-NB15 flow files onto all 3,540,241 CICFlowMeter rows by
5-tuple (both orientations) + Argus time window, with an **empirically recovered clock offset of
+14400 s** (98.4% of samples within ±2 s). Coverage **93.1% direct + 4.6% reversed + 2.3%
ip-pair-modal fallback = 0.00% unmatched**, in 36 s. The methodology is reusable independently of what
it was used to reject.

---

## 2. Empirical contributions

### 2.1 Service identity is the missing ingredient — predicted externally, then cashed in — **ESTABLISHED**

`final_model_v6_unsw.ipynb` → `results/cic_unsw_svc/` — **+0.0982 [+0.0885, +0.1081], p<0.001**

> ⚠️ **NARROWED 2026-08-19 after a literature check. "Ports/service features help" is ALREADY KNOWN**
> — published work on CIC-IDS2017 reports that every model leans heavily on `Destination Port`, that it
> is the one feature selected by all train/test attack pairs, and explicitly **flags it as a possible
> bias with "contaminating consequences"**. **Do not claim the discovery.** Note the literature's
> concern is *the same concern we reached independently* via the IXIA service-targeting caveat (§6.3) —
> say so; converging on it from the other direction is worth more than pretending we got there first.

The defensible novelty is the **shape of the argument**, not the number and not the feature:

1. **Diagnosis** (`knn_class_overlap.ipynb`): the plateau is class overlap in CICFlowMeter space —
   overlap explains model confusion at **ρ=0.827** where exact label ambiguity explains it at 0.538,
   and explains pairs exact matching cannot (Exploits↔Recon: 451 confusions, **zero** exact ambiguity).
2. **External test** (`final_model_v6_unsw.ipynb`): the same traffic through a *different extractor*,
   **zero shared columns**. The prediction "richer host/temporal context will fix it" was **refuted** —
   the 14 `ct_*` columns carry only 6.8% of importance — and replaced: the top five are **service
   identity and TTL**, which CICFlowMeter does not emit.
3. **Cash-in** (`results/cic_unsw_svc/`): recovering ports/protocol from the official release
   delivered **0.6478 → 0.7460**, twice the entire previous stack (+0.0502), and the prediction
   transferred **almost quantitatively** — Shellcode 0.376→**0.658** (UNSW predicted 0.650), Recon
   0.642→**0.833** (0.807), Fuzzers 0.723→**0.899**.
4. **Mechanism visible in the geometry**: the overlap attractors dissolve — svc splits Shellcode
   (ephemeral ports) from Fuzzers (BGP 179 / SMB 445) and Recon (RPC 111) from Exploits. e_1NN
   0.3632→**0.2098**. Gate precision 0.9267→0.9411 at unchanged recall — what gate re-weighting could
   never buy.

**Hypothesis → external confirmation → quantitative transfer → ablation-established gain** is a
complete scientific arc, and it is rare in this subfield. **That arc is the claim** — that we predicted
the size and the per-family shape of the gain from a representation with *zero shared columns* before
measuring it, not that service features are useful.

### 2.2 Moderate, not full, rebalancing — and `class_weight='balanced'` is a bad default — **ESTABLISHED**

`two_stage_stageB_weighting.ipynb`; **+0.0385 [0.0246, 0.0512], p<0.001** — the single largest
architectural lever in the project, **more than twice the entire 7-member ensemble** (+0.0177).

- **`balanced` ranked 12th of 14, below no weighting at all.** Use **α≈0.25** in `(1/n_c)^α`.
- **Interior optimum confirmed by two independent families.** α peaks at 0.25; effective-number peaks
  at β=0.999. Expressed as a Worms:Exploits **weight ratio**, both agree the useful band is **3–11×**,
  where `balanced` demands **128×**.
- **One principle unifies weighting and resampling:** full SMOTE underperformed class weighting
  (0.5807→0.5088, Worms inflated ~950×); capped SMOTE beat full SMOTE. Same shape, both levers.
- **Focal weighting failed** (γ=2 collapses the model to 0.3497) because the hard samples here are
  **label-ambiguous**, not informative — a reportable negative result about when focal loss applies.

### 2.3 The value of architecture is conditional on the problem being hard — **MEASURED, three demonstrations**

| Setting | e_1NN | Spread across all combiner variants | Full-stack gain |
|---|---|---|---|
| CIC-UNSW curated (hard) | 0.3629 | wide | **+0.0502** |
| CIC-IDS2017 (easy) | **0.0026** | **0.0007** | ~0 |
| Full release at real base rate | 0.2084 | **0.0038** | ~0 (best single ≈ SELV ensemble) |

On CIC-IDS2017 equal-average and a plain ExtraTrees **match** the winner. Confirmed from the other
direction by the v6 interaction result (`v6_significance.ipynb`): ensembling is **strongly
established on DERIVED features (+0.0231, p<0.001)** and **was not established on BASE** — the
combiner only pays once the features are pure enough. Neither ingredient is established alone
(features without ensembling +0.0100, p=0.071; dynamic-vs-static on BASE +0.0029, p=0.555) but
together they are (+0.0152, p=0.012) — so this is an **interaction**, and the effect is *ensembling*,
not *dynamism*.

**Consequence for the literature:** an architecture benchmarked only on a near-separable dataset has
not been benchmarked at all.

### 2.4 Dynamic / competence-based combination is closed — **mechanistically, not just empirically** — **NULL**

Five runs (`two_stage_stageB_des.ipynb`, `_dynamic_v4`, `_dynamic_v5`, `results/v71d_region_w/`,
`results/v71e_view_pool/`). The novelty is that the confounder was **eliminated by construction**:

- v5 **retracted** v4's "competence cannot be estimated" — per-member AUC is 0.82–0.85. The real
  finding: **98.1% shared row-difficulty variance vs 1.9% member-specific**. Competence is estimable,
  but *what is estimable is how hard the row is, not who should handle it.*
- v7.1e then **built a pool where that is false** — disjoint feature views drove shared variance
  93.4%→**51.2%**, median within-row weight spread 0.0000→**0.2956**, region acc-vs-spread ρ
  −0.857→**+0.071**. Dynamism **still did not pay** (pure-dynamic −0.0119 / −0.0123 / −0.0149).
- **THE MECHANISM: competence is local *accuracy*, so it minimises error rate while the objective is
  macro-F1.** Every pure-dynamic variant in every pool has the **highest accuracy and macro-precision
  and the lowest macro-recall** — buying accuracy already saturated at 98.7% and paying in rare-class
  recall.
- This is the mirror image of §1.4, and it **predicts the failure of any local-accuracy-derived
  weighting (including META-DES-class) without running it.**
- The granularity axis is closed at both ends: per-region W +0.0018 (p≈0.55), per-instance +0.0011.

### 2.5 Pool members should be selected for complementarity, not individual quality — **MEASURED, replicated 3×**

- v2: XGB **α=1.0** disagrees with the rest on 15–23% of rows while the others cluster at 5–9%, and is
  individually one of the *worst* members (0.5595) — yet it carries the gain.
- v7.1e: `V:FLAG` (0.2279) and `V:TIME` (0.3296) are **the worst single models in the project**, and
  adding them to the pool helps (**ALPHA static 0.6881 → UNION static 0.6976**).
- **With a caveat that sharpens it:** `VIEW EQUAL-AVG` is catastrophic (0.6345). The weak views only
  pay **with** learned per-class weights — complementarity is not a licence to average.
- Also: **diversity must come from the weighting axis, not the algorithm axis** — the RF/XGB
  complementarity that originally motivated stacking was an **artifact of `balanced` weighting** and
  vanishes under α=0.25.

### 2.6 TTL / `sttl` is a measured testbed label proxy — **MEASURED**

`results/ttl_join/`

- Benign is **97% sttl ∈ {31, 29}** — and those values are **0.0000 attack across 3.35M rows**. Every
  attack family is **≥98% sttl ∈ {62, 254}**.
- The rule `sttl ∈ {62,254}` **alone** recovers ~100% attack recall at fpr ≈1.6% — **matching the
  entire 99-feature learned gate.**
- TTL here encodes *which machine generated the traffic* (one router hop), and machine ↔ label by
  testbed construction. **Adding it is equivalent to adding the source IP.**
- **This retro-explains `sttl`'s notorious dominance in the UNSW-NB15 literature** — papers using it
  inherit the artifact. It also forced a **self-correction**: the "ask for TTL" half of this project's
  own earlier prescription is **withdrawn**. The service-identity half stands, because svc was
  validated by ablation and services are what attacks *target*, not who sends them.

### 2.7 The deduplication effect is protocol-specific, not universal — **MEASURED, and it corrects this project's own earlier claim**

`results/unsw_dedupe_ablation/`

| | DEDUPE | NO-DEDUPE | Δ |
|---|---|---|---|
| macro-F1 | 0.6378 | 0.6403 | **+0.0025** |
| accuracy | 0.8299 | 0.8228 | **−0.0071** |
| test rows with an exact train twin | 0.00% | **42.70%** | — |

- **No inflation despite 42.7% leakage**, and accuracy moves the *wrong* way. The widely-repeated
  "dedupe is worth 10–15 points" figure is real **only** in a binary + sequential-stream setting; it
  does **not** replicate under a multiclass batch pipeline with a stratified split.
- **Decisive, confound-free diagnostic: leaked rows score *worse* than clean ones** — 0.8064
  (n=22,007) vs 0.8350 (n=29,528), same model, same test set. **The model is not memorising.**
- **Real mechanism: duplication re-weights classes, it does not memorise.** Duplicate rates are wildly
  uneven (Generic 95.8%, Worms 2.9%); Generic inflates 0.914→0.990 but it cancels inside the
  DoS/Exploits/Analysis triangle.
- **⚠️ But dedupe *is* essential for the distance-based diagnostics** — e_1NN moves −0.061, larger
  than the entire round-1 feature gain, because the test row's own twin sits in the reference set.
- **Standing rule produced: always state task, split protocol and learner when quoting a dedupe
  effect.**

### 2.8 The model is brittle to undirected noise, and the brittleness is model-bound — **ESTABLISHED**

`results/v73_adv_budget/` → `v73b` → `v73c` → `v73d`

> **This section is the supporting evidence for §0.** The headline claim (robust detection, fragile
> attribution) is stated there; what follows is the full measurement chain behind it. Read §0 first.

- **The control is the headline.** *Undirected random* noise at 0.002 IQR (L2 0.008, **no model
  access**) costs **0.13 macro-F1** and drops attack recall 0.8994→0.7737. Adversarial search only
  doubles it. This is far broader than evasion — it implies degradation under measurement noise, a
  different flow timeout, or another CICFlowMeter build.
- **⚠️ A pre-registered hypothesis was refuted:** feature importance does **not** predict robustness.
  The prediction was a flat curve because svc is 70.4% of importance and outside attacker control;
  attack recall still halves at 0.020 IQR. **A model can lean 70% on unforgeable evidence and still be
  evaded through the other 30%.**
- **It is model-bound, not representation-bound**: at matched fpr, RF retains **0.93** of clean recall
  at budget 0.20 where the deployed XGB retains **0.46**. Depth is not the lever; it is the
  averaging-of-independent-trees bias. Survives the integer-rounding validity check (+0.0085).
- **The gate is the lever, not Stage B** — a second pre-registered prediction, also **refuted**.
  Gate recall under noise: XGB 0.9862→**0.4686**, RF 0.9823→**0.9688**. Swapping Stage B buys nothing
  (−0.0056) and costs −0.0248 clean.
- **Established in both directions** (`v73d`, 2,000 paired resamples): clean **−0.0038 [−0.0064,
  −0.0013], p=0.005** and **+0.0089 → +0.0563** under noise at all six budgets. Report as an
  **operating choice**, never as a better model.
- **The residual problem is honest and unsolved:** even with the RF gate, macro-F1 falls
  0.6827→0.2996 while attack recall holds at 0.79. **Detection can be made robust; family attribution
  cannot, by either model family.** SOC framing: *you still get the alert, you cannot trust the label
  on it.*

### 2.9 A reject class at the real base rate — and the sink **inverts** rather than dissolves — **ESTABLISHED**

`final_model_v7_1_reject.ipynb` — **+0.0242 [+0.0142, +0.0345], p<0.001**

Stage B extended to 10 classes: 9 families + "Benign-that-fooled-the-gate", trained on 34,101 OOF gate
FPs. Benign-as-attack false alarms **8,370 → 2,093 (4×)**. Shellcode **0.515 → 0.725**.

- **⚠️ The honest finding is the inversion.** Fuzzers precision 0.44→0.68 but recall 0.94→**0.58** —
  the reject class swallows 2,003 real Fuzzers. **The Fuzzers↔Benign boundary is genuinely inseparable
  in this representation; the reject class lets you *choose which side pays*** (FP flood vs FN
  swallow). State it as a SOC operating choice, never as "Fuzzers fixed".
- **It renders the gate threshold redundant** — a lever that mattered for the baseline arm
  (0.6449→0.6570 on SELV) becomes a null once the reject class exists (t=0.7 ≡ t=0.5 on test).
- **Boundary estimation is saturated**: enriching the reject set with all near-boundary benign
  (p>0.2) added only 5,240 rows and returned **+0.0009, p≈0.82** — the gate is well-calibrated, so
  there was no denser picture of the boundary to learn.

### 2.10 The plateau is a property of the representation, not of the modelling — **MEASURED**

Three independent instruments, one of which **refutes the obvious explanation**:

- **Label noise is NOT the cause** (`label_ambiguity_ceiling.ipynb`): only **1,068 rows (0.35%)** are
  ambiguous; Bayes-optimal accuracy 0.9981. Treat this notebook as **eliminative, not aspirational** —
  the bound is attained only by memorisation, and a perfect memoriser scores macro-F1 **0.0848**.
- **Class overlap IS** (`knn_class_overlap.ipynb`): ρ=0.827; Exploits and Fuzzers are **attractors**
  (Analysis is 51.4% Exploits, Shellcode 46.5% Fuzzers). **One geometric fact explains the Fuzzers FP
  sink, the collision pairs and the block-structured confusion matrix.**
- **The modelling is not trivial either**: best k-NN macro-F1 0.4458 vs Stage-B 0.6211 (**+0.175**).
  The data is hard *and* the pipeline recovers a lot.
- **Feature algebra is exhausted, and that is a measured claim**: round 2 added 38 targeted columns
  carrying **43.9% of importance** and returned **nothing** (−0.0073 single), because `handshake_bal`
  — the #1 feature overall — is a **strictly monotone rewrite** of round 1's `r_syn_fin`. 3 of the top
  5 are bounded monotone rewrites of existing columns. **That is why the bracket moved and the score
  did not.**

### 2.11 Engineered flow features that replicate across three independent extractors — **MEASURED**

`feature_engineering.ipynb`; replicated on CIC-IDS2017 (`results/cicids2017_v6/`)

31 scale-free features (ratios, per-packet normalisations, IAT/length CVs, **flags as fractions**,
header overhead, window ratio). Engineered = 41.3% of columns but **46.7% of importance** on
CIC-UNSW, and 39.7% of columns / **59.0% of importance** on CIC-IDS2017 with the top three all
derived. **`frac_PSH` is top-2 on both datasets independently.** This survives whatever the
CIC-IDS2017 headline turns out to mean.

---

## 3. Literature-audit contribution

### 3.1 A fifth gap: claimed-but-unavailable artifacts — **MEASURED**

`literature_artifact_audit.md`

The review named four gaps (weak rare-class recall, unaddressed imbalance, single-dataset evaluation,
missing explainability — **the fourth is restated in §3.2**). This audit evidences a fifth.

- **0 of 10 reviewed papers release their own code.**
- The corpus's **single GitHub link is third-party dataset code** — different owner from the paper's
  authors, contents are five baseline notebooks with **no stacked ensemble and no Boruta**, i.e. none
  of the paper's actual method, and it appears in the PDF attached to the sentence introducing *the
  dataset*. **A keyword scan counts it as a code release; reading it does not.** Every hit in the audit
  was read in context for exactly this reason.
- The most common availability statement points at a **public benchmark dataset**, which is not the
  pipeline. The dataset is what everyone already has; the pipeline decides whether 32–40% duplicate
  rows are removed before the split.
- **Why this is more than a courtesy complaint:** without the code a reader cannot tell which protocol
  a published number was produced under — **and this project has now priced several of those protocol
  choices** (§2.7 dedupe, §2.6 TTL, §1.7 tuning, curated-vs-full base rate). The protocol is worth
  more than the model.
- Two corpus corrections found while auditing: 11 PDFs are **10 unique papers** (one preprint/published
  duplicate), and this repo's own `CLAUDE.md` was stale about the corpus — corrected.

### 3.2 The explainability gap is **unverifiability**, not absence — **MEASURED**

`Clones/CyberAdaptAI/README.md` + `RESULTS.md`; corpus scan over all 10 papers

The review's original framing — "missing explainability" — **does not survive the corpus scan** and is
restated here. XAI is not absent; it is **unauditable**, which folds this gap into §3.1 rather than
standing beside it.

| XAI status | Papers |
|---|---|
| **Implements** SHAP | **1** — CyberAdaptAI (§3.5, Eq 9, Table 7) |
| Names XAI as **future work only** | 2 — "we intend to integrate SHAP and LIME"; "future work could explore XAI" |
| Passing mention / none | 7 |

- **A keyword scan would have miscounted this too** (cf. §3.1). One apparent `shap` hit is the word
  "**shap**ing"; another paper's SHAP/LIME mentions are all inside its *future work* paragraph. Only
  reading in context separates an implementation from an intention.
- **THE FINDING: CyberAdaptAI's Table 7 is almost certainly not SHAP output.** Its columns each sum to
  **exactly 1.00**, values come in **0.05 increments**, and only **five** features are listed. Real
  SHAP values sum to `f(x) − E[f(x)]`, are frequently negative, and span the full feature set. The
  caption itself hedges: *"SHAP-**Like** analysis."*
- **This project can demonstrate the contrast, not merely allege it.** The reimplementation executes
  the paper's own Eq 9 with exact TreeSHAP and reports an **additivity residual of 3.8e-06**
  (CIC-IDS2017) / **4.8e-06** (UNSW-NB15) across 40+ features — i.e. `f(x) = φ₀ + Σφⱼ` holds to
  float32 precision. **That is what the equation the paper prints actually produces.**
- **A second, independent defect in the same table:** Table 8 — captioned as the cross-dataset
  class-wise F1 scores — is a **byte-for-byte duplicate of Table 7**. So the cross-dataset F1 numbers
  **appear nowhere in the paper**, although the prose discusses them and cross-dataset generalisation
  is one of its three headline novelty claims.
- **Why this is the sharper gap:** an unreleased pipeline (§3.1) makes a *score* uncheckable; an
  unreleased XAI artifact makes the *explanation* uncheckable — and explanation is precisely what is
  offered to justify trusting the score. Reproducing the paper is what made both visible.

---

## 4. Negative results as a deliverable

Twelve pre-registered nulls, each with a diagnosed mechanism rather than a shrug. This is a
contribution because it maps the dead ends of a design space that the literature only ever reports
successes from.

| Lever | Verdict | Diagnosed mechanism |
|---|---|---|
| Full SMOTE | worse than class weighting | rarest classes inflated ~950×; interpolation noise |
| Focal weighting | collapses at γ=2 | hard samples are label-ambiguous, not informative |
| DES v1 (algorithm-diverse pool) | no transfer | pool not complementary; `balanced` artifact |
| Per-class thresholds | inverted rank order | **algebraically redundant with `W`** — verified on 100.0000% of rows |
| Soft routing | +0.0000 | gate already optimally calibrated |
| Gate threshold (×2) + gate weighting | null | weighting and thresholding are **one** degree of freedom |
| α > 1.0 | degraded | the α axis saturates at 1.0 |
| D2 feature algebra | −0.0073 | top features are monotone rewrites of existing ones |
| XGB hyperparameters | p≈0.23 | absorbed by the weighting+ensemble machinery |
| Reject-set enrichment | p≈0.82 | boundary estimation saturated (only 5,240 rows in the band) |
| Per-region W (any K) | p≈0.55 | fits in-sample (+0.0076), does not transfer |
| Dynamic weighting, **any** pool | closed | competence = local accuracy = **wrong objective** |
| **Controllability-aware feature allocation** (v7.3g) | **refuted**, clean cost −0.3354 | restricting the gate to uncontrollable features **concentrates** risk — gate recall 0.9862→0.4603 clean, then **0.0000 the instant the attacker retargets the service**, where the full-feature gate degrades gracefully |

⭐ **The generalisable claim from that last one, worth its own paragraph:** *robustness through
feature restriction concentrates risk rather than removing it.* It is the mirror image of the
importance-≠-robustness result — a model leaning **70%** on unforgeable evidence is still evaded
through the other 30% (v7.3), and a model leaning **100%** on it is destroyed the moment that
evidence becomes forgeable (v7.3g). **Feature restriction is not a robustness mechanism in either
direction.**

⚠️ **Measurement lesson from the same run, applies retroactively:** `misattributed` is
**upper-bounded by gate recall** — a row that never passes the gate cannot be mislabelled. Never
compare attribution metrics across arms with different gate recall without normalising by the rows
that actually reached Stage B.

Plus three **refuted pre-registered predictions**, which are stronger evidence than the nulls:
host/temporal context would fix the plateau (**refuted** — it is service identity); feature importance
predicts robustness (**refuted**); Stage B matters more than the gate under noise (**refuted**).

Two **self-corrections** are also on the record and should be presented as such: the "hierarchy helps
on its own" claim (retracted — it is an **interaction** with tempered weighting, +0.0178 p=0.005
jointly, −0.0061 p=0.174 alone), and "competence cannot be estimated" (retracted by v5).

---

## 5. Where the novelty maps onto the named literature gaps

| Gap from the review | What this thesis delivers | Status |
|---|---|---|
| Weak rare-class recall | α≈0.25 tempered weighting (+0.0385); reject class (+0.0242); service identity (+0.0982) — Shellcode 0.376→0.725 | ESTABLISHED |
| Unaddressed class imbalance | **Moderate-not-full rebalancing**, one principle across weighting *and* resampling; `balanced` shown to be a bad default | ESTABLISHED |
| Single-dataset evaluation | 4 datasets/representations (CIC-UNSW-2024 curated + full release, UNSW-NB15, CIC-IDS2017, TON-IoT audited) | MEASURED — **see §6** |
| ~~Missing explainability (XAI)~~ → **unverifiable explainability** | ⚠️ **GAP RESTATED — see §3.2.** The gap is not absence: CyberAdaptAI implements SHAP. It is that its published SHAP table **fails SHAP's defining property**, and no code was released to check it | MEASURED |
| **Claimed-but-unavailable artifacts (new)** | 0/10 audit; protocol choices priced | MEASURED |

⚠️ **SHAP on this project's own model is a diagnostic instrument, not a novelty claim** — see §6.1.

**The headline contribution (§0) sits OUTSIDE this table**, and that is deliberate. It is not "filling
a gap the review found" — it is **a sixth gap the review did not notice**, and should be introduced as
such rather than retrofitted onto the original four.

**Verified by scan + read-in-context, 2026-08-19:** *none of the 10 reviewed papers runs a robustness
or adversarial experiment on its own proposed model.* The terms do appear — but every `adversarial`
hit in CyberAdaptAI is a **related-work citation or reference-list entry** (it cites others' adversarial
augmentation and white-box attack work while testing none), and the one paper with substantial hits is
a **survey**, whose job is precisely to review others' adversarial work (FGSM / JSMA / DeepFool / CW).
Most uses of "robustness" elsewhere in the corpus are the informal sense — *"robust performance"* — not
a measurement. **This is the third time in this project that a keyword count and a reading disagree**
(cf. code links §3.1, SHAP §3.2); state the method, not just the count.

---

## 6. What is *not* novel — the honesty section

Put this in the thesis. It is what makes the rest credible.

1. **⏳ SHAP is not done — and it would not be novel if it were.** CyberAdaptAI (2025) already applies
   SHAP to a dynamic NIDS ensemble (§3.2), so **"we add explainability" is not available as a
   contribution** and must not be claimed as one. Two consequences: (a) the review's "missing
   explainability" gap is **restated** as *unverifiable* explainability, where this project does have
   something original (§3.2); (b) SHAP on this project's own model is worth running as a **diagnostic
   instrument** — what the reject class keys on, why `svc_rpc` carries 0.372 importance, and the
   overlap pairs service identity did *not* fix — but it belongs in the analysis chapter, not the
   novelty claim. It remains the last undelivered *planned* item.
2. **The single-dataset gap is narrowed, not closed.** CIC-UNSW-2024 and UNSW-NB15 are the *same
   traffic* through different extractors. CIC-IDS2017 and TON-IoT are genuinely different but both sit
   in the **too-easy regime** (e_1NN 0.0026 and 0.0306) and cannot test the architecture. **No
   chronological split has been run**, so the evaluation-realism gap this project criticises in others
   is not yet closed in its own work.
3. **Service identity may be partly a testbed artifact.** IXIA PerfectStorm targets attacks at
   specific services, so the +0.0982 could be partly an artifact of how the traffic was generated. A
   cross-network claim still needs chronological or cross-dataset evidence. Also note the port
   consensus was computed over the full 3.54M file including test rows — label-free and deterministic,
   the same class as the pre-split prune, **but it must be stated**.
4. **No model-capacity contribution exists, by design and by measurement.** Algorithms, ensembles,
   dynamic selection, thresholds and hyperparameters were all tested; **none was established.** Every
   gain came from weighting, information, or objectives. This is a finding, but it means the thesis
   should not claim a novel classifier.
5. **The headline is low against the literature** (0.687 vs 99%+). The defence is §1–§3, not a better
   number — and it needs the **protocol price list** to land properly (see `NEXT_STEPS.md`).
6. **Rare-class claims are not individually supportable.** Worms has 48 test rows (±0.13), Analysis 77.
   Several per-class movements quoted historically in this project are **not established**; report them
   as intervals or not at all.
7. **Intervals capture test-set sampling only.** Split and seed variance are excluded, so every CI here
   is **optimistic**. The multi-seed run is 1 of 10 complete.
8. **One confound is un-removed** in the robustness series: the reject set, Stage-B pool, `W_e2e` and
   competence reference all derive from the cached **XGB** OOF gate, so the RF arm is scored on a
   population its reject class never saw. It works **against** RF, so those results are conservative —
   but a rebuild around RF is untested.
9. ~~RF under a targeted white-box attack is untested.~~ **RESOLVED 2026-08-19** — and the resolution
   is a caveat, not a clearance. RF's advantage survives (+0.1830, H1 supported) but **RF loses 79% of
   its detection to a targeted attacker**, and **+0.6773 of its apparent robustness in v7.3c/d was
   surrogate artifact**. Never write "RF is robust" unqualified; write "more robust than XGB, against
   noise strongly and against a targeted attacker only relatively". `results/v73e_rf_whitebox/`.
10. **One load-bearing citation is unverified** — the Sarhan/Layeghy/Portmann TTL "hidden label" quote
    comes from search snippets, not a retrieved PDF.

---

## 7. The four sentences to lead a defence with

0. **"Under a realistic attacker budget, the system keeps detecting and stops explaining — you get the
   alert, you cannot trust the label on it."** — gate recall retains 99% while macro-F1 retains 44%,
   in **both** model families; the nearest published multiclass work has the data and never separates
   the two. **Lead with this one** (§0).
1. **"Every capacity-side lever is tested and none is established; every gain in this project came
   from weighting, information, or the objective."** — twelve nulls, three refuted predictions, and a
   40-config hyperparameter sweep back this.
2. **"The plateau is a property of the CICFlowMeter representation, and we proved it by removing
   it."** — diagnosed at ρ=0.827, confirmed from a second extractor with zero shared columns, then
   cashed in for +0.0982.
3. **"The protocol is worth more than the model, and we can price it."** — dedupe, TTL, base rate,
   tuning and the adversarial threat model each measured; 0 of 10 reviewed papers ship the code that
   would let a reader check theirs.
