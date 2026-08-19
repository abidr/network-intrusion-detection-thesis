# Artifact availability audit of the reviewed literature

_Run 2026-08-09 over every PDF in `Papers/`._

A companion to the literature review: for each reviewed paper, **is there a usable code artifact?**
The review already names four gaps (weak rare-class recall, unaddressed imbalance, single-dataset
evaluation, missing explainability). This audit supports a fifth: **claimed but unavailable
artifacts** — and, in the companion scan added 2026-08-19, **restates the fourth**: explainability is
not missing, it is unverifiable.

## Method

Every PDF in `Papers/` was text-extracted (`pypdf`) and scanned for code-hosting domains
(`github.com`, `gitlab.com`, `bitbucket.org`, `zenodo.org`, `figshare`, `osf.io`, `codeocean`,
`sourceforge`, `kaggle.com`, `huggingface.co`, `drive.google`, `dropbox`) and for availability
phrases (`code availability`, `data availability`, `source code`, `code is available`,
`publicly available`, `upon reasonable request`, `supplementary material`, `reproducib*`).

Every hit was then read **in context** rather than counted, because a keyword scan cannot tell a
code release from a dataset citation — and in this corpus, that distinction turned out to be the
whole result.

## Result: 0 of 10 papers release their own code

| Paper | Code artifact | What the availability statement actually says |
|---|---|---|
| ASENIDSB — Stacked Ensemble + Boruta (Dasari et al.) | **none** | contains a `github.com` URL, but it is a **dataset citation** — see below |
| CyberAdaptAI (Uddamari & Sammulal, 2025) | **none** | *"modularized in Python files and documented in a Git-based repository"* — **no URL, no name, no DOI** |
| A feature selection-driven ML framework | none | *"No datasets were generated or analysed during the current study"* |
| A novel ensemble learning-based model | none | dataset download links only (KDD'99, UNSW-NB15, CIC-IDS2017) |
| A deep learning/ML approach for anomaly detection | none | *"included in the article/supplementary material, further inquiries … to the corresponding authors"* |
| Enhancing accuracy … smart cities | none | Kaggle and UCI dataset links only |
| Ensemble Learning … Based on Correlation | none | *"No new data were created in this study"* |
| Network-based intrusion detection using deep learning (Farhan et al.) | none | *"available from the corresponding authors upon request"* |
| Anomaly-Detection Technique … Ensemble Algorithm | none | **no availability statement at all** |
| NIDS: A systematic study | none | none |

### The single GitHub link is third-party dataset code

`https://github.com/anandhumohan/UNR-IDD-Classification`, cited by ASENIDSB. Verified by fetching it:

- **Owner is `anandhumohan`; the paper is by Dasari et al.** Different parties.
- Contents are five notebooks — AdaBoost, Bagging, Decision Tree, KNN, Random Forest — plus
  `UNR-IDD.csv` and a README.
- **No stacked ensemble and no Boruta feature selection**, i.e. none of ASENIDSB's actual method.
- In the PDF it appears as footnote *"Note 1"*, attached to the sentence introducing **the UNR-IDD
  dataset**.

It is a dataset source, not an artifact release. An automated scan counts it as a code link; reading
it does not.

## Two corpus corrections found while auditing

1. **The corpus is 10 unique papers, not 11 files.** `Network-based intrusion detection.pdf` and
   `Network-based_intrusion_detection_using_deep_learn.pdf` are the **same paper** — Farhan et al.,
   *"Network-based intrusion detection using deep learning technique"* — identical title, author
   list and abstract, differing only in extracted length (54,375 vs 59,595 characters) and page
   count (15 vs 16). Most likely preprint versus published version. **Do not count it twice in the
   review.**
2. **`CLAUDE.md` was stale.** It described "21 reviewed papers (`Papers/Abid/`, `Papers/Ovi/`)".
   Those subdirectories do not exist; there are 11 PDFs at the top level of `Papers/`. Corrected.

## Why this belongs in the review chapter

The CyberAdaptAI reproduction (`Clones/CyberAdaptAI/`) was undertaken because that paper's
reproducibility claim could not be honoured. This audit shows **that is not an outlier — it is the
norm across the entire reviewed set.** Not one of the ten papers ships the code behind its numbers,
and the most common availability statement points at a *public benchmark dataset*, which is not the
same thing at all: the dataset is what everyone already has; the pipeline is what determines whether
32-40% duplicate rows get removed before the split.

That connects the gap directly to this project's central methodological finding — **but the effect must
be scoped, and the scoping makes the argument stronger, not weaker.**

> ⚠️ **CORRECTED 2026-08-19.** This paragraph previously claimed deduplication moves reported accuracy
> by "10-15 points" on CIC-UNSW-NB15-2024 and UNSW-NB15 alike. **That is measured only in
> CyberAdaptAI's binary + sequential-stream setting** (`Clones/CyberAdaptAI/RESULTS.md`:
> dedup+sequential 0.8065 acc / 0.7323 F1 → no-dedup+sequential 0.9019 / 0.8825; **shuffled, the gap
> shrinks to +0.034 acc / +0.066 F1**). Under a **multiclass batch pipeline with a stratified random
> split it does not replicate at all: +0.0025 macro-F1 and −0.0071 accuracy** — despite **42.7% of
> test rows having an exact train twin** (`results/unsw_dedupe_ablation/`). **Always state task, split
> protocol and learner when quoting a dedupe effect.**

The correction is the point. The size of the leakage penalty depends on **task, split protocol and
learner** — three things papers routinely under-report — so a reader cannot even estimate the penalty
from the paper alone, let alone check it. Without the code they cannot tell which side of the line a
published number falls on, and the near-ceiling figures common in this literature are consistent with
duplicate leakage under precisely the protocols where it bites hardest.

**Suggested framing:** "artifact unavailability" is not a courtesy complaint; in this specific
subfield it is what makes the headline numbers unfalsifiable, because the single decision that most
affects them is a preprocessing step that papers do not report in enough detail to reconstruct.

### A checkable corollary — the index-column test

One consequence of the reproduction work is a rule that needs **no code at all**, only the methods
section (measured in `Clones/CyberAdaptAI/RESULTS.md`):

> **A paper that reports removing duplicate records while retaining an `id` / index column has not
> deduplicated.** With a unique row index in the feature matrix, no two rows are duplicates, so the
> dedup pass removes nothing.

Verified on UNSW-NB15: with `id` kept, the dedup and no-dedup runs are **identical to four decimal
places** in both sequential and shuffled order, and accuracy rises 0.9142 → 0.9803. The index column
does not merely add a leaky feature — it reintroduces duplicate leakage silently, while the paper can
still truthfully claim a deduplication step.

This is worth applying across the review set, since it is checkable from any methods section and the
datasets in question carry 32–40% duplicate rows.

## Companion audit: explainability is present but unverifiable

_Run 2026-08-19, same method — scan all 10 PDFs for `shap`, `lime`, `explainab`, `interpretab`,
`xai`, `feature importance`, `saliency`, then **read every hit in context**._

This was run to test the review's fourth named gap, "missing explainability". **The gap does not
survive the scan and is restated.**

| XAI status | Papers |
|---|---|
| **Implements** SHAP | **1** — CyberAdaptAI (§3.5, Eq 9, Table 7; 18 `shap` / 38 `interpretab` hits) |
| Names XAI as **future work only** | 2 — *"we intend to integrate … SHAP and LIME"*; *"future work could explore XAI"* |
| Passing mention (1–3 hits) or none | 7 |

**Reading in context mattered exactly as it did for the code links.** One paper's lone `shap` hit is
the word "**shap**ing" ("shaping the success of future research"). Another's SHAP/LIME mentions sit
entirely inside its *future work* paragraph, plus one related-work citation. A third claims
interpretability via Boruta feature ranking while conceding in the same section that its stacked
ensemble's decision process "can be a challenge" to interpret. A keyword count would score all three
as XAI papers.

### The finding: the one SHAP table fails SHAP's defining property

CyberAdaptAI's **Table 7** is almost certainly not SHAP output:

- its columns each sum to **exactly 1.00**, values come in **0.05 increments**, and only **five**
  features are listed;
- real SHAP values sum to `f(x) − E[f(x)]`, are frequently negative, and span the full feature set;
- the caption itself hedges: *"SHAP-**Like** analysis."*

**Table 8 compounds it** — captioned as the cross-dataset class-wise F1 scores, its contents are a
**byte-for-byte duplicate of Table 7** (same five features, same values, same Instance 1/2/3 columns).
The cross-dataset F1 numbers therefore **appear nowhere in the paper**, although the prose discusses
them and cross-dataset generalisation is one of its three headline novelty claims.

The reimplementation (`Clones/CyberAdaptAI/`) executes **the paper's own Eq 9** with exact TreeSHAP and
reports an additivity residual of **3.8e-06** (CIC-IDS2017) / **4.8e-06** (UNSW-NB15) across 40+
features — `f(x) = φ₀ + Σφⱼ` to float32 precision. So the contrast can be **demonstrated**, not merely
alleged.

**Why this belongs in the same chapter as the code audit.** An unreleased pipeline makes a *score*
uncheckable; an unreleased XAI artifact makes the *explanation* uncheckable — and the explanation is
what is offered to justify trusting the score. Both defects were found the same way: by attempting the
reproduction. The restated gap is **unverifiable explainability**, and it is a special case of artifact
unavailability rather than a separate deficiency.

⚠️ **Consequence for this thesis: "we add SHAP" is not available as a contribution** and must not be
claimed as one. SHAP on our own model is a diagnostic instrument for the analysis chapter. See
`novelty.md` §3.2.

## Caveats

- Text extraction can miss URLs rendered inside figures, headers/footers, or as untagged hyperlink
  annotations. The scan covers extracted body text only.
- The XAI scan shares that limitation, and counts term *occurrences* in body text: a paper could in
  principle implement XAI in a figure or appendix the extractor mangles. The 1-implements / 2-future-work
  split was confirmed by reading, not by the counts.
- Absence of a link in the PDF does not prove no repository exists anywhere; it proves the paper does
  not point to one, which is the claim being made here.
- Only the ASENIDSB URL was resolvable and therefore checked live. The CyberAdaptAI repository claim
  has no URL to check — searches for `CyberAdaptAI` and `AdaptEnsembleNet` return no matching
  repository (recorded in `Clones/CyberAdaptAI/README.md` under *Provenance*).
