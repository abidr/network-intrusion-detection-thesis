# Artifact availability audit of the reviewed literature

_Run 2026-08-09 over every PDF in `Papers/`._

A companion to the literature review: for each reviewed paper, **is there a usable code artifact?**
The review already names four gaps (weak rare-class recall, unaddressed imbalance, single-dataset
evaluation, missing explainability). This audit supports a fifth: **claimed but unavailable
artifacts.**

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

That connects the gap directly to this project's central methodological finding. On CIC-UNSW-NB15-2024
and UNSW-NB15 alike, whether duplicates are deduplicated moves reported accuracy by **10-15 points**
(see `Clones/CyberAdaptAI/RESULTS.md`). Without the code, a reader cannot tell which side of that
line a published number falls on — and the near-ceiling figures common in this literature are exactly
what duplicate leakage produces.

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

## Caveats

- Text extraction can miss URLs rendered inside figures, headers/footers, or as untagged hyperlink
  annotations. The scan covers extracted body text only.
- Absence of a link in the PDF does not prove no repository exists anywhere; it proves the paper does
  not point to one, which is the claim being made here.
- Only the ASENIDSB URL was resolvable and therefore checked live. The CyberAdaptAI repository claim
  has no URL to check — searches for `CyberAdaptAI` and `AdaptEnsembleNet` return no matching
  repository (recorded in `Clones/CyberAdaptAI/README.md` under *Provenance*).
