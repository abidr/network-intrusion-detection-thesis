# Network Intrusion Detection (NIDS) — Undergraduate Thesis (AIUB)

Machine-learning intrusion detection on **CIC-UNSW-NB15-2024**, built around a two-stage
hierarchical detector with tempered class weighting. See `CLAUDE.md` for the full design
record and `results_summary.md` for results.

## Dataset — not included in this repository

`Resource/CIC-UNSW-Data.csv` is **187 MB**, above GitHub's 100 MB per-file limit, so it is
git-ignored. `Resource/CIC-UNSW-Label.csv` (0.9 MB) *is* tracked.

To run the notebooks, obtain the CIC-UNSW-NB15-2024 flow features (Canadian Institute for
Cybersecurity, <https://www.unb.ca/cic/datasets/>) and place the feature file at:

```
Resource/CIC-UNSW-Data.csv
```

It must be row-aligned by position with the tracked label file — there is no shared key,
so never sort one without the other.

| file | rows | columns |
|---|---|---|
| `Resource/CIC-UNSW-Data.csv` | 447,915 | 76 numeric features (`int64`/`float64`) |
| `Resource/CIC-UNSW-Label.csv` | 447,915 | 1 (`Label`, values 0–9) |

Label mapping (attack families alphabetical, Benign = 0):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Benign | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode | Worms |

## Environment

Python 3.12 with `scikit-learn`, `xgboost`, `imbalanced-learn`, `openpyxl`.
On the author's machine these live in the `ai4beg` conda environment.
