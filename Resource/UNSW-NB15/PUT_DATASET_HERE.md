# UNSW-NB15 — place the CSVs in this directory

This is the **original** UNSW-NB15 (Moustafa & Slay, 2015), the dataset CyberAdaptAI reports
Table 4 on. It is *not* the same as `../CIC-UNSW-Data.csv`, which is a 2024 CICFlowMeter
re-extraction with different features.

Download: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>

## Preferred — the pre-split form (smaller, has a header)

```
UNSW_NB15_training-set.csv     175,341 rows, 45 columns
UNSW_NB15_testing-set.csv       82,332 rows, 45 columns
```

This is what most papers mean by "UNSW-NB15". The adapter concatenates both files and lets the
framework build its own 80:20 sequential stream split, since that is what §3.2 specifies — the
official train/test split is not used as-is.

## Alternative — the full form (headerless)

```
UNSW-NB15_1.csv .. UNSW-NB15_4.csv    2,540,044 rows total, 49 columns, NO header
NUSW-NB15_features.csv                required to name the columns
```

Note the typo in the features filename (`NUSW`, not `UNSW`) — that is how it ships.

## Then run

```bash
cd ../../Clones/CyberAdaptAI

# Binary (attack vs benign) -- the setting comparable to the paper's Table 4
C:/Users/night/miniconda3/envs/ai4beg/python.exe run_experiment.py \
    --dataset unsw-nb15 --task binary --out results/unsw_binary

# 10-class, for comparison against the thesis baselines
C:/Users/night/miniconda3/envs/ai4beg/python.exe run_experiment.py \
    --dataset unsw-nb15 --task multiclass --out results/unsw_multiclass
```

The adapter auto-detects which form is present. Everything is wired and tested against
synthetic files of the correct shape; only the real data is missing.
