# CIC-IDS2017 — place the 8 day-CSVs in this directory

The `MachineLearningCSV` release (CICFlowMeter flow features + `Label`). This is the
**genuinely different traffic** the project still needs: `CIC-UNSW-Data.csv` and
`Resource/UNSW-NB15/` are two extractions of the *same* traffic, so neither closes the
single-dataset-evaluation gap identified in the literature review.

Source: <https://www.unb.ca/cic/datasets/ids-2017.html> → `CSVs/MachineLearningCSV.zip`
(Kaggle mirror of the same raw CSVs: `chethuhn/network-intrusion-dataset`).

## Expected files (flat, original names)

```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

Note the shipped spelling quirks — `Wednesday-workingHours` (lowercase `w`) and
`Infilteration` (sic). Keep them; do not "fix" the names.

Monday is benign-only. Concatenate all 8 to get the full ~2.83 M flows.

## Known defects — handle before modelling

These are the reasons a CIC-IDS2017 number in a paper is hard to trust, and they are the same
class of problem `literature_artifact_audit.md` documents:

1. **Duplicates.** Measured: **10.89%** on features+label — a third of CIC-UNSW's 32%. Dedupe
   **before** the split regardless. Do *not* expect the "10–15 point accuracy swing" quoted
   elsewhere in this repo: that figure is specific to CyberAdaptAI's binary + sequential-stream
   protocol and did **not** replicate under the v6 batch pipeline
   (`results/unsw_dedupe_ablation/`). Dedupe still matters critically for the Cover–Hart bracket
   and every k-NN diagnostic.
2. **`Flow Bytes/s` / `Flow Packets/s` contain `inf` and `NaN`** (division by zero-duration
   flows). Measured on this copy: `Flow Bytes/s` 1,358 NaN + 1,509 inf; `Flow Packets/s`
   2,867 inf. Small (~0.05–0.1%) but they break `RobustScaler` and every distance
   computation. Unlike CIC-UNSW-2024, this file is *not* clean — the "0 NaNs, 0 infs" fact in
   CLAUDE.md applies only to `CIC-UNSW-Data.csv`.
3. **Column names carry leading spaces** — measured: **65 of 79** on this copy (`" Flow Duration"`).
   Naming is older CICFlowMeter v3 (`Fwd Avg Bytes/Bulk`) vs the 2024 file's `Fwd Bytes/Bulk Avg`.
   Strip and remap before any cross-dataset comparison; the two feature sets overlap but are not
   identical. All 8 files agree on the column list once stripped, so `concat` is safe.
4. **`Fwd Header Length` IS duplicated** — pandas silently renames the second copy to
   `Fwd Header Length.1`, so a duplicate-name check finds nothing. It is caught by the
   correlation prune at r=1.000. Don't rely on a name check to spot it.
5. **The three `Web Attack` labels contain a non-ASCII en-dash** (`Web Attack – Brute Force`).
   Reading as UTF-8 fails; use `encoding="latin-1"` and normalise the label strings, or the
   three web classes silently split into variants.
6. **Label errors in the original release** (Engelen et al. 2021; Liu et al. 2022). The
   cleaned variant is on Kaggle as `dhoogla/cicids2017` — useful as a **second arm** to
   quantify the cost of those errors, but do not make it the primary: it is pre-deduplicated,
   which destroys finding (1).

## Different taxonomy — this is not a label-transfer test

15 labels (BENIGN + 14 attacks: FTP/SSH-Patator, DoS Hulk/GoldenEye/slowloris/Slowhttptest,
DDoS, PortScan, Bot, Web Attack ×3, Infiltration, Heartbleed) — **not** the UNSW 10-family
taxonomy. No row-level correspondence exists. Compare *architecture behaviour* (does the
hierarchy + α≈0.25 tempered weighting + α-diverse ensemble still pay?), never absolute
macro-F1 levels against the 0.6485 headline.

`Destination Port` **is** present here, so the service-identity features that
`final_model_v6_unsw.ipynb` found carry the top importances (`svc_dns`, `svc_http`) are
buildable on this file — unlike on `CIC-UNSW-Data.csv`. TTL is still absent (CICFlowMeter does
not emit it).

If you want IPs and timestamps as well — for host/session aggregation, or for a chronological
rather than random split — that is the separate `GeneratedLabelledFlows.zip` download, not
this one.

## Git

`Resource/CIC-IDS2017/*.csv` is gitignored (~1 GB). Only this file is tracked.
