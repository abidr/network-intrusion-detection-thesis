# TON-IoT — place the network flow CSV(s) in this directory

TON-IoT (Moustafa, UNSW Canberra, 2020) — IoT/IIoT testbed telemetry plus Zeek-derived
network flows. This is the **genuinely different traffic** the project still needs:
`CIC-UNSW-Data.csv` and `Resource/UNSW-NB15/` are two extractions of the *same* traffic, and
`Resource/CIC-IDS2017/` turned out too easy to test anything (`e_1NN = 0.0026`). Neither closes
the single-dataset-evaluation gap the literature review identified.

Source: <https://research.unsw.edu.au/projects/toniot-datasets> (also on IEEE DataPort and
Kaggle). Take the **Network** subset — `Processed_Network_dataset/` or the sampled
`Train_Test_Network.csv`.

## Expected files

```
Train_Test_Network.csv          # the ~461 K-row sampled set — start here
```

or the full release, which is far larger (~22 M rows across `Network_dataset_1..23.csv`).
The audit script caps reads at 3 M rows by default; pass `--rows N` to change it, and note the
`TRUNCATED` flag it prints.

The IoT-telemetry (`Train_Test_IoT_*.csv`) and OS-log (`Train_Test_Windows/Linux_*.csv`)
subsets are **different problems with different feature spaces**. Do not concatenate them with
the network flows. Audit them separately if you intend to use them at all.

## RUN THE AUDIT FIRST — it is a gate, not a formality

```bash
C:/Users/night/miniconda3/envs/ai4beg/python.exe results/toniot_audit/toniot_audit.py
```

It fits nothing and takes minutes. See `results/toniot_audit/README.md`. Two verdicts matter:

1. **Leakage.** TON-IoT ships `src_ip`, `dst_ip`, `src_port` and `ts`. On a testbed, endpoint
   identity ↔ label by construction — this is exactly the failure mode that killed `sttl`
   (`results/ttl_join/`), where `sttl ∈ {62,254}` alone matched the entire 99-feature learned
   gate. Any column flagged SUSPECT must be excluded before modelling.
   > Keep the distinction the project established: **service/port identity survived ablation on
   > our data (+0.0982) because services are what attacks TARGET; endpoint identity is WHO SENT
   > the traffic and did not survive.** A flagged `src_ip` is disqualifying; a flagged `service`
   > needs thought, not reflex exclusion.
2. **Difficulty.** `e_1NN` has predicted the architecture's value three times. Read the
   **attack-only** figure — that is the Stage-B analogue comparable to our 0.3629. Near zero
   means every combiner variant will tie and the honest write-up is a conditional-value
   finding, not a benchmark win.

## Measured facts (audited 2026-08-19, `results/toniot_audit/`)

The copy in this directory: **211,043 rows × 44 columns** — note this is *not* the 461k-row version
usually cited; the file is complete (211,044 lines incl. header), just a smaller sample.

- **Author-balanced, so the base rate is artificial**: `normal` 50,000 (23.7%), eight families at
  exactly 20,000 each, `mitm` 1,043. **Attack share 76.31%**, imbalance 48:1.
- **Duplicates 50.07%** on features+label — the highest in the project (CIC-UNSW 32%, UNSW-NB15 36.8%,
  CIC-IDS2017 10.86%); 1,781 rows carry conflicting labels.
- **LEAKAGE GATE FAILS**: `src_ip` reaches out-of-sample recall **0.9998 at fpr 0.0402** (in-sample
  0.9999). Exclude it. `dst_ip` (recall 0.9949 / fpr 0.7733) and `src_port` (in-sample 0.9720 →
  out-of-sample 0.2797) are correctly *not* flagged.
- **15 columns are >99% missing** — all `ssl_*`, `http_*`, `weird_*`. The application-layer richness
  TON-IoT is known for is essentially absent here. Prune 52 → 46 after dropping them.
- **TOO EASY to test the architecture**: attack-only **e_1NN 0.0306**, accuracy ceiling [0.969, 0.985].
  That is the CIC-IDS2017 regime, where all 17 combiner variants landed within 0.0007 — not the
  CIC-UNSW regime (0.3629). Eight of nine families have k=10 purity ≥ 0.91; only `mitm` is below.
- **Service identity is worth only −0.0028 e_1NN here vs −0.153 on CIC-UNSW** — as predicted, because
  Zeek already ships `service` and `dst_port`, so there is no omission to recover.

**Best use: a pre-registered prediction test.** Write down in advance that every combiner variant will
tie within ~0.005, then run it — a *predicted* null on a fourth dataset is much stronger evidence for
"architecture value is conditional on hardness" than three post-hoc ones. Do not report it as a
benchmark win. For a genuine hardness test, try `Processed_datasets/Processed_Network_dataset/` at the
real base rate instead.

## Known handling quirks

1. **Missing values are a bare dash `-`**, not empty or `NaN`. Pandas will not catch this by
   default, and it silently turns numeric Zeek columns (`duration`, `src_bytes`, …) into
   `object`. The audit passes `na_values=["-", …]`; do the same in any pipeline.
2. **High-cardinality text columns** — `dns_query`, `http_uri`, `http_user_agent`,
   `ssl_subject`, `ssl_issuer`. These are content, not flow behaviour, and are also plausible
   label proxies on a scripted testbed. The audit excludes them from the feature matrix and
   reports them separately. Decide deliberately before using any of them.
3. **Two label columns** — `label` (binary 0/1) and `type` (multiclass). Use `type`; the audit
   prefers it automatically and derives the binary view from it.
4. **Tiny classes.** `mitm` in particular is small enough that it may not support a stratified
   split. Print the per-class support table before any metric, and never put a class with a
   handful of test rows into a headline macro average — the same rule that applies to
   Heartbleed and Infiltration on CIC-IDS2017, and to Worms (49 rows, ±0.13) on our own data.
5. **Duplicates.** Measure before assuming. Dedupe before the split regardless — and do *not*
   expect the "10–15 point accuracy swing" quoted elsewhere in this repo: that figure is
   specific to CyberAdaptAI's binary + sequential-stream protocol and did **not** replicate
   under the v6 batch pipeline (`results/unsw_dedupe_ablation/`, +0.0025 macro-F1 despite 42.7%
   leakage). Dedupe still matters critically for the Cover–Hart bracket and every k-NN
   diagnostic.

## Different taxonomy — this is not a label-transfer test

10 labels (normal + backdoor, ddos, dos, injection, mitm, password, ransomware, scanning, xss)
— **not** the UNSW 10-family taxonomy, despite both having ten. `backdoor`, `ddos`/`dos` and
`scanning` are near-cognates of UNSW's Backdoor, DoS and Reconnaissance; `injection`, `xss`,
`password` and `ransomware` have no UNSW counterpart, and UNSW's Exploits/Fuzzers/Generic/
Shellcode/Worms have none here. No row-level correspondence exists.

Compare **architecture behaviour** — does the hierarchy + α≈0.25 tempered weighting + reject
class + end-to-end `W` objective still pay? — never absolute macro-F1 levels against the
0.6874 headline. Report the paired same-pipeline gap.

**Let SELV re-select from scratch.** Do not port `HYBRID W × OLA-cw k=30 p=3`: SELV picked
dynamic on CIC-UNSW and static on UNSW-NB15. The configuration is a property of the data; the
protocol is the contribution.

## Git

`Resource/TON-IoT/*.csv` is gitignored. Only this file is tracked.
