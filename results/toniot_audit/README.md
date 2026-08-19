# TON-IoT audit — run before any modelling

`toniot_audit.py` is a **gate**, not an experiment. It fits nothing. It answers the two
questions that decide whether pointing the v7.1b protocol at TON-IoT is worth an evening:

1. **Is it honest?** — does any single column solve attack-vs-benign on its own?
2. **Is it hard?** — what is `e_1NN`, and therefore can the architecture matter at all?

## Running it

```bash
# drop the CSV in Resource/TON-IoT/ first, then:
C:/Users/night/miniconda3/envs/ai4beg/python.exe results/toniot_audit/toniot_audit.py

# or point it anywhere
... toniot_audit.py path/to/Train_Test_Network.csv
... toniot_audit.py path/to/dir --rows 2000000     # cap for the 22M-row full release
```

Writes `toniot_audit.txt` (UTF-8 log) and `toniot_audit.json` beside the script. Needs the
`ai4beg` env — base python has no sklearn.

If several CSVs are present it audits the one with `network` in the name, else the largest,
and lists the rest. The IoT-telemetry and OS-log CSVs are different problems; audit them
separately if you intend to use them.

## The leakage gate

A column is **SUSPECT** if, on its own, it reaches ≥ 0.95 attack recall at ≤ 0.05 fpr.
That is the bar `sttl` cleared on our own data (`results/ttl_join/`), where
`sttl ∈ {62,254}` matched the entire 99-feature learned gate and was rejected as an
endpoint label proxy — adding it is adding the source IP.

**The rule is evaluated out of sample**: the value→attack map is learned on half the rows
and scored on the other half, and a value needs ≥ 20 supporting rows to be usable. Both
guards matter. Scored in-sample, any high-cardinality column trivially wins because nearly
every value is unique to one row — that is memorisation inside the audit. Validated against
a synthetic file with a planted `src_ip` leak:

| column | values | recall (OOS) | rec_in | flagged |
|---|---|---|---|---|
| `src_ip` (planted) | 44 | **1.0000** | 1.0000 | SUSPECT ✓ |
| `ts` | 9,189 | 0.0000 | 0.9519 | no ✓ |
| `src_port` | 9,045 | 0.0000 | 0.9375 | no ✓ |

A genuine fingerprint has in-sample ≈ out-of-sample. A large gap means cardinality.

**If the gate fails**, exclude the named columns before modelling. Note the distinction the
project already established: **service/port identity survived ablation on our data (+0.0982)
because services are what attacks TARGET; endpoint identity is WHO SENT the traffic and did
not survive.** A flagged `service` column needs thought, not automatic exclusion; a flagged
`src_ip` does not.

## The difficulty read

`e_1NN` has predicted the architecture's value three times:

| dataset | e_1NN | full stack worth |
|---|---|---|
| CIC-UNSW-2024 | 0.3629 | +0.0502 |
| UNSW-NB15 | 0.3043 | +0.045 (ensembling) |
| CIC-IDS2017 | **0.0026** | **+0.0007** — too easy to test anything |

Read the **attack-only** figure; that is the Stage-B analogue comparable to ours. Near zero
means every combiner variant will tie and the honest result is a conditional-value finding,
not a benchmark win. Much above ~0.45 means checking the feature matrix before concluding
"hard dataset" — the excluded high-cardinality columns may be where the family signal lives.

## After a PASS

Run the v7.1b protocol with **SELV re-selecting from scratch**. Do not port
`HYBRID W × OLA-cw k=30 p=3` — SELV picked dynamic on CIC-UNSW and static on UNSW-NB15.
The configuration is a property of the data; the protocol is the contribution.

Levels are not comparable across datasets. Report the paired same-pipeline gap.
