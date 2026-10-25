# v7.1b protocol on TON-IoT (`toniot_v71.py`)

Final-model protocol (`final_model_v7_1_reject_stages.ipynb`) run on `Train_Test_Network.csv`,
with SELV re-selecting from scratch. Runtime 14.3 min. Log `toniot_v71.txt`, results `toniot_v71.json`.

Deviations, all dataset-forced: `src_ip`/`src_port`/`dst_ip` excluded (audit: `src_ip` alone gives
0.9998 OOS attack recall), `dns_query`/`http_uri` excluded as text; Zeek has no CICFlowMeter
columns, so 16 Zeek-field analogues replace the 31 DERIVED features; service from Zeek's native
`service` one-hot plus the same dst-port svc block. Dedupe 211,043 -> 105,333 (dedupe runs after
the >99%-missing columns are dropped, hence 42 fewer rows than the audit's 105,375).

**Test: macro-F1 0.9601 [0.9555, 0.9650] / acc 0.9803 / ROC-AUC 0.9955** (SELV winner
`HYBRID W x OLA-cw k=30 p=1`, gate t=0.6). Weakest class mitm 0.768 (208 test rows).

- Gate near-perfect (recall 0.9988 / precision 0.9989); reject set only 91 rows, test rejects 6.
- **Architecture adds nothing**: FINAL vs 9-class baseline +0.0007 [-0.0021, +0.0033], p~0.58;
  reject -0.0005, e2e +0.0013, neither established. All 21 rules span 0.0059 on test; best single
  member 0.9603. Matches the pre-registered audit prediction (e_1NN 0.0306) and the CIC-IDS2017
  pattern: the architecture's value is conditional on the problem being hard.
