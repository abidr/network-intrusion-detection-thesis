"""
Diagnose the v7.3e reproduction-check failure.

v7.3e's "XGB RANDOM control" deviated from v7.3c arm A by up to 0.0718 gate recall.
Hypothesis: NOT a prep divergence, but a different noise model --

    v7.3c perturb():  budget * IQR on EVERY column, uniformly
    v7.3e craft(random): FREE at budget*IQR, COSTLY at budget*0.5*IQR (attacker structure)

so v7.3e's "random" is strictly WEAKER noise and must damage less. If that is the whole
story, re-running the arm with the UNIFORM noise model must reproduce v7.3c arm A gate
recall to within refit jitter.

Gate recall only -- no Stage-B pool needed, so this is ~2 min instead of ~5.
"""
import io, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
TARGET_FPR = 0.0168
LOG = []
say = lambda s="": (print(s), sys.stdout.flush(), LOG.append(str(s)))
dump = lambda: io.open(OUT / "verify_repro.txt", "w", encoding="utf-8").write("\n".join(LOG))

t0 = time.time()
t_start = t0                      # the spliced prep block references t_start
_src = open(OUT / "rf_whitebox.py", encoding="utf-8").read()
_prep = _src.split("# ---------------------------------------------------------------- prep")[1]
_prep = _prep.split("# ---------------------------------------------------------------- fixed Stage-B")[0]
_prep = _prep.split("\n", 1)[1]           # drop the tail of the marker comment line
exec(_prep, globals())
say("prep reused from rf_whitebox.py | %.1fm | %d int cols | FREE %d COSTLY %d"
    % ((time.time() - t0) / 60, len(INT_COLS), len(FREE_COLS), len(COSTLY_COLS)))


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
Xg = X_trf.iloc[sub].reset_index(drop=True); yg = (y_arr[sub] != 0).astype(int)
gate = make_xgb().fit(Xg, yg, sample_weight=compute_sample_weight("balanced", yg))
say("gate trained %.1fm" % ((time.time() - t0) / 60))

att_te = y_te != 0
i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))
pb = gate.predict_proba(X_ben)[:, 1]
THR = float(np.quantile(pb, 1.0 - TARGET_FPR))
say("threshold %.4f -> benign fpr %.4f" % (THR, float((pb > THR).mean())))


def noise(budget, uniform):
    """uniform=True reproduces v7.3c perturb(); uniform=False is v7.3e's attacker structure."""
    F_ = F_att0.copy(); rng = np.random.RandomState(RS)
    for c in FULL.columns:
        cap = budget * IQR[c] if (uniform or c in FREE_COLS) else budget * COSTLY_RATIO * IQR[c]
        F_[c] = F_att0[c].values + rng.uniform(-cap, cap, len(F_att0)).astype(np.float32)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    F_ = repair(F_)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    return F_


V73C_GR = {0.0: 0.9862, 0.002: 0.9425, 0.01: 0.9226, 0.02: 0.8363,
           0.05: 0.6687, 0.10: 0.5611, 0.20: 0.4686}
say("\n%8s %12s %12s %12s %10s" % ("budget", "UNIFORM", "v73c armA", "d(uniform)", "ATTACKER"))
dmax_u = 0.0; rows = []
for b in BUDGETS:
    if b == 0.0:
        gu = ga = float((gate.predict_proba(assemble(F_att0, S_att0))[:, 1] > THR).mean())
    else:
        gu = float((gate.predict_proba(assemble(noise(b, True), S_att0))[:, 1] > THR).mean())
        ga = float((gate.predict_proba(assemble(noise(b, False), S_att0))[:, 1] > THR).mean())
    d = gu - V73C_GR[b]; dmax_u = max(dmax_u, abs(d))
    say("%8.3f %12.4f %12.4f %+12.4f %10.4f" % (b, gu, V73C_GR[b], d, ga))
    rows.append(dict(budget=b, uniform=round(gu, 4), v73c=V73C_GR[b],
                     dev=round(d, 4), attacker_structured=round(ga, 4)))
    dump()

say("\nmax |deviation| of UNIFORM vs v7.3c arm A: %.4f" % dmax_u)
if dmax_u < 0.005:
    say("=> CONFIRMED: prep and gate reproduce exactly. The v7.3e repro-check failure was")
    say("   a MIS-SPECIFIED CHECK (different noise model), not a divergence. v7.3e's greedy")
    say("   arms are unaffected -- they never used the v7.3c noise model.")
else:
    say("=> NOT EXPLAINED by the noise model. A real prep/gate divergence exists;")
    say("   do NOT quote v7.3e until it is found.")
say("\ntotal %.1fm" % ((time.time() - t0) / 60))
dump()
