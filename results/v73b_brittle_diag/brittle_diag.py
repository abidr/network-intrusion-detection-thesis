"""
v7.3b: is the brittleness REAL, and is it the MODEL or the REPRESENTATION?

v7.3 found that undirected random noise of 0.002 IQR costs 0.13 macro-F1 and drops attack
recall 0.8994 -> 0.7737. Two things must be settled before anyone tries to fix that.

PART 1 -- IS IT REAL? v7.3 perturbed integer-valued columns continuously. Packet counts and
flag counts are integers; adding 0.3 of a packet is not a flow anyone can send. At the small
budgets where the claim is strongest, much of the perturbation on those columns was
sub-integer and therefore physically meaningless. Here every integer-valued column is
detected and ROUNDED after perturbation (and clipped at 0), with the continuous curve kept
alongside so the difference is visible rather than assumed.

PART 2 -- MODEL OR REPRESENTATION? The same eliminative move that killed the label-noise
hypothesis. Seven model families spanning the capacity range are trained on identical data
and hit with identical noise:
    XGB d8/400   the deployed gate           high capacity
    XGB d3/400   shallower
    XGB d8/50    fewer rounds
    DT d6        single shallow tree         very low capacity
    RF 100       different inductive bias
    1-NN         nonparametric, no smoothing
    LogReg       smooth linear boundary      maximally smooth
If ALL of them are brittle, brittleness is a property of the CICFlowMeter representation --
the same conclusion the plateau reached, and consistent with Cover-Hart saying 21-38% of
attack rows are unclassifiable (rows near boundaries flip under small displacement). It
would then be a finding, not a defect. If the deployed depth-8 stack is markedly worse than
a shallow tree, it is capacity, and it is fixable -- which would also turn v7.2's "40
hyperparameter configs make no difference" into "capacity is irrelevant to accuracy and
decisive for robustness".

The attack here is RANDOM noise only, deliberately: it needs no model access, so it is a
fair test across all seven families, and it was already the stronger finding in v7.3.
Only test ATTACK rows are perturbed. Scope is the binary attack-vs-benign gate task, which
is where v7.3 localised the collapse (attack recall tracked gate recall almost exactly).
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
BUDGETS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
SUB_TRAIN = 400_000          # stratified subsample for the 7-family comparison
KNN_REF = 150_000
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "brittle_diag.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

# ---------------------------------------------------------------- prep (v7-identical)
hdr = pd.read_csv(CFM, nrows=0).columns.tolist()
FEAT76 = hdr[7:-1]
dtypes = {c: np.float32 for c in FEAT76}
dtypes["Dst Port"] = np.int32; dtypes["Protocol"] = np.int16
df = pd.read_csv(CFM, usecols=["Dst Port", "Protocol"] + FEAT76 + ["Label"], dtype=dtypes)
df = df.drop_duplicates().reset_index(drop=True)
y = df["Label"].map(CAT).astype(int).values

Xraw = df[FEAT76]
nun = Xraw.nunique(); zero = nun[nun <= 1].index.tolist()
FULL = Xraw.drop(columns=zero)
corr = FULL.corr(); upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
BASE_COLS = [c for c in FULL.columns if c not in red]


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def derive(F):
    g = lambda c: F[c] if c in F.columns else pd.Series(0.0, index=F.index)
    fpk, bpk = g("Total Fwd Packet"), g("Total Bwd packets")
    fby, bby = g("Total Length of Fwd Packet"), g("Total Length of Bwd Packet")
    tot_pk, tot_by = fpk + bpk, fby + bby
    d = {}
    d["r_pkt_fwd_bwd"] = safe(fpk, bpk); d["r_byte_fwd_bwd"] = safe(fby, bby)
    d["asym_pkt"] = safe(fpk - bpk, tot_pk); d["asym_byte"] = safe(fby - bby, tot_by)
    d["bpp_fwd"] = safe(fby, fpk); d["bpp_bwd"] = safe(bby, bpk); d["bpp_flow"] = safe(tot_by, tot_pk)
    d["r_bpp"] = safe(d["bpp_fwd"], d["bpp_bwd"])
    d["cv_flow_iat"] = safe(g("Flow IAT Std"), g("Flow IAT Mean"))
    d["cv_fwd_iat"] = safe(g("Fwd IAT Std"), g("Fwd IAT Mean"))
    d["cv_bwd_iat"] = safe(g("Bwd IAT Std"), g("Bwd IAT Mean"))
    d["iat_span_rel"] = safe(g("Flow IAT Max") - g("Flow IAT Min"), g("Flow IAT Mean"))
    d["cv_fwd_len"] = safe(g("Fwd Packet Length Std"), g("Fwd Packet Length Mean"))
    d["cv_bwd_len"] = safe(g("Bwd Packet Length Std"), g("Bwd Packet Length Mean"))
    d["cv_pkt_len"] = safe(g("Packet Length Std"), g("Packet Length Mean"))
    d["fwd_len_span_rel"] = safe(g("Fwd Packet Length Max") - g("Fwd Packet Length Min"),
                                 g("Fwd Packet Length Mean"))
    d["bwd_len_span_rel"] = safe(g("Bwd Packet Length Max") - g("Bwd Packet Length Min"),
                                 g("Bwd Packet Length Mean"))
    for flag in ("FIN", "SYN", "RST", "PSH", "ACK"):
        d["frac_" + flag] = safe(g(flag + " Flag Count"), tot_pk)
    d["r_syn_fin"] = safe(g("SYN Flag Count"), g("FIN Flag Count"))
    d["hdr_per_pkt_fwd"] = safe(g("Fwd Header Length"), fpk)
    d["hdr_per_pkt_bwd"] = safe(g("Bwd Header Length"), bpk)
    d["payload_share_fwd"] = safe(fby - g("Fwd Header Length"), fby)
    dur = g("Flow Duration")
    d["active_share"] = safe(g("Active Mean"), dur); d["idle_share"] = safe(g("Idle Mean"), dur)
    d["r_active_idle"] = safe(g("Active Mean"), g("Idle Mean")); d["dur_per_pkt"] = safe(dur, tot_pk)
    d["r_init_win"] = safe(g("FWD Init Win Bytes"), g("Bwd Init Win Bytes"))
    return pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


def repair(F):
    F = F.copy(); has = lambda c: c in F.columns
    fpk = F["Total Fwd Packet"] if has("Total Fwd Packet") else None
    bpk = F["Total Bwd packets"] if has("Total Bwd packets") else None
    fby = F["Total Length of Fwd Packet"] if has("Total Length of Fwd Packet") else None
    bby = F["Total Length of Bwd Packet"] if has("Total Length of Bwd Packet") else None
    dur_s = (F["Flow Duration"] / 1e6) if has("Flow Duration") else None
    if fpk is not None and bpk is not None:
        tot_pk = fpk + bpk
        if has("Down/Up Ratio"): F["Down/Up Ratio"] = safe(bpk, fpk)
        if dur_s is not None:
            if has("Flow Packets/s"): F["Flow Packets/s"] = safe(tot_pk, dur_s)
            if has("Fwd Packets/s"):  F["Fwd Packets/s"] = safe(fpk, dur_s)
            if has("Bwd Packets/s"):  F["Bwd Packets/s"] = safe(bpk, dur_s)
        if fby is not None and bby is not None:
            tot_by = fby + bby
            if has("Average Packet Size"): F["Average Packet Size"] = safe(tot_by, tot_pk)
            if has("Packet Length Mean"):  F["Packet Length Mean"] = safe(tot_by, tot_pk)
            if dur_s is not None and has("Flow Bytes/s"): F["Flow Bytes/s"] = safe(tot_by, dur_s)
        if fby is not None and has("Fwd Segment Size Avg"): F["Fwd Segment Size Avg"] = safe(fby, fpk)
        if bby is not None and has("Bwd Segment Size Avg"): F["Bwd Segment Size Avg"] = safe(bby, bpk)
    num = F.select_dtypes(include=[np.number]).columns
    F[num] = F[num].clip(lower=0.0)
    return F


DER_ALL = derive(FULL)
dz = DER_ALL.nunique(); DER_COLS = [c for c in DER_ALL.columns if c not in dz[dz <= 1].index]
port = df["Dst Port"].values; proto = df["Protocol"].values
SVC_PORTS = [(20, "svc_ftpdata"), (21, "svc_ftp"), (22, "svc_ssh"), (23, "svc_telnet"),
             (25, "svc_smtp"), (53, "svc_dns"), (80, "svc_http"), (110, "svc_pop3"),
             (111, "svc_rpc"), (139, "svc_netbios"), (143, "svc_imap"), (179, "svc_bgp"),
             (443, "svc_https"), (445, "svc_smb"), (514, "svc_syslog"), (520, "svc_rip"),
             (1723, "svc_pptp"), (5190, "svc_aim"), (6881, "svc_bt"), (8080, "svc_httpalt")]
KNOWN = np.array([p for p, _ in SVC_PORTS])
s = {nm: (port == p).astype(np.float32) for p, nm in SVC_PORTS}
s["svc_other_low"] = ((port < 1024) & ~np.isin(port, KNOWN)).astype(np.float32)
s["svc_registered"] = ((port >= 1024) & (port < 49152) & ~np.isin(port, KNOWN)).astype(np.float32)
s["svc_ephemeral"] = (port >= 49152).astype(np.float32)
s["proto_tcp"] = (proto == 6).astype(np.float32)
s["proto_udp"] = (proto == 17).astype(np.float32)
s["dst_port_num"] = port.astype(np.float32)
SVC = pd.DataFrame(s, index=df.index)
sz = SVC.nunique(); SVC = SVC.drop(columns=sz[sz <= 1].index.tolist())
FEAT_ORDER = BASE_COLS + DER_COLS + list(SVC.columns)


def assemble(F, S):
    return pd.concat([F[BASE_COLS].astype(np.float32), derive(F)[DER_COLS], S], axis=1)[FEAT_ORDER]


X = assemble(FULL, SVC)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
yb_trf = (y_arr != 0).astype(int); yb_te = (y_te != 0).astype(int)
FULL_te = FULL.iloc[i_test].reset_index(drop=True)
SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
del X, corr, upper, DER_ALL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("prep done %.0fm -- splits match v7/v7.1/v7.1b/v7.3" % ((time.time() - t_start) / 60))

# ---------------------------------------------------------------- integer columns
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns
            if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
say("\n=== INTEGER-VALUED COLUMNS (%d of %d) ===" % (len(INT_COLS), FULL.shape[1]))
say("  %s" % ", ".join(INT_COLS))
say("  v7.3 perturbed these continuously. Sub-integer moves are not realisable flows;")
say("  here they are ROUNDED after perturbation, and the continuous curve is kept for contrast.")
dump()

att_te = y_te != 0
i_att = np.where(att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True)
S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
rng_master = np.random.RandomState(RS)


def perturb(budget, round_ints):
    """Uniform noise within +/- budget*IQR per column, then consistency repair."""
    F_ = F_att0.copy()
    rng = np.random.RandomState(RS)
    for c in FULL.columns:
        F_[c] = F_att0[c].values + rng.uniform(-budget * IQR[c], budget * IQR[c],
                                               len(F_att0)).astype(np.float32)
    if round_ints:
        for c in INT_COLS:
            F_[c] = np.round(F_[c].values)
    F_ = repair(F_)
    if round_ints:                       # repair may reintroduce fractions on int columns
        for c in INT_COLS:
            F_[c] = np.round(F_[c].values)
    return F_, S_att0


# ---------------------------------------------------------------- PART 1: real gate
def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


t0 = time.time()
gate = make_xgb()
gate.fit(X_trf, yb_trf, sample_weight=compute_sample_weight("balanced", yb_trf))
say("\ngate refit %.1fm" % ((time.time() - t0) / 60))
clean_rec = float((gate.predict_proba(assemble(F_att0, S_att0))[:, 1] > 0.5).mean())
ben_te = np.where(~att_te)[0][:200_000]          # fpr on a benign slice, for context only
clean_fpr = float((gate.predict_proba(assemble(FULL_te.iloc[ben_te].reset_index(drop=True),
                                               SVC_te.iloc[ben_te].reset_index(drop=True))
                                      )[:, 1] > 0.5).mean())
say("clean gate: attack recall %.4f | benign fpr %.4f  [v7.1b: 0.9893 / ~0.0166]"
    % (clean_rec, clean_fpr))

say("\n=== PART 1: does integer rounding change the brittleness curve? ===")
say("  %8s %14s %14s %10s" % ("budget", "continuous", "int-rounded", "difference"))
p1 = []
for b in BUDGETS:
    if b == 0.0:
        rc, rr = clean_rec, clean_rec
    else:
        Fc, Sc = perturb(b, False); Fr, Sr = perturb(b, True)
        rc = float((gate.predict_proba(assemble(Fc, Sc))[:, 1] > 0.5).mean())
        rr = float((gate.predict_proba(assemble(Fr, Sr))[:, 1] > 0.5).mean())
    say("  %8.3f %14.4f %14.4f %10.4f" % (b, rc, rr, rr - rc))
    p1.append(dict(budget=b, recall_continuous=round(rc, 4), recall_rounded=round(rr, 4)))
dump()

# ---------------------------------------------------------------- PART 2: model families
say("\n=== PART 2: model or representation? (identical data, identical noise) ===")
sub, _ = train_test_split(np.arange(len(y_arr)), train_size=SUB_TRAIN, random_state=RS,
                          stratify=yb_trf)
Xs = X_trf.iloc[sub].reset_index(drop=True); ys = yb_trf[sub]
sw = compute_sample_weight("balanced", ys)
sc = RobustScaler().fit(Xs)
Xs_s = np.nan_to_num(sc.transform(Xs).astype(np.float32), posinf=0, neginf=0)
knn_i = np.random.RandomState(RS).choice(len(Xs), min(KNN_REF, len(Xs)), replace=False)

FAMILIES = {}
t0 = time.time()
FAMILIES["XGB d8/400 (the gate)"] = (make_xgb().fit(Xs, ys, sample_weight=sw), False)
FAMILIES["XGB d3/400"] = (make_xgb(max_depth=3).fit(Xs, ys, sample_weight=sw), False)
FAMILIES["XGB d8/50"] = (make_xgb(n_estimators=50).fit(Xs, ys, sample_weight=sw), False)
FAMILIES["DT d6"] = (DecisionTreeClassifier(max_depth=6, random_state=RS,
                                            class_weight="balanced").fit(Xs, ys), False)
FAMILIES["RF 100"] = (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS,
                                             class_weight="balanced").fit(Xs, ys), False)
FAMILIES["15-NN"] = (KNeighborsClassifier(n_neighbors=15, algorithm="brute", n_jobs=-1
                                          ).fit(Xs_s[knn_i], ys[knn_i]), True)
FAMILIES["LogReg"] = (LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1
                                         ).fit(Xs_s, ys), True)
say("  %d families trained on %d rows (%.1fm)" % (len(FAMILIES), SUB_TRAIN,
                                                  (time.time() - t0) / 60))


def proba_of(mdl, scaled, F_, S_):
    Xp = assemble(F_, S_)
    if scaled:
        Xp = np.nan_to_num(sc.transform(Xp).astype(np.float32), posinf=0, neginf=0)
    return mdl.predict_proba(Xp)[:, 1]


# Recall alone is not comparable across families: a model that shouts "attack" scores high
# recall AND looks robust for free. Every family is therefore thresholded to the DEPLOYED
# GATE'S benign false-positive rate first, so all seven sit at the same operating point.
F_ben = FULL_te.iloc[ben_te].reset_index(drop=True)
S_ben = SVC_te.iloc[ben_te].reset_index(drop=True)
say("\n  matching every family to the deployed gate's benign fpr = %.4f" % clean_fpr)
THR = {}
for nm, (mdl, scaled) in FAMILIES.items():
    pb = proba_of(mdl, scaled, F_ben, S_ben)
    THR[nm] = float(np.quantile(pb, 1.0 - clean_fpr))
    say("    %-22s threshold %.4f (realised fpr %.4f)"
        % (nm, THR[nm], float((pb > THR[nm]).mean())))

PERT = {b: (perturb(b, True) if b > 0 else (F_att0, S_att0)) for b in BUDGETS}
rows = []
say("\n  attack recall at MATCHED fpr:")
say("  %-22s %8s" % ("family", "clean") + "".join("%9.3f" % b for b in BUDGETS if b > 0))
for nm, (mdl, scaled) in FAMILIES.items():
    r = [float((proba_of(mdl, scaled, *PERT[b]) > THR[nm]).mean()) for b in BUDGETS]
    rel = [x / max(r[0], 1e-9) for x in r]
    say("  %-22s %8.4f" % (nm, r[0]) + "".join("%9.4f" % x for x in r[1:]))
    rows.append(dict(family=nm, threshold=round(THR[nm], 4), clean_recall=round(r[0], 4),
                     recall={str(b): round(x, 4) for b, x in zip(BUDGETS, r)},
                     retained={str(b): round(x, 4) for b, x in zip(BUDGETS, rel)}))
    dump()

say("\n  RETAINED FRACTION of clean recall at matched fpr (1.000 = fully robust):")
say("  %-22s %8s" % ("family", "clean") + "".join("%9.3f" % b for b in BUDGETS if b > 0))
for r_ in rows:
    say("  %-22s %8.4f" % (r_["family"], r_["clean_recall"])
        + "".join("%9.4f" % r_["retained"][str(b)] for b in BUDGETS if b > 0))
say("\n  (read BOTH columns: a family with low clean recall can retain a high fraction of")
say("   a bad score. Robustness only counts where clean recall is competitive.)")

# ---------------------------------------------------------------- verdict
say("\n" + "=" * 78); say("VERDICT"); say("=" * 78)
d1 = [p for p in p1 if p["budget"] == 0.002][0]
say("PART 1  integer rounding at budget 0.002: recall %.4f -> %.4f (%+.4f)"
    % (d1["recall_continuous"], d1["recall_rounded"],
       d1["recall_rounded"] - d1["recall_continuous"]))
say("        %s" % ("the v7.3 small-budget claim SURVIVES rounding"
                    if d1["recall_rounded"] - d1["recall_continuous"] < 0.05 else
                    "the v7.3 small-budget claim was INFLATED by sub-integer moves"))
ret02 = {r_["family"]: r_["retained"]["0.02"] for r_ in rows}
lo, hi = min(ret02.values()), max(ret02.values())
say("\nPART 2  retained recall at budget 0.02 spans %.3f (%s) to %.3f (%s)"
    % (lo, min(ret02, key=ret02.get), hi, max(ret02, key=ret02.get)))
say("        spread across seven families = %.3f" % (hi - lo))
say("        %s" % ("ALL families brittle -> REPRESENTATION-bound, report as a finding"
                    if hi < 0.8 else
                    "some family is robust -> MODEL-bound, capacity/smoothing is worth fixing"))

res = dict(experiment="brittleness: integer-rounding validity check + model-family diagnostic",
           integer_columns=INT_COLS, budgets=list(BUDGETS),
           clean_gate=dict(attack_recall=round(clean_rec, 4), benign_fpr=round(clean_fpr, 4)),
           part1_rounding=p1, part2_families=rows,
           sub_train=SUB_TRAIN, knn_ref=int(min(KNN_REF, SUB_TRAIN)),
           attack_rows=int(len(i_att)), minutes=round((time.time() - t_start) / 60, 1))
(OUT / "brittle_diag.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
