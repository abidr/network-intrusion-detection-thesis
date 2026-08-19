"""
v7.3e: does the RF gate's robustness survive a TARGETED white-box attack?

v7.3c/v7.3d established that swapping the Stage-A gate XGB -> RF buys +0.0089..+0.0563
macro-F1 under UNDIRECTED noise for a real but small clean cost (-0.0038, p=0.005), and
that the gate is the lever rather than Stage B. Every one of those numbers was measured
under noise, or under an adversarial search aimed at the XGBoost gate.

That leaves the recommendation open on its most obvious flank. RF's decision surface is
smoother (averaged independent trees), which is the very property credited for its noise
robustness -- and smoothness is also what makes a surface easier to descend deliberately.
So the untested question is:

    Is RF actually more robust, or was it merely never the target?

Four attacker settings, all at MATCHED benign fpr (v7.3b showed that comparing families at
their default 0.5 cut measures threshold placement, not robustness -- it inverted the answer
there), all under the SAME attacker-controllable budget as v7.3:

    XGB<-XGB    white-box greedy search against the XGB gate, XGB gate deployed   (v7.3 arm)
    RF <-RF     white-box greedy search against the RF  gate, RF  gate deployed   <- THE CELL
    RF <-XGB    attack crafted on XGB, faced by the RF gate       (transfer / surrogate)
    RANDOM      same budget, no optimisation, both gates                          (control)

The RF<-XGB arm matters independently: if RF only looks robust because the v7.3 attack was
aimed at XGB, then RF<-RF will collapse while RF<-XGB holds, and the difference between those
two columns is exactly the amount of the v7.3c result that was surrogate artifact.

PRE-REGISTERED PREDICTION AND DECISION RULE ARE PRINTED BEFORE ANY RESULT (see PREREG below).

HELD FIXED, and stated rather than hidden (identical to v7.3c): the reject-class definition,
the Stage-B pool, W_e2e and the competence reference set all derive from the CACHED XGB
out-of-fold gate. So this answers "swap the deployed gate", not "rebuild the system around
RF". The confound works AGAINST RF, so an RF win here is conservative.

Attack: greedy per-feature sign search, 2 passes, cumulative offset clipped to the cap, so
passes refine direction without exceeding budget. Budget b in units of each column's TRAIN
IQR; COSTLY columns get b*0.5; IMPOSSIBLE (svc/proto/dst port) frozen; integer columns
rounded; CICFlowMeter identities and all 31 derived columns recomputed after every move.
Only test ATTACK rows are perturbed. ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
PASSES = 2
TARGET_FPR = 0.0168
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "rf_whitebox.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTION -- written before the run, printed before any result
==============================================================================
Claim under test: v7.3c/d's "RF gate is more robust" is a property of the MODEL,
not an artifact of the attack having been aimed at XGBoost.

H1 (recommendation survives): at budget 0.20 IQR, RF<-RF retains a clearly larger
    fraction of clean GATE RECALL than XGB<-XGB does. Threshold for "clearly":
    retained(RF<-RF) - retained(XGB<-XGB) > +0.10.

H0 / INVERSION (recommendation must be narrowed): the gap is <= +0.10, i.e. once the
    attacker aims at RF, its advantage shrinks into the range where it is no longer a
    useful deployment recommendation -- or reverses.

SURROGATE TEST: RF<-XGB minus RF<-RF is the share of v7.3c's RF advantage that came
    from the attack being aimed elsewhere. If that difference is large, v7.3c's headline
    overstates what a real adversary would face.

DECISION RULE, committed in advance:
  * H1 holds  -> novelty.md section 0 supporting result 3 ("detection is fixable")
                 stands as written, now covering targeted attack as well as noise.
  * H0 holds  -> section 0 result 3 must be narrowed to "detection is fixable against
                 undirected noise ONLY", and the RF recommendation in results_summary.md
                 / CLAUDE.md restricted to the drift/measurement-noise setting.
  * Either way the HEADLINE (robust detection, fragile attribution) is unaffected: the
    attribution collapse was measured in BOTH model families, so no outcome here can
    rescue family attribution. This run can only move the detection half.

NOTE ON ATTACKING AN RF: RF probabilities are piecewise-constant with granularity 1/n_trees
(0.01 at 100 trees). A greedy search can stall on ties where no single-feature move changes
the vote. The fraction of rows the search actually moved is reported per arm as
'moved' -- a low value on RF<-RF is itself a finding (search stalls), NOT evidence of
robustness, and must be reported as such.
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

# ---------------------------------------------------------------- prep (v7.3c-identical)
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
SVC_COLS = list(SVC.columns)
FEAT_ORDER = BASE_COLS + DER_COLS + SVC_COLS


def assemble(F, S):
    return pd.concat([F[BASE_COLS].astype(np.float32), derive(F)[DER_COLS], S], axis=1)[FEAT_ORDER]


X = assemble(FULL, SVC)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
del X, corr, upper, DER_ALL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("\nprep done %.0fm | %d integer columns" % ((time.time() - t_start) / 60, len(INT_COLS)))

FREE_PREFIX = ("Flow Duration", "Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle",
               "Fwd Packet Length", "Bwd Packet Length", "Packet Length", "Fwd Seg",
               "Bwd Seg", "Fwd Segment", "Bwd Segment", "Average Packet Size",
               "Fwd Header", "Bwd Header", "FWD Init Win", "Bwd Init Win", "Down/Up",
               "Flow Bytes", "Flow Packets", "Fwd Packets/s", "Bwd Packets/s",
               "Subflow", "Fwd Act Data", "Fwd Bytes", "Fwd Packet/Bulk", "Bwd Bytes",
               "Bwd Packet/Bulk", "Fwd Bulk", "Bwd Bulk")
COSTLY_PREFIX = ("Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                 "Total Length of Bwd Packet", "FIN Flag", "SYN Flag", "RST Flag",
                 "PSH Flag", "ACK Flag", "Fwd PSH", "Fwd URG", "Bwd URG", "URG Flag",
                 "CWR Flag", "ECE Flag", "Bwd PSH")


def bucket(c):
    if c.startswith(COSTLY_PREFIX): return "COSTLY"
    if c.startswith(FREE_PREFIX):   return "FREE"
    return "FREE"


BK = {c: bucket(c) for c in FULL.columns}
FREE_COLS = [c for c in FULL.columns if BK[c] == "FREE"]
COSTLY_COLS = [c for c in FULL.columns if BK[c] == "COSTLY"]
assert len(FREE_COLS) + len(COSTLY_COLS) == FULL.shape[1]
say("  FREE %d | COSTLY %d | IMPOSSIBLE %d (svc/proto/dst port, frozen)"
    % (len(FREE_COLS), len(COSTLY_COLS), len(SVC_COLS)))

# ---------------------------------------------------------------- fixed Stage-B machinery
z = np.load(ROOT / "cache_oof_v7_full.npz"); oof_gate = z["gate"]
p_gate_oof = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
M = 7; OOF10 = [zr[f"m{m}"] for m in range(M)]
attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
NC = len(attack_labels); NCR = NC + 1; IREJ = NC
LAB10 = np.array(attack_labels + [0])
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
REJECT = (y_arr == 0) & (oof_gate == 1)
yr10 = np.where(REJECT, IREJ, fam_trf)
comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
prof = lambda ps: np.hstack(ps).astype(np.float32)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


gpos_comb = comb_idx[p_gate_oof[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]; P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate_oof[comb_idx] <= 0.5]
BG = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)): BG[c, 0] = cnt


def fuse(P, W, Wd=None, p=1.0):
    if Wd is not None:
        Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(P[0].shape, np.float64)
    for m in range(M):
        coef = W[m][None, :] if Wd is None else Wp[:, m:m + 1] * W[m][None, :]
        S += coef * P[m]
    return S.argmax(1)


W = np.ones((M, NCR)) / M
def _sc(Wc):
    yp = LAB10[fuse(P_GPOS, Wc)]
    return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)
best = _sc(W)
t0 = time.time()
for _ in range(3):
    for c in range(NCR):
        for m_ in range(M):
            cur = W[m_, c]
            for gg in GRID_W:
                W[m_, c] = gg
                if W[:, c].sum() == 0: continue
                s_ = _sc(W)
                if s_ > best + 1e-7: best, cur = s_, gg
            W[m_, c] = cur
W_e2e = W
say("W_e2e fitted (%.0fs)" % (time.time() - t0))

ref = np.where(in_comb & (yr10 >= 0))[0]; y_ref = yr10[ref]
CORRECT = np.stack([(OOF10[m][ref].argmax(1) == y_ref) for m in range(M)])
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = ((n_ref.sum() / n_ref) ** 0.25); WC_REF = WC_REF / WC_REF.mean()
NN_D = NearestNeighbors(n_neighbors=30, algorithm="brute", n_jobs=-1).fit(
    prof([OOF10[m][ref] for m in range(M)]))


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


def fit_member(kind, param, Xa, ya):
    n = np.bincount(np.asarray(ya), minlength=NCR)
    if kind == "smote":
        strat = {c: param for c in range(NCR) if 0 < n[c] < param}
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat).fit_resample(Xa, ya)
        m = make_xgb(); m.fit(Xs, ys); return m
    if kind == "alpha":  m = make_xgb()
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    w = ((n.sum() / np.maximum(n, 1)) ** param); w = w / w.mean()
    sw = w[np.asarray(ya)].astype(float); sw = sw * len(sw) / sw.sum()
    m.fit(Xa, ya, sample_weight=sw); return m


t0 = time.time()
m10 = yr10 >= 0
POOL = [fit_member(k, p, X_trf[m10].reset_index(drop=True), pd.Series(yr10[m10]))
        for _, (k, p) in SPECS]
say("Stage-B pool refit %.1fm (held FIXED across all arms)" % ((time.time() - t0) / 60))

# ---------------------------------------------------------------- the two gates
sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
Xg = X_trf.iloc[sub].reset_index(drop=True); yg = (y_arr[sub] != 0).astype(int)
swg = compute_sample_weight("balanced", yg)
t0 = time.time()
GATES = {"XGB": make_xgb().fit(Xg, yg, sample_weight=swg)}
say("XGB gate trained (%.1fm)" % ((time.time() - t0) / 60))
t0 = time.time()
GATES["RF"] = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS,
                                     class_weight="balanced").fit(Xg, yg)
say("RF  gate trained (%.1fm)" % ((time.time() - t0) / 60))
del X_trf, Xg; gc.collect()

att_te = y_te != 0
i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))
say("test attack rows %d | benign %d" % (len(i_att), len(i_ben)))

THR = {}
for nm, gm in GATES.items():
    pb = gm.predict_proba(X_ben)[:, 1]
    THR[nm] = float(np.quantile(pb, 1.0 - TARGET_FPR))
    say("  gate %-3s threshold %.4f -> benign fpr %.4f" % (nm, THR[nm], float((pb > THR[nm]).mean())))
dump()


def gate_p(model, F_, S_):
    return model.predict_proba(assemble(F_, S_))[:, 1].astype(np.float32)


def _caps(budget):
    caps = {c: budget * IQR[c] for c in FREE_COLS}
    caps.update({c: budget * COSTLY_RATIO * IQR[c] for c in COSTLY_COLS})
    return caps


def _norms(off, caps):
    if not off: return 0.0, 0.0
    Z = np.stack([off[c] / IQR[c] for c in caps], 1)
    return float(np.abs(Z).max(1).mean()), float(np.linalg.norm(Z, axis=1).mean())


def craft(budget, surrogate, mode="greedy"):
    """Return perturbed attack rows. `surrogate` is the model the attacker optimises
    against -- white-box when it equals the deployed gate, a transfer attack otherwise."""
    caps = _caps(budget)
    off = {c: np.zeros(len(F_att0), np.float32) for c in caps}

    def build(o):
        # Integer columns are rounded (round -> repair -> round), identical to v7.3c's
        # perturb(). Consequence: the "XGB RANDOM control" arm here reproduces v7.3c arm A
        # exactly, which is this run's correctness check -- see REPRO block at the end.
        # v7.3's own adversarial arm did NOT round and used a 0.5 cut on a full-data gate,
        # so its REALISABLE numbers are NOT comparable to the greedy arms here.
        F_ = F_att0.copy()
        for c in caps: F_[c] = F_att0[c].values + o[c]
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        F_ = repair(F_)
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        return F_

    if mode == "random":
        rng = np.random.RandomState(RS)
        for c in caps:
            off[c] = rng.uniform(-caps[c], caps[c], len(F_att0)).astype(np.float32)
        return build(off), _norms(off, caps), float("nan")

    model = GATES[surrogate]
    p0 = gate_p(model, build(off), S_att0)
    cur = p0.copy()
    for _ in range(PASSES):
        for c in caps:
            best_p, best_o = cur, off[c]
            for dsign in (+1.0, -1.0):
                cand = np.clip(off[c] + dsign * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                p_ = gate_p(model, build(o2), S_att0)
                take = p_ < best_p
                if take.any():
                    best_p = np.where(take, p_, best_p)
                    best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_p
    moved = float((cur < p0 - 1e-9).mean())
    return build(off), _norms(off, caps), moved


def stageB(Xp):
    Pg = [m.predict_proba(Xp).astype(np.float32) for m in POOL]
    _, ii = NN_D.kneighbors(prof(Pg))
    wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
    Wd = np.empty((len(Xp), M))
    for m in range(M):
        Wd[:, m] = (CORRECT[m][ii] * wn).sum(1) / den
    return LAB10[fuse(Pg, W_e2e, Wd=Wd, p=3)]


BEN_PRED = {}
for nm, gm in GATES.items():
    gmb = gm.predict_proba(X_ben)[:, 1] > THR[nm]
    yp_ben = np.zeros(len(i_ben), int)
    if gmb.any(): yp_ben[gmb] = stageB(X_ben[gmb])
    BEN_PRED[nm] = yp_ben

# deployed gate, surrogate the attacker optimises against, mode
ARMS = [("XGB<-XGB  white-box on XGB", "XGB", "XGB", "greedy"),
        ("RF <-RF   white-box on RF", "RF", "RF", "greedy"),
        ("RF <-XGB  transfer from XGB", "RF", "XGB", "greedy"),
        ("XGB RANDOM control", "XGB", None, "random"),
        ("RF  RANDOM control", "RF", None, "random")]

rows = []
CRAFT_CACHE = {}
for tag, dep, sur, mode in ARMS:
    gm, thr = GATES[dep], THR[dep]
    say("\n=== %s ===" % tag)
    say("  %8s %10s %10s %12s %12s %8s %9s" % ("budget", "macro-F1", "accuracy",
                                               "att-recall", "gate-recall", "moved", "L2(IQR)"))
    for b in BUDGETS:
        t_arm = time.time()
        if b == 0.0:
            F_p, nrm, moved = F_att0, (0.0, 0.0), 0.0
        else:
            key = (b, sur, mode)
            if key not in CRAFT_CACHE:
                CRAFT_CACHE[key] = craft(b, sur, mode)
            F_p, nrm, moved = CRAFT_CACHE[key]
        Xp = assemble(F_p, S_att0)
        pb = gm.predict_proba(Xp)[:, 1]
        gma = pb > thr
        yp_a = np.zeros(len(Xp), int)
        if gma.any(): yp_a[gma] = stageB(Xp[gma])
        yp = np.zeros(len(y_te), int); yp[i_ben] = BEN_PRED[dep]; yp[i_att] = yp_a
        mf = f1_score(y_te, yp, average="macro", zero_division=0)
        rec = dict(arm=tag, deployed=dep, surrogate=sur, mode=mode, budget=b,
                   macro_f1=round(mf, 4), accuracy=round(float((yp == y_te).mean()), 4),
                   attack_recall=round(float((yp_a != 0).mean()), 4),
                   gate_recall=round(float(gma.mean()), 4),
                   moved=round(moved, 4), linf_iqr=round(nrm[0], 4), l2_iqr=round(nrm[1], 4),
                   secs=round(time.time() - t_arm, 1))
        say("  %8.3f %10.4f %10.4f %12.4f %12.4f %8.3f %9.3f"
            % (b, mf, rec["accuracy"], rec["attack_recall"], rec["gate_recall"],
               moved, nrm[1]))
        rows.append(rec); dump()

R = pd.DataFrame(rows)
say("\n" + "=" * 78); say("GATE RECALL by budget -- the primary question"); say("=" * 78)
pg = R.pivot(index="budget", columns="arm", values="gate_recall")
say(pg.to_string())
say("\nretained fraction of CLEAN gate recall:")
rg = (pg / pg.loc[0.0])
say(rg.round(4).to_string())

A_X = "XGB<-XGB  white-box on XGB"; A_R = "RF <-RF   white-box on RF"; A_T = "RF <-XGB  transfer from XGB"
bmax = max(BUDGETS)
gap = float(rg.loc[bmax, A_R] - rg.loc[bmax, A_X])
surrogate_infl = float(rg.loc[bmax, A_T] - rg.loc[bmax, A_R])
say("\n" + "=" * 78); say("PRE-REGISTERED TEST"); say("=" * 78)
say("  retained gate recall at budget %.2f:" % bmax)
say("    XGB<-XGB (white-box)  %.4f" % rg.loc[bmax, A_X])
say("    RF <-RF  (white-box)  %.4f" % rg.loc[bmax, A_R])
say("    RF <-XGB (transfer)   %.4f" % rg.loc[bmax, A_T])
say("  H1 statistic: retained(RF<-RF) - retained(XGB<-XGB) = %+.4f  (threshold > +0.10)" % gap)
verdict = "H1 SUPPORTED -- RF advantage survives a targeted attack" if gap > 0.10 else \
          "H0 -- RF advantage does NOT survive targeted attack; narrow the claim to noise only"
say("  VERDICT: %s" % verdict)
say("  surrogate inflation: retained(RF<-XGB) - retained(RF<-RF) = %+.4f" % surrogate_infl)
say("    (how much of v7.3c's RF advantage came from the attack being aimed at XGB)")
say("\nmacro-F1 by budget:")
say(R.pivot(index="budget", columns="arm", values="macro_f1").to_string())
say("\nsearch 'moved' fraction (low on a white-box arm = search stalled, NOT robustness):")
say(R.pivot(index="budget", columns="arm", values="moved").to_string())

# ---- correctness check: the XGB random arm must reproduce v7.3c arm A -----------------
V73C_A_MF = {0.0: 0.6865, 0.002: 0.6089, 0.01: 0.5575, 0.02: 0.4797,
             0.05: 0.3621, 0.10: 0.2978, 0.20: 0.2433}
V73C_A_GR = {0.0: 0.9862, 0.002: 0.9425, 0.01: 0.9226, 0.02: 0.8363,
             0.05: 0.6687, 0.10: 0.5611, 0.20: 0.4686}
say("\n" + "=" * 78)
say("REPRODUCTION CHECK -- 'XGB RANDOM control' vs v7.3c arm A (same gate, thr, noise)")
say("=" * 78)
say("  %8s %10s %10s %9s | %11s %11s %9s"
    % ("budget", "mf here", "mf v73c", "d", "gaterec here", "gaterec v73c", "d"))
dmax = 0.0
for b in BUDGETS:
    r_ = R[(R["arm"] == "XGB RANDOM control") & (R["budget"] == b)].iloc[0]
    d1 = r_["macro_f1"] - V73C_A_MF[b]; d2 = r_["gate_recall"] - V73C_A_GR[b]
    dmax = max(dmax, abs(d1), abs(d2))
    say("  %8.3f %10.4f %10.4f %+9.4f | %11.4f %11.4f %+9.4f"
        % (b, r_["macro_f1"], V73C_A_MF[b], d1, r_["gate_recall"], V73C_A_GR[b], d2))
say("  max abs deviation %.4f -- %s" % (dmax, "OK (within refit jitter)" if dmax < 0.005
                                        else "WARNING: does not reproduce, investigate before quoting"))

res = dict(experiment="v7.3e -- does the RF gate's robustness survive a TARGETED white-box attack",
           prereg=dict(h1="retained(RF<-RF) - retained(XGB<-XGB) > +0.10 at budget 0.20",
                       gap=round(gap, 4), verdict=verdict,
                       surrogate_inflation=round(surrogate_infl, 4)),
           held_fixed="reject set, Stage-B pool, W_e2e, competence reference (all from cached XGB OOF gate)",
           gate_train_rows=GATE_TRAIN, target_fpr=TARGET_FPR, thresholds=THR,
           passes=PASSES, costly_ratio=COSTLY_RATIO,
           buckets=dict(free=len(FREE_COLS), costly=len(COSTLY_COLS), impossible=len(SVC_COLS)),
           budgets=list(BUDGETS), results=rows,
           repro_check=dict(against="results/v73c_rf_gate arm A (XGB gate + 7-member, noise)",
                            arm="XGB RANDOM control", max_abs_deviation=round(dmax, 4),
                            ok=bool(dmax < 0.005)),
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "rf_whitebox.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
