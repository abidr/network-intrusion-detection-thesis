"""
XAI: VERIFIABLE explanations of the deployed v7.1b pipeline (gate + 7-member hybrid combiner).

"We add SHAP" is not a contribution -- CyberAdaptAI already has it (novelty.md 3.2). The gap is
UNVERIFIABLE explainability: its Table 7 cannot be SHAP output and its Eq 9 was never checked.
This run therefore makes every explanation carry its own verification:

  LEVEL 1  column-level exact TreeSHAP (path-dependent) on every XGBoost model -- the gate and the
           five XGB Stage-B members -- via xgboost pred_contribs (log-odds margin space).
           Verified by ADDITIVITY (sum phi + bias == margin) and by a SECOND, INDEPENDENT
           implementation (shap 0.51 TreeExplainer).
  LEVEL 2  group-level exact interventional Shapley on the DEPLOYED ensemble score, in
           probability space, over the v7.1e feature views {SVC, TIME, SIZE, FLAG}. With 4 groups
           all 16 coalitions are enumerated, so the values are exact (no sampling). Shapley is
           linear, and the fused score is S_c(x) = sum_m coef_mc(x) P_m(c|x), so
           phi_g(S_c) = sum_m coef_mc(x) phi_g(P_m(c)) EXACTLY -- the real 7-member system is
           explained, RF and ET included, not a surrogate.
           Scope: coef_mc(x) = Wp_m(x) W_e2e[m,c] holds the per-instance competence routing FIXED
           at its value for x (standard treatment of gated mixtures). Grouping also sidesteps the
           arbitrary credit split between monotone-equivalent derived columns.

Analyses (PRE-REGISTERED below): additivity, global group shares vs gain importance, the overlap
pairs (which group pushes a confused row toward the WRONG family), faithfulness (top-k deletion vs
random-k deletion), and explanation stability under the v7.3 random-noise threat model.

Same prep/splits/caches/W protocol as v7.1b (results/v71_w_e2e/), asserted; the refit reproduces
the v7.1b test score as a sanity gate. ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
GROUPS = ("SVC", "TIME", "SIZE", "FLAG")
PER_CLASS, PER_PAIR, BG_PER_CLASS = 200, 200, 4
GATE_N = 2000                                   # per side (benign / attack) for gate SHAP
KS = (1, 3, 5, 10); RAND_REPS = 5
NOISE = (0.002, 0.02); COSTLY_RATIO = 0.5
# (true label, predicted label) on gate-positive rows; 0 as prediction = routed to reject
PAIRS = [(7, 4), (4, 7), (3, 4), (4, 3), (8, 5), (5, 8), (1, 4), (0, 5), (5, 0)]
MIN_PAIR = 15
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "xai_treeshap.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTIONS -- written before the run, printed before any result
==============================================================================
P1 VERIFIABILITY. (a) every XGBoost model (gate + 5 XGB members): max |sum phi + bias - margin|
   < 1e-3 (float32 over 400 trees); (b) shap 0.51 TreeExplainer agrees with xgboost
   pred_contribs to < 1e-4 (two independent TreeSHAP implementations); (c) group-Shapley
   efficiency residual < 1e-6 AND the full coalition reproduces the deployed Stage-B decision
   on 100% of explained rows. (c) is algebraically guaranteed -- it is a correctness check of
   the code, not evidence. (a)+(b) are the substantive contrast with CyberAdaptAI.
P2 WHERE THE EVIDENCE LIVES. Stage-B ensemble: SVC has the largest share of mean|phi| (gain
   importance put 70.4% on svc). Gate: SVC is NOT the largest share (v7.3g: an svc-only gate
   reaches recall 0.46 -- service identity does not say "malicious").
P3 OVERLAP PAIRS. In the majority of evaluable pairs (>= 15 confused rows), the group with the
   largest mean push toward the WRONG family is NOT SVC: service identity already separated
   these pairs (+0.098), so the residual confusions should live in behaviour groups.
P4 FAITHFULNESS. Deleting the top-5 SHAP columns (to the train median) lowers the explained
   margin by >= 2x the drop from deleting 5 random columns, for both the gate and XGB a=0.25.
P5 STABILITY. Under random noise at 0.002 IQR (v7.3 machinery, svc frozen), among rows whose
   decision does NOT flip: median per-row Spearman(phi_clean, phi_noisy) >= 0.90 and median
   top-5 Jaccard >= 0.60, for both models. (v7.3: predictions are brittle at this budget; the
   question is whether explanations are brittle where predictions hold.)
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

# ---------------------------------------------------------------- prep (v7.3f-identical)
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
X_test = X.iloc[i_test].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
del X, corr, upper, DER_ALL, smp, df, Xraw; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361

COSTLY_PREFIX = ("Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                 "Total Length of Bwd Packet", "FIN Flag", "SYN Flag", "RST Flag",
                 "PSH Flag", "ACK Flag", "Fwd PSH", "Fwd URG", "Bwd URG", "URG Flag",
                 "CWR Flag", "ECE Flag", "Bwd PSH")
COSTLY_COLS = [c for c in FULL.columns if c.startswith(COSTLY_PREFIX)]   # half budget; rest FREE


def view_of(c):                                  # v7.1e partition, verbatim
    if c in SVC_COLS:
        return "SVC"
    lc = c.lower()
    if "flag" in lc or lc.startswith("frac_") or lc == "r_syn_fin":
        return "FLAG"
    if any(t in lc for t in ("iat", "duration", "active", "idle", "/s", "dur_per_pkt")):
        return "TIME"
    return "SIZE"


GOF = np.array([GROUPS.index(view_of(c)) for c in FEAT_ORDER])
GIDX = [np.where(GOF == g)[0] for g in range(len(GROUPS))]
assert sum(len(v) for v in GIDX) == len(FEAT_ORDER)
say("\nprep done %.0fm | %d features | groups %s"
    % ((time.time() - t_start) / 60, len(FEAT_ORDER),
       ", ".join("%s %d" % (GROUPS[g], len(GIDX[g])) for g in range(len(GROUPS)))))

# ---------------------------------------------------------------- caches + W (v7.1b-identical)
M = 7
oof_gate = np.load(ROOT / "cache_oof_v7_full.npz")["gate"]
p_gate = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF10 = [zr[f"m{m}"] for m in range(M)]
attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
NC = len(attack_labels); NCR = NC + 1; IREJ = NC
LAB10 = np.array(attack_labels + [0])
LAB2HEAD = {int(l): i for i, l in enumerate(LAB10)}
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
REJECT = (y_arr == 0) & (oof_gate == 1)
yr10 = np.where(REJECT, IREJ, fam_trf)
comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
assert REJECT.sum() == 34101

GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]
XGB_MEMBERS = [0, 1, 2, 3, 4]
IDX_A25 = MODELS.index("XGB a=0.25")


def fuse_static(probas, Wclass):
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M):
        S += probas[m] * Wclass[m][None, :]
    return S.argmax(1)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


ref = np.where(in_comb & (yr10 >= 0))[0]; y_ref = yr10[ref]
CORRECT = np.stack([(OOF10[m][ref].argmax(1) == y_ref) for m in range(M)])
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = ((n_ref.sum() / n_ref) ** 0.25); WC_REF = WC_REF / WC_REF.mean()
prof = lambda ps: np.hstack(ps).astype(np.float32)
NN_D = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
    prof([OOF10[m][ref] for m in range(M)]))


def comp_ola(idx, k):
    nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
    Wd = np.empty((nb.shape[0], M))
    for m in range(M):
        Wd[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
    return Wd


def ascent(score_fn, W0):
    W = W0.copy(); best = score_fn(W)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = score_fn(W)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best


P_REF = [OOF10[m][ref] for m in range(M)]
W_sb, _ = ascent(lambda W: f1_score(y_ref, fuse_static(P_REF, W), average="macro", zero_division=0),
                 np.ones((M, NCR)) / M)
gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]; P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)): BG[c, 0] = cnt


def e2e_of_W(W):
    yp = LAB10[fuse_static(P_GPOS, W)]
    return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)


W_u, e_u = ascent(e2e_of_W, np.ones((M, NCR)) / M)
W_w, e_w = ascent(e2e_of_W, W_sb)
W_e2e = W_u if e_u >= e_w else W_w
say("W_e2e refit: COMB e2e %.4f (v7.1b log: 0.6906, warm init)" % max(e_u, e_w))
del OOF10, P_REF, P_GPOS, zr; gc.collect()
dump()

# ---------------------------------------------------------------- refit deployed system
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
    m.fit(Xa, ya, sample_weight=sw)
    return m


t0 = time.time()
GATE = make_xgb()
yb = (y_arr != 0).astype(int)
GATE.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
m10 = yr10 >= 0
Xa10 = X_trf[m10].reset_index(drop=True)
ya10 = yr10[m10]
POOL = [fit_member(k, p, Xa10, pd.Series(ya10)) for _, (k, p) in SPECS]
say("gate + pool refit %.1fm" % ((time.time() - t0) / 60))
MED_ALL = X_trf.sample(300_000, random_state=RS).median().values.astype(np.float32)
MED_B = Xa10.median().values.astype(np.float32)
rngb = np.random.RandomState(RS)
bg_rows = np.concatenate([rngb.choice(np.where(ya10 == c)[0], BG_PER_CLASS, replace=False)
                          for c in range(NCR)])
BG_X = Xa10.iloc[bg_rows].values.astype(np.float32)
del X_trf; gc.collect()
dump()


def stageB_probas(Xdf):
    return [m.predict_proba(Xdf).astype(np.float32) for m in POOL]


def routing_coef(P):
    """deployed HYBRID W_e2e x OLA-cw k=30 p=3: coef[n,m,c] = Wp_m(x) * W_e2e[m,c]"""
    _, ii = NN_D.kneighbors(prof(P))
    Wd = comp_ola(ii, 30)
    Wp = np.power(np.clip(Wd, 1e-6, None), 3.0); Wp = Wp / Wp.sum(1, keepdims=True)
    return Wp[:, :, None] * W_e2e[None, :, :]


def fused(P, coef):
    return np.einsum("nmc,mnc->nc", coef, np.stack(P).astype(np.float64))


p_test = GATE.predict_proba(X_test)[:, 1].astype(np.float32)
gmask = p_test > 0.5
gpos = np.where(gmask)[0]
P_test = stageB_probas(X_test.iloc[gpos])
COEF_test = routing_coef(P_test)
head_test = fused(P_test, COEF_test).argmax(1)
yp = np.zeros(len(y_te), int); yp[gpos] = LAB10[head_test]
mf1 = f1_score(y_te, yp, average="macro", zero_division=0)
say("REPRODUCTION: deployed test macro-F1 %.4f acc %.4f (v7.1b 0.6874 / 0.9866; same-run refits "
    "seen 0.6874-0.6888)" % (mf1, (yp == y_te).mean()))
REPRO_OK = abs(mf1 - 0.6880) < 0.004
say("  reproduction %s" % ("OK" if REPRO_OK else "OUTSIDE known jitter -- read results with care"))
dump()

# ---------------------------------------------------------------- explain set (Stage B)
rng = np.random.RandomState(RS)
yt_g = y_te[gpos]; yp_g = yp[gpos]
pick, tags = [], []
for c in range(10):
    w = np.where(yt_g == c)[0]
    w = rng.choice(w, min(PER_CLASS, len(w)), replace=False)
    pick.append(w); tags += ["strat"] * len(w)
for a, b in PAIRS:
    w = np.where((yt_g == a) & (yp_g == b))[0]
    w = rng.choice(w, min(PER_PAIR, len(w)), replace=False)
    pick.append(w); tags += ["pair"] * len(w)
pick = np.concatenate(pick)
E_rows, first = np.unique(pick, return_index=True)       # positions within gpos
STRAT = np.zeros(len(E_rows), bool)
for i, p_ in enumerate(pick):
    if tags[i] == "strat": STRAT[np.searchsorted(E_rows, p_)] = True
E_test = gpos[E_rows]                                     # positions within test
Xe = X_test.iloc[E_test].reset_index(drop=True)
ye = y_te[E_test]; ype = yp[E_test]
N = len(Xe)
say("\nexplain set: %d gate-positive test rows (%d stratified, rest confusion-enriched)"
    % (N, STRAT.sum()))

# ================================================================ LEVEL 1: column TreeSHAP
say("\n" + "=" * 78); say("LEVEL 1 -- column-level exact TreeSHAP (log-odds), additivity + 2nd implementation")
say("=" * 78)


def contribs(model, Xdf):
    dm = xgb.DMatrix(Xdf)
    bst = model.get_booster()
    return bst.predict(dm, pred_contribs=True), bst.predict(dm, output_margin=True)


ADD = {}
gi_b = rng.choice(np.where(y_te == 0)[0], GATE_N, replace=False)
gi_a = rng.choice(np.where(y_te != 0)[0], GATE_N, replace=False)
G_rows = np.concatenate([gi_b, gi_a])
Xg = X_test.iloc[G_rows].reset_index(drop=True)
PHI_G, MG_G = contribs(GATE, Xg)
res = np.abs(PHI_G.sum(-1) - MG_G)
ADD["gate"] = dict(max=float(res.max()), mean=float(res.mean()))
PHI_M, MG_M = {}, {}
for m in XGB_MEMBERS:
    ph, mg = contribs(POOL[m], Xe)
    PHI_M[m], MG_M[m] = ph, mg
    r = np.abs(ph.sum(-1) - mg)
    ADD[MODELS[m]] = dict(max=float(r.max()), mean=float(r.mean()))
for k_, v in ADD.items():
    say("  additivity %-15s max |resid| %.2e  mean %.2e" % (k_, v["max"], v["mean"]))

XIMPL = {}
try:
    sv = np.asarray(shap.TreeExplainer(GATE).shap_values(Xg.iloc[:300]))
    XIMPL["gate"] = float(np.abs(sv - PHI_G[:300, :-1]).max())
    sv = np.asarray(shap.TreeExplainer(POOL[IDX_A25]).shap_values(Xe.iloc[:300]))   # (n, F, K)
    XIMPL["XGB a=0.25"] = float(np.abs(sv - np.transpose(PHI_M[IDX_A25][:300, :, :-1], (0, 2, 1))).max())
except Exception as ex:
    XIMPL["error"] = repr(ex)[:200]
say("  shap %s vs xgboost pred_contribs (max |diff|): %s" % (shap.__version__, XIMPL))
P1a = all(v["max"] < 1e-3 for v in ADD.values())
P1b = ("error" not in XIMPL) and all(v < 1e-4 for v in XIMPL.values())
dump()

# ================================================================ LEVEL 2: group Shapley, ensemble
say("\n" + "=" * 78); say("LEVEL 2 -- exact interventional group Shapley of the DEPLOYED fused score")
say("=" * 78)
t0 = time.time()
NG = len(GROUPS); B = len(BG_X)
Xe_np = Xe.values.astype(np.float32)
V = {}                                                    # V[mask] = list over m of (N, 10)
P_full = stageB_probas(Xe)
V[(1 << NG) - 1] = P_full
P_bg = stageB_probas(pd.DataFrame(BG_X, columns=FEAT_ORDER))
V[0] = [np.repeat(p.mean(0, keepdims=True), N, 0) for p in P_bg]
for mask in range(1, (1 << NG) - 1):
    absent = np.concatenate([GIDX[g] for g in range(NG) if not (mask >> g) & 1])
    rep = np.repeat(Xe_np, B, axis=0)
    rep[:, absent] = np.tile(BG_X[:, absent], (N, 1))
    rdf = pd.DataFrame(rep, columns=FEAT_ORDER)
    V[mask] = [m.predict_proba(rdf).astype(np.float32).reshape(N, B, NCR).mean(1) for m in POOL]
    del rep, rdf; gc.collect()
say("  %d coalitions x %d members x %d rows x %d background: %.1fm"
    % (1 << NG, M, N, B, (time.time() - t0) / 60))

from math import factorial
wS = lambda s_: factorial(s_) * factorial(NG - s_ - 1) / factorial(NG)
PHI_MEM = np.zeros((M, N, NG, NCR))
for g in range(NG):
    for mask in range(1 << NG):
        if (mask >> g) & 1: continue
        w_ = wS(bin(mask).count("1"))
        for m in range(M):
            PHI_MEM[m, :, g, :] += w_ * (V[mask | (1 << g)][m] - V[mask][m])
COEF_e = routing_coef(P_full)
PHI = np.einsum("nmc,mngc->ngc", COEF_e, PHI_MEM)          # (N, groups, classes)
BASE_S = fused(V[0], COEF_e)
S_FULL = fused(P_full, COEF_e)
eff = np.abs(PHI.sum(1) + BASE_S - S_FULL)
head_e = S_FULL.argmax(1)
repro_e = float((LAB10[head_e] == ype).mean())
say("  efficiency max |sum phi + base - S(x)| = %.2e | full coalition reproduces deployed "
    "decision on %.4f%% of rows" % (eff.max(), 100 * repro_e))
P1c = eff.max() < 1e-6 and repro_e == 1.0
dump()

# ================================================================ global shares
say("\n" + "=" * 78); say("GLOBAL -- where the evidence lives (share of mean |phi| by group)"); say("=" * 78)


def share(v):
    v = np.asarray(v, float); return v / v.sum()


phiG_grp = np.stack([PHI_G[:, GIDX[g]].sum(1) for g in range(NG)], 1)     # (2*GATE_N, NG)
gate_all = share(np.abs(phiG_grp).mean(0))
gate_ben = share(np.abs(phiG_grp[:GATE_N]).mean(0))
gate_att = share(np.abs(phiG_grp[GATE_N:]).mean(0))
st = np.where(STRAT)[0]
sb_pred = share(np.abs(PHI[st, :, head_e[st]]).mean(0))
gain_gate = share([GATE.feature_importances_[GIDX[g]].sum() for g in range(NG)])
gain_a25 = share([POOL[IDX_A25].feature_importances_[GIDX[g]].sum() for g in range(NG)])
col_share = share([len(GIDX[g]) for g in range(NG)])
say("  %-34s %s" % ("", "  ".join("%6s" % g for g in GROUPS)))
for nm, v in [("column share", col_share), ("gate SHAP (all)", gate_all),
              ("gate SHAP (benign rows)", gate_ben), ("gate SHAP (attack rows)", gate_att),
              ("gate gain importance", gain_gate),
              ("Stage-B ENSEMBLE SHAP (S_pred)", sb_pred), ("XGB a=0.25 gain importance", gain_a25)]:
    say("  %-34s %s" % (nm, "  ".join("%6.3f" % x for x in v)))

say("\n  Stage-B ensemble, per TRUE class, share of mean |phi(S_true)|:")
PER_CLASS_SHARE = {}
for c in range(10):
    w = st[ye[st] == c]
    if len(w) == 0: continue
    hc = LAB2HEAD[c]
    v = share(np.abs(PHI[w, :, hc]).mean(0)); PER_CLASS_SHARE[NAMES[c]] = [round(float(x), 4) for x in v]
    say("    %-15s n=%4d  %s" % (NAMES[c], len(w), "  ".join("%6.3f" % x for x in v)))

# column-level agreement with gain importance
AGREE = {}
for nm, phi_abs, gain in [
        ("gate", np.abs(PHI_G[:, :-1]).mean(0), GATE.feature_importances_),
        ("XGB a=0.25", np.abs(PHI_M[IDX_A25][st, MG_M[IDX_A25][st].argmax(1), :-1]).mean(0),
         POOL[IDX_A25].feature_importances_)]:
    rho = spearmanr(phi_abs, gain).correlation
    top_s = set(np.argsort(-phi_abs)[:10]); top_g = set(np.argsort(-gain)[:10])
    AGREE[nm] = dict(spearman=round(float(rho), 4), top10_overlap=len(top_s & top_g),
                     top5_shap=[FEAT_ORDER[i] for i in np.argsort(-phi_abs)[:5]],
                     top5_gain=[FEAT_ORDER[i] for i in np.argsort(-gain)[:5]])
    say("\n  %s: Spearman(mean|SHAP|, gain) %.3f | top-10 overlap %d/10" % (nm, rho, len(top_s & top_g)))
    say("    top-5 SHAP: %s" % ", ".join(AGREE[nm]["top5_shap"]))
    say("    top-5 gain: %s" % ", ".join(AGREE[nm]["top5_gain"]))
P2 = (GROUPS[int(np.argmax(sb_pred))] == "SVC") and (GROUPS[int(np.argmax(gate_all))] != "SVC")
dump()

# ================================================================ overlap pairs
say("\n" + "=" * 78)
say("OVERLAP PAIRS -- D_g = phi_g(S_wrong) - phi_g(S_true); positive = pushes toward the WRONG class")
say("=" * 78)
PAIR_RES = []
for a, b in PAIRS:
    ha, hb = LAB2HEAD[a], LAB2HEAD[b]
    conf = np.where((ye == a) & (ype == b))[0]
    ctrl = np.where((ye == a) & (ype == a))[0]
    pname = "%s -> %s" % (NAMES[a], "REJECT" if b == 0 else NAMES[b])
    if len(conf) < MIN_PAIR:
        say("  %-26s n=%d  (below %d, not evaluated)" % (pname, len(conf), MIN_PAIR)); continue
    D = PHI[conf][:, :, hb] - PHI[conf][:, :, ha]
    Dc = PHI[ctrl][:, :, hb] - PHI[ctrl][:, :, ha] if len(ctrl) else np.full((1, NG), np.nan)
    mu = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(len(conf))
    topd = np.bincount(D.argmax(1), minlength=NG) / len(conf)
    driver = GROUPS[int(np.argmax(mu))]
    say("  %-26s n=%3d | driver %-4s" % (pname, len(conf), driver))
    for g in range(NG):
        say("      %-4s mean D %+.4f (+/- %.4f) | top driver on %5.1f%% of rows | correct-%s rows %+.4f"
            % (GROUPS[g], mu[g], 1.96 * se[g], 100 * topd[g], NAMES[a], np.nanmean(Dc[:, g])))
    PAIR_RES.append(dict(pair=pname, n=int(len(conf)), n_control=int(len(ctrl)), driver=driver,
                         mean_D={GROUPS[g]: round(float(mu[g]), 4) for g in range(NG)},
                         ci95={GROUPS[g]: round(float(1.96 * se[g]), 4) for g in range(NG)},
                         top_driver_frac={GROUPS[g]: round(float(topd[g]), 4) for g in range(NG)},
                         control_mean_D={GROUPS[g]: round(float(np.nanmean(Dc[:, g])), 4)
                                         for g in range(NG)}))
n_eval = len(PAIR_RES); n_nonsvc = sum(r["driver"] != "SVC" for r in PAIR_RES)
P3 = n_eval > 0 and n_nonsvc > n_eval / 2
say("\n  pairs evaluated %d | driver is NOT SVC in %d" % (n_eval, n_nonsvc))
dump()

# ================================================================ faithfulness
say("\n" + "=" * 78); say("FAITHFULNESS -- top-k SHAP deletion vs random-k deletion (to train median)")
say("=" * 78)


def deletion(model, Xdf, phi, cls, med):
    """phi: (n, F) contributions toward the explained output; cls: None (binary) or (n,) class."""
    Xn = Xdf.values.astype(np.float32); n, F = Xn.shape
    bst = model.get_booster()
    marg = lambda A: bst.predict(xgb.DMatrix(pd.DataFrame(A, columns=FEAT_ORDER)), output_margin=True)
    pick_m = (lambda mg: mg) if cls is None else (lambda mg: mg[np.arange(n), cls])
    base = pick_m(marg(Xn))
    order = np.argsort(-phi, axis=1)
    out = {}
    for k in KS:
        A = Xn.copy(); cols = order[:, :k]
        A[np.arange(n)[:, None], cols] = med[cols]
        top = float((base - pick_m(marg(A))).mean())
        rr = []
        for r_ in range(RAND_REPS):
            rs = np.random.RandomState(RS + r_)
            cols = np.argsort(rs.rand(n, F), axis=1)[:, :k]
            A = Xn.copy(); A[np.arange(n)[:, None], cols] = med[cols]
            rr.append(float((base - pick_m(marg(A))).mean()))
        out[k] = dict(top=round(top, 4), random=round(float(np.mean(rr)), 4),
                      ratio=round(top / max(np.mean(rr), 1e-9), 2))
    return out


ga = GATE_N + np.where(MG_G[GATE_N:] > 0)[0]                  # attack rows the gate fires on
FAITH = {"gate": deletion(GATE, Xg.iloc[ga], PHI_G[ga, :-1], None, MED_ALL)}
pc = MG_M[IDX_A25][st].argmax(1)
FAITH["XGB a=0.25"] = deletion(POOL[IDX_A25], Xe.iloc[st], PHI_M[IDX_A25][st, pc, :-1], pc, MED_B)
for nm, d in FAITH.items():
    for k in KS:
        say("  %-11s k=%2d  top-k drop %.4f | random-k drop %.4f | ratio %.2f"
            % (nm, k, d[k]["top"], d[k]["random"], d[k]["ratio"]))
P4 = all(FAITH[nm][5]["ratio"] >= 2.0 for nm in FAITH)
dump()

# ================================================================ stability under noise
say("\n" + "=" * 78); say("STABILITY -- column SHAP under v7.3 random noise (svc frozen, identities repaired)")
say("=" * 78)


def perturb(test_pos, budget):
    F0 = FULL_te.iloc[test_pos].reset_index(drop=True); S0 = SVC_te.iloc[test_pos].reset_index(drop=True)
    rs = np.random.RandomState(RS); F_ = F0.copy()
    for c in FULL.columns:
        cap = budget * IQR[c] * (COSTLY_RATIO if c in COSTLY_COLS else 1.0)
        F_[c] = F0[c].values + rs.uniform(-cap, cap, len(F0)).astype(np.float32)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    F_ = repair(F_)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    return assemble(F_, S0)


def row_spearman(A, Bm):
    ra = rankdata(A, axis=1); rb = rankdata(Bm, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)


def jacc5(A, Bm):
    ta = np.argsort(-np.abs(A), 1)[:, :5]; tb = np.argsort(-np.abs(Bm), 1)[:, :5]
    return np.array([len(set(x) & set(z)) / len(set(x) | set(z)) for x, z in zip(ta, tb)])


STAB = {}
g_att_pos = G_rows[ga]
st_pos = E_test[st]
for b in NOISE:
    Xn_g = perturb(g_att_pos, b)
    ph, mg = contribs(GATE, Xn_g)
    flip = (mg > 0) != (MG_G[ga] > 0)
    rho = row_spearman(PHI_G[ga, :-1], ph[:, :-1]); jj = jacc5(PHI_G[ga, :-1], ph[:, :-1])
    Xn_b = perturb(st_pos, b)
    ph2, mg2 = contribs(POOL[IDX_A25], Xn_b)
    flip2 = mg2.argmax(1) != pc
    A0 = PHI_M[IDX_A25][st, pc, :-1]; A1 = ph2[np.arange(len(st)), pc, :-1]
    rho2 = row_spearman(A0, A1); jj2 = jacc5(A0, A1)
    for nm, fl, r_, j_ in [("gate", flip, rho, jj), ("XGB a=0.25", flip2, rho2, jj2)]:
        keep = ~fl
        d = dict(flip_rate=round(float(fl.mean()), 4),
                 stable_rows_median_spearman=round(float(np.median(r_[keep])), 4) if keep.any() else None,
                 stable_rows_median_jaccard5=round(float(np.median(j_[keep])), 4) if keep.any() else None,
                 flipped_rows_median_spearman=round(float(np.median(r_[fl])), 4) if fl.any() else None,
                 flipped_rows_median_jaccard5=round(float(np.median(j_[fl])), 4) if fl.any() else None)
        STAB.setdefault(nm, {})[b] = d
        say("  %-11s budget %.3f | flip %.4f | unchanged rows: Spearman %s Jaccard5 %s | "
            "flipped rows: Spearman %s Jaccard5 %s"
            % (nm, b, d["flip_rate"], d["stable_rows_median_spearman"], d["stable_rows_median_jaccard5"],
               d["flipped_rows_median_spearman"], d["flipped_rows_median_jaccard5"]))
P5 = all((STAB[nm][0.002]["stable_rows_median_spearman"] or 0) >= 0.90
         and (STAB[nm][0.002]["stable_rows_median_jaccard5"] or 0) >= 0.60 for nm in STAB)
dump()

# ================================================================ verdicts
say("\n" + "=" * 78); say("PRE-REGISTERED VERDICTS"); say("=" * 78)
VERD = {"P1a additivity < 1e-3": P1a, "P1b shap == pred_contribs < 1e-4": P1b,
        "P1c group efficiency + 100% reproduction": P1c,
        "P2 SVC top for Stage B, not for gate": P2,
        "P3 non-SVC driver in majority of pairs": P3,
        "P4 top-5 deletion >= 2x random": P4,
        "P5 explanations stable where predictions hold (0.002)": P5}
for k_, v in VERD.items():
    say("  %-55s %s" % (k_, "SUPPORTED" if v else "REFUTED"))

out = dict(experiment="XAI -- verifiable TreeSHAP of the deployed v7.1b pipeline",
           reproduction=dict(test_macro_f1=round(float(mf1), 4),
                             test_acc=round(float((yp == y_te).mean()), 4), ok=bool(REPRO_OK)),
           groups={GROUPS[g]: int(len(GIDX[g])) for g in range(NG)},
           explain_rows=int(N), stratified_rows=int(STRAT.sum()), background_rows=int(B),
           additivity=ADD, cross_implementation=XIMPL,
           group_efficiency_max=float(eff.max()), full_coalition_reproduction=repro_e,
           shares=dict(columns=col_share.round(4).tolist(), gate_shap=gate_all.round(4).tolist(),
                       gate_shap_benign=gate_ben.round(4).tolist(),
                       gate_shap_attack=gate_att.round(4).tolist(),
                       gate_gain=gain_gate.round(4).tolist(),
                       stageB_ensemble_shap=sb_pred.round(4).tolist(),
                       xgb_a025_gain=gain_a25.round(4).tolist(), order=list(GROUPS)),
           per_class_share=PER_CLASS_SHARE, agreement=AGREE, pairs=PAIR_RES,
           faithfulness={nm: {str(k): v for k, v in d.items()} for nm, d in FAITH.items()},
           stability={nm: {str(b): v for b, v in d.items()} for nm, d in STAB.items()},
           verdicts={k_: bool(v) for k_, v in VERD.items()},
           versions=dict(shap=shap.__version__, xgboost=xgb.__version__, numpy=np.__version__),
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "xai_treeshap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
np.savez_compressed(OUT / "xai_phi.npz", phi_group=PHI.astype(np.float32), base=BASE_S.astype(np.float32),
                    y_true=ye, y_pred=ype, strat=STRAT, phi_gate=PHI_G.astype(np.float32),
                    gate_rows=G_rows, explain_test_pos=E_test)
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
