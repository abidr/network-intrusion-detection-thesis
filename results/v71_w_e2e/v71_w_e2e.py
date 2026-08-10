"""
v7.1 follow-up: refit the reject-arm per-class W under the END-TO-END objective.

v7.1's W was fitted to maximise Stage-B 10-class macro-F1, where the reject class
is 1/10 of the objective. End-to-end, a correct reject barely moves Benign F1
(support 505k) while a swallowed attack costs its family's recall directly. This
script refits W by the same coordinate ascent but scoring the full 10-class
END-TO-END macro-F1 over the COMB half (reject -> Benign, gate-negative rows as
fixed background). Zero new parameters; only the objective changes.

Protocol: same prep/splits as v7/v7.1 (asserted); reuses cache_oof_v7_full.npz,
cache_oof_v71_gateproba.npz, cache_oof_v71_reject.npz. SELV (t=0.5) selects among
the W_e2e variants; test once; paired bootstrap vs the v7.1 winner (same-run refit).
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "v71_w_e2e.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
BASE = FULL.drop(columns=red)


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


F = FULL
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
d["fwd_len_span_rel"] = safe(g("Fwd Packet Length Max") - g("Fwd Packet Length Min"), g("Fwd Packet Length Mean"))
d["bwd_len_span_rel"] = safe(g("Bwd Packet Length Max") - g("Bwd Packet Length Min"), g("Bwd Packet Length Mean"))
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
DER = pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
dz = DER.nunique(); DER = DER.drop(columns=dz[dz <= 1].index.tolist())

port = df["Dst Port"].values
proto = df["Protocol"].values
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
X = pd.concat([BASE.astype(np.float32), DER, SVC], axis=1)
del d, Xraw, FULL, BASE, DER, SVC, corr, upper, s, df, F
del fpk, bpk, fby, bby, tot_pk, tot_by, dur; gc.collect()

idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True)
X_test = X.iloc[i_test].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
del X; gc.collect()

attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
NC = len(attack_labels)
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values

comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
assert len(y_arr) == 2085442 and len(y_te) == 521361 and att_trf.sum() == 65191
say("prep done %.0fm -- splits match v7/v7.1" % ((time.time() - t_start) / 60))

# ---------------------------------------------------------------- caches
M = 7
z = np.load(ROOT / "cache_oof_v7_full.npz")
oof_gate = z["gate"]
p_gate = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF10 = [zr[f"m{m}"] for m in range(M)]
REJECT = (y_arr == 0) & (oof_gate == 1)
NCR = NC + 1; IREJ = NC
yr10 = np.where(REJECT, IREJ, fam_trf)
LAB10 = np.array(attack_labels + [0])
say("caches loaded; reject set %d" % REJECT.sum())
dump()

# ---------------------------------------------------------------- combiner machinery
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]


def fuse_static(probas, Wclass):
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M):
        S += probas[m] * Wclass[m][None, :]
    return S.argmax(1)


def fuse_dyn(probas, Wd, p=1.0, Wclass=None):
    Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M):
        coef = Wp[:, m:m + 1] if Wclass is None else Wp[:, m:m + 1] * Wclass[m][None, :]
        S += coef * probas[m]
    return S.argmax(1)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


# v7.1 reference state (kNN ref = comb attacks + comb reject rows; unchanged)
ref = np.where(in_comb & (yr10 >= 0))[0]
y_ref = yr10[ref]
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


# ---------------------------------------------------------------- W under both objectives
# Stage-B objective (v7.1's): macro-F1 of the 10-class problem on ref rows
def fit_W_stageB():
    P = [OOF10[m][ref] for m in range(M)]
    W = np.ones((M, NCR)) / M
    best = f1_score(y_ref, fuse_static(P, W), average="macro", zero_division=0)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = f1_score(y_ref, fuse_static(P, W), average="macro", zero_division=0)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best


# E2E objective: 10-class end-to-end macro-F1 over the whole COMB half.
# Gate-negative rows are fixed background (predicted Benign); only the
# gate-positive rows' fusion changes with W.
gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]
P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG[c, 0] = cnt                                   # gate-negatives predicted Benign
say("COMB gate-positives %d | background rows %d" % (len(gpos_comb), len(neg_comb)))


def e2e_of_W(W):
    yp = LAB10[fuse_static(P_GPOS, W)]
    cm = np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG
    return macro_f1_from_counts(cm)


def fit_W_e2e(W0):
    W = W0.copy()
    best = e2e_of_W(W)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = e2e_of_W(W)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best


t0 = time.time()
W_sb, sb_score = fit_W_stageB()
say("W_stageB refit %.0fs (Stage-B mF1 %.4f -- v7.1 notebook: 0.6604)" % (time.time() - t0, sb_score))
t0 = time.time()
W_e2e_u, e_u = fit_W_e2e(np.ones((M, NCR)) / M)
W_e2e_w, e_w = fit_W_e2e(W_sb)
W_e2e, e_best, init_used = (W_e2e_u, e_u, "uniform") if e_u >= e_w else (W_e2e_w, e_w, "warm(W_sb)")
say("W_e2e fitted %.0fs: COMB e2e mF1 uniform-init %.4f | warm-init %.4f -> using %s"
    % (time.time() - t0, e_u, e_w, init_used))
say("reference COMB e2e mF1: W_stageB %.4f | uniform %.4f"
    % (e2e_of_W(W_sb), e2e_of_W(np.ones((M, NCR)) / M)))
say("\nreject-column weights (member x reject): W_sb %s -> W_e2e %s"
    % (np.array2string(W_sb[:, IREJ], precision=2), np.array2string(W_e2e[:, IREJ], precision=2)))
dump()

# ---------------------------------------------------------------- SELV selection (t=0.5)
gmask_selv = p_gate[selv_idx] > 0.5
gpos_selv = selv_idx[gmask_selv]
P_selv = [OOF10[m][gpos_selv] for m in range(M)]
_, i_selv = NN_D.kneighbors(prof(P_selv))
Wd30 = comp_ola(i_selv, 30)


def score_e2e(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int)
    yp[gmask] = LAB10[fam_g]
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


def cand(P_g, Wd):
    return {"STATIC W_e2e": fuse_static(P_g, W_e2e),
            "HYBRID W_e2e x OLA-cw k=30 p=1": fuse_dyn(P_g, Wd, p=1, Wclass=W_e2e),
            "HYBRID W_e2e x OLA-cw k=30 p=3": fuse_dyn(P_g, Wd, p=3, Wclass=W_e2e),
            "v7.1 ref: HYBRID W_sb x OLA-cw k=30 p=3": fuse_dyn(P_g, Wd, p=3, Wclass=W_sb)}


selv_tab = pd.DataFrame([score_e2e(fg, y_arr[selv_idx], gmask_selv, v)
                         for v, fg in cand(P_selv, Wd30).items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV (t=0.5) ---"); say(selv_tab.to_string(index=False))
new_variants = [v for v in selv_tab.variant if not v.startswith("v7.1 ref")]
BESTV = selv_tab[selv_tab.variant.isin(new_variants)].iloc[0]["variant"]
say("SELV winner among W_e2e variants: %s" % BESTV)
del P_selv, i_selv, Wd30; gc.collect()
dump()

# ---------------------------------------------------------------- refit pool, test once
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
gate_f = make_xgb()
yb = (y_arr != 0).astype(int)
gate_f.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
m10 = yr10 >= 0
Xa10 = X_trf[m10].reset_index(drop=True)
ya10s = pd.Series(yr10[m10])
FINAL10 = [fit_member(k, p, Xa10, ya10s) for _, (k, p) in SPECS]
say("gate + pool refit %.0fm" % ((time.time() - t0) / 60))
dump()

p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
gmask_test = p_test > 0.5
gpos_test = np.where(gmask_test)[0]
P_test = [m.predict_proba(X_test.iloc[gpos_test]).astype(np.float32) for m in FINAL10]
_, i_test_nn = NN_D.kneighbors(prof(P_test))
Wd30_t = comp_ola(i_test_nn, 30)

test_preds = cand(P_test, Wd30_t)
test_tab = pd.DataFrame([score_e2e(fg, y_te, gmask_test, v) for v, fg in test_preds.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- TEST (selection made on SELV) ---"); say(test_tab.to_string(index=False))


def to_yp(fam_g):
    yp = np.zeros(len(y_te), int); yp[gmask_test] = LAB10[fam_g]; return yp


yp_new = to_yp(test_preds[BESTV])
yp_ref = to_yp(test_preds["v7.1 ref: HYBRID W_sb x OLA-cw k=30 p=3"])
sel = test_tab[test_tab.variant == BESTV].iloc[0]
say("\nSELV-selected '%s': test mF1 %.4f acc %.4f" % (BESTV, sel["mF1"], sel["acc"]))

# paired bootstrap vs same-run v7.1 reference
code_a = y_te * 10 + yp_ref
code_b = y_te * 10 + yp_new
rng = np.random.RandomState(RS)
cls_idx = [np.where(y_te == c)[0] for c in range(10)]
diffs = np.empty(2000); f1b = np.empty(2000)
for it in range(2000):
    bidx = np.concatenate([ci[rng.randint(0, len(ci), len(ci))] for ci in cls_idx])
    cma = np.bincount(code_a[bidx], minlength=100).reshape(10, 10)
    cmb = np.bincount(code_b[bidx], minlength=100).reshape(10, 10)
    fb = macro_f1_from_counts(cmb); f1b[it] = fb
    diffs[it] = fb - macro_f1_from_counts(cma)
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
mf_a = f1_score(y_te, yp_ref, average="macro", zero_division=0)
mf_b = f1_score(y_te, yp_new, average="macro", zero_division=0)
blo, bhi = np.percentile(f1b, [2.5, 97.5])
say("\npaired delta (W_e2e - W_sb): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (mf_b - mf_a, lo, hi, min(1.0, p_two)))
say("W_e2e marginal: %.4f [%.4f, %.4f]" % (mf_b, blo, bhi))

fa = f1_score(y_te, yp_ref, average=None, labels=range(10), zero_division=0)
fb_ = f1_score(y_te, yp_new, average=None, labels=range(10), zero_division=0)
ra = recall_score(y_te, yp_ref, average=None, labels=range(10), zero_division=0)
rb = recall_score(y_te, yp_new, average=None, labels=range(10), zero_division=0)
say("\nper-class F1 (W_sb -> W_e2e) and recall:")
for i, n in enumerate(NAMES):
    say("  %-16s F1 %.4f -> %.4f (%+.4f) | R %.4f -> %.4f"
        % (n, fa[i], fb_[i], fb_[i] - fa[i], ra[i], rb[i]))

n_leak_a = int(((y_te == 0) & (yp_ref != 0)).sum())
n_leak_b = int(((y_te == 0) & (yp_new != 0)).sum())
sw_a = int((gmask_test & (yp_ref == 0) & (y_te != 0)).sum())
sw_b = int((gmask_test & (yp_new == 0) & (y_te != 0)).sum())
say("\nbenign-as-attack: %d -> %d | real attacks swallowed by reject: %d -> %d"
    % (n_leak_a, n_leak_b, sw_a, sw_b))
say("\n" + classification_report(y_te, yp_new, labels=list(range(10)), target_names=NAMES,
                                 digits=3, zero_division=0))

res = dict(experiment="reject-arm W refit under end-to-end objective",
           comb_e2e=dict(W_stageB=round(float(e2e_of_W(W_sb)), 4),
                         uniform=round(float(e2e_of_W(np.ones((M, NCR)) / M)), 4),
                         W_e2e=round(float(e_best), 4), init=init_used),
           selv_winner=BESTV,
           test=dict(W_e2e=round(float(mf_b), 4), W_sb_ref=round(float(mf_a), 4)),
           paired_delta=round(float(mf_b - mf_a), 4),
           paired_ci=[round(float(lo), 4), round(float(hi), 4)],
           paired_p=round(float(min(1.0, p_two)), 4),
           marginal_ci=[round(float(blo), 4), round(float(bhi), 4)],
           per_class_f1={NAMES[i]: round(float(fb_[i]), 4) for i in range(10)},
           benign_as_attack=[n_leak_a, n_leak_b], attacks_swallowed=[sw_a, sw_b],
           reject_col_W_sb=[round(float(v), 2) for v in W_sb[:, IREJ]],
           reject_col_W_e2e=[round(float(v), 2) for v in W_e2e[:, IREJ]],
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "v71_w_e2e.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
