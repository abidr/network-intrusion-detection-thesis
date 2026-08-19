"""
v7.1d: per-REGION W -- does dynamism at a coarser granularity than per-instance work?

Motivation. The combiner ladder has two endpoints and nothing in between:
  K = 1      global per-class W          (STATIC W_e2e, v7.1b)          -- established
  K = N      per-instance competence     (OLA-cw / LCA, v4/v5)          -- inert
v5 explained the inert end: decomposing w_m(x) gives 98.1% shared row-difficulty
variance vs 1.9% member-specific, and the shared part is *algebraically cancelled*
by the across-member normalisation w~_m = w_m^p / sum_j w_j^p. So per-instance
weighting leans entirely on the 1.9%.

This script fills the gap: partition the 70-dim DECISION space (v5: decision space
beats feature space) into K regions by KMeans, and fit a separate 7x10 W per region
under the SAME end-to-end objective as v7.1b. K=1 recovers v7.1b's STATIC W_e2e
exactly (asserted on 100% of rows), so any difference is attributable to regions
alone. Shrinkage lam toward the global W guards the small-region overfitting that
v2 warned about (constrained combiners beat stacking when the fitting set is small).

Pre-registered reading of the outcome:
  - region W beats global   -> real regional specialisation; dynamism works at a
                              granularity the data supports.
  - region W ties global    -> the ladder is bracketed: global works, per-instance
                              does not, and nothing in between does either. The
                              "dynamic component is inert" observation becomes a
                              demonstrated limit rather than an anecdote.
The DIAG block below decides this before the ascent runs: if the locally-best
member is the same in every region, there is no specialisation to exploit.

Protocol: same prep/splits/caches as v7/v7.1/v7.1b (asserted). All selection on
SELV (t=0.5); test touched once; paired bootstrap vs the same-run v7.1b winner.
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
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

K_GRID = (2, 3, 5, 8)          # K=1 is the global control, added separately
LAM_GRID = (1.0, 0.5)          # 1.0 = pure region W, 0.5 = shrunk halfway to global
MIN_REGION = 500               # regions below this keep the global W (no refit)
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "region_w.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
say("prep done %.0fm -- splits match v7/v7.1/v7.1b" % ((time.time() - t_start) / 60))

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
prof = lambda ps: np.hstack(ps).astype(np.float32)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


def fuse_regional(P, Wl, reg, Wd=None, p=1.0):
    """Wl: list of K arrays (M, NCR). reg: region id per row. Wd: optional (n, M)."""
    Wstack = np.stack(Wl)                                  # (K, M, NCR)
    if Wd is not None:
        Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(P[0].shape, np.float64)
    for m in range(M):
        coef = Wstack[:, m, :][reg]                        # (n, NCR)
        if Wd is not None:
            coef = coef * Wp[:, m:m + 1]
        S += coef * P[m]
    return S.argmax(1)


def fuse_static(P, W):
    return fuse_regional(P, [W], np.zeros(P[0].shape[0], np.int64))


# v7.1b reference state (kNN ref = comb attacks + comb reject rows; unchanged)
ref = np.where(in_comb & (yr10 >= 0))[0]
y_ref = yr10[ref]
CORRECT = np.stack([(OOF10[m][ref].argmax(1) == y_ref) for m in range(M)])
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = ((n_ref.sum() / n_ref) ** 0.25); WC_REF = WC_REF / WC_REF.mean()
NN_D = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
    prof([OOF10[m][ref] for m in range(M)]))


def comp_ola(idx, k):
    nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
    Wd = np.empty((nb.shape[0], M))
    for m in range(M):
        Wd[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
    return Wd


# ---------------------------------------------------------------- e2e objective (v7.1b)
gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]
P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG[c, 0] = cnt                                   # gate-negatives predicted Benign
say("COMB gate-positives %d | background rows %d" % (len(gpos_comb), len(neg_comb)))


def cm_rows(yt, yp):
    return np.bincount(yt * 10 + yp, minlength=100).reshape(10, 10)


def e2e_of_regional(Wl, reg):
    yp = LAB10[fuse_regional(P_GPOS, Wl, reg)]
    return macro_f1_from_counts(cm_rows(yt_gpos, yp) + BG)


def fit_W_e2e(W0):
    """Global (K=1) coordinate ascent -- reproduces v7.1b."""
    W = W0.copy(); zeros = np.zeros(len(gpos_comb), np.int64)
    best = e2e_of_regional([W], zeros)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = e2e_of_regional([W], zeros)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best


t0 = time.time()
W_e2e_u, e_u = fit_W_e2e(np.ones((M, NCR)) / M)
W_e2e, e_glob = (W_e2e_u, e_u)
say("W_e2e (global, K=1) fitted %.0fs: COMB e2e mF1 %.4f  [v7.1b run: 0.6906]"
    % (time.time() - t0, e_glob))
dump()

# ---------------------------------------------------------------- regions in DECISION space
PROF_GPOS = prof(P_GPOS)                                # (n_gpos, 70)
say("\n=== REGION DIAGNOSTICS (decision space, %d-dim) ===" % PROF_GPOS.shape[1])
KM = {}
for K in K_GRID:
    km = KMeans(n_clusters=K, n_init=4, random_state=RS).fit(PROF_GPOS)
    KM[K] = km
    reg = km.labels_
    say("\nK=%d" % K)
    for k in range(K):
        ii = np.where(reg == k)[0]
        comp = np.bincount(yt_gpos[ii], minlength=10)
        top = np.argsort(-comp)[:3]
        accs = np.array([(LAB10[P_GPOS[m][ii].argmax(1)] == yt_gpos[ii]).mean() for m in range(M)])
        say("  region %d  n=%6d  top: %-42s  best member: %-14s (acc %.3f, spread %.3f)"
            % (k, len(ii),
               ", ".join("%s %.0f%%" % (NAMES[t], 100.0 * comp[t] / max(len(ii), 1)) for t in top),
               MODELS[int(accs.argmax())], accs.max(), accs.max() - accs.min()))
    winners = set()
    for k in range(K):
        ii = np.where(reg == k)[0]
        accs = [(LAB10[P_GPOS[m][ii].argmax(1)] == yt_gpos[ii]).mean() for m in range(M)]
        winners.add(MODELS[int(np.argmax(accs))])
    say("  distinct locally-best members across regions: %d of %d  -> %s"
        % (len(winners), K, sorted(winners)))
dump()

# ---------------------------------------------------------------- fit per-region W
def fit_regional_W(K, reg):
    """Coordinate ascent on the GLOBAL e2e objective, one region's W at a time.
    Regions below MIN_REGION keep the global W (never refit)."""
    Wl = [W_e2e.copy() for _ in range(K)]
    idx_k = [np.where(reg == k)[0] for k in range(K)]
    fit_k = [k for k in range(K) if len(idx_k[k]) >= MIN_REGION]
    yp_all = LAB10[fuse_regional(P_GPOS, Wl, reg)]

    def region_pred(k, W):
        ii = idx_k[k]
        S = np.zeros((len(ii), NCR), np.float64)
        for m in range(M):
            S += P_GPOS[m][ii] * W[m][None, :]
        return LAB10[S.argmax(1)]

    best = macro_f1_from_counts(cm_rows(yt_gpos, yp_all) + BG)
    for _ in range(3):
        for k in fit_k:
            ii = idx_k[k]
            other = np.ones(len(yt_gpos), bool); other[ii] = False
            CM_BASE = cm_rows(yt_gpos[other], yp_all[other]) + BG
            for c in range(NCR):
                for m_ in range(M):
                    cur = Wl[k][m_, c]; cur_pred = None
                    for gg in GRID_W:
                        Wl[k][m_, c] = gg
                        if Wl[k][:, c].sum() == 0: continue
                        pk = region_pred(k, Wl[k])
                        s_ = macro_f1_from_counts(CM_BASE + cm_rows(yt_gpos[ii], pk))
                        if s_ > best + 1e-7: best, cur, cur_pred = s_, gg, pk
                    Wl[k][m_, c] = cur
                    if cur_pred is not None: yp_all[ii] = cur_pred
    return Wl, best, [len(i) for i in idx_k], fit_k


REG_W = {}
for K in K_GRID:
    t0 = time.time()
    Wl, sc, sizes, fit_k = fit_regional_W(K, KM[K].labels_)
    REG_W[K] = Wl
    dev = np.mean([np.abs(Wl[k] - W_e2e).mean() for k in range(K)])
    mx = np.max([np.abs(Wl[k] - W_e2e).max() for k in range(K)])
    pair = np.mean([np.abs(Wl[a] - Wl[b]).mean()
                    for a in range(K) for b in range(a + 1, K)]) if K > 1 else 0.0
    chg = float((LAB10[fuse_regional(P_GPOS, Wl, KM[K].labels_)]
                 != LAB10[fuse_static(P_GPOS, W_e2e)]).mean())
    say("\nK=%d fitted %.0fs | COMB e2e %.4f (global %.4f, %+.4f) | regions refit %d/%d"
        % (K, time.time() - t0, sc, e_glob, sc - e_glob, len(fit_k), K))
    say("   mean |W_k - W_glob| %.4f | max %.4f | mean pairwise |W_a - W_b| %.4f | rows changed %.2f%%"
        % (dev, mx, pair, 100 * chg))
    dump()

# ---------------------------------------------------------------- SELV selection (t=0.5)
gmask_selv = p_gate[selv_idx] > 0.5
gpos_selv = selv_idx[gmask_selv]
P_selv = [OOF10[m][gpos_selv] for m in range(M)]
PROF_selv = prof(P_selv)
_, i_selv = NN_D.kneighbors(PROF_selv)
Wd30_s = comp_ola(i_selv, 30)


def shrink(Wl, lam):
    return [(1 - lam) * W_e2e + lam * W for W in Wl]


def build_cands(P_g, PROF_g, Wd):
    n = P_g[0].shape[0]
    z0 = np.zeros(n, np.int64)
    out = {"K=1 STATIC W_e2e (v7.1b static)": fuse_regional(P_g, [W_e2e], z0),
           "K=1 HYBRID W_e2e x OLA-cw k=30 p=3 (v7.1b FINAL)":
               fuse_regional(P_g, [W_e2e], z0, Wd=Wd, p=3)}
    for K in K_GRID:
        reg = KM[K].predict(PROF_g)
        for lam in LAM_GRID:
            Wl = shrink(REG_W[K], lam)
            out["REGION K=%d lam=%.1f" % (K, lam)] = fuse_regional(P_g, Wl, reg)
        Wl = REG_W[K]
        out["REGION K=%d x OLA-cw k=30 p=3" % K] = fuse_regional(P_g, Wl, reg, Wd=Wd, p=3)
    return out


def score_e2e(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int)
    yp[gmask] = LAB10[fam_g]
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


selv_cands = build_cands(P_selv, PROF_selv, Wd30_s)
# mechanical control: K=1 regional must reproduce the plain static fusion exactly
assert (selv_cands["K=1 STATIC W_e2e (v7.1b static)"]
        == fuse_static(P_selv, W_e2e)).mean() == 1.0
say("\ncontrol: K=1 regional reproduces STATIC W_e2e on 100.0000% of SELV rows")

selv_tab = pd.DataFrame([score_e2e(fg, y_arr[selv_idx], gmask_selv, v)
                         for v, fg in selv_cands.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV (t=0.5) ---"); say(selv_tab.to_string(index=False))
REFV = "K=1 HYBRID W_e2e x OLA-cw k=30 p=3 (v7.1b FINAL)"
new_v = [v for v in selv_tab.variant if v.startswith("REGION")]
BESTV = selv_tab[selv_tab.variant.isin(new_v)].iloc[0]["variant"]
say("SELV winner among REGION variants: %s" % BESTV)
del P_selv, PROF_selv, i_selv, Wd30_s; gc.collect()
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
PROF_test = prof(P_test)
_, i_test_nn = NN_D.kneighbors(PROF_test)
Wd30_t = comp_ola(i_test_nn, 30)

test_preds = build_cands(P_test, PROF_test, Wd30_t)
test_tab = pd.DataFrame([score_e2e(fg, y_te, gmask_test, v) for v, fg in test_preds.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- TEST (selection made on SELV) ---"); say(test_tab.to_string(index=False))


def to_yp(fam_g):
    yp = np.zeros(len(y_te), int); yp[gmask_test] = LAB10[fam_g]; return yp


yp_new = to_yp(test_preds[BESTV])
yp_ref = to_yp(test_preds[REFV])
sel = test_tab[test_tab.variant == BESTV].iloc[0]
rank = int(test_tab.index[test_tab.variant == BESTV][0]) + 1
say("\nSELV-selected '%s': test mF1 %.4f acc %.4f (rank %d of %d on test)"
    % (BESTV, sel["mF1"], sel["acc"], rank, len(test_tab)))

# paired bootstrap vs same-run v7.1b reference
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
say("\npaired delta (region W - v7.1b): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (mf_b - mf_a, lo, hi, min(1.0, p_two)))
say("region W marginal: %.4f [%.4f, %.4f]" % (mf_b, blo, bhi))
say("v7.1b reference this run: %.4f  [previous run: 0.6874]" % mf_a)

fa = f1_score(y_te, yp_ref, average=None, labels=range(10), zero_division=0)
fb_ = f1_score(y_te, yp_new, average=None, labels=range(10), zero_division=0)
say("\nper-class F1 (v7.1b -> region W):")
for i, n in enumerate(NAMES):
    say("  %-16s %.4f -> %.4f (%+.4f)" % (n, fa[i], fb_[i], fb_[i] - fa[i]))
say("\n" + classification_report(y_te, yp_new, labels=list(range(10)), target_names=NAMES,
                                digits=3, zero_division=0))

res = dict(experiment="per-region W in decision space (K=1 global .. K=8) under e2e objective",
           k_grid=list(K_GRID), lam_grid=list(LAM_GRID), min_region=MIN_REGION,
           comb_e2e_global=round(float(e_glob), 4),
           comb_e2e_region={str(K): round(float(e2e_of_regional(REG_W[K], KM[K].labels_)), 4)
                            for K in K_GRID},
           w_divergence={str(K): round(float(np.mean([np.abs(REG_W[K][k] - W_e2e).mean()
                                                      for k in range(K)])), 4) for K in K_GRID},
           selv_winner=BESTV, selv_table=selv_tab.to_dict("records"),
           test_table=test_tab.to_dict("records"), test_rank_of_selv_winner=rank,
           test=dict(region=round(float(mf_b), 4), v71b_ref=round(float(mf_a), 4)),
           paired_delta=round(float(mf_b - mf_a), 4),
           paired_ci=[round(float(lo), 4), round(float(hi), 4)],
           paired_p=round(float(min(1.0, p_two)), 4),
           marginal_ci=[round(float(blo), 4), round(float(bhi), 4)],
           per_class_f1={NAMES[i]: round(float(fb_[i]), 4) for i in range(10)},
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "region_w.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
