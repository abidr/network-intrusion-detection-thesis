"""
v7.3d: paired-bootstrap significance for the RF-vs-XGB gate swap.

v7.3c measured, end-to-end, that swapping the Stage-A gate from XGBoost to Random Forest
costs -0.0038 macro-F1 clean and gains +0.028 .. +0.056 under input noise. Neither number
carried an interval, so neither is quotable yet. Two questions:

  (1) Is the clean cost distinguishable from zero? The project convention is that a single
      config carries a marginal 95% CI of ~+/-0.017 and a paired comparison ~+/-0.012, so a
      -0.0038 point estimate is almost certainly inside noise -- but "almost certainly"
      is not a measurement.
  (2) Are the robustness gains established, budget by budget?

Method matches ablation_significance.ipynb and every other significance run in this repo:
2,000 stratified bootstrap resamples of the test set, with BOTH arms scored on the SAME
resample each iteration so the difference is paired. Arms C and D from v7.3c are dropped --
they were settled (Stage-B swap buys no robustness and costs clean performance).

Everything else is pinned to v7.3c: same splits, same 800k gate training subsample, both
gates thresholded to the same benign fpr, Stage-B pool / W_e2e / reject set / competence
reference all held fixed and all derived from the cached XGB out-of-fold gate. That last
point is a confound that works AGAINST the RF arm, and it is restated in the output.

Predictions are cached to rf_gate_sig_preds.npz so the bootstrap can be re-analysed without
refitting. ASCII-only prints; log written UTF-8.
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
PRED = OUT / "rf_gate_sig_preds.npz"
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
GATE_TRAIN, TARGET_FPR, N_BOOT = 800_000, 0.0168, 2000
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "rf_gate_sig.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
del X, corr, upper, DER_ALL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("prep done %.0fm" % ((time.time() - t_start) / 60))


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


# ---------------------------------------------------------------- predictions (cached)
if PRED.exists():
    zc = np.load(PRED)
    PREDS = {k: zc[k] for k in zc.files}
    say("loaded cached predictions: %s" % PRED.name)
else:
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
    gpos_comb = comb_idx[p_gate_oof[comb_idx] > 0.5]
    yt_gpos = y_arr[gpos_comb]; P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
    neg_comb = comb_idx[p_gate_oof[comb_idx] <= 0.5]
    BG = np.zeros((10, 10), np.int64)
    for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)): BG[c, 0] = cnt

    def fuse(P, Wc, Wd=None, p=1.0):
        if Wd is not None:
            Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
        S = np.zeros(P[0].shape, np.float64)
        for m in range(M):
            coef = Wc[m][None, :] if Wd is None else Wp[:, m:m + 1] * Wc[m][None, :]
            S += coef * P[m]
        return S.argmax(1)

    Wm = np.ones((M, NCR)) / M
    def _sc(Wc):
        yp = LAB10[fuse(P_GPOS, Wc)]
        return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)
    best = _sc(Wm)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = Wm[m_, c]
                for gg in GRID_W:
                    Wm[m_, c] = gg
                    if Wm[:, c].sum() == 0: continue
                    s_ = _sc(Wm)
                    if s_ > best + 1e-7: best, cur = s_, gg
                Wm[m_, c] = cur
    W_e2e = Wm
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
    say("Stage-B pool refit %.1fm (identical for both arms)" % ((time.time() - t0) / 60))
    sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                              stratify=(y_arr != 0).astype(int))
    Xg = X_trf.iloc[sub].reset_index(drop=True); yg = (y_arr[sub] != 0).astype(int)
    swg = compute_sample_weight("balanced", yg)
    GATES = {"XGB": make_xgb().fit(Xg, yg, sample_weight=swg),
             "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS,
                                          class_weight="balanced").fit(Xg, yg)}
    say("gates trained on %d rows" % GATE_TRAIN)
    del X_trf, Xg; gc.collect()

    att_te = y_te != 0
    i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
    F_att0 = FULL_te.iloc[i_att].reset_index(drop=True)
    S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
    X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True),
                     SVC_te.iloc[i_ben].reset_index(drop=True))
    THR = {}
    for nm, gm in GATES.items():
        pb = gm.predict_proba(X_ben)[:, 1]
        THR[nm] = float(np.quantile(pb, 1.0 - TARGET_FPR))
        say("  gate %-3s threshold %.4f -> benign fpr %.4f"
            % (nm, THR[nm], float((pb > THR[nm]).mean())))

    def perturb(budget):
        F_ = F_att0.copy(); rng = np.random.RandomState(RS)
        for c in FULL.columns:
            F_[c] = F_att0[c].values + rng.uniform(-budget * IQR[c], budget * IQR[c],
                                                   len(F_att0)).astype(np.float32)
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        F_ = repair(F_)
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        return F_, S_att0

    def stageB(Xp):
        Pg = [m.predict_proba(Xp).astype(np.float32) for m in POOL]
        _, ii = NN_D.kneighbors(prof(Pg))
        wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
        Wd = np.empty((len(Xp), M))
        for m in range(M):
            Wd[:, m] = (CORRECT[m][ii] * wn).sum(1) / den
        return LAB10[fuse(Pg, W_e2e, Wd=Wd, p=3)]

    XP = {b: assemble(*(perturb(b) if b > 0 else (F_att0, S_att0))) for b in BUDGETS}
    PREDS = {}
    for gname in ("XGB", "RF"):
        gm, thr = GATES[gname], THR[gname]
        gmb = gm.predict_proba(X_ben)[:, 1] > thr
        yp_ben = np.zeros(len(i_ben), int)
        if gmb.any(): yp_ben[gmb] = stageB(X_ben[gmb])
        for b in BUDGETS:
            Xp = XP[b]
            gma = gm.predict_proba(Xp)[:, 1] > thr
            yp_a = np.zeros(len(Xp), int)
            if gma.any(): yp_a[gma] = stageB(Xp[gma])
            yp = np.zeros(len(y_te), int); yp[i_ben] = yp_ben; yp[i_att] = yp_a
            PREDS["%s_%g" % (gname, b)] = yp.astype(np.int8)
            say("  predicted %s @ budget %.3f" % (gname, b)); dump()
    PREDS["y_te"] = y_te.astype(np.int8)
    np.savez_compressed(PRED, **PREDS)
    say("predictions cached -> %s" % PRED.name)

y_true = PREDS["y_te"].astype(int)
dump()

# ---------------------------------------------------------------- paired bootstrap
rng = np.random.RandomState(RS)
cls_idx = [np.where(y_true == c)[0] for c in range(10)]
BIDX = [np.concatenate([ci[rng.randint(0, len(ci), len(ci))] for ci in cls_idx])
        for _ in range(N_BOOT)]


def boot(yp_a, yp_b):
    ca = y_true * 10 + yp_a; cb = y_true * 10 + yp_b
    d = np.empty(N_BOOT); fa = np.empty(N_BOOT); fb = np.empty(N_BOOT)
    for it, bi in enumerate(BIDX):
        cma = np.bincount(ca[bi], minlength=100).reshape(10, 10)
        cmb = np.bincount(cb[bi], minlength=100).reshape(10, 10)
        fa[it] = macro_f1_from_counts(cma); fb[it] = macro_f1_from_counts(cmb)
        d[it] = fb[it] - fa[it]
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d.mean(), lo, hi, min(1.0, p), np.percentile(fa, [2.5, 97.5]), np.percentile(fb, [2.5, 97.5])


say("\n" + "=" * 88)
say("PAIRED BOOTSTRAP -- RF gate minus XGB gate, %d stratified resamples" % N_BOOT)
say("=" * 88)
say("  %8s %9s %9s %10s %20s %8s %s"
    % ("budget", "XGB", "RF", "delta", "95% CI", "p", "verdict"))
rows = []
for b in BUDGETS:
    a = PREDS["XGB_%g" % b].astype(int); c = PREDS["RF_%g" % b].astype(int)
    mfa = f1_score(y_true, a, average="macro", zero_division=0)
    mfb = f1_score(y_true, c, average="macro", zero_division=0)
    d, lo, hi, p, cia, cib = boot(a, c)
    verdict = "ESTABLISHED" if lo > 0 else ("worse, established" if hi < 0 else "not established")
    say("  %8.3f %9.4f %9.4f %+10.4f  [%+.4f, %+.4f] %8.3f  %s"
        % (b, mfa, mfb, mfb - mfa, lo, hi, p, verdict))
    rows.append(dict(budget=b, xgb=round(float(mfa), 4), rf=round(float(mfb), 4),
                     delta=round(float(mfb - mfa), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
                     p=round(float(p), 4), verdict=verdict,
                     xgb_marginal_ci=[round(float(cia[0]), 4), round(float(cia[1]), 4)],
                     rf_marginal_ci=[round(float(cib[0]), 4), round(float(cib[1]), 4)]))
    dump()

r0 = rows[0]
say("\n" + "=" * 88); say("READOUT"); say("=" * 88)
say("CLEAN (budget 0): delta %+.4f, CI [%+.4f, %+.4f], p = %.3f -> %s"
    % (r0["delta"], r0["ci"][0], r0["ci"][1], r0["p"], r0["verdict"]))
say("  XGB %.4f %s | RF %.4f %s" % (r0["xgb"], r0["xgb_marginal_ci"], r0["rf"], r0["rf_marginal_ci"]))
if r0["verdict"] == "not established":
    say("  -> the gate swap is FREE on clean performance: the cost is not distinguishable")
    say("     from zero, consistent with the +/-0.012 paired band from ablation_significance.")
est = [r for r in rows[1:] if r["verdict"] == "ESTABLISHED"]
if est:
    say("\nUNDER NOISE: established at %d of %d budgets, from %.3f IQR upward."
        % (len(est), len(rows) - 1, est[0]["budget"]))
    say("  largest gain %+.4f at budget %.2f, CI [%+.4f, %+.4f]"
        % (max(r["delta"] for r in est), max(est, key=lambda r: r["delta"])["budget"],
           max(est, key=lambda r: r["delta"])["ci"][0],
           max(est, key=lambda r: r["delta"])["ci"][1]))
else:
    say("\nUNDER NOISE: no budget reaches significance -- the v7.3c gains are NOT established.")
say("\nCONFOUND (restated): the reject set, Stage-B pool, W_e2e and competence reference all")
say("derive from the cached XGB out-of-fold gate. The RF arm is handed a gate-positive")
say("population its reject class never saw, so this confound works AGAINST RF.")

res = dict(experiment="paired bootstrap, RF vs XGB Stage-A gate, clean and under input noise",
           n_boot=N_BOOT, gate_train_rows=GATE_TRAIN, target_fpr=TARGET_FPR,
           budgets=list(BUDGETS), results=rows,
           confound="reject set / pool / W_e2e / competence reference all from cached XGB OOF gate",
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "rf_gate_sig.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
