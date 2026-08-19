"""
v7.3: adversarial evasion under an ATTACKER-CONTROLLABLE budget.

The standard adversarial-NIDS experiment perturbs the feature vector freely, as if it
were an image. But a flow record is not a picture -- it is a receipt of packets that
actually traversed a network and still had to work as an attack. Two consequences the
usual setup ignores:

  (1) SOME FEATURES CANNOT MOVE AT ALL. If the attack exploits an RPC service, the
      traffic goes to port 111. Setting svc_rpc = 0 does not describe a stealthier
      attack; it describes a different (or no) attack. On our model these columns carry
      70.4% of total importance -- so an unconstrained attack "succeeds" almost entirely
      by editing evidence no attacker can edit.
  (2) THE VECTOR MUST STAY INTERNALLY CONSISTENT. Flow Bytes/s is bytes over duration;
      frac_PSH is PSH count over packet count; bpp_fwd is bytes over packets. Perturbing
      those independently produces a record no packet capture could ever generate. Here
      the attacker sets PRIMITIVES only, and every derivable column -- CICFlowMeter rates
      and ratios plus all 31 engineered features -- is RECOMPUTED afterwards.

Three buckets (see BUCKETS below; every one of the 99 columns is assigned, asserted):
  FREE       timing, padding, packet-size shape      -- attacker changes at will
  COSTLY     packet/byte counts, flag counts         -- changeable, but costs the attacker
  IMPOSSIBLE service identity, protocol, dst port    -- changing it abandons the attack

Attacker model: white-box on the Stage-A gate, greedy sign search per feature (2 passes),
budget b measured in units of each column's TRAIN IQR, with COSTLY columns allowed b*
COSTLY_RATIO. Only test ATTACK rows are perturbed -- an attacker controls their own
traffic, not the benign background. Objective is to minimise the gate's p(attack); the
end-to-end effect (macro-F1, per-family recall) is then measured through the full
v7.1b pipeline.

Three arms, so the headline claim is falsifiable rather than assumed:
  REALISABLE   FREE + COSTLY move, IMPOSSIBLE frozen, consistency repaired   <- the claim
  UNCONSTRAINED everything moves incl. svc, no repair  <- what the literature reports
  SENSITIVITY  counts demoted FREE->frozen              <- does the bucketing drive it?

Protocol: same prep/splits/caches as v7/v7.1/v7.1b (asserted). No retraining; the model
is fixed and only test inputs move. ASCII-only prints; log written UTF-8.
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
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
BUDGETS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
COSTLY_RATIO = 0.5          # costly columns get half the free budget
PASSES = 2                  # greedy sign-search sweeps
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "adv_budget.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
    """All 31 engineered columns, as a pure function of the (possibly perturbed) FULL frame.
    Factored out precisely so it can be RE-RUN after an attack."""
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
    D = pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
    return D


def repair(F):
    """Re-impose CICFlowMeter's own internal identities after a perturbation.
    Without this the attacker can produce a record with a duration and a byte count that
    contradict its own Flow Bytes/s -- i.e. a flow no capture could produce."""
    F = F.copy()
    has = lambda c: c in F.columns
    fpk = F["Total Fwd Packet"] if has("Total Fwd Packet") else None
    bpk = F["Total Bwd packets"] if has("Total Bwd packets") else None
    fby = F["Total Length of Fwd Packet"] if has("Total Length of Fwd Packet") else None
    bby = F["Total Length of Bwd Packet"] if has("Total Length of Bwd Packet") else None
    dur_s = (F["Flow Duration"] / 1e6) if has("Flow Duration") else None      # us -> s
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
            if dur_s is not None and has("Flow Bytes/s"):
                F["Flow Bytes/s"] = safe(tot_by, dur_s)
        if fby is not None and has("Fwd Segment Size Avg"):
            F["Fwd Segment Size Avg"] = safe(fby, fpk)
        if bby is not None and has("Bwd Segment Size Avg"):
            F["Bwd Segment Size Avg"] = safe(bby, bpk)
    num = F.select_dtypes(include=[np.number]).columns
    F[num] = F[num].clip(lower=0.0)                     # counts/sizes cannot go negative
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
FULL_te = FULL.iloc[i_test].reset_index(drop=True)
SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
del X, corr, upper, DER_ALL; gc.collect()

attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
NC = len(attack_labels); NCR = NC + 1; IREJ = NC
LAB10 = np.array(attack_labels + [0]); M = 7
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
assert len(y_arr) == 2085442 and len(y_te) == 521361 and att_trf.sum() == 65191
say("prep done %.0fm -- splits match v7/v7.1/v7.1b" % ((time.time() - t_start) / 60))

# ---------------------------------------------------------------- BUCKETS
FREE_PREFIX = ("Flow Duration", "Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle",
               "Fwd Packet Length", "Bwd Packet Length", "Packet Length",
               "Fwd Header Length", "Bwd Header Length", "Fwd Seg Size Min",
               "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
               "FWD Init Win Bytes", "Bwd Init Win Bytes", "Flow Bytes/s", "Flow Packets/s",
               "Fwd Packets/s", "Bwd Packets/s", "Down/Up Ratio")
COSTLY_PREFIX = ("Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                 "Total Length of Bwd Packet", "Fwd Act Data Pkts", "Subflow",
                 "FIN Flag", "SYN Flag", "RST Flag", "PSH Flag", "ACK Flag",
                 "Fwd PSH Flags", "Bwd Bytes/Bulk", "Bwd Packet/Bulk", "Bwd Bulk Rate")


def bucket(c):
    if c.startswith(COSTLY_PREFIX): return "COSTLY"
    if c.startswith(FREE_PREFIX):   return "FREE"
    return "FREE"                                   # remaining raw flow stats are shape-like


BK = {c: bucket(c) for c in FULL.columns}
FREE_COLS = [c for c in FULL.columns if BK[c] == "FREE"]
COSTLY_COLS = [c for c in FULL.columns if BK[c] == "COSTLY"]
assert len(FREE_COLS) + len(COSTLY_COLS) == FULL.shape[1]
say("\n=== ATTACKER BUCKETS ===")
say("  FREE       %2d raw cols : %s ..." % (len(FREE_COLS), ", ".join(FREE_COLS[:6])))
say("  COSTLY     %2d raw cols : %s ..." % (len(COSTLY_COLS), ", ".join(COSTLY_COLS[:6])))
say("  IMPOSSIBLE %2d cols     : %s (service identity / protocol / dst port)"
    % (len(SVC_COLS), ", ".join(SVC_COLS[:6])))
say("  derived    %2d cols     : never set directly -- RECOMPUTED from perturbed primitives"
    % len(DER_COLS))
say("  total modelled features: %d" % len(FEAT_ORDER))
dump()

# ---------------------------------------------------------------- model (v7.1b), fixed
z = np.load(ROOT / "cache_oof_v7_full.npz"); oof_gate = z["gate"]
p_gate_oof = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF10 = [zr[f"m{m}"] for m in range(M)]
REJECT = (y_arr == 0) & (oof_gate == 1)
yr10 = np.where(REJECT, IREJ, fam_trf)
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


def fit_W_e2e():
    W = np.ones((M, NCR)) / M

    def sc(Wc):
        yp = LAB10[fuse(P_GPOS, Wc)]
        return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)

    best = sc(W)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = sc(W)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W


t0 = time.time(); W_e2e = fit_W_e2e()
say("\nW_e2e fitted (%.0fs)" % (time.time() - t0))

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
gate_f = make_xgb(); yb = (y_arr != 0).astype(int)
gate_f.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
m10 = yr10 >= 0
POOL = [fit_member(k, p, X_trf[m10].reset_index(drop=True), pd.Series(yr10[m10]))
        for _, (k, p) in SPECS]
say("gate + pool refit %.0fm  (model is now FIXED; only test inputs move)"
    % ((time.time() - t0) / 60))
del X_trf; gc.collect()
dump()

# ---------------------------------------------------------------- clean baseline
att_te = y_te != 0
i_att = np.where(att_te)[0]
say("\ntest attack rows to perturb: %d (benign background left untouched)" % len(i_att))


def predict_rows(Fp, Sp):
    """Full v7.1b pipeline on a set of rows -> predicted labels."""
    Xp = assemble(Fp, Sp)
    pg = gate_f.predict_proba(Xp)[:, 1].astype(np.float32)
    yp = np.zeros(len(Xp), int)
    gm = pg > 0.5
    if gm.any():
        Pg = [m.predict_proba(Xp[gm]).astype(np.float32) for m in POOL]
        _, ii = NN_D.kneighbors(prof(Pg))
        wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
        Wd = np.empty((int(gm.sum()), M))
        for m in range(M):
            Wd[:, m] = (CORRECT[m][ii] * wn).sum(1) / den
        yp[gm] = LAB10[fuse(Pg, W_e2e, Wd=Wd, p=3)]
    return yp, pg


yp_clean, pg_clean = predict_rows(FULL_te, SVC_te)
mf_clean = f1_score(y_te, yp_clean, average="macro", zero_division=0)
say("clean pipeline: macro-F1 %.4f | acc %.4f   [v7.1b published 0.6874 / 0.9866]"
    % (mf_clean, (yp_clean == y_te).mean()))
dump()

# ---------------------------------------------------------------- the attack
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True)
S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
y_att = y_te[i_att]


def gate_p(F_, S_):
    return gate_f.predict_proba(assemble(F_, S_))[:, 1].astype(np.float32)


def _caps(budget, cols_free, cols_costly):
    caps = {c: budget * IQR[c] for c in cols_free}
    caps.update({c: budget * COSTLY_RATIO * IQR[c] for c in cols_costly})
    return caps


def _norms(off, caps):
    """Realised displacement, in IQR units -- the honest size of the perturbation.
    An L-inf budget of b applied across d columns is up to b*sqrt(d) in L2, so the
    per-feature budget on its own understates how far the record actually moved."""
    if not off: return 0.0, 0.0, 0.0
    Z = np.stack([off[c] / IQR[c] for c in caps], 1)          # (n, d) in IQR units
    return (float(np.abs(Z).max(1).mean()), float(np.linalg.norm(Z, axis=1).mean()),
            float((np.abs(Z) > 1e-9).sum(1).mean()))


def attack(budget, cols_free, cols_costly, do_repair, svc_free=False, mode="greedy"):
    """Perturb within a per-feature cap of `budget` IQRs. Cumulative offset is CLIPPED to
    the cap, so multiple passes refine the direction without exceeding the budget."""
    S_ = S_att0.copy()
    caps = _caps(budget, cols_free, cols_costly)
    off = {c: np.zeros(len(F_att0), np.float32) for c in caps}

    def build(o):
        F_ = F_att0.copy()
        for c in caps: F_[c] = F_att0[c].values + o[c]
        return repair(F_) if do_repair else F_

    if mode == "random":                       # control: same budget, no optimisation
        rng = np.random.RandomState(RS)
        for c in caps:
            off[c] = rng.uniform(-caps[c], caps[c], len(F_att0)).astype(np.float32)
        return build(off), S_, _norms(off, caps)

    cur = gate_p(build(off), S_)
    for _ in range(PASSES):
        for c in caps:
            best_p, best_o = cur, off[c]
            for dsign in (+1.0, -1.0):
                cand = np.clip(off[c] + dsign * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                p_ = gate_p(build(o2), S_)
                take = p_ < best_p
                if take.any():
                    best_p = np.where(take, p_, best_p)
                    best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_p
        if svc_free:                       # unconstrained arm: let svc move too
            for c in S_.columns:
                if c == "dst_port_num": continue
                trial = S_.copy(); trial[c] = 0.0
                p_ = gate_p(build(off), trial)
                take = p_ < cur
                S_.loc[take, c] = 0.0
                cur = np.where(take, p_, cur)
    return build(off), S_, _norms(off, caps)


ARMS = {
    "REALISABLE (free+costly, svc frozen, repaired)":
        dict(cf=FREE_COLS, cc=COSTLY_COLS, rep=True, svc=False, mode="greedy"),
    "UNCONSTRAINED (everything incl. svc, no repair)":
        dict(cf=FREE_COLS, cc=COSTLY_COLS, rep=False, svc=True, mode="greedy"),
    "SENSITIVITY (counts frozen too -- timing/size only)":
        dict(cf=FREE_COLS, cc=[], rep=True, svc=False, mode="greedy"),
    "RANDOM CONTROL (same budget, no optimisation)":
        dict(cf=FREE_COLS, cc=COSTLY_COLS, rep=True, svc=False, mode="random"),
}

rows = []
for arm, cfg in ARMS.items():
    say("\n=== %s ===" % arm)
    say("  %6s %10s %10s %12s %12s  %8s %8s %7s"
        % ("budget", "macro-F1", "accuracy", "att-recall", "gate-recall",
           "Linf", "L2", "n_moved"))
    for b in BUDGETS:
        t0 = time.time()
        if b == 0.0:
            Fp, Sp, nrm = F_att0, S_att0, (0.0, 0.0, 0.0)
        else:
            Fp, Sp, nrm = attack(b, cfg["cf"], cfg["cc"], cfg["rep"], cfg["svc"], cfg["mode"])
        yp_a, pg_a = predict_rows(Fp, Sp)
        yp = yp_clean.copy(); yp[i_att] = yp_a
        mf = f1_score(y_te, yp, average="macro", zero_division=0)
        acc = float((yp == y_te).mean())
        arec = float((yp_a != 0).mean())               # still called SOME attack
        grec = float((pg_a > 0.5).mean())              # still caught by the gate
        perc = {NAMES[c]: round(float((yp_a[y_att == c] != 0).mean()), 4)
                for c in range(1, 10) if (y_att == c).any()}
        say("  %6.3f %10.4f %10.4f %12.4f %12.4f  %8.3f %8.3f %7.1f   (%.0fs)"
            % (b, mf, acc, arec, grec, nrm[0], nrm[1], nrm[2], time.time() - t0))
        rows.append(dict(arm=arm, budget=b, macro_f1=round(mf, 4), accuracy=round(acc, 4),
                         attack_recall=round(arec, 4), gate_recall=round(grec, 4),
                         linf_iqr=round(nrm[0], 4), l2_iqr=round(nrm[1], 4),
                         n_moved=round(nrm[2], 1), per_family_recall=perc))
        dump()

# ---------------------------------------------------------------- readout
R = pd.DataFrame(rows)
say("\n" + "=" * 78); say("SUMMARY -- macro-F1 by attacker budget"); say("=" * 78)
piv = R.pivot(index="budget", columns="arm", values="macro_f1")
say(piv.to_string())
say("\nattack recall (fraction of attack flows still flagged as SOME attack):")
say(R.pivot(index="budget", columns="arm", values="attack_recall").to_string())

real = R[R.arm.str.startswith("REALISABLE")].set_index("budget")
unc = R[R.arm.str.startswith("UNCONSTRAINED")].set_index("budget")
rnd = R[R.arm.str.startswith("RANDOM")].set_index("budget")

say("\nrealisability gap at EVERY budget (unconstrained is the standard experiment;")
say("the gap is what that experiment overstates because it edits evidence no attacker owns):")
say("  %8s %12s %14s %10s" % ("budget", "realisable", "unconstrained", "gap"))
for b in BUDGETS:
    say("  %8.3f %12.4f %14.4f %10.4f"
        % (b, real.loc[b, "macro_f1"], unc.loc[b, "macro_f1"],
           real.loc[b, "macro_f1"] - unc.loc[b, "macro_f1"]))

say("\nADVERSARIAL vs RANDOM at the same budget -- the control that separates")
say("'the model is attackable' from 'the model is merely brittle':")
say("  %8s %12s %10s %10s" % ("budget", "adversarial", "random", "difference"))
for b in BUDGETS:
    say("  %8.3f %12.4f %10.4f %10.4f"
        % (b, real.loc[b, "macro_f1"], rnd.loc[b, "macro_f1"],
           rnd.loc[b, "macro_f1"] - real.loc[b, "macro_f1"]))

half = [b for b in BUDGETS if real.loc[b, "attack_recall"] < 0.5 * real.loc[0.0, "attack_recall"]]
if half:
    say("\n  attack recall halves at a realisable budget of %.3f IQR "
        "(realised L2 %.3f IQR across %.0f columns)"
        % (half[0], real.loc[half[0], "l2_iqr"], real.loc[half[0], "n_moved"]))

bmax = max(BUDGETS)
say("\nper-family attack recall, clean -> largest realisable budget (%.2f IQR):" % bmax)
pf = real.loc[bmax, "per_family_recall"]; pf0 = real.loc[0.0, "per_family_recall"]
for k in pf:
    say("  %-16s %.4f -> %.4f (%+.4f)" % (k, pf0[k], pf[k], pf[k] - pf0[k]))

res = dict(experiment="adversarial evasion under an attacker-controllable budget",
           clean=dict(macro_f1=round(float(mf_clean), 4),
                      accuracy=round(float((yp_clean == y_te).mean()), 4)),
           buckets=dict(free=FREE_COLS, costly=COSTLY_COLS, impossible=SVC_COLS,
                        derived_recomputed=DER_COLS),
           budgets=list(BUDGETS), costly_ratio=COSTLY_RATIO, passes=PASSES,
           attack_rows=int(len(i_att)), results=rows,
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "adv_budget.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
