"""
v7.1e: DISJOINT FEATURE-VIEW pool -- the last untested axis for dynamic combination.

Why this exists. Three dynamic-combination attempts have now returned nulls
(v4 feature-space DES, v5 decision-space DES, v7.1d per-region W), and v7.1d
supplied the mechanism: across K=8 decision-space regions, best-member accuracy
and best-worst member spread are anti-correlated at Spearman rho = -0.86. Members
disagree only where all of them are wrong (regions at acc 0.52/0.59 carry spread
0.070) and agree where they are right (acc 0.898 carries spread 0.009). So local
selection can only act where selection cannot pay.

The cause is pool construction: all 7 current members see the SAME 99 features and
differ only in class weighting. Weighting moves decision boundaries, not
information, so every member is blind on the same rows -- v5's 98.1% shared
row-difficulty variance vs 1.9% member-specific.

Hypothesis under test: members trained on DISJOINT FEATURE VIEWS fail on DIFFERENT
rows, producing member-specific competence rather than row difficulty. A port-111
Recon flow should be easy for the service view and blind to the timing view.

PRE-REGISTERED GO/NO-GO, evaluated before the combiner grid runs (DIAG block):
  (1) shared-variance share of the competence weights must fall meaningfully below
      v5's 98.1%;
  (2) the anti-correlation must weaken -- i.e. high-spread regions must stop being
      the low-accuracy ones (rho materially above -0.86);
  (3) median within-row weight spread must rise above v5's 0.0000.
If all three stay put, dynamic combination is closed on this data and the negative
result is complete: three combiner families and two pool constructions, one
mechanism, one predictive diagnostic.

Three pools are scored under one protocol:
  ALPHA  7 members, class-weight diversity  (cached OOF -- the v7.1b reference)
  VIEW   5 members, information diversity   (cross-fitted here)
  UNION  12 members
Protocol: same prep/splits/caches as v7/v7.1/v7.1b (asserted). W fitted under the
end-to-end objective (v7.1b). All selection on SELV (t=0.5); test touched once;
paired bootstrap vs the same-run v7.1b reference. ASCII-only prints; log UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold
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
CACHE = ROOT / "cache_oof_v71e_views.npz"
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
K_DIAG = 8              # regions for the anti-correlation diagnostic (matches v7.1d)
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "view_pool.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
SVC_COLS = set(SVC.columns)
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
say("prep done %.0fm -- splits match v7/v7.1/v7.1b/v7.1d" % ((time.time() - t_start) / 60))

# ---------------------------------------------------------------- caches / labels
z = np.load(ROOT / "cache_oof_v7_full.npz")
oof_gate = z["gate"]
p_gate = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF_ALPHA_FULL = [zr[f"m{m}"] for m in range(7)]
REJECT = (y_arr == 0) & (oof_gate == 1)
NCR = NC + 1; IREJ = NC
yr10 = np.where(REJECT, IREJ, fam_trf)
LAB10 = np.array(attack_labels + [0])
m10 = yr10 >= 0
i10 = np.where(m10)[0]
pos = np.full(len(y_arr), -1, np.int64); pos[i10] = np.arange(len(i10))
say("caches loaded; reject set %d; stage-B rows %d" % (REJECT.sum(), len(i10)))

# ---------------------------------------------------------------- FEATURE VIEWS
def view_of(c):
    if c in SVC_COLS:
        return "SVC"
    lc = c.lower()
    if "flag" in lc or lc.startswith("frac_") or lc == "r_syn_fin":
        return "FLAG"
    if any(t in lc for t in ("iat", "duration", "active", "idle", "/s", "dur_per_pkt")):
        return "TIME"
    return "SIZE"


VIEW_OF = {c: view_of(c) for c in X_trf.columns}
VIEW_COLS = {v: [c for c in X_trf.columns if VIEW_OF[c] == v] for v in ("SVC", "TIME", "SIZE", "FLAG")}
# disjoint + exhaustive by construction; assert it so the "disjoint views" claim is checked
assert sum(len(v) for v in VIEW_COLS.values()) == X_trf.shape[1]
assert all(len(v) > 0 for v in VIEW_COLS.values())
VIEW_COLS["ALL"] = list(X_trf.columns)          # full-feature control member
say("\n=== FEATURE VIEWS (disjoint partition of %d columns) ===" % X_trf.shape[1])
for v in ("SVC", "TIME", "SIZE", "FLAG"):
    say("  %-5s %3d cols | %s%s" % (v, len(VIEW_COLS[v]), ", ".join(VIEW_COLS[v][:6]),
                                    " ..." if len(VIEW_COLS[v]) > 6 else ""))
say("  ALL   %3d cols (full-feature control member)" % len(VIEW_COLS["ALL"]))
dump()

VIEW_SPECS = [("V:SVC", "SVC"), ("V:TIME", "TIME"), ("V:SIZE", "SIZE"),
              ("V:FLAG", "FLAG"), ("V:ALL", "ALL")]
ALPHA_SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
               + [("XGB smote2000", ("smote", 2000)),
                  ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


def alpha_w(ya, a):
    n = np.bincount(np.asarray(ya), minlength=NCR)
    w = ((n.sum() / np.maximum(n, 1)) ** a); w = w / w.mean()
    sw = w[np.asarray(ya)].astype(float)
    return sw * len(sw) / sw.sum()


def fit_alpha_member(kind, param, Xa, ya):
    n = np.bincount(np.asarray(ya), minlength=NCR)
    if kind == "smote":
        strat = {c: param for c in range(NCR) if 0 < n[c] < param}
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat).fit_resample(Xa, ya)
        m = make_xgb(); m.fit(Xs, ys); return m
    if kind == "alpha":  m = make_xgb()
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    m.fit(Xa, ya, sample_weight=alpha_w(ya, param))
    return m


# ---------------------------------------------------------------- cross-fit the VIEW pool
Xa10 = X_trf.iloc[i10].reset_index(drop=True)
ya10 = yr10[i10]
if CACHE.exists():
    zc = np.load(CACHE)
    OOF_VIEW = [zc[f"v{j}"] for j in range(len(VIEW_SPECS))]
    say("\nview OOF loaded from cache %s" % CACHE.name)
else:
    t0 = time.time()
    OOF_VIEW = [np.zeros((len(i10), NCR), np.float32) for _ in VIEW_SPECS]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    for f, (tr, va) in enumerate(skf.split(Xa10, ya10)):
        ytr = ya10[tr]; sw = alpha_w(ytr, 0.25)
        for j, (nm, vw) in enumerate(VIEW_SPECS):
            cols = VIEW_COLS[vw]
            mdl = make_xgb()
            mdl.fit(Xa10.iloc[tr][cols], ytr, sample_weight=sw)
            OOF_VIEW[j][va] = mdl.predict_proba(Xa10.iloc[va][cols]).astype(np.float32)
            del mdl
        gc.collect()
        say("  cross-fit fold %d/5 done (%.0fm)" % (f + 1, (time.time() - t0) / 60))
        dump()
    np.savez_compressed(CACHE, **{f"v{j}": OOF_VIEW[j] for j in range(len(VIEW_SPECS))})
    say("view cross-fit %.0fm -> %s" % ((time.time() - t0) / 60, CACHE.name))
dump()

# ---------------------------------------------------------------- e2e objective scaffolding
gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
assert (pos[gpos_comb] >= 0).all()
yt_gpos = y_arr[gpos_comb]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG[c, 0] = cnt
say("\nCOMB gate-positives %d | background rows %d" % (len(gpos_comb), len(neg_comb)))

GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
prof = lambda ps: np.hstack(ps).astype(np.float32)


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


def fuse(P, W=None, Wd=None, p=1.0):
    Mm = len(P)
    if Wd is not None:
        Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(P[0].shape, np.float64)
    for m in range(Mm):
        coef = np.ones((1, P[0].shape[1])) if W is None else W[m][None, :]
        if Wd is not None:
            coef = coef * Wp[:, m:m + 1]
        S += coef * P[m]
    return S.argmax(1)


def fit_W_e2e(P_gpos):
    Mm = len(P_gpos)
    W = np.ones((Mm, NCR)) / Mm

    def sc(Wc):
        yp = LAB10[fuse(P_gpos, W=Wc)]
        return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)

    best = sc(W)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(Mm):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = sc(W)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best


# ---------------------------------------------------------------- pools
ref = np.where(in_comb & m10)[0]
y_ref = yr10[ref]
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = ((n_ref.sum() / n_ref) ** 0.25); WC_REF = WC_REF / WC_REF.mean()

POOLS = {
    "ALPHA": dict(names=[s_[0] for s_ in ALPHA_SPECS],
                  oof=[OOF_ALPHA_FULL[m][i10] for m in range(7)]),
    "VIEW":  dict(names=[s_[0] for s_ in VIEW_SPECS], oof=OOF_VIEW),
}
POOLS["UNION"] = dict(names=POOLS["ALPHA"]["names"] + POOLS["VIEW"]["names"],
                      oof=POOLS["ALPHA"]["oof"] + POOLS["VIEW"]["oof"])

say("\n=== INDIVIDUAL MEMBER QUALITY (OOF, stage-B 10-class on COMB ref rows) ===")
for m, nm in enumerate(POOLS["UNION"]["names"]):
    P = POOLS["UNION"]["oof"][m][pos[ref]]
    yp = P.argmax(1)
    say("  %-14s acc %.4f  mF1 %.4f" % (nm, (yp == y_ref).mean(),
                                        f1_score(y_ref, yp, average="macro", zero_division=0)))
dump()

# ---------------------------------------------------------------- PRE-REGISTERED DIAGNOSTIC
say("\n=== PRE-REGISTERED DIAGNOSTIC (go/no-go, before any combiner grid) ===")
say("v5 reference on the ALPHA pool: shared variance 98.1%, median within-row spread 0.0000,")
say("v7.1d reference: region acc-vs-spread Spearman rho = -0.86\n")
DIAG = {}
for pname, pd_ in POOLS.items():
    Mm = len(pd_["names"])
    Pref = [o[pos[ref]] for o in pd_["oof"]]
    CORRECT = np.stack([(Pref[m].argmax(1) == y_ref) for m in range(Mm)])
    nn = NearestNeighbors(n_neighbors=30, algorithm="brute", n_jobs=-1).fit(prof(Pref))
    Pg = [o[pos[gpos_comb]] for o in pd_["oof"]]
    _, ii = nn.kneighbors(prof(Pg))
    wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
    Wd = np.empty((len(gpos_comb), Mm))
    for m in range(Mm):
        Wd[:, m] = (CORRECT[m][ii] * wn).sum(1) / den
    # v5-style variance decomposition of the competence weights
    rm = Wd.mean(1, keepdims=True); grand = Wd.mean()
    shared = float(np.mean((rm - grand) ** 2))
    specific = float(np.mean(Wd.var(1)))
    pct_shared = 100 * shared / max(shared + specific, 1e-12)
    spread_row = float(np.median(Wd.max(1) - Wd.min(1)))
    corrmat = np.corrcoef(Wd.T)
    pairwise = float(corrmat[np.triu_indices(Mm, 1)].mean()) if Mm > 1 else 1.0
    # region acc-vs-spread anti-correlation
    km = KMeans(n_clusters=K_DIAG, n_init=4, random_state=RS).fit(prof(Pg))
    accs_b, spreads, sizes = [], [], []
    for k in range(K_DIAG):
        jj = np.where(km.labels_ == k)[0]
        a = np.array([(LAB10[Pg[m][jj].argmax(1)] == yt_gpos[jj]).mean() for m in range(Mm)])
        accs_b.append(a.max()); spreads.append(a.max() - a.min()); sizes.append(len(jj))
    rho, prho = spearmanr(accs_b, spreads)
    dis = float(np.mean([(Pg[a].argmax(1) != Pg[b].argmax(1)).mean()
                         for a in range(Mm) for b in range(a + 1, Mm)]))
    DIAG[pname] = dict(members=Mm, pct_shared=round(pct_shared, 1),
                       median_within_row_spread=round(spread_row, 4),
                       mean_pairwise_weight_corr=round(pairwise, 3),
                       region_rho=round(float(rho), 3), region_rho_p=round(float(prho), 4),
                       mean_pairwise_disagreement=round(dis, 4))
    say("%-6s M=%2d | shared var %5.1f%% | median within-row spread %.4f | weight corr %.3f"
        % (pname, Mm, pct_shared, spread_row, pairwise))
    say("       region acc-vs-spread rho %+.3f (p %.3f) | mean pairwise disagreement %.3f"
        % (rho, prho, dis))
    for k in np.argsort(-np.array(sizes)):
        say("         region %d n=%6d best-acc %.3f spread %.3f"
            % (k, sizes[k], accs_b[k], spreads[k]))
    pd_["nn"] = nn; pd_["CORRECT"] = CORRECT
    del Pref, Pg, ii, wn, Wd; gc.collect()
    dump()

go = (DIAG["VIEW"]["pct_shared"] < DIAG["ALPHA"]["pct_shared"] - 5
      and DIAG["VIEW"]["region_rho"] > DIAG["ALPHA"]["region_rho"] + 0.2)
say("\nGO/NO-GO: VIEW pool %s the pre-registered bar vs ALPHA "
    "(shared %.1f%% vs %.1f%%, rho %+.3f vs %+.3f)"
    % ("CLEARS" if go else "does NOT clear", DIAG["VIEW"]["pct_shared"],
       DIAG["ALPHA"]["pct_shared"], DIAG["VIEW"]["region_rho"], DIAG["ALPHA"]["region_rho"]))
say("(the combiner grid runs either way; the bar is recorded for honest reporting)")
dump()

# ---------------------------------------------------------------- fit W per pool
for pname, pd_ in POOLS.items():
    t0 = time.time()
    Pg = [o[pos[gpos_comb]] for o in pd_["oof"]]
    W, sc = fit_W_e2e(Pg)
    pd_["W"] = W; pd_["comb_e2e"] = sc
    say("W_e2e %-6s fitted %3.0fs | COMB e2e mF1 %.4f" % (pname, time.time() - t0, sc))
    del Pg; gc.collect()
    dump()

# ---------------------------------------------------------------- SELV
gmask_selv = p_gate[selv_idx] > 0.5
gpos_selv = selv_idx[gmask_selv]
assert (pos[gpos_selv] >= 0).all()


def comp_ola(pd_, PROF_g, k=30):
    Mm = len(pd_["names"])
    _, ii = pd_["nn"].kneighbors(PROF_g, n_neighbors=k)
    wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
    Wd = np.empty((PROF_g.shape[0], Mm))
    for m in range(Mm):
        Wd[:, m] = (pd_["CORRECT"][m][ii] * wn).sum(1) / den
    return Wd


def build_cands(getP):
    out = {}
    for pname, pd_ in POOLS.items():
        P = getP(pname)
        Wd = comp_ola(pd_, prof(P), 30)
        W = pd_["W"]
        out["%s EQUAL-AVG" % pname] = fuse(P)
        out["%s STATIC W" % pname] = fuse(P, W=W)
        for p_ in (1, 3, 8):
            out["%s DYN OLA-cw p=%d" % (pname, p_)] = fuse(P, Wd=Wd, p=p_)
        for p_ in (1, 3):
            out["%s HYBRID W x OLA-cw p=%d" % (pname, p_)] = fuse(P, W=W, Wd=Wd, p=p_)
    return out


def score_e2e(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int); yp[gmask] = LAB10[fam_g]
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


selv_cands = build_cands(lambda pn: [o[pos[gpos_selv]] for o in POOLS[pn]["oof"]])
selv_tab = pd.DataFrame([score_e2e(fg, y_arr[selv_idx], gmask_selv, v)
                         for v, fg in selv_cands.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV (t=0.5) ---"); say(selv_tab.to_string(index=False))
REFV = "ALPHA HYBRID W x OLA-cw p=3"                 # the v7.1b architecture
BESTV = selv_tab.iloc[0]["variant"]
BESTNEW = selv_tab[~selv_tab.variant.str.startswith("ALPHA")].iloc[0]["variant"]
say("SELV winner overall: %s" % BESTV)
say("SELV winner among VIEW/UNION variants: %s" % BESTNEW)
del selv_cands; gc.collect()
dump()

# ---------------------------------------------------------------- refit, test once
t0 = time.time()
gate_f = make_xgb()
yb = (y_arr != 0).astype(int)
gate_f.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
ya10s = pd.Series(ya10)
FIT = {}
FIT["ALPHA"] = [fit_alpha_member(k, p, Xa10, ya10s) for _, (k, p) in ALPHA_SPECS]
FIT["VIEW"] = []
for nm, vw in VIEW_SPECS:
    mdl = make_xgb(); mdl.fit(Xa10[VIEW_COLS[vw]], ya10, sample_weight=alpha_w(ya10, 0.25))
    FIT["VIEW"].append(mdl)
say("gate + pools refit %.0fm" % ((time.time() - t0) / 60))
dump()

p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
gmask_test = p_test > 0.5
Xg = X_test.iloc[np.where(gmask_test)[0]]
PT = {"ALPHA": [m.predict_proba(Xg).astype(np.float32) for m in FIT["ALPHA"]],
      "VIEW": [FIT["VIEW"][j].predict_proba(Xg[VIEW_COLS[vw]]).astype(np.float32)
               for j, (nm, vw) in enumerate(VIEW_SPECS)]}
PT["UNION"] = PT["ALPHA"] + PT["VIEW"]
test_preds = build_cands(lambda pn: PT[pn])
test_tab = pd.DataFrame([score_e2e(fg, y_te, gmask_test, v) for v, fg in test_preds.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- TEST (selection made on SELV) ---"); say(test_tab.to_string(index=False))


def to_yp(fam_g):
    yp = np.zeros(len(y_te), int); yp[gmask_test] = LAB10[fam_g]; return yp


yp_new = to_yp(test_preds[BESTNEW]); yp_ref = to_yp(test_preds[REFV])
rank = int(test_tab.index[test_tab.variant == BESTNEW][0]) + 1
sel = test_tab[test_tab.variant == BESTNEW].iloc[0]
say("\nSELV-selected new-pool variant '%s': test mF1 %.4f acc %.4f (rank %d of %d)"
    % (BESTNEW, sel["mF1"], sel["acc"], rank, len(test_tab)))

code_a = y_te * 10 + yp_ref; code_b = y_te * 10 + yp_new
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
say("\npaired delta (new pool - v7.1b arch): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (mf_b - mf_a, lo, hi, min(1.0, p_two)))
say("new-pool marginal: %.4f [%.4f, %.4f]" % (mf_b, blo, bhi))
say("v7.1b reference this run: %.4f  [v7.1b run: 0.6874]" % mf_a)

# dynamic-vs-static within each pool, on test
say("\ndynamic-vs-static within pool (test macro-F1):")
for pname in POOLS:
    st = test_tab[test_tab.variant == "%s STATIC W" % pname].iloc[0]["mF1"]
    hy = test_tab[test_tab.variant == "%s HYBRID W x OLA-cw p=3" % pname].iloc[0]["mF1"]
    dy = test_tab[test_tab.variant == "%s DYN OLA-cw p=3" % pname].iloc[0]["mF1"]
    say("  %-6s static %.4f | hybrid %.4f (%+.4f) | pure-dyn %.4f (%+.4f)"
        % (pname, st, hy, hy - st, dy, dy - st))

fa = f1_score(y_te, yp_ref, average=None, labels=range(10), zero_division=0)
fb_ = f1_score(y_te, yp_new, average=None, labels=range(10), zero_division=0)
say("\nper-class F1 (v7.1b -> new pool):")
for i, n in enumerate(NAMES):
    say("  %-16s %.4f -> %.4f (%+.4f)" % (n, fa[i], fb_[i], fb_[i] - fa[i]))
say("\n" + classification_report(y_te, yp_new, labels=list(range(10)), target_names=NAMES,
                                digits=3, zero_division=0))

res = dict(experiment="disjoint feature-view pool vs class-weight pool; dynamic combination",
           views={v: len(VIEW_COLS[v]) for v in ("SVC", "TIME", "SIZE", "FLAG", "ALL")},
           diagnostic=DIAG, go_no_go=bool(go),
           comb_e2e={p: round(float(POOLS[p]["comb_e2e"]), 4) for p in POOLS},
           selv_winner=BESTV, selv_winner_newpool=BESTNEW,
           selv_table=selv_tab.to_dict("records"), test_table=test_tab.to_dict("records"),
           test_rank_of_selv_newpool=rank,
           test=dict(new=round(float(mf_b), 4), v71b_ref=round(float(mf_a), 4)),
           paired_delta=round(float(mf_b - mf_a), 4),
           paired_ci=[round(float(lo), 4), round(float(hi), 4)],
           paired_p=round(float(min(1.0, p_two)), 4),
           marginal_ci=[round(float(blo), 4), round(float(bhi), 4)],
           per_class_f1={NAMES[i]: round(float(fb_[i]), 4) for i in range(10)},
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "view_pool.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
