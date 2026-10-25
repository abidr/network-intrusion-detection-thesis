"""v7.1b protocol on TON-IoT (Train_Test_Network.csv).

Same architecture and selection protocol as final_model_v7_1_reject_stages.ipynb:
dedupe -> label-free prune -> stratified 80/20 -> COMB/SELV halves -> 5-fold cross-fitted
gate -> reject set from OOF gate false positives -> 10-class 7-member alpha-diverse pool ->
per-class W under the Stage-B and the end-to-end objective -> 21 rules + gate threshold
selected on SELV -> refit on train+val -> test once -> paired bootstrap.

Dataset-forced differences, all recorded in the JSON:
  * src_ip / src_port / dst_ip are EXCLUDED (results/toniot_audit/: src_ip alone gives
    0.9998 out-of-sample attack recall). dns_query / http_uri are high-cardinality text.
  * TON-IoT is a Zeek extraction, so the 31 CICFlowMeter DERIVED features do not exist.
    The scale-free analogues that Zeek's fields allow are built instead (ratios,
    bytes-per-packet, header overhead, asymmetry).
  * Service identity: Zeek ships `service` natively (one-hot, from the categorical block);
    the same 20-port svc one-hots / port bins / proto flags are added from dst_port.

Run:  C:/Users/night/miniconda3/envs/ai4beg/python.exe results/toniot_v71/toniot_v71.py
"""
import io, json, sys, time, gc, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(r"E:\AIUB\THESIS")
SRC = ROOT / "Resource" / "TON-IoT" / "Train_Test_Network.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
NA = ["-", "(empty)", "", " ", "NaN", "nan", "None", "null"]
EXCLUDE = ["src_ip", "src_port", "dst_ip"]          # endpoint identity, audit-flagged
TEXT = ["dns_query", "http_uri"]                     # high-cardinality text
LOG, RES = [], {}


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "toniot_v71.txt", "w", encoding="utf-8").write("\n".join(LOG))
    io.open(OUT / "toniot_v71.json", "w", encoding="utf-8").write(json.dumps(RES, indent=2))


T0 = time.time()
say("=" * 78)
say("v7.1b protocol on TON-IoT -- %s" % SRC.name)
say("=" * 78)

# ------------------------------------------------------------------ 1. load
df = pd.read_csv(SRC, na_values=NA, keep_default_na=True, low_memory=False)
df.columns = [c.strip() for c in df.columns]
df["type"] = df["type"].astype(str).str.strip().str.lower()
say("rows %d | columns %d" % df.shape)

fams = sorted(t for t in df["type"].unique() if t != "normal")
NAMES = ["normal"] + fams
CAT = {n: i for i, n in enumerate(NAMES)}
K = len(NAMES)
say("classes (%d): %s" % (K, NAMES))

feat = [c for c in df.columns if c not in EXCLUDE + TEXT + ["label", "type"]]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
cat = [c for c in feat if c not in num]
miss = df[feat].isna().mean()
dead = miss[miss > 0.99].index.tolist()
num = [c for c in num if c not in dead]
cat = [c for c in cat if c not in dead]
say("excluded: %s (endpoint) + %s (text) + %d >99%%-missing" % (EXCLUDE, TEXT, len(dead)))
RES["excluded_endpoint"] = EXCLUDE; RES["excluded_text"] = TEXT; RES["dropped_missing"] = dead

# ------------------------------------------------------------------ 2.1-2.3 dedupe
n0 = len(df)
df = df.drop_duplicates(subset=num + cat + ["type"]).reset_index(drop=True)
say("dedupe on features+label: %d -> %d (%.2f%% removed)" % (n0, len(df), 100 * (1 - len(df) / n0)))
y = df["type"].map(CAT).values.astype(int)
cnt = np.bincount(y, minlength=K)
for i, n in enumerate(NAMES):
    say("  %-12s %7d  %6.3f%%" % (n, cnt[i], 100 * cnt[i] / len(y)))
RES["rows_raw"] = int(n0); RES["rows_dedup"] = int(len(df))
RES["class_counts"] = {n: int(cnt[i]) for i, n in enumerate(NAMES)}

# ------------------------------------------------------------------ 2.4-2.6 prune (label-free)
Xn = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
Xc = pd.get_dummies(df[cat].astype(str).fillna("__NA__"), prefix=cat, dtype=np.float32)
FULLX = pd.concat([Xn.astype(np.float32), Xc], axis=1)
nz = FULLX.nunique()
FULLX = FULLX.drop(columns=nz[nz <= 1].index.tolist())
corr = FULLX.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
BASE = FULLX.drop(columns=red)
say("prune: %d -> %d (zero-var) -> %d (|r|>=0.95)" % (len(nz), FULLX.shape[1], BASE.shape[1]))

# ------------------------------------------------------------------ 2.7 Zeek analogues of DERIVED
def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

g = lambda c: Xn[c] if c in Xn.columns else pd.Series(0.0, index=Xn.index)
sp, dp, sb, db = g("src_pkts"), g("dst_pkts"), g("src_bytes"), g("dst_bytes")
sib, dib, dur = g("src_ip_bytes"), g("dst_ip_bytes"), g("duration")
d = {}
d["r_pkt_fwd_bwd"] = safe(sp, dp);              d["r_byte_fwd_bwd"] = safe(sb, db)
d["asym_pkt"] = safe(sp - dp, sp + dp);         d["asym_byte"] = safe(sb - db, sb + db)
d["bpp_fwd"] = safe(sb, sp);                    d["bpp_bwd"] = safe(db, dp)
d["bpp_flow"] = safe(sb + db, sp + dp);         d["r_bpp"] = safe(d["bpp_fwd"], d["bpp_bwd"])
d["hdr_per_pkt_fwd"] = safe(sib - sb, sp);      d["hdr_per_pkt_bwd"] = safe(dib - db, dp)
d["payload_share_fwd"] = safe(sb, sib);         d["payload_share_bwd"] = safe(db, dib)
d["dur_per_pkt"] = safe(dur, sp + dp);          d["byte_rate"] = safe(sb + db, dur)
d["pkt_rate"] = safe(sp + dp, dur);             d["missed_share"] = safe(g("missed_bytes"), sb + db)
DER = pd.DataFrame(d).astype(np.float32)
dz = DER.nunique(); DER = DER.drop(columns=dz[dz <= 1].index.tolist())

# ------------------------------------------------------------------ 2.8 service identity from dst_port
port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1).astype(int).values
proto = df["proto"].astype(str).str.lower().values
SVC_PORTS = [(20, "svc_ftpdata"), (21, "svc_ftp"), (22, "svc_ssh"), (23, "svc_telnet"),
             (25, "svc_smtp"), (53, "svc_dns"), (80, "svc_http"), (110, "svc_pop3"),
             (111, "svc_rpc"), (139, "svc_netbios"), (143, "svc_imap"), (179, "svc_bgp"),
             (443, "svc_https"), (445, "svc_smb"), (514, "svc_syslog"), (520, "svc_rip"),
             (1723, "svc_pptp"), (5190, "svc_aim"), (6881, "svc_bt"), (8080, "svc_httpalt")]
KNOWN = np.array([p for p, _ in SVC_PORTS])
s = {nm: (port == p).astype(np.float32) for p, nm in SVC_PORTS}
s["svc_other_low"] = ((port >= 0) & (port < 1024) & ~np.isin(port, KNOWN)).astype(np.float32)
s["svc_registered"] = ((port >= 1024) & (port < 49152) & ~np.isin(port, KNOWN)).astype(np.float32)
s["svc_ephemeral"] = (port >= 49152).astype(np.float32)
s["proto_tcp"] = (proto == "tcp").astype(np.float32)
s["proto_udp"] = (proto == "udp").astype(np.float32)
s["dst_port_num"] = port.astype(np.float32)
SVC = pd.DataFrame(s)
SVC = SVC.drop(columns=[c for c in SVC.columns if c in BASE.columns])   # dst_port already in BASE
sz = SVC.nunique(); SVC = SVC.drop(columns=sz[sz <= 1].index.tolist())

X = pd.concat([BASE.reset_index(drop=True), DER, SVC], axis=1)
X = X.loc[:, ~X.columns.duplicated()].astype(np.float32)
X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
say("BASE %d + DERIVED-analogue %d + SVC %d = %d features"
    % (BASE.shape[1], DER.shape[1], SVC.shape[1], X.shape[1]))
RES["features"] = dict(base=int(BASE.shape[1]), derived=int(DER.shape[1]), svc=int(SVC.shape[1]),
                       total=int(X.shape[1]))
del df, Xn, Xc, FULLX, corr, upper; gc.collect()
dump()

# ------------------------------------------------------------------ 2.9-2.10 splits
i_trf, i_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True); X_test = X.iloc[i_test].reset_index(drop=True)
y_arr, y_te = y[i_trf], y[i_test]
att_trf, att_te = y_arr != 0, y_te != 0
attack_labels = list(range(1, K))
NC = len(attack_labels)
fam_trf = np.where(att_trf, y_arr - 1, -1)
comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
say("\ntrain+val %d (%d attacks) | test %d (%d attacks) | COMB %d | SELV %d"
    % (len(y_arr), att_trf.sum(), len(y_te), att_te.sum(), len(comb_idx), len(selv_idx)))
RES["split"] = dict(trainval=int(len(y_arr)), test=int(len(y_te)),
                    test_support={n: int((y_te == i).sum()) for i, n in enumerate(NAMES)})

# ------------------------------------------------------------------ 3. models (verbatim from the notebook)
def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw)
    return XGBClassifier(**p)

def wc_alpha(n, a):
    n = np.maximum(n, 1)
    w = (n.sum() / n) ** a
    return w / w.mean()

def sample_w(ya, wc):
    w = wc[np.asarray(ya)].astype(float)
    return w * len(w) / w.sum()

SPECS = ([("XGB a=%s" % a, ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)), ("RF a=0.25", ("rf", 0.25)),
            ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]
M = len(SPECS)


class Full:
    """Wraps a member so predict_proba always returns ncls columns, even if a class was
    absent from that fold (the reject class can be tiny on an attack-majority dataset)."""
    def __init__(self, m, present, ncls):
        self.m, self.present, self.ncls = m, present, ncls

    def predict_proba(self, Xq):
        P = np.zeros((len(Xq), self.ncls), np.float32)
        P[:, self.present] = self.m.predict_proba(Xq)
        return P


def fit_member(kind, param, Xa, ya, ncls):
    ya = np.asarray(ya)
    present = np.unique(ya)
    remap = {c: i for i, c in enumerate(present)}
    yl = np.array([remap[c] for c in ya])
    k = len(present)
    n = np.bincount(yl, minlength=k)
    if kind == "smote":
        strat = {c: param for c in range(k) if 0 < n[c] < param}
        kn = max(1, min(5, int(n[n > 0].min()) - 1))
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat, k_neighbors=kn).fit_resample(Xa, yl)
        m = make_xgb(); m.fit(Xs, ys)
        return Full(m, present, ncls)
    if kind == "alpha": m = make_xgb()
    elif kind == "rf":  m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    else:               m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    m.fit(Xa, yl, sample_weight=sample_w(yl, wc_alpha(n, param)))
    return Full(m, present, ncls)


def fit_gate(Xa, ya):
    yb = (ya != 0).astype(int)
    gm = make_xgb()
    gm.fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb))
    return gm

# ------------------------------------------------------------------ 4.1 cross-fitted gate
folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr))
t0 = time.time()
p_gate = np.zeros(len(y_arr), np.float32)
for tr, va in folds:
    gm = fit_gate(X_trf.iloc[tr], y_arr[tr])
    p_gate[va] = gm.predict_proba(X_trf.iloc[va])[:, 1]
oof_gate = (p_gate > 0.5).astype(np.int8)
yb_trf = att_trf.astype(int)
say("\nOOF gate @0.5: attack recall %.4f | precision %.4f (%.0fs)"
    % (recall_score(yb_trf, oof_gate), precision_score(yb_trf, oof_gate), time.time() - t0))
RES["oof_gate"] = dict(recall=round(float(recall_score(yb_trf, oof_gate)), 4),
                       precision=round(float(precision_score(yb_trf, oof_gate)), 4))

# ------------------------------------------------------------------ 4.2 reject class
REJECT = (y_arr == 0) & (oof_gate == 1)
NCR, IREJ = NC + 1, NC
FAMR = NAMES[1:] + ["REJECT->normal"]
yr10 = np.where(REJECT, IREJ, fam_trf)
LAB9 = np.array(attack_labels)
LAB10 = np.array(attack_labels + [0])
say("reject set: %d benign OOF-gate false positives (%.3f%% of benign train+val)"
    % (REJECT.sum(), 100 * REJECT.sum() / max((y_arr == 0).sum(), 1)))
RES["reject_set"] = int(REJECT.sum())
dump()

# ------------------------------------------------------------------ 4.3 cross-fit both pools
def crossfit(ylab, ncls, tag):
    OOF = [np.zeros((len(y_arr), ncls), np.float32) for _ in range(M)]
    t0 = time.time()
    for f_, (tr, va) in enumerate(folds):
        use = ylab[tr] >= 0
        Xa = X_trf.iloc[tr[use]].reset_index(drop=True)
        ya = ylab[tr[use]]
        for mi, (_, (kind, param)) in enumerate(SPECS):
            mdl = fit_member(kind, param, Xa, ya, ncls)
            OOF[mi][va] = mdl.predict_proba(X_trf.iloc[va])
            del mdl; gc.collect()
        say("  %s fold %d/5 (%.0fs elapsed)" % (tag, f_ + 1, time.time() - t0))
    return OOF

say("\ncross-fitting the 10-class reject pool ...")
OOF10 = crossfit(yr10, NCR, "reject")
say("cross-fitting the 9-class baseline pool ...")
OOF9 = crossfit(fam_trf, NC, "baseline")
dump()

# ------------------------------------------------------------------ 4.4 combiner (verbatim)
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
prof = lambda ps: np.hstack(ps).astype(np.float32)

def fuse(P_g, Wd=None, p=1.0, Wclass=None):
    if Wd is not None:
        Wp = np.power(np.clip(Wd, 1e-6, None), p)
        Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(P_g[0].shape, np.float64)
    for m in range(M):
        coef = 1.0 if Wd is None else Wp[:, m:m + 1]
        if Wclass is not None:
            coef = coef * Wclass[m][None, :]
        S += coef * P_g[m]
    return S

def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()

def build_arm(OOF, ref_mask, yref_all, ncls, lab, name):
    ref = np.where(ref_mask)[0]
    y_ref = yref_all[ref]
    P_ref = [OOF[m][ref] for m in range(M)]
    st = dict(name=name, OOF=OOF, ncls=ncls, LAB=np.asarray(lab), ref=ref, y_ref=y_ref)
    st["CORRECT"] = np.stack([(P_ref[m].argmax(1) == y_ref) for m in range(M)])
    st["WC"] = wc_alpha(np.bincount(y_ref, minlength=ncls), 0.25)
    st["NN"] = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(prof(P_ref))
    st["P_ref"] = P_ref
    return st

def comp_ola(st, idx, k):
    nb = idx[:, :k]
    wn = st["WC"][st["y_ref"][nb]]
    den = wn.sum(1)
    Wd = np.empty((nb.shape[0], M))
    for m in range(M):
        Wd[:, m] = (st["CORRECT"][m][nb] * wn).sum(1) / den
    return Wd

def comp_lca(st, P_g, idx, k):
    nb = idx[:, :k]
    yn = st["y_ref"][nb]
    Wd = np.empty((nb.shape[0], M))
    for m in range(M):
        pq = P_g[m].argmax(1)
        mask = yn == pq[:, None]
        cnt_ = mask.sum(1)
        Wd[:, m] = np.where(cnt_ > 0, (st["CORRECT"][m][nb] & mask).sum(1) / np.maximum(cnt_, 1),
                            st["CORRECT"][m][nb].mean(1))
    return Wd

def fit_W(st, score_fn, W0=None, passes=3):
    W = (np.ones((M, st["ncls"])) / M) if W0 is None else W0.copy()
    best = score_fn(W)
    for _ in range(passes):
        for c in range(st["ncls"]):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = score_fn(W)
                    if s_ > best + 1e-7: best, cur = s_, gg
                W[m_, c] = cur
    return W, best

ARM_R = build_arm(OOF10, in_comb & (yr10 >= 0), yr10, NCR, LAB10, "reject")
ARM_B = build_arm(OOF9, in_comb & att_trf, fam_trf, NC, LAB9, "baseline")

t0 = time.time()
sb_fn = lambda W: f1_score(ARM_R["y_ref"], fuse(ARM_R["P_ref"], Wclass=W).argmax(1),
                           average="macro", zero_division=0)
W_sb, sb_score = fit_W(ARM_R, sb_fn)
gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]
P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG = np.zeros((K, K), np.int64)
for c, n_ in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG[c, 0] = n_

def e2e_fn(W):
    yp = LAB10[fuse(P_GPOS, Wclass=W).argmax(1)]
    cm = np.bincount(yt_gpos * K + yp, minlength=K * K).reshape(K, K) + BG
    return macro_f1_from_cm(cm)

W_u, e_u = fit_W(ARM_R, e2e_fn)
W_w, e_w = fit_W(ARM_R, e2e_fn, W0=W_sb)
W_e2e = W_w if e_w >= e_u else W_u
b_fn = lambda W: f1_score(ARM_B["y_ref"], fuse(ARM_B["P_ref"], Wclass=W).argmax(1),
                          average="macro", zero_division=0)
W_base, _ = fit_W(ARM_B, b_fn)
say("\nW fitted (%.0fs): COMB Stage-B mF1 %.4f | COMB e2e uniform %.4f / W_sb %.4f / W_e2e %.4f"
    % (time.time() - t0, sb_score, e2e_fn(np.ones((M, NCR)) / M), e2e_fn(W_sb), max(e_u, e_w)))
dump()

# ------------------------------------------------------------------ 4.5 SELV selection
def rule_names():
    out = ["STATIC: equal average", "STATIC: per-class W"]
    out += ["single: " + m for m in MODELS]
    out += ["DYN OLA-cw k=%d p=%d" % (k, p) for k in (30, 100) for p in (1, 3, 8)]
    out += ["DYN LCA k=30"]
    out += ["HYBRID W x OLA-cw k=%d p=%d" % (k, p) for k in (30, 100) for p in (1, 3)]
    out += ["HYBRID W x LCA k=30 p=1"]
    return out

def rule_scores(st, P_g, idx, Wclass, name):
    if name.startswith("single: "):
        return P_g[MODELS.index(name[8:])]
    if name == "STATIC: equal average":
        return fuse(P_g, Wclass=np.ones((M, st["ncls"])) / M)
    if name == "STATIC: per-class W":
        return fuse(P_g, Wclass=Wclass)
    k = int(name.split("k=")[1].split()[0])
    p = float(name.split("p=")[1]) if "p=" in name else 1.0
    Wd = comp_lca(st, P_g, idx, k) if "LCA" in name else comp_ola(st, idx, k)
    return fuse(P_g, Wd=Wd, p=p, Wclass=(Wclass if name.startswith("HYBRID") else None))

def score(ye, yp, tag):
    return dict(rule=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))

def selv_pred(st, t, Wclass, rule):
    gm = p_gate[selv_idx] > t
    gp = selv_idx[gm]
    P = [st["OOF"][m][gp] for m in range(M)]
    _, idx = st["NN"].kneighbors(prof(P))
    yp = np.zeros(len(selv_idx), int)
    if len(gp):
        yp[gm] = st["LAB"][rule_scores(st, P, idx, Wclass, rule).argmax(1)]
    return yp

y_selv = y_arr[selv_idx]
tab = pd.DataFrame([score(y_selv, selv_pred(ARM_R, 0.5, W_e2e, r), r) for r in rule_names()]
                   ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV: 21 rules, reject arm, gate t=0.5 ---")
say(tab.to_string(index=False))
WINNER = tab.iloc[0]["rule"]
say("SELV winner: %s (%.4f) | SELV spread %.4f" % (WINNER, tab.iloc[0]["mF1"],
                                                   tab["mF1"].max() - tab["mF1"].min()))
sw = pd.DataFrame([dict(t=t, **score(y_selv, selv_pred(ARM_R, t, W_e2e, WINNER), "t=%.1f" % t))
                   for t in (0.3, 0.4, 0.5, 0.6, 0.7)]).sort_values("mF1", ascending=False)
FINAL_T = float(sw.iloc[0]["t"])
say("gate threshold sweep: " + ", ".join("t=%.1f %.4f" % (r.t, r.mF1) for r in sw.itertuples()))
say("SELV-selected threshold %.1f" % FINAL_T)
# baseline arm gets its own SELV winner the same way
tabB = pd.DataFrame([score(y_selv, selv_pred(ARM_B, 0.5, W_base, r), r) for r in rule_names()]
                    ).sort_values("mF1", ascending=False).reset_index(drop=True)
WIN_B = tabB.iloc[0]["rule"]
say("baseline-arm SELV winner: %s (%.4f)" % (WIN_B, tabB.iloc[0]["mF1"]))
RES["selv"] = dict(winner=WINNER, winner_mF1=float(tab.iloc[0]["mF1"]), threshold=FINAL_T,
                   spread=float(tab["mF1"].max() - tab["mF1"].min()), baseline_winner=WIN_B)
dump()

# ------------------------------------------------------------------ 4.6 refit on train+val
t0 = time.time()
gate_f = fit_gate(X_trf, y_arr)
FINAL9 = [fit_member(k, p, X_trf[att_trf].reset_index(drop=True), fam_trf[att_trf], NC)
          for _, (k, p) in SPECS]
m10 = yr10 >= 0
FINAL10 = [fit_member(k, p, X_trf[m10].reset_index(drop=True), yr10[m10], NCR)
           for _, (k, p) in SPECS]
p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
say("\nrefit done (%.1f min)" % ((time.time() - t0) / 60))

# ------------------------------------------------------------------ 5. test, touched once
def test_predict(st, POOL, Wclass, rule, t):
    gm = p_test > t
    gp = np.where(gm)[0]
    Pp = [m.predict_proba(X_test.iloc[gp]) for m in POOL]
    _, ix = st["NN"].kneighbors(prof(Pp))
    S = rule_scores(st, Pp, ix, Wclass, rule)
    yp = np.zeros(len(y_te), int)
    yp[gm] = st["LAB"][S.argmax(1)]
    return yp, gm, gp, S, Pp, ix

y_pred, gmask_t, gpos_t, S_t, P_t, idx_t = test_predict(ARM_R, FINAL10, W_e2e, WINNER, FINAL_T)

q = S_t / np.maximum(S_t.sum(1, keepdims=True), 1e-12)
pa = p_test[gpos_t]
y_proba = np.zeros((len(y_te), K)); y_proba[:, 0] = 1.0
y_proba[gpos_t, 0] = (1 - pa) + pa * q[:, IREJ]
for k_ in range(NC):
    y_proba[gpos_t, attack_labels[k_]] = pa * q[:, k_]
y_proba /= y_proba.sum(1, keepdims=True)
auc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")

say("\n" + "=" * 78)
say("TEST -- %s @ gate t=%.1f" % (WINNER, FINAL_T))
say("=" * 78)
mf = f1_score(y_te, y_pred, average="macro", zero_division=0)
acc = accuracy_score(y_te, y_pred)
say("Accuracy %.4f | macro-F1 %.4f | macro-P %.4f | macro-R %.4f | ROC-AUC %.4f"
    % (acc, mf, precision_score(y_te, y_pred, average="macro", zero_division=0),
       recall_score(y_te, y_pred, average="macro", zero_division=0), auc))
say("weighted F1 %.4f" % f1_score(y_te, y_pred, average="weighted", zero_division=0))
say("")
say(classification_report(y_te, y_pred, labels=list(range(K)), target_names=NAMES,
                          digits=4, zero_division=0))
cm = confusion_matrix(y_te, y_pred, labels=list(range(K)))
say(pd.DataFrame(cm, index=NAMES, columns=NAMES).to_string())

n_leak = int(((y_te == 0) & (y_pred != 0)).sum())
n_rej = int((gmask_t & (y_pred == 0)).sum())
n_rej_ok = int((gmask_t & (y_pred == 0) & (y_te == 0)).sum())
say("\ngate @%.1f: attack recall %.4f | precision %.4f"
    % (FINAL_T, recall_score(att_te.astype(int), gmask_t.astype(int)),
       precision_score(att_te.astype(int), gmask_t.astype(int))))
say("benign -> attack %d | sent back to normal %d (correct %d, attacks %d)"
    % (n_leak, n_rej, n_rej_ok, n_rej - n_rej_ok))

# ------------------------------------------------------------------ 6. ladder + paired bootstrap
yp_v7, *_ = test_predict(ARM_B, FINAL9, W_base, WIN_B, 0.5)
yp_v71, *_ = test_predict(ARM_R, FINAL10, W_sb, WINNER, 0.5)
yp_v71b, *_ = test_predict(ARM_R, FINAL10, W_e2e, WINNER, 0.5)
# architecture-value check: every rule scored on test (reporting only, NOT selection)
gm5 = p_test > FINAL_T; gp5 = np.where(gm5)[0]
all_test = []
for r in rule_names():
    yp = np.zeros(len(y_te), int)
    yp[gm5] = LAB10[rule_scores(ARM_R, P_t, idx_t, W_e2e, r).argmax(1)]
    all_test.append(score(y_te, yp, r))
all_test = pd.DataFrame(all_test).sort_values("mF1", ascending=False).reset_index(drop=True)

def paired_bootstrap(yp_a, yp_b, n_boot=2000, seed=RS):
    ca, cb = y_te * K + yp_a, y_te * K + yp_b
    rng = np.random.RandomState(seed)
    cls_i = [np.where(y_te == c)[0] for c in range(K)]
    diffs, marg = np.empty(n_boot), np.empty(n_boot)
    for it in range(n_boot):
        b = np.concatenate([ci[rng.randint(0, len(ci), len(ci))] for ci in cls_i])
        fa = macro_f1_from_cm(np.bincount(ca[b], minlength=K * K).reshape(K, K))
        fb = macro_f1_from_cm(np.bincount(cb[b], minlength=K * K).reshape(K, K))
        diffs[it], marg[it] = fb - fa, fb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(diffs.mean()), float(lo), float(hi), float(p), np.percentile(marg, [2.5, 97.5])

say("\n--- ladder (test) ---")
ladder = [score(y_te, yp, tag) for tag, yp in [("v7    9-class baseline", yp_v7),
                                               ("v7.1  + reject class", yp_v71),
                                               ("v7.1b + e2e objective", yp_v71b),
                                               ("FINAL (SELV-selected)", y_pred)]]
say(pd.DataFrame(ladder).to_string(index=False))
sig = {}
for tag, a, b in [("reject vs 9-class baseline", yp_v7, yp_v71),
                  ("e2e vs Stage-B objective", yp_v71, yp_v71b),
                  ("FINAL vs 9-class baseline", yp_v7, y_pred)]:
    dd, lo, hi, p, _ = paired_bootstrap(a, b)
    sig[tag] = dict(delta=round(dd, 4), lo=round(lo, 4), hi=round(hi, 4), p=round(p, 3))
    say("%-30s %+.4f [%+.4f, %+.4f] p~%.3f %s" % (tag, dd, lo, hi, p,
                                                  "ESTABLISHED" if lo > 0 or hi < 0 else "not established"))
_, _, _, _, mci = paired_bootstrap(yp_v7, y_pred)
say("\nHEADLINE macro-F1 %.4f, 95%% CI [%.4f, %.4f] -> quote %.3f [%.3f, %.3f] / acc %.4f"
    % (mf, mci[0], mci[1], mf, mci[0], mci[1], acc))
say("\n--- architecture-value check: all 21 rules on TEST (reporting only) ---")
say(all_test.to_string(index=False))
say("test spread across 21 rules: %.4f | best single member on test: %.4f"
    % (all_test["mF1"].max() - all_test["mF1"].min(),
       all_test[all_test["rule"].str.startswith("single")]["mF1"].max()))

pcf = f1_score(y_te, y_pred, average=None, labels=list(range(K)), zero_division=0)
RES["test"] = dict(rule=WINNER, threshold=FINAL_T, macro_f1=round(float(mf), 4),
                   macro_f1_ci=[round(float(mci[0]), 4), round(float(mci[1]), 4)],
                   accuracy=round(float(acc), 4), roc_auc=round(float(auc), 4),
                   per_class_f1={n: round(float(pcf[i]), 4) for i, n in enumerate(NAMES)},
                   confusion=cm.tolist(), benign_to_attack=n_leak,
                   rejected_ok=n_rej_ok, rejected_attacks=n_rej - n_rej_ok)
RES["ladder"] = ladder; RES["paired"] = sig
RES["test_all_rules"] = all_test.to_dict("records")
RES["runtime_min"] = round((time.time() - T0) / 60, 1)
say("\nruntime %.1f min" % RES["runtime_min"])
dump()
