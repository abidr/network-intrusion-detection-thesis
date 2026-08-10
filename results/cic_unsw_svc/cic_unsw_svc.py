"""
Service-identity features on CIC-UNSW-2024: the v6 pipeline with and without
dst-port/protocol columns recovered from CICFlowMeter_out.csv.

Motivation: final_model_v6_unsw.ipynb showed that on the same traffic under the
UNSW extractor, the top importances are service identity (svc_dns, svc_http) and
TTL -- fields CICFlowMeter does not emit. The released CICFlowMeter_out.csv
carries Dst Port / Protocol for all 3.54M flows; Data.csv rows were verified to
be an exact-verbatim subset (all 447,915 rows found by feature-vector match,
labels 100% consistent with the alphabetical 0-9 mapping).

Design:
  ARM "DERIVED"     -- exact replication of final_model_v6.ipynb (control; must
                       reproduce 0.6485 / 0.9123 or the paired claim is void).
  ARM "DERIVED_SVC" -- same + service columns. All splits stratify on y only,
                       so both arms share identical train/test/fold/SELV rows.

Service columns are LABEL-FREE metadata: for each row, the set of
(Dst Port, Protocol) pairs among its exact feature-vector matches in the full
CICFlowMeter_out.csv. Port is used only when the match set is unanimous
(77.3% of raw rows); otherwise dst_port = -1 and svc_ambiguous = 1.

NOTES / DEVIATIONS (stated, not silent):
  N1  Port consensus is computed over the full 3.54M-row file (train and test
      alike). It is a deterministic, label-free annotation of the feature
      vector -- the same class of operation as the pre-split variance prune.
  N2  Protocol was verified 100% unambiguous; port ambiguity is flagged, never
      guessed.
  N3  e_1NN here uses ALL train+val attack rows as reference and all test
      attack rows as queries (no subsampling), RobustScaler fitted on train+val
      attack rows.
  N4  Paired bootstrap (2000 stratified resamples) compares the two arms'
      final prediction vectors on the same test rows.

ASCII-only prints; full log written UTF-8.
"""
import hashlib, io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
DATA = ROOT / "Resource" / "CIC-UNSW-Data.csv"
LABEL = ROOT / "Resource" / "CIC-UNSW-Label.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
LABELS = {0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
          5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"}
NAMES = [LABELS[i] for i in range(10)]
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "cic_unsw_svc.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

# =========================================================================
say("=" * 78); say("1. IDENTITY RECOVERY (dst port / protocol via feature-vector join)")
say("=" * 78)

ID_CACHE = OUT / "cache_identity.npz"
if ID_CACHE.exists():
    z = np.load(ID_CACHE)
    port_arr, proto_arr, npair_arr = z["port"], z["proto"], z["npair"]
    say("loaded %s (%d rows)" % (ID_CACHE.name, len(port_arr)))
else:
    def h64(s):
        return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")

    idx = {}
    t0 = time.time()
    with open(CFM, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\r\n")
            if not c:
                continue
            p = c.split(",", 7)
            dp = (int(p[4]), int(p[5]))
            feats = p[7].rsplit(",", 1)[0]
            h = h64(feats)
            s = idx.get(h)
            if s is None:
                idx[h] = {dp}
            else:
                s.add(dp)
    say("indexed %d unique feature vectors from CICFlowMeter_out (%.0fs)"
        % (len(idx), time.time() - t0))

    ports, protos, npairs = [], [], []
    miss = 0
    with open(DATA, "r", encoding="utf-8", errors="replace") as fd:
        fd.readline()
        for dline in fd:
            s = idx.get(h64(dline.rstrip("\r\n")))
            if s is None:
                miss += 1; ports.append(-1); protos.append(-1); npairs.append(0)
                continue
            pset = {a for a, b in s}
            prset = {b for a, b in s}
            ports.append(next(iter(pset)) if len(pset) == 1 else -1)
            protos.append(next(iter(prset)) if len(prset) == 1 else -1)
            npairs.append(len(s))
    port_arr = np.array(ports, np.int32)
    proto_arr = np.array(protos, np.int16)
    npair_arr = np.array(npairs, np.int32)
    del idx; gc.collect()
    if miss:
        say("!! %d Data.csv rows had NO match -- should be 0" % miss)
    np.savez_compressed(ID_CACHE, port=port_arr, proto=proto_arr, npair=npair_arr)
    say("identity cached (%.0fs total)" % (time.time() - t0))

say("raw rows: port unambiguous %.2f%% | protocol unambiguous %.2f%%"
    % (100 * (port_arr >= 0).mean(), 100 * (proto_arr >= 0).mean()))
dump()

# =========================================================================
say(""); say("=" * 78); say("2. LOAD + DEDUPE + PRUNE + DERIVED (exact final_model_v6.ipynb)")
say("=" * 78)

Xraw = pd.read_csv(DATA)
yraw = pd.read_csv(LABEL)["Label"]
ORIG_COLS = list(Xraw.columns)
df = pd.concat([Xraw, yraw], axis=1)
df["_port"] = port_arr
df["_proto"] = proto_arr
# dedupe key = features + label exactly as the notebook; _port/_proto are
# deterministic functions of the feature vector so the row set is identical
df = df.drop_duplicates(subset=ORIG_COLS + ["Label"]).reset_index(drop=True)
Xraw2, y = df[ORIG_COLS], df["Label"]
say("dedupe: %d -> %d rows" % (len(port_arr), len(df)))

nun = Xraw2.nunique(); zero = nun[nun <= 1].index.tolist()
FULL = Xraw2.drop(columns=zero)
corr = FULL.corr(); upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
BASE = FULL.drop(columns=red)
say("prune: 76 -> %d (zero-var %d, |r|>=0.95 %d)" % (BASE.shape[1], len(zero), len(red)))


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
DER = pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)
dz = DER.nunique(); DER = DER.drop(columns=dz[dz <= 1].index.tolist())
X_DERIVED = pd.concat([BASE, DER], axis=1)
say("DERIVED: %d = %d BASE + %d engineered" % (X_DERIVED.shape[1], BASE.shape[1], DER.shape[1]))

# --- service columns ---------------------------------------------------------
port = df["_port"].values
proto = df["_proto"].values
SVC_PORTS = [(20, "svc_ftpdata"), (21, "svc_ftp"), (22, "svc_ssh"), (23, "svc_telnet"),
             (25, "svc_smtp"), (53, "svc_dns"), (80, "svc_http"), (110, "svc_pop3"),
             (111, "svc_rpc"), (139, "svc_netbios"), (143, "svc_imap"), (179, "svc_bgp"),
             (443, "svc_https"), (445, "svc_smb"), (514, "svc_syslog"), (520, "svc_rip"),
             (1723, "svc_pptp"), (5190, "svc_aim"), (6881, "svc_bt"), (8080, "svc_httpalt")]
KNOWN = {p for p, _ in SVC_PORTS}
s = {}
for p, nm in SVC_PORTS:
    s[nm] = (port == p).astype(np.float64)
s["svc_other_low"] = ((port >= 0) & (port < 1024) & ~np.isin(port, list(KNOWN))).astype(np.float64)
s["svc_registered"] = ((port >= 1024) & (port < 49152) & ~np.isin(port, list(KNOWN))).astype(np.float64)
s["svc_ephemeral"] = (port >= 49152).astype(np.float64)
s["svc_ambiguous"] = (port < 0).astype(np.float64)
s["proto_tcp"] = (proto == 6).astype(np.float64)
s["proto_udp"] = (proto == 17).astype(np.float64)
s["dst_port_num"] = port.astype(np.float64)
SVC = pd.DataFrame(s, index=df.index)
sz = SVC.nunique(); svc_const = sz[sz <= 1].index.tolist()
SVC = SVC.drop(columns=svc_const)
say("SVC columns: %d (constant-dropped: %s)" % (SVC.shape[1], svc_const or "none"))
X_SVC = pd.concat([X_DERIVED, SVC], axis=1)
SVC_NAMES = set(SVC.columns)
say("DERIVED_SVC: %d features" % X_SVC.shape[1])

say("\ndeduped rows: port unambiguous %.2f%% overall, by class:" % (100 * (port >= 0).mean()))
for k in range(10):
    m = (y.values == k)
    say("  %-16s %7d rows | port known %6.2f%% | top ports: %s"
        % (NAMES[k], m.sum(), 100 * (port[m] >= 0).mean(),
           dict(pd.Series(port[m][port[m] >= 0]).value_counts().head(4))))
dump()

# =========================================================================
say(""); say("=" * 78); say("3. SHARED SPLITS (identical across arms; stratify on y only)")
say("=" * 78)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
y_trf = y.iloc[i_trf].reset_index(drop=True)
y_te = y.iloc[i_test].values
y_arr = y_trf.values

attack_labels = sorted(l for l in y_trf.unique() if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
idx_to_fam = {i: f for f, i in fam_to_idx.items()}
NC = len(attack_labels)
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
att_te = y_te != 0
fam_te = pd.Series(y_te[att_te]).map(fam_to_idx).values
say("train+val %d (%d attack) | test %d (%d attack)"
    % (len(y_arr), att_trf.sum(), len(y_te), att_te.sum()))

comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
ref_idx = comb_idx[att_trf[comb_idx]]
y_ref = fam_trf[ref_idx]
say("COMB attack reference: %d rows | SELV: %d rows" % (len(ref_idx), len(selv_idx)))

# =========================================================================
# v6 machinery (verbatim from final_model_v6.ipynb)
# =========================================================================


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


def wc_alpha(n, a):
    w = (n.sum() / n) ** a; return w / w.mean()


def sample_w(ya, wc):
    w = wc[ya].astype(float); return w * len(w) / w.sum()


SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [sp[0] for sp in SPECS]; M = len(MODELS)
I25 = MODELS.index("XGB a=0.25")


def fit_member(kind, param, Xa, ya):
    n = np.bincount(np.asarray(ya), minlength=NC)
    if kind == "smote":
        strat = {c: param for c in range(NC) if n[c] < param}
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat).fit_resample(Xa, ya)
        m = make_xgb(); m.fit(Xs, ys); return m
    if kind == "alpha":  m = make_xgb()
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    m.fit(Xa, ya, sample_weight=sample_w(np.asarray(ya), wc_alpha(n, param)))
    return m


def fit_gate(Xa, ya10):
    yb = (ya10 != 0).astype(int)
    g_ = make_xgb(); g_.fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb))
    return g_


GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def run_arm(tag, X):
    """Full v6 protocol on feature matrix X. Returns result dict."""
    say(""); say("=" * 78); say("ARM %s (%d features)" % (tag, X.shape[1])); say("=" * 78)
    X_trf = X.iloc[i_trf].reset_index(drop=True)
    X_test = X.iloc[i_test].reset_index(drop=True)

    # --- e_1NN (N3) ---
    sc = RobustScaler().fit(X_trf[att_trf])
    Ztr = sc.transform(X_trf[att_trf]).astype(np.float32)
    Zte = sc.transform(X_test[att_te]).astype(np.float32)
    nn1 = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Ztr)
    _, ii1 = nn1.kneighbors(Zte)
    e1 = float((fam_trf[att_trf][ii1[:, 0]] != fam_te).mean())
    say("e_1NN %.4f  (DERIVED reference from feature_engineering.ipynb: 0.3629)" % e1)
    del Ztr, Zte, nn1; gc.collect()
    dump()

    # --- cross-fit ---
    cache = OUT / ("cache_oof_%s.npz" % tag)
    if cache.exists():
        z = np.load(cache)
        OOF = [z[f"m{m}"] for m in range(M)]; oof_gate = z["gate"]
        say("loaded %s" % cache.name)
    else:
        OOF = [np.zeros((len(y_arr), NC), np.float32) for _ in range(M)]
        oof_gate = np.zeros(len(y_arr), int)
        t0 = time.time()
        for f_, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr)):
            tf = time.time()
            Xtr_f, ytr_f = X_trf.iloc[tr], y_arr[tr]
            oof_gate[va] = fit_gate(Xtr_f, ytr_f).predict(X_trf.iloc[va])
            m_att = ytr_f != 0
            Xa = Xtr_f[m_att].reset_index(drop=True)
            ya = pd.Series(fam_trf[tr][m_att])
            for mi, (nm, (kind, param)) in enumerate(SPECS):
                OOF[mi][va] = fit_member(kind, param, Xa, ya).predict_proba(X_trf.iloc[va]).astype(np.float32)
            say("  fold %d/5 done (%.0fs)" % (f_ + 1, time.time() - tf)); dump()
        say("cross-fit %.0fs" % (time.time() - t0))
        np.savez_compressed(cache, gate=oof_gate, **{f"m{m}": OOF[m] for m in range(M)})
    say("OOF gate: attack recall %.4f / precision %.4f"
        % (recall_score((y_arr != 0).astype(int), oof_gate),
           precision_score((y_arr != 0).astype(int), oof_gate)))
    dump()

    # --- combiner ---
    CORRECT = np.stack([(OOF[m][ref_idx].argmax(1) == y_ref) for m in range(M)])
    WC_REF = wc_alpha(np.bincount(y_ref, minlength=NC), 0.25)
    prof = lambda ps: np.hstack(ps).astype(np.float32)
    NN_D = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
        prof([OOF[m][ref_idx] for m in range(M)]))

    def comp_ola(idx, k):
        nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
        W = np.empty((nb.shape[0], M))
        for m in range(M):
            W[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
        return W

    def comp_lca(probas, idx, k):
        nb = idx[:, :k]; yn = y_ref[nb]; W = np.empty((nb.shape[0], M))
        for m in range(M):
            pq = probas[m].argmax(1); mask = yn == pq[:, None]; cnt = mask.sum(1)
            W[:, m] = np.where(cnt > 0, (CORRECT[m][nb] & mask).sum(1) / np.maximum(cnt, 1),
                               CORRECT[m][nb].mean(1))
        return W

    def fuse_dyn(probas, Wd, p=1.0, Wclass=None):
        Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
        S = np.zeros(probas[0].shape, np.float64)
        for m in range(M):
            coef = Wp[:, m:m + 1] if Wclass is None else Wp[:, m:m + 1] * Wclass[m][None, :]
            S += coef * probas[m]
        return S.argmax(1)

    def fuse_static(probas, Wclass):
        S = np.zeros(probas[0].shape, np.float64)
        for m in range(M):
            S += probas[m] * Wclass[m][None, :]
        return S.argmax(1)

    def fit_W(probas, ytrue, passes=3):
        W = np.ones((M, NC)) / M
        best = f1_score(ytrue, fuse_static(probas, W), average="macro", zero_division=0)
        for _ in range(passes):
            for c in range(NC):
                for m_ in range(M):
                    cur = W[m_, c]
                    for gg in GRID_W:
                        W[m_, c] = gg
                        if W[:, c].sum() == 0: continue
                        sc_ = f1_score(ytrue, fuse_static(probas, W), average="macro", zero_division=0)
                        if sc_ > best + 1e-7: best, cur = sc_, gg
                    W[m_, c] = cur
        return W

    t0 = time.time()
    W_class = fit_W([OOF[m][ref_idx] for m in range(M)], y_ref)
    say("W fitted %.0fs" % (time.time() - t0)); dump()

    def score10_g(fam_g, ye, gmask, vtag):
        yp = np.zeros(len(ye), int)
        if len(fam_g):
            yp[gmask] = np.vectorize(idx_to_fam.get, otypes=[int])(fam_g)
        return dict(variant=vtag,
                    e2e_mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                    e2e_mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                    e2e_mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                    e2e_acc=round(float((yp == ye).mean()), 4))

    def run_all(P_g, idx, ye, gmask):
        out = [score10_g(fuse_static(P_g, W_class), ye, gmask, "STATIC: per-class W"),
               score10_g(fuse_static(P_g, np.ones((M, NC)) / M), ye, gmask, "STATIC: equal average")]
        for m in range(M):
            out.append(score10_g(P_g[m].argmax(1), ye, gmask, "single: " + MODELS[m]))
        for k in (30, 100):
            for p in (1, 3, 8):
                out.append(score10_g(fuse_dyn(P_g, comp_ola(idx, k), p=p), ye, gmask,
                                     f"DYN OLA-cw k={k} p={p}"))
        out.append(score10_g(fuse_dyn(P_g, comp_lca(P_g, idx, 30)), ye, gmask, "DYN LCA k=30"))
        for k in (30, 100):
            for p in (1, 3):
                out.append(score10_g(fuse_dyn(P_g, comp_ola(idx, k), p=p, Wclass=W_class), ye, gmask,
                                     f"HYBRID W x OLA-cw k={k} p={p}"))
        out.append(score10_g(fuse_dyn(P_g, comp_lca(P_g, idx, 30), p=1, Wclass=W_class), ye, gmask,
                             "HYBRID W x LCA k=30 p=1"))
        return pd.DataFrame(out)

    gmask_selv = oof_gate[selv_idx] == 1
    gpos_selv = selv_idx[gmask_selv]
    P_selv = [OOF[m][gpos_selv] for m in range(M)]
    _, i_selv = NN_D.kneighbors(prof(P_selv))
    selv_tab = run_all(P_selv, i_selv, y_arr[selv_idx], gmask_selv).sort_values(
        "e2e_mF1", ascending=False).reset_index(drop=True)
    say("\n--- SELV (%s) ---" % tag); say(selv_tab.to_string(index=False))
    BESTV = selv_tab.iloc[0]["variant"]
    say("SELV winner: %s (%.4f)" % (BESTV, selv_tab.iloc[0]["e2e_mF1"]))
    del P_selv, i_selv; gc.collect(); dump()

    # --- refit + test once ---
    t0 = time.time()
    gate_f = fit_gate(X_trf, y_arr)
    Xa_f = X_trf[att_trf].reset_index(drop=True)
    ya_f = pd.Series(fam_trf[att_trf])
    FINAL = [fit_member(k, p, Xa_f, ya_f) for _, (k, p) in SPECS]
    say("refit %.0fs" % (time.time() - t0)); dump()

    gmask_test = gate_f.predict(X_test) == 1
    gpos_test = np.where(gmask_test)[0]
    P_test = [m.predict_proba(X_test.iloc[gpos_test]).astype(np.float32) for m in FINAL]
    _, i_test_nn = NN_D.kneighbors(prof(P_test))

    test_tab = run_all(P_test, i_test_nn, y_te, gmask_test).sort_values(
        "e2e_mF1", ascending=False).reset_index(drop=True)
    say("\n--- TEST (%s; selection already made on SELV) ---" % tag)
    say(test_tab.to_string(index=False))
    sel = test_tab[test_tab.variant == BESTV].iloc[0]
    rank = 1 + list(test_tab.variant).index(BESTV)
    say("SELV-selected '%s' ranks %d of %d on test -> mF1 %.4f acc %.4f"
        % (BESTV, rank, len(test_tab), sel["e2e_mF1"], sel["e2e_acc"]))

    def build(t, P_g, idx):
        if t == "STATIC: per-class W":   return fuse_static(P_g, W_class)
        if t == "STATIC: equal average": return fuse_static(P_g, np.ones((M, NC)) / M)
        if t.startswith("single: "):     return P_g[MODELS.index(t[8:])].argmax(1)
        if t == "DYN LCA k=30":          return fuse_dyn(P_g, comp_lca(P_g, idx, 30))
        if t == "HYBRID W x LCA k=30 p=1":
            return fuse_dyn(P_g, comp_lca(P_g, idx, 30), p=1, Wclass=W_class)
        k = 100 if "k=100" in t else 30
        p = 8 if "p=8" in t else (3 if "p=3" in t else 1)
        return fuse_dyn(P_g, comp_ola(idx, k), p=p,
                        Wclass=(W_class if t.startswith("HYBRID") else None))

    fam_pred = build(BESTV, P_test, i_test_nn)
    yp = np.zeros(len(y_te), int)
    yp[gmask_test] = np.vectorize(idx_to_fam.get, otypes=[int])(fam_pred)
    say("\n" + classification_report(y_te, yp, labels=list(range(10)),
                                     target_names=NAMES, digits=3, zero_division=0))
    f1c = f1_score(y_te, yp, average=None, labels=range(10), zero_division=0)
    gr = recall_score((y_te != 0).astype(int), gmask_test.astype(int))
    gp = precision_score((y_te != 0).astype(int), gmask_test.astype(int))
    say("gate on test: attack recall %.4f / precision %.4f" % (gr, gp))

    imp = pd.Series(FINAL[I25].feature_importances_, index=X.columns).sort_values(ascending=False)
    say("\ntop 25 Stage-B features (XGB a=0.25):")
    for i, (kk, vv) in enumerate(imp.head(25).items()):
        mark = "(svc)" if kk in SVC_NAMES else ("(der)" if kk in DER.columns else "")
        say("  %2d. %-24s %.4f %s" % (i + 1, kk, vv, mark))
    svc_share = float(imp[[c for c in imp.index if c in SVC_NAMES]].sum() / imp.sum()) if SVC_NAMES & set(imp.index) else 0.0
    if tag == "DERIVED_SVC":
        say("svc columns = %.1f%% of features, %.1f%% of importance"
            % (100 * len(SVC_NAMES & set(X.columns)) / X.shape[1], 100 * svc_share))

    dump()
    return dict(tag=tag, n_features=int(X.shape[1]), e_1nn=round(e1, 4),
                selv_winner=BESTV, test_rank=int(rank),
                mF1=float(sel["e2e_mF1"]), acc=float(sel["e2e_acc"]),
                mP=float(sel["e2e_mP"]), mR=float(sel["e2e_mR"]),
                per_class_f1={NAMES[i]: round(float(f1c[i]), 4) for i in range(10)},
                gate_recall=round(float(gr), 4), gate_precision=round(float(gp), 4),
                svc_importance_share=round(svc_share, 4),
                selv_table=selv_tab.to_dict("records"),
                test_table=test_tab.to_dict("records")), yp


res_a, yp_a = run_arm("DERIVED", X_DERIVED)
res_b, yp_b = run_arm("DERIVED_SVC", X_SVC)

# =========================================================================
say(""); say("=" * 78); say("4. PAIRED COMPARISON (N4: 2000 stratified bootstrap resamples)")
say("=" * 78)
say("DERIVED     : mF1 %.4f acc %.4f  (published v6: 0.6485 / 0.9123)" % (res_a["mF1"], res_a["acc"]))
say("DERIVED_SVC : mF1 %.4f acc %.4f" % (res_b["mF1"], res_b["acc"]))
repro_ok = abs(res_a["mF1"] - 0.6485) < 1e-4 and abs(res_a["acc"] - 0.9123) < 1e-4
say("control reproduces published v6: %s" % ("YES (exact)" if repro_ok else "NO -- INVESTIGATE"))


def macro_f1_from_counts(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return f1.mean()


code_a = y_te * 10 + yp_a
code_b = y_te * 10 + yp_b
rng = np.random.RandomState(RS)
cls_idx = [np.where(y_te == c)[0] for c in range(10)]
diffs = np.empty(2000); f1s_b = np.empty(2000)
for it in range(2000):
    bidx = np.concatenate([ci[rng.randint(0, len(ci), len(ci))] for ci in cls_idx])
    cma = np.bincount(code_a[bidx], minlength=100).reshape(10, 10)
    cmb = np.bincount(code_b[bidx], minlength=100).reshape(10, 10)
    fb = macro_f1_from_counts(cmb)
    f1s_b[it] = fb
    diffs[it] = fb - macro_f1_from_counts(cma)
d_obs = res_b["mF1"] - res_a["mF1"]
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
say("\npaired delta (SVC - DERIVED): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (d_obs, lo, hi, min(1.0, p_two)))
blo, bhi = np.percentile(f1s_b, [2.5, 97.5])
say("DERIVED_SVC marginal: %.4f [%.4f, %.4f]" % (res_b["mF1"], blo, bhi))
say("\nper-class F1 delta (SVC - DERIVED):")
for i in range(10):
    say("  %-16s %.4f -> %.4f  (%+.4f)"
        % (NAMES[i], res_a["per_class_f1"][NAMES[i]], res_b["per_class_f1"][NAMES[i]],
           res_b["per_class_f1"][NAMES[i]] - res_a["per_class_f1"][NAMES[i]]))

res = dict(experiment="service-identity features from CICFlowMeter_out.csv",
           notes=["N1 port consensus over full 3.54M file (label-free)",
                  "N2 port used only when unanimous; else svc_ambiguous",
                  "N3 e_1NN on all attack rows, no subsampling",
                  "N4 paired stratified bootstrap 2000x"],
           control_reproduces_v6=bool(repro_ok),
           arms=[res_a, res_b],
           paired_delta=round(float(d_obs), 4),
           paired_ci=[round(float(lo), 4), round(float(hi), 4)],
           paired_p=round(float(min(1.0, p_two)), 4),
           svc_marginal_ci=[round(float(blo), 4), round(float(bhi), 4)],
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "cic_unsw_svc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
