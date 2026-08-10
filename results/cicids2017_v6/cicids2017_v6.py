"""
v6 architecture on CIC-IDS2017.

v6 = DERIVED features (BASE + 31 engineered) + two-stage gate/family hierarchy
     + 7-member alpha-diverse pool + combiner selected on SELV.

Faithful to final_model_v6.ipynb / final_model_v6_unsw.ipynb except where the
10x scale forces a change. Every deviation is listed in DEVIATIONS below and
echoed into the JSON so it cannot be lost.

DEVIATIONS (stated, not silent):
  D1  Web Attack Brute Force / XSS / Sql Injection collapsed to one "Web Attack"
      class, per the audit. DoS variants kept separate. -> 13 classes.
  D2  Combiner/DES reference set capped at 40,000 stratified attack rows.
      v6 used 32,468 and v6-unsw 30,809, so this matches the regime the combiner
      was validated in; the uncapped set here would be ~170k and makes the
      brute-force kNN quadratic blow-up intractable.
  D3  Rows carrying inf/NaN (Flow Bytes/s, Flow Packets/s) are dropped, not
      imputed -- 0.1% of rows. Per-class breakdown of the loss is reported.
  D4  SMOTE k_neighbors clamped to min(5, n_min-1); Heartbleed has 11 rows total
      and the default k=5 would raise.
  D5  Features held as float32 numpy, not float64 DataFrame (memory).

ASCII-only prints; full log written UTF-8.
"""
import io, json, sys, time, gc
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
                             classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

D = Path(r"E:\AIUB\THESIS\Resource\CIC-IDS2017")
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
REF_CAP = 40000
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "cicids2017_v6.txt", "w", encoding="utf-8").write("\n".join(LOG))


FILES = ["Monday-WorkingHours.pcap_ISCX.csv",
         "Tuesday-WorkingHours.pcap_ISCX.csv",
         "Wednesday-workingHours.pcap_ISCX.csv",
         "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         "Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

t_start = time.time()
say("=" * 78); say("1. LOAD + COLLAPSE WEB ATTACKS"); say("=" * 78)

fr = []
for f in FILES:
    d_ = pd.read_csv(D / f, low_memory=False, encoding="latin-1")
    d_.columns = [c.strip() for c in d_.columns]
    fr.append(d_)
df = pd.concat(fr, ignore_index=True); del fr; gc.collect()
say("pooled %d x %d" % df.shape)

df["Label"] = df["Label"].astype(str).str.strip()
n_web = int(df["Label"].str.startswith("Web Attack").sum())
df.loc[df["Label"].str.startswith("Web Attack"), "Label"] = "Web Attack"
say("D1: collapsed %d Web Attack rows (BruteForce/XSS/SqlInjection) into one class" % n_web)

FEAT = [c for c in df.columns if c != "Label"]

# --- D3: inf/NaN rows ---
num = df[FEAT].to_numpy(dtype=np.float64)
bad = ~np.isfinite(num).all(axis=1)
say("D3: dropping %d rows (%.4f%%) with inf/NaN" % (bad.sum(), 100 * bad.mean()))
lost = df.loc[bad, "Label"].value_counts()
for k, v in lost.items():
    say("      %-22s -%d" % (k, v))
df = df.loc[~bad].reset_index(drop=True); del num; gc.collect()

# --- dedupe ---
n0 = len(df)
df = df.drop_duplicates().reset_index(drop=True)
say("dedupe: %d -> %d (%.2f%% duplicates)" % (n0, len(df), 100 * (1 - len(df) / n0)))

# --- label encoding: BENIGN=0, attacks alphabetical (case-insensitive) ---
atk = sorted([l for l in df["Label"].unique() if l != "BENIGN"], key=str.lower)
CAT = {"BENIGN": 0}
for i, a in enumerate(atk): CAT[a] = i + 1
NAMES = ["BENIGN"] + atk
NCLS = len(NAMES)
y = df["Label"].map(CAT).astype(int).values
say("\n%d classes" % NCLS)
vc = pd.Series(y).value_counts().sort_index()
for k, v in vc.items():
    say("  %d %-22s %8d  %7.4f%%" % (k, NAMES[k], v, 100 * v / len(y)))

say("\n" + "=" * 78); say("2. PRUNE + DERIVED FEATURES"); say("=" * 78)
Xdf = df[FEAT]
nun = Xdf.nunique(); zero = nun[nun <= 1].index.tolist()
FULL = Xdf.drop(columns=zero)
corr = FULL.corr(numeric_only=True)
upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
BASE = FULL.drop(columns=red)
say("zero-var dropped (%d)" % len(zero))
say("|r|>=0.95 dropped (%d)" % len(red))
say("FULL %d -> BASE %d" % (FULL.shape[1], BASE.shape[1]))

# v3 -> the names build_derived expects
REN = {"Total Fwd Packets": "Total Fwd Packet",
       "Total Backward Packets": "Total Bwd packets",
       "Total Length of Fwd Packets": "Total Length of Fwd Packet",
       "Total Length of Bwd Packets": "Total Length of Bwd Packet",
       "Init_Win_bytes_forward": "FWD Init Win Bytes",
       "Init_Win_bytes_backward": "Bwd Init Win Bytes"}
FULLr = FULL.rename(columns=REN)


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_derived(F):
    d = {}
    missing = []
    def g(c):
        if c in F.columns: return F[c]
        missing.append(c); return pd.Series(0.0, index=F.index)
    fpk, bpk = g("Total Fwd Packet"), g("Total Bwd packets")
    fby, bby = g("Total Length of Fwd Packet"), g("Total Length of Bwd Packet")
    tot_pk, tot_by = fpk + bpk, fby + bby
    d["r_pkt_fwd_bwd"] = safe(fpk, bpk);   d["r_byte_fwd_bwd"] = safe(fby, bby)
    d["asym_pkt"] = safe(fpk - bpk, tot_pk); d["asym_byte"] = safe(fby - bby, tot_by)
    d["bpp_fwd"] = safe(fby, fpk); d["bpp_bwd"] = safe(bby, bpk)
    d["bpp_flow"] = safe(tot_by, tot_pk); d["r_bpp"] = safe(d["bpp_fwd"], d["bpp_bwd"])
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
    d["active_share"] = safe(g("Active Mean"), dur)
    d["idle_share"] = safe(g("Idle Mean"), dur)
    d["r_active_idle"] = safe(g("Active Mean"), g("Idle Mean"))
    d["dur_per_pkt"] = safe(dur, tot_pk)
    d["r_init_win"] = safe(g("FWD Init Win Bytes"), g("Bwd Init Win Bytes"))
    O = pd.DataFrame(d, index=F.index).astype(np.float64)
    return O.replace([np.inf, -np.inf], 0.0).fillna(0.0), sorted(set(missing))


DER, missing = build_derived(FULLr)
say("derived built: %d columns" % DER.shape[1])
if missing:
    say("!! WARNING - these inputs were ABSENT and silently zeroed: %s" % missing)
else:
    say("all derived inputs present -- no silent zero-fill (checked)")
dz = DER.nunique(); dropped_der = dz[dz <= 1].index.tolist()
DER = DER.drop(columns=dropped_der)
say("derived constant-dropped (%d): %s" % (len(dropped_der), dropped_der or "none"))

Xdf = pd.concat([BASE, DER], axis=1)
COLS = list(Xdf.columns)
DERIVED_NAMES = set(DER.columns)
say("DERIVED space: %d features (CIC-UNSW v6 had 75)" % len(COLS))
X = Xdf.to_numpy(dtype=np.float32)
del Xdf, DER, FULL, FULLr, BASE, df, corr, upper; gc.collect()
say("X %s  %.0f MB" % (X.shape, X.nbytes / 1e6))
dump()

say("\n" + "=" * 78); say("3. SPLIT"); say("=" * 78)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf, y_arr = X[i_trf], y[i_trf]
X_test, y_te = X[i_test], y[i_test]
del X; gc.collect()

attack_labels = sorted(l for l in np.unique(y) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
idx_to_fam = {i: f for f, i in fam_to_idx.items()}
NC = len(attack_labels)
FAM = [NAMES[idx_to_fam[i]] for i in range(NC)]
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = np.array([fam_to_idx[v] for v in y_arr[att_trf]])
att_te = y_te != 0
fam_te = np.array([fam_to_idx[v] for v in y_te[att_te]])
say("train+val %d (%d attack) | test %d (%d attack) | %d families"
    % (len(y_arr), att_trf.sum(), len(y_te), att_te.sum(), NC))
say("test support per class:")
for k, v in pd.Series(y_te).value_counts().sort_index().items():
    say("  %-22s %7d" % (NAMES[k], v))

say("\n" + "=" * 78); say("4. COVER-HART BRACKET (attack rows, RobustScaler space)"); say("=" * 78)


def cover_hart(e1nn, c):
    inner = 1 - (c / (c - 1)) * e1nn
    return np.nan if inner < 0 else (c - 1) / c * (1 - np.sqrt(inner))


sc = RobustScaler().fit(X_trf[att_trf])
SUB = 60000
rng = np.random.RandomState(RS)
ref_pool = np.where(att_trf)[0]
sub = rng.choice(ref_pool, min(SUB, len(ref_pool)), replace=False)
Ztr = sc.transform(X_trf[sub]).astype(np.float32)
q = np.where(att_te)[0]
qs = rng.choice(len(q), min(60000, len(q)), replace=False)
Zte = sc.transform(X_test[q[qs]]).astype(np.float32)
nn1 = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Ztr)
_, ii1 = nn1.kneighbors(Zte)
e1 = float((fam_trf[sub][ii1[:, 0]] != fam_te[qs]).mean())
say("e_1NN %.4f (ref %d, query %d)  ->  e* in [%.4f, %.4f]  acc ceiling [%.4f, %.4f]"
    % (e1, len(sub), len(qs), cover_hart(e1, NC), e1, 1 - e1, 1 - cover_hart(e1, NC)))
say("CIC-UNSW DERIVED reference: e_1NN 0.3629 | UNSW-NB15: 0.3043")
del Ztr, Zte; gc.collect()
dump()

say("\n" + "=" * 78); say("5. v6 POOL -- 5-fold cross-fit"); say("=" * 78)


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


wc_alpha = lambda n, a: ((n.sum() / np.maximum(n, 1)) ** a) / (((n.sum() / np.maximum(n, 1)) ** a).mean())


def sample_w(ya, wc):
    w = wc[ya].astype(float); return w * len(w) / w.sum()


SPECS = ([("XGB a=%s" % a, ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [s[0] for s in SPECS]; M = len(MODELS); I25 = MODELS.index("XGB a=0.25")


def fit_member(kind, param, Xa, ya):
    n = np.bincount(np.asarray(ya), minlength=NC)
    if kind == "smote":
        nz = n[n > 0]
        k = int(max(1, min(5, nz.min() - 1)))            # D4
        strat = {c: param for c in range(NC) if 0 < n[c] < param}
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat,
                       k_neighbors=k).fit_resample(Xa, ya)
        m = make_xgb(); m.fit(Xs, ys); return m
    if kind == "alpha":  m = make_xgb()
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    m.fit(Xa, ya, sample_weight=sample_w(np.asarray(ya), wc_alpha(n, param)))
    return m


def fit_gate(Xa, ya10):
    yb = (ya10 != 0).astype(int)
    m = make_xgb(); m.fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb)); return m


CACHE = OUT / "cache_oof_cicids2017.npz"
if CACHE.exists():
    z = np.load(CACHE); OOF = [z["m%d" % m] for m in range(M)]; oof_gate = z["gate"]
    say("loaded %s" % CACHE.name)
else:
    OOF = [np.zeros((len(y_arr), NC), np.float32) for _ in range(M)]
    oof_gate = np.zeros(len(y_arr), np.int8)
    t0 = time.time()
    for f_, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr)):
        tf = time.time()
        oof_gate[va] = fit_gate(X_trf[tr], y_arr[tr]).predict(X_trf[va])
        say("  fold %d gate %.0fs" % (f_ + 1, time.time() - tf))
        ma = y_arr[tr] != 0
        Xa = X_trf[tr][ma]; ya = fam_trf[tr][ma]
        for mi, (nm, (k, p)) in enumerate(SPECS):
            tm = time.time()
            mdl = fit_member(k, p, Xa, ya)
            OOF[mi][va] = mdl.predict_proba(X_trf[va]).astype(np.float32)
            del mdl; gc.collect()
            say("    %-16s %.0fs" % (nm, time.time() - tm))
        say("  fold %d/5 total %.0fs (elapsed %.0fm)" % (f_ + 1, time.time() - tf, (time.time() - t0) / 60))
        dump()
    say("cross-fit %.0fs" % (time.time() - t0))
    np.savez_compressed(CACHE, gate=oof_gate, **{"m%d" % m: OOF[m] for m in range(M)})

yb_tr = (y_arr != 0).astype(int)
say("OOF gate: attack recall %.4f / precision %.4f" % (recall_score(yb_tr, oof_gate),
                                                      precision_score(yb_tr, oof_gate)))
dump()

say("\n" + "=" * 78); say("6. COMBINER"); say("=" * 78)
comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                      random_state=RS + 1, stratify=y_arr)
ref_full = comb_idx[att_trf[comb_idx]]
if len(ref_full) > REF_CAP:                                   # D2
    rf_lab = fam_trf[ref_full]
    ref_idx, _ = train_test_split(ref_full, train_size=REF_CAP,
                                  random_state=RS, stratify=rf_lab)
    say("D2: reference capped %d -> %d stratified attack rows" % (len(ref_full), len(ref_idx)))
else:
    ref_idx = ref_full
y_ref = fam_trf[ref_idx]
CORRECT = np.stack([(OOF[m][ref_idx].argmax(1) == y_ref) for m in range(M)])
WC_REF = wc_alpha(np.bincount(y_ref, minlength=NC), 0.25)
prof = lambda ps: np.hstack(ps).astype(np.float32)
NN_D = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
    prof([OOF[m][ref_idx] for m in range(M)]))


def comp_ola(idx, k):
    nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
    W = np.empty((nb.shape[0], M))
    for m in range(M): W[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
    return W


def fuse_static(probas, Wc):
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M): S += probas[m] * Wc[m][None, :]
    return S.argmax(1)


def fuse_dyn(probas, Wd, p=1.0, Wclass=None):
    Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M):
        coef = Wp[:, m:m + 1] if Wclass is None else Wp[:, m:m + 1] * Wclass[m][None, :]
        S += coef * probas[m]
    return S.argmax(1)


GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


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
                    s = f1_score(ytrue, fuse_static(probas, W), average="macro", zero_division=0)
                    if s > best + 1e-7: best, cur = s, gg
                W[m_, c] = cur
    return W


t0 = time.time()
W_class = fit_W([OOF[m][ref_idx] for m in range(M)], y_ref)
say("W fitted %.0fs on %d rows" % (time.time() - t0, len(ref_idx)))
dump()


def score_all(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int)
    if len(fam_g): yp[gmask] = np.array([idx_to_fam[v] for v in fam_g])
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


def run_all(P_g, idx, ye, gmask):
    out = [score_all(fuse_static(P_g, W_class), ye, gmask, "STATIC: per-class W"),
           score_all(fuse_static(P_g, np.ones((M, NC)) / M), ye, gmask, "STATIC: equal average")]
    for m in range(M):
        out.append(score_all(P_g[m].argmax(1), ye, gmask, "single: " + MODELS[m]))
    for k in (30, 100):
        for p in (1, 3, 8):
            out.append(score_all(fuse_dyn(P_g, comp_ola(idx, k), p=p), ye, gmask,
                                 "DYN OLA-cw k=%d p=%d" % (k, p)))
    for k in (30, 100):
        out.append(score_all(fuse_dyn(P_g, comp_ola(idx, k), p=8, Wclass=W_class), ye, gmask,
                             "HYBRID W x OLA-cw k=%d p=8" % k))
    return pd.DataFrame(out)


t0 = time.time()
gmask_selv = oof_gate[selv_idx] == 1
gpos = selv_idx[gmask_selv]
P_selv = [OOF[m][gpos] for m in range(M)]
_, i_selv = NN_D.kneighbors(prof(P_selv))
say("SELV kNN %.0fs (%d queries)" % (time.time() - t0, len(gpos)))
selv_tab = run_all(P_selv, i_selv, y_arr[selv_idx], gmask_selv).sort_values(
    "mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV ---"); say(selv_tab.to_string(index=False))
BESTV = selv_tab.iloc[0]["variant"]
say("SELV winner: %s (%.4f)" % (BESTV, selv_tab.iloc[0]["mF1"]))
del P_selv, i_selv; gc.collect()
dump()

say("\n" + "=" * 78); say("7. REFIT ON train+val, TEST ONCE"); say("=" * 78)
t0 = time.time()
gate_f = fit_gate(X_trf, y_arr)
say("gate refit %.0fs" % (time.time() - t0))
Xa_f = X_trf[att_trf]; ya_f = fam_trf[att_trf]
FINAL = []
for nm, (k, p) in SPECS:
    tm = time.time(); FINAL.append(fit_member(k, p, Xa_f, ya_f))
    say("  %-16s %.0fs" % (nm, time.time() - tm)); dump()
gmask_test = gate_f.predict(X_test) == 1
tp = np.where(gmask_test)[0]
P_test = [m.predict_proba(X_test[tp]).astype(np.float32) for m in FINAL]
_, i_test = NN_D.kneighbors(prof(P_test))
say("refit+test %.0fs" % (time.time() - t0))

test_tab = run_all(P_test, i_test, y_te, gmask_test).sort_values(
    "mF1", ascending=False).reset_index(drop=True)
say("\n--- TEST (selection already made on SELV) ---")
say(test_tab.to_string(index=False))
sel = test_tab[test_tab.variant == BESTV].iloc[0]
rank = 1 + list(test_tab.variant).index(BESTV)
say("\nSELV-selected '%s' ranks %d of %d on test" % (BESTV, rank, len(test_tab)))
say("CIC-IDS2017: macro-F1 %.4f | accuracy %.4f" % (sel["mF1"], sel["acc"]))
say("reference levels (NOT comparable - different taxonomy):")
say("  CIC-UNSW-2024 v6 : 0.6485 / 0.9123")
say("  UNSW-NB15 v6     : 0.6387 / 0.8303")


def build(t, P_g, idx):
    if t == "STATIC: per-class W":   return fuse_static(P_g, W_class)
    if t == "STATIC: equal average": return fuse_static(P_g, np.ones((M, NC)) / M)
    if t.startswith("single: "):     return P_g[MODELS.index(t[8:])].argmax(1)
    k = 100 if "k=100" in t else 30
    p = 8 if "p=8" in t else (3 if "p=3" in t else 1)
    return fuse_dyn(P_g, comp_ola(idx, k), p=p,
                    Wclass=(W_class if t.startswith("HYBRID") else None))


fam_pred = build(BESTV, P_test, i_test)
yp = np.zeros(len(y_te), int)
yp[gmask_test] = np.array([idx_to_fam[v] for v in fam_pred])
say("\n" + classification_report(y_te, yp, labels=list(range(NCLS)), target_names=NAMES,
                                 digits=3, zero_division=0))

f1c = f1_score(y_te, yp, average=None, labels=range(NCLS), zero_division=0)
tiny = [NAMES.index(n) for n in ("Heartbleed", "Infiltration") if n in NAMES]
keep = [i for i in range(NCLS) if i not in tiny]
say("macro-F1 all %d classes            : %.4f" % (NCLS, f1c.mean()))
say("macro-F1 excl. Heartbleed/Infiltr. : %.4f  (%d classes)" % (f1c[keep].mean(), len(keep)))
say("  -> the two tiny classes have %d and %d test rows"
    % tuple(int((y_te == t).sum()) for t in tiny))

gr = recall_score((y_te != 0).astype(int), gmask_test.astype(int))
gp = precision_score((y_te != 0).astype(int), gmask_test.astype(int))
say("\ngate on test: attack recall %.4f / precision %.4f" % (gr, gp))
say("  CIC-UNSW v6 gate 0.9961/0.9280 | UNSW-NB15 0.9176/0.9209")

imp = pd.Series(FINAL[I25].feature_importances_, index=COLS).sort_values(ascending=False)
say("\ntop 20 Stage-B features")
for i, (k, v) in enumerate(imp.head(20).items()):
    say("  %2d. %-28s %.4f %s" % (i + 1, k, v, "(der)" if k in DERIVED_NAMES else ""))
der_share = imp[[c for c in imp.index if c in DERIVED_NAMES]].sum() / imp.sum()
say("derived = %.1f%% of columns, %.1f%% of importance"
    % (100 * len(DERIVED_NAMES) / len(COLS), 100 * der_share))
if "Destination Port" in imp.index:
    say("Destination Port importance rank %d (%.4f)"
        % (1 + list(imp.index).index("Destination Port"), imp["Destination Port"]))

res = dict(dataset="CIC-IDS2017", deviations=["D1 web-attack collapse", "D2 ref cap %d" % REF_CAP,
                                              "D3 drop inf/NaN rows", "D4 SMOTE k clamp",
                                              "D5 float32"],
           n_classes=NCLS, names=NAMES, n_features=len(COLS),
           rows_after_clean_dedupe=int(len(y)), n_train=int(len(y_arr)), n_test=int(len(y_te)),
           e_1nn=round(e1, 4), selv_winner=BESTV, test_rank=int(rank),
           mF1=float(sel["mF1"]), acc=float(sel["acc"]),
           mP=float(sel["mP"]), mR=float(sel["mR"]),
           macroF1_excl_tiny=round(float(f1c[keep].mean()), 4),
           per_class_f1={NAMES[i]: round(float(f1c[i]), 4) for i in range(NCLS)},
           gate_recall=round(float(gr), 4), gate_precision=round(float(gp), 4),
           derived_importance_share=round(float(der_share), 4),
           selv_table=selv_tab.to_dict("records"), test_table=test_tab.to_dict("records"),
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "cicids2017_v6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
