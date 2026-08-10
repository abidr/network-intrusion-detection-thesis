"""
v7.2: hyperparameter tuning of the Stage-B XGBoost -- the one conventional lever
never tested in this project (every XGB has run n_estimators=400, max_depth=8,
lr=0.1, subsample=0.9, colsample=0.9 since the first baseline).

PRE-REGISTERED PROTOCOL:
  Stage 1 (probe): 40 random configs + the default. Each trains a single
    10-class reject-arm XGB (alpha=0.25 weights) on the COMB half's
    attack+reject rows and is scored by END-TO-END macro-F1 on the SELV half
    (gate fixed via cached OOF probabilities, t=0.5). Test is not touched.
  Decision rule: promote to Stage 2 only if best SELV score exceeds the
    default's by > +0.005 (below that is rare-class selection noise).
  Stage 2 (only if bar cleared): re-cross-fit the 5 XGB pool members with the
    tuned params (RF/ET unchanged -- stated), refit W under the e2e objective,
    SELV-select the combiner, refit on train+val, test ONCE, paired bootstrap
    vs the same-run v7.1b reference.

EXPECTATION (pre-registered): the project record says gains come from
weighting/data/information, never capacity -- a null is the likely and
publishable outcome.

Caveat (stated): Stage-1 candidates train on half the data; params that
interact with dataset size may rank differently at full scale -- that is why
Stage 2 re-validates instead of trusting the probe ranking.

ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc, itertools, random
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
BAR = 0.005
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "v72_hparam.txt", "w", encoding="utf-8").write("\n".join(LOG))


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

M = 7
z = np.load(ROOT / "cache_oof_v7_full.npz")
oof_gate = z["gate"]
p_gate = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
REJECT = (y_arr == 0) & (oof_gate == 1)
NCR = NC + 1; IREJ = NC
yr10 = np.where(REJECT, IREJ, fam_trf)
LAB10 = np.array(attack_labels + [0])
say("reject set %d" % REJECT.sum())
dump()

# ---------------------------------------------------------------- stage-1 machinery
comb_ar = comb_idx[yr10[comb_idx] >= 0]           # COMB attacks + rejects (train rows)
Xc = X_trf.iloc[comb_ar].reset_index(drop=True)
yc = yr10[comb_ar]
n_c = np.bincount(yc, minlength=NCR)
w_c = (n_c.sum() / n_c) ** 0.25; w_c = w_c / w_c.mean()
sw_c = w_c[yc].astype(float); sw_c = sw_c * len(sw_c) / sw_c.sum()

gmask_selv = p_gate[selv_idx] > 0.5
gpos_selv = selv_idx[gmask_selv]
X_selv_g = X_trf.iloc[gpos_selv].reset_index(drop=True)
y_selv_g = y_arr[gpos_selv]
BG_selv = np.zeros((10, 10), np.int64)
neg_selv = selv_idx[~gmask_selv]
for c, cnt in zip(*np.unique(y_arr[neg_selv], return_counts=True)):
    BG_selv[c, 0] = cnt
say("stage-1: train %d COMB rows | eval %d SELV gate-positives (+%d background)"
    % (len(yc), len(gpos_selv), len(neg_selv)))


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


def selv_e2e(model):
    yp = LAB10[model.predict_proba(X_selv_g).argmax(1)]
    cm = np.bincount(y_selv_g * 10 + yp, minlength=100).reshape(10, 10) + BG_selv
    return macro_f1_from_cm(cm)


DEFAULT = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
               min_child_weight=1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1)


def xgb_of(cfg):
    p = dict(tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=RS)
    p.update(cfg)
    return XGBClassifier(**p)


say(""); say("=" * 74); say("STAGE 1 -- random search (40 configs + default)"); say("=" * 74)

SPACE = dict(max_depth=[6, 8, 10, 12],
             lr_rounds=[(0.05, 800), (0.1, 400), (0.1, 800), (0.2, 400)],
             min_child_weight=[1, 5, 20],
             subsample=[0.7, 0.9, 1.0],
             colsample_bytree=[0.7, 0.9, 1.0],
             reg_lambda=[1, 10])
rng = random.Random(RS)
seen, CFGS = set(), []
while len(CFGS) < 40:
    lr, ne = rng.choice(SPACE["lr_rounds"])
    cfg = dict(max_depth=rng.choice(SPACE["max_depth"]), learning_rate=lr, n_estimators=ne,
               min_child_weight=rng.choice(SPACE["min_child_weight"]),
               subsample=rng.choice(SPACE["subsample"]),
               colsample_bytree=rng.choice(SPACE["colsample_bytree"]),
               reg_lambda=rng.choice(SPACE["reg_lambda"]))
    key = tuple(sorted(cfg.items()))
    if key in seen or cfg == DEFAULT:
        continue
    seen.add(key); CFGS.append(cfg)

CACHE_S1 = OUT / "cache_stage1.json"
if CACHE_S1.exists():
    rows = json.loads(CACHE_S1.read_text(encoding="utf-8"))
    s0 = float(next(r["selv"] for r in rows if r["cfg"] == "DEFAULT"))
    say("loaded %s (%d configs)" % (CACHE_S1.name, len(rows)))
else:
    rows = []
    t0 = time.time()
    m0 = xgb_of(DEFAULT); m0.fit(Xc, yc, sample_weight=sw_c)
    s0 = selv_e2e(m0); del m0; gc.collect()
    rows.append(dict(cfg="DEFAULT", selv=round(s0, 4), **DEFAULT))
    say("  0/40 DEFAULT                                selv %.4f (%.0fs)" % (s0, time.time() - t0))
    dump()

    for i, cfg in enumerate(CFGS):
        tc = time.time()
        m = xgb_of(cfg); m.fit(Xc, yc, sample_weight=sw_c)
        sc = selv_e2e(m); del m; gc.collect()
        rows.append(dict(cfg="C%02d" % (i + 1), selv=round(sc, 4), **cfg))
        say(" %2d/40 d%-2d lr%.2f n%-3d mcw%-2d sub%.1f col%.1f l%-2d selv %.4f  %+.4f (%.0fs)"
            % (i + 1, cfg["max_depth"], cfg["learning_rate"], cfg["n_estimators"],
               cfg["min_child_weight"], cfg["subsample"], cfg["colsample_bytree"],
               cfg["reg_lambda"], sc, sc - s0, time.time() - tc))
        if (i + 1) % 5 == 0:
            dump()
    CACHE_S1.write_text(json.dumps(rows), encoding="utf-8")

tab = pd.DataFrame(rows).sort_values("selv", ascending=False).reset_index(drop=True)
say("\n--- STAGE 1 RANKING (top 10) ---")
say(tab.head(10).to_string(index=False))
best = tab.iloc[0]
gain = float(best["selv"]) - s0
say("\ndefault SELV %.4f | best %.4f (%s) | gain %+.4f | pre-registered bar +%.3f"
    % (s0, best["selv"], best["cfg"], gain, BAR))
dump()

res = dict(default_selv=round(s0, 4), best_cfg=str(best["cfg"]),
           best_selv=round(float(best["selv"]), 4), gain=round(gain, 4), bar=BAR,
           promoted=bool(gain > BAR), stage1_table=tab.to_dict("records"))

if gain <= BAR:
    say("\nBAR NOT CLEARED -- stopping (pre-registered). Capacity is not the binding")
    say("constraint; the sweep table above is the evidence. Test was never touched.")
    res["minutes"] = round((time.time() - t_start) / 60, 1)
    (OUT / "v72_hparam.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    say("\ntotal %.1f min" % res["minutes"]); dump()
    sys.exit(0)

# ---------------------------------------------------------------- stage 2
say(""); say("=" * 74); say("STAGE 2 -- bar cleared: tuned pool, W_e2e, SELV select, test once"); say("=" * 74)
TUNED = {k: best[k] for k in ("max_depth", "learning_rate", "n_estimators",
                              "min_child_weight", "subsample", "colsample_bytree", "reg_lambda")}
TUNED = {k: (int(v) if k in ("max_depth", "n_estimators", "min_child_weight", "reg_lambda")
             else float(v)) for k, v in TUNED.items()}
say("tuned params: %s  (RF/ET members stay default -- not tuned)" % TUNED)


def make_xgb_cfg(cfg, **kw):
    p = dict(tree_method="hist", eval_metric="mlogloss", n_jobs=-1, random_state=RS)
    p.update(cfg); p.update(kw); return XGBClassifier(**p)


SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]


def fit_member_t(kind, param, Xa, ya, cfg):
    n = np.bincount(np.asarray(ya), minlength=NCR)
    if kind == "smote":
        strat = {c: param for c in range(NCR) if 0 < n[c] < param}
        Xs, ys = SMOTE(random_state=RS, sampling_strategy=strat).fit_resample(Xa, ya)
        m = make_xgb_cfg(cfg); m.fit(Xs, ys); return m
    if kind == "alpha":  m = make_xgb_cfg(cfg)
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    w = (n.sum() / np.maximum(n, 1)) ** param; w = w / w.mean()
    sw = w[np.asarray(ya)].astype(float); sw = sw * len(sw) / sw.sum()
    m.fit(Xa, ya, sample_weight=sw)
    return m


folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr))
CACHE_T = OUT / "cache_oof_v72_tuned.npz"
if CACHE_T.exists():
    zt = np.load(CACHE_T)
    OOF_T = [zt[f"m{m}"] for m in range(M)]
    say("loaded %s" % CACHE_T.name)
else:
    OOF_T = [np.zeros((len(y_arr), NCR), np.float32) for _ in range(M)]
    t0 = time.time()
    for f_, (tr, va) in enumerate(folds):
        tf = time.time()
        m_use = yr10[tr] >= 0
        Xa = X_trf.iloc[tr[m_use]].reset_index(drop=True)
        ya = pd.Series(yr10[tr[m_use]])
        for mi, (nm, (kind, param)) in enumerate(SPECS):
            mdl = fit_member_t(kind, param, Xa, ya, TUNED)
            OOF_T[mi][va] = mdl.predict_proba(X_trf.iloc[va]).astype(np.float32)
            del mdl; gc.collect()
        say("  fold %d/5 (%.0fs, elapsed %.0fm)" % (f_ + 1, time.time() - tf, (time.time() - t0) / 60))
        dump()
    np.savez_compressed(CACHE_T, **{f"m{m}": OOF_T[m] for m in range(M)})

# combiner: ref state on tuned OOF + W under e2e objective
ref = np.where(in_comb & (yr10 >= 0))[0]
y_ref = yr10[ref]
CORRECT = np.stack([(OOF_T[m][ref].argmax(1) == y_ref) for m in range(M)])
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = (n_ref.sum() / n_ref) ** 0.25; WC_REF = WC_REF / WC_REF.mean()
prof = lambda ps: np.hstack(ps).astype(np.float32)
NN_D = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
    prof([OOF_T[m][ref] for m in range(M)]))
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def fuse_static(probas, W):
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M): S += probas[m] * W[m][None, :]
    return S.argmax(1)


def fuse_dyn(probas, Wd, p=1.0, Wclass=None):
    Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(probas[0].shape, np.float64)
    for m in range(M):
        coef = Wp[:, m:m + 1] if Wclass is None else Wp[:, m:m + 1] * Wclass[m][None, :]
        S += coef * probas[m]
    return S.argmax(1)


def comp_ola(idx, k):
    nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
    Wd = np.empty((nb.shape[0], M))
    for m in range(M): Wd[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
    return Wd


gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]
P_GPOS = [OOF_T[m][gpos_comb] for m in range(M)]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG_comb = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG_comb[c, 0] = cnt


def e2e_of_W(W):
    yp = LAB10[fuse_static(P_GPOS, W)]
    cm = np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG_comb
    return macro_f1_from_cm(cm)


def fit_W_e2e(W0):
    W = W0.copy(); best_ = e2e_of_W(W)
    for _ in range(3):
        for c in range(NCR):
            for m_ in range(M):
                cur = W[m_, c]
                for gg in GRID_W:
                    W[m_, c] = gg
                    if W[:, c].sum() == 0: continue
                    s_ = e2e_of_W(W)
                    if s_ > best_ + 1e-7: best_, cur = s_, gg
                W[m_, c] = cur
    return W


t0 = time.time()
W_e2e = fit_W_e2e(np.ones((M, NCR)) / M)
say("tuned W_e2e fitted %.0fs (COMB e2e %.4f)" % (time.time() - t0, e2e_of_W(W_e2e)))


def score_e2e(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int); yp[gmask] = LAB10[fam_g]
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


P_selv = [OOF_T[m][gpos_selv] for m in range(M)]
_, i_selv = NN_D.kneighbors(prof(P_selv))
Wd30 = comp_ola(i_selv, 30)
I25 = MODELS.index("XGB a=0.25")
CAND = {"single tuned XGB a=0.25": P_selv[I25].argmax(1),
        "STATIC W_e2e (tuned pool)": fuse_static(P_selv, W_e2e),
        "HYBRID W_e2e x OLA-cw k=30 p=1": fuse_dyn(P_selv, Wd30, p=1, Wclass=W_e2e),
        "HYBRID W_e2e x OLA-cw k=30 p=3": fuse_dyn(P_selv, Wd30, p=3, Wclass=W_e2e)}
selv_tab = pd.DataFrame([score_e2e(fg, y_arr[selv_idx], gmask_selv, v) for v, fg in CAND.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV (tuned pool) ---"); say(selv_tab.to_string(index=False))
BESTV = selv_tab.iloc[0]["variant"]
say("SELV winner: %s" % BESTV)
del P_selv, i_selv, Wd30; gc.collect()
dump()

# refit both pools on train+val, test once
def fit_member_d(kind, param, Xa, ya):
    return fit_member_t(kind, param, Xa, ya, DEFAULT)


t0 = time.time()
gate_f = make_xgb_cfg(DEFAULT)
yb = (y_arr != 0).astype(int)
gate_f.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
m10 = yr10 >= 0
Xa10 = X_trf[m10].reset_index(drop=True)
ya10s = pd.Series(yr10[m10])
FINAL_T = [fit_member_t(k, p, Xa10, ya10s, TUNED) for _, (k, p) in SPECS]
FINAL_D = [fit_member_t(k, p, Xa10, ya10s, DEFAULT) for _, (k, p) in SPECS]
say("gate + tuned pool + default pool refit %.0fm" % ((time.time() - t0) / 60))
dump()

# default-arm reference (v7.1b same-run): needs default OOF + default W_e2e
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF_D = [zr[f"m{m}"] for m in range(M)]
CORRECT_D = np.stack([(OOF_D[m][ref].argmax(1) == y_ref) for m in range(M)])
NN_DD = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
    prof([OOF_D[m][ref] for m in range(M)]))
P_GPOS = [OOF_D[m][gpos_comb] for m in range(M)]     # e2e_of_W now scores the default pool
W_e2e_D = fit_W_e2e(np.ones((M, NCR)) / M)

p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
gmask_test = p_test > 0.5
gpos_t = np.where(gmask_test)[0]

P_t_T = [m.predict_proba(X_test.iloc[gpos_t]).astype(np.float32) for m in FINAL_T]
_, i_t = NN_D.kneighbors(prof(P_t_T))
Wd30_t = comp_ola(i_t, 30)
CAND_T = {"single tuned XGB a=0.25": P_t_T[I25].argmax(1),
          "STATIC W_e2e (tuned pool)": fuse_static(P_t_T, W_e2e),
          "HYBRID W_e2e x OLA-cw k=30 p=1": fuse_dyn(P_t_T, Wd30_t, p=1, Wclass=W_e2e),
          "HYBRID W_e2e x OLA-cw k=30 p=3": fuse_dyn(P_t_T, Wd30_t, p=3, Wclass=W_e2e)}

P_t_D = [m.predict_proba(X_test.iloc[gpos_t]).astype(np.float32) for m in FINAL_D]
_, i_tD = NN_DD.kneighbors(prof(P_t_D))
nbD = i_tD[:, :30]; wnD = WC_REF[y_ref[nbD]]; denD = wnD.sum(1)
WdD = np.empty((nbD.shape[0], M))
for m in range(M): WdD[:, m] = (CORRECT_D[m][nbD] * wnD).sum(1) / denD
fam_ref = fuse_dyn(P_t_D, WdD, p=3, Wclass=W_e2e_D)

test_tab = pd.DataFrame([score_e2e(fg, y_te, gmask_test, v) for v, fg in CAND_T.items()]
                        + [score_e2e(fam_ref, y_te, gmask_test, "v7.1b ref (default pool)")]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- TEST (selection made on SELV) ---"); say(test_tab.to_string(index=False))

yp_new = np.zeros(len(y_te), int); yp_new[gmask_test] = LAB10[CAND_T[BESTV]]
yp_ref = np.zeros(len(y_te), int); yp_ref[gmask_test] = LAB10[fam_ref]
mf_new = f1_score(y_te, yp_new, average="macro", zero_division=0)
mf_ref = f1_score(y_te, yp_ref, average="macro", zero_division=0)
say("\nSELV-selected '%s': test mF1 %.4f | v7.1b ref %.4f" % (BESTV, mf_new, mf_ref))

code_a = y_te * 10 + yp_ref
code_b = y_te * 10 + yp_new
rngb = np.random.RandomState(RS)
cls_i = [np.where(y_te == c)[0] for c in range(10)]
diffs = np.empty(2000)
for it in range(2000):
    bidx = np.concatenate([ci[rngb.randint(0, len(ci), len(ci))] for ci in cls_i])
    cma = np.bincount(code_a[bidx], minlength=100).reshape(10, 10)
    cmb = np.bincount(code_b[bidx], minlength=100).reshape(10, 10)
    diffs[it] = macro_f1_from_cm(cmb) - macro_f1_from_cm(cma)
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
say("paired delta (tuned - v7.1b): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (mf_new - mf_ref, lo, hi, min(1.0, p_two)))
say("\n" + classification_report(y_te, yp_new, labels=list(range(10)), target_names=NAMES,
                                 digits=3, zero_division=0))

res.update(stage2=dict(tuned=TUNED, selv_winner=BESTV,
                       test_tuned=round(float(mf_new), 4), test_ref=round(float(mf_ref), 4),
                       paired_delta=round(float(mf_new - mf_ref), 4),
                       paired_ci=[round(float(lo), 4), round(float(hi), 4)],
                       paired_p=round(float(min(1.0, p_two)), 4),
                       selv_table=selv_tab.to_dict("records"),
                       test_table=test_tab.to_dict("records")))
res["minutes"] = round((time.time() - t_start) / 60, 1)
(OUT / "v72_hparam.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % res["minutes"])
dump()
