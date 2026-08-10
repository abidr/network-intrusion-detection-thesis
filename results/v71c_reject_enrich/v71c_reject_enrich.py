"""
v7.1c: reject-set enrichment -- the last untested in-scope lever.

v7.1b trains the reject class on the 34,101 benign rows the OOF gate got wrong
(p > 0.5). Here the reject class instead trains on ALL near-boundary benign
rows (OOF gate p > 0.2), giving the members a denser picture of the same
benign/attack boundary. Everything else is held fixed:
  - inference unchanged (gate threshold stays 0.5);
  - the combiner reference set (kNN/CORRECT/W fitting rows) stays the v7.1b
    ref = COMB attacks + COMB p>0.5 rejects, so only member TRAINING changes;
  - W refit under the end-to-end objective (v7.1b protocol);
  - SELV selects among {STATIC, HYBRID p1, HYBRID p3}; test ONCE; paired
    bootstrap vs a same-run v7.1b default reference.

PRE-REGISTERED EXPECTATION: the boundary is proven inseparable; enrichment can
only improve its ESTIMATION. Gain 0 to +0.005 plausible; a null closes the
question. ASCII-only prints; log written UTF-8.
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
P_ENRICH = 0.2
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "v71c_reject_enrich.txt", "w", encoding="utf-8").write("\n".join(LOG))


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
oof_gate = np.load(ROOT / "cache_oof_v7_full.npz")["gate"]
p_gate = np.load(ROOT / "cache_oof_v71_gateproba.npz")["p"]
NCR = NC + 1; IREJ = NC
LAB10 = np.array(attack_labels + [0])

REJ05 = (y_arr == 0) & (oof_gate == 1)                       # v7.1b reject set
ENR = (y_arr == 0) & (p_gate > P_ENRICH)                     # enriched training set
yr_ref = np.where(REJ05, IREJ, fam_trf)                      # ref labels (unchanged)
yr_train = np.where(ENR, IREJ, fam_trf)                      # member training labels
say("reject train set: %d (v7.1b: %d) -- +%d near-boundary benign (p in (%.1f, 0.5])"
    % (ENR.sum(), REJ05.sum(), ENR.sum() - REJ05.sum(), P_ENRICH))
dump()

SPECS = ([(f"XGB a={a}", ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]


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
    w = (n.sum() / np.maximum(n, 1)) ** param; w = w / w.mean()
    sw = w[np.asarray(ya)].astype(float); sw = sw * len(sw) / sw.sum()
    m.fit(Xa, ya, sample_weight=sw)
    return m


folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr))
CACHE_E = OUT / "cache_oof_v71c_enriched.npz"
if CACHE_E.exists():
    ze = np.load(CACHE_E)
    OOF_E = [ze[f"m{m}"] for m in range(M)]
    say("loaded %s" % CACHE_E.name)
else:
    OOF_E = [np.zeros((len(y_arr), NCR), np.float32) for _ in range(M)]
    t0 = time.time()
    for f_, (tr, va) in enumerate(folds):
        tf = time.time()
        m_use = yr_train[tr] >= 0
        Xa = X_trf.iloc[tr[m_use]].reset_index(drop=True)
        ya = pd.Series(yr_train[tr[m_use]])
        for mi, (nm, (kind, param)) in enumerate(SPECS):
            mdl = fit_member(kind, param, Xa, ya)
            OOF_E[mi][va] = mdl.predict_proba(X_trf.iloc[va]).astype(np.float32)
            del mdl; gc.collect()
        say("  fold %d/5 (%.0fs, elapsed %.0fm, train rows %d)"
            % (f_ + 1, time.time() - tf, (time.time() - t0) / 60, len(ya)))
        dump()
    np.savez_compressed(CACHE_E, **{f"m{m}": OOF_E[m] for m in range(M)})

# ---------------------------------------------------------------- combiner machinery
ref = np.where(in_comb & (yr_ref >= 0))[0]                    # v7.1b ref rows, unchanged
y_ref = yr_ref[ref]
n_ref = np.bincount(y_ref, minlength=NCR)
WC_REF = (n_ref.sum() / n_ref) ** 0.25; WC_REF = WC_REF / WC_REF.mean()
prof = lambda ps: np.hstack(ps).astype(np.float32)
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


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


gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
yt_gpos = y_arr[gpos_comb]
neg_comb = comb_idx[p_gate[comb_idx] <= 0.5]
BG_comb = np.zeros((10, 10), np.int64)
for c, cnt in zip(*np.unique(y_arr[neg_comb], return_counts=True)):
    BG_comb[c, 0] = cnt


def build_side(OOF):
    st = {}
    st["CORRECT"] = np.stack([(OOF[m][ref].argmax(1) == y_ref) for m in range(M)])
    st["NN"] = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
        prof([OOF[m][ref] for m in range(M)]))
    P_G = [OOF[m][gpos_comb] for m in range(M)]

    def e2e_of_W(W):
        yp = LAB10[fuse_static(P_G, W)]
        cm = np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG_comb
        return macro_f1_from_cm(cm)

    W = np.ones((M, NCR)) / M
    best_ = e2e_of_W(W)
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
    st["W"] = W; st["comb_e2e"] = best_
    return st


def comp_ola(st, idx, k):
    nb = idx[:, :k]; wn = WC_REF[y_ref[nb]]; den = wn.sum(1)
    Wd = np.empty((nb.shape[0], M))
    for m in range(M): Wd[:, m] = (st["CORRECT"][m][nb] * wn).sum(1) / den
    return Wd


t0 = time.time()
SIDE_E = build_side(OOF_E)
say("enriched W_e2e fitted %.0fs (COMB e2e %.4f)" % (time.time() - t0, SIDE_E["comb_e2e"]))
dump()


def score_e2e(fam_g, ye, gmask, tag):
    yp = np.zeros(len(ye), int); yp[gmask] = LAB10[fam_g]
    return dict(variant=tag,
                mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                acc=round(float((yp == ye).mean()), 4))


gmask_selv = p_gate[selv_idx] > 0.5
gpos_selv = selv_idx[gmask_selv]
P_selv = [OOF_E[m][gpos_selv] for m in range(M)]
_, i_selv = SIDE_E["NN"].kneighbors(prof(P_selv))
Wd30 = comp_ola(SIDE_E, i_selv, 30)
CAND = {"STATIC W_e2e (enriched)": fuse_static(P_selv, SIDE_E["W"]),
        "HYBRID W_e2e x OLA-cw k=30 p=1": fuse_dyn(P_selv, Wd30, p=1, Wclass=SIDE_E["W"]),
        "HYBRID W_e2e x OLA-cw k=30 p=3": fuse_dyn(P_selv, Wd30, p=3, Wclass=SIDE_E["W"])}
selv_tab = pd.DataFrame([score_e2e(fg, y_arr[selv_idx], gmask_selv, v) for v, fg in CAND.items()]
                        ).sort_values("mF1", ascending=False).reset_index(drop=True)
say("\n--- SELV (enriched) ---"); say(selv_tab.to_string(index=False))
BESTV = selv_tab.iloc[0]["variant"]
say("SELV winner: %s" % BESTV)
del P_selv, i_selv, Wd30; gc.collect()
dump()

# ---------------------------------------------------------------- refit, test once
t0 = time.time()
gate_f = make_xgb()
yb = (y_arr != 0).astype(int)
gate_f.fit(X_trf, yb, sample_weight=compute_sample_weight("balanced", yb))
mE = yr_train >= 0
FINAL_E = [fit_member(k, p, X_trf[mE].reset_index(drop=True), pd.Series(yr_train[mE]))
           for _, (k, p) in SPECS]
mD = yr_ref >= 0
FINAL_D = [fit_member(k, p, X_trf[mD].reset_index(drop=True), pd.Series(yr_ref[mD]))
           for _, (k, p) in SPECS]
say("gate + enriched pool + default pool refit %.0fm" % ((time.time() - t0) / 60))
dump()

# default-arm reference state (v7.1b, same-run)
zr = np.load(ROOT / "cache_oof_v71_reject.npz")
OOF_D = [zr[f"m{m}"] for m in range(M)]
SIDE_D = build_side(OOF_D)
say("default W_e2e (COMB e2e %.4f)" % SIDE_D["comb_e2e"])

p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
gmask_test = p_test > 0.5
gpos_t = np.where(gmask_test)[0]

P_t_E = [m.predict_proba(X_test.iloc[gpos_t]).astype(np.float32) for m in FINAL_E]
_, i_tE = SIDE_E["NN"].kneighbors(prof(P_t_E))
Wd30_E = comp_ola(SIDE_E, i_tE, 30)
CAND_T = {"STATIC W_e2e (enriched)": fuse_static(P_t_E, SIDE_E["W"]),
          "HYBRID W_e2e x OLA-cw k=30 p=1": fuse_dyn(P_t_E, Wd30_E, p=1, Wclass=SIDE_E["W"]),
          "HYBRID W_e2e x OLA-cw k=30 p=3": fuse_dyn(P_t_E, Wd30_E, p=3, Wclass=SIDE_E["W"])}

P_t_D = [m.predict_proba(X_test.iloc[gpos_t]).astype(np.float32) for m in FINAL_D]
_, i_tD = SIDE_D["NN"].kneighbors(prof(P_t_D))
Wd30_D = comp_ola(SIDE_D, i_tD, 30)
fam_ref = fuse_dyn(P_t_D, Wd30_D, p=3, Wclass=SIDE_D["W"])

test_tab = pd.DataFrame([score_e2e(fg, y_te, gmask_test, v) for v, fg in CAND_T.items()]
                        + [score_e2e(fam_ref, y_te, gmask_test, "v7.1b ref (default reject set)")]
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
say("paired delta (enriched - v7.1b): %+.4f  95%% CI [%+.4f, %+.4f]  p ~ %.3f"
    % (mf_new - mf_ref, lo, hi, min(1.0, p_two)))

fa = f1_score(y_te, yp_ref, average=None, labels=range(10), zero_division=0)
fb_ = f1_score(y_te, yp_new, average=None, labels=range(10), zero_division=0)
say("\nper-class F1 (default -> enriched reject set):")
for i, n in enumerate(NAMES):
    say("  %-16s %.4f -> %.4f  (%+.4f)" % (n, fa[i], fb_[i], fb_[i] - fa[i]))
sw_a = int((gmask_test & (yp_ref == 0) & (y_te != 0)).sum())
sw_b = int((gmask_test & (yp_new == 0) & (y_te != 0)).sum())
fp_a = int(((y_te == 0) & (yp_ref != 0)).sum())
fp_b = int(((y_te == 0) & (yp_new != 0)).sum())
say("\nbenign-as-attack: %d -> %d | attacks swallowed by reject: %d -> %d"
    % (fp_a, fp_b, sw_a, sw_b))
say("\n" + classification_report(y_te, yp_new, labels=list(range(10)), target_names=NAMES,
                                 digits=3, zero_division=0))

res = dict(experiment="reject-set enrichment (train reject on OOF gate p>%.1f)" % P_ENRICH,
           n_reject_train=int(ENR.sum()), n_reject_default=int(REJ05.sum()),
           selv_winner=BESTV,
           test=dict(enriched=round(float(mf_new), 4), ref=round(float(mf_ref), 4)),
           paired_delta=round(float(mf_new - mf_ref), 4),
           paired_ci=[round(float(lo), 4), round(float(hi), 4)],
           paired_p=round(float(min(1.0, p_two)), 4),
           benign_as_attack=[fp_a, fp_b], attacks_swallowed=[sw_a, sw_b],
           per_class_f1={NAMES[i]: round(float(fb_[i]), 4) for i in range(10)},
           selv_table=selv_tab.to_dict("records"), test_table=test_tab.to_dict("records"),
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "v71c_reject_enrich.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
