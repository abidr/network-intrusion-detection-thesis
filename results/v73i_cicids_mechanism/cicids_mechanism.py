"""
v7.3i: is brittleness caused by CLASS OVERLAP? A pre-registered mechanism test on CIC-IDS2017.

The v7.3 series explained the model's brittleness by Cover-Hart: rows sit near decision
boundaries, so a tiny perturbation flips them. That explanation is not a story -- it is a
FALSIFIABLE PREDICTION ACROSS DATASETS, because 1-NN error is measured independently of any
attack:

    CIC-UNSW (full release)   e_1NN 0.2084   -> brittle    (MEASURED: 0.13 macro-F1 lost
                                                            at 0.002 IQR of random noise)
    CIC-IDS2017              e_1NN 0.0026   -> should be DRAMATICALLY more robust

If CIC-IDS2017 resists the same budget, brittleness is a property of CLASS OVERLAP, and the
robustness chapter welds onto this project's plateau diagnosis -- two findings become one.
If CIC-IDS2017 is equally brittle, the Cover-Hart explanation is WRONG and brittleness is
architectural. Either outcome is decisive, and either way this run also removes the
single-dataset objection from the robustness claim.

Same budget machinery as v7.3-v7.3h: attacker sets PRIMITIVES only (FREE / COSTLY at half /
IMPOSSIBLE frozen), every derivable column recomputed so the record stays physically
consistent, budgets in train-IQR units, integer columns rounded, only test ATTACK rows move,
gate matched to the same benign fpr before any comparison.

DEVIATIONS from the CIC-UNSW runs, stated not silent -- forced by the dataset, not chosen:
  D1  NO PROTOCOL COLUMN in CIC-IDS2017. The IMPOSSIBLE tier is `Destination Port` plus the
      service one-hots derived from it. CIC-UNSW had proto_tcp/proto_udp as well.
  D2  NO REJECT CLASS. It requires cached OOF gate predictions, which exist for CIC-UNSW and
      not here. Stage B is a plain family classifier. The comparison quantity is the RETAINED
      FRACTION, which is internally normalised, so this affects levels far more than ratios.
  D3  Web Attack variants collapsed to one class (per the audit) -> 13 classes; inf/NaN rows
      dropped (0.1%); read latin-1 (the Web Attack labels carry a non-ASCII en-dash).
  D4  Stage B is a single XGB a=0.25. On THIS dataset the v6 run found all 17 combiner
      variants within 0.0007, so the ensemble is provably irrelevant here.
  D5  e_1NN is re-measured IN THIS RUN rather than quoted from results/cicids2017_v6/, so the
      predictor and the outcome come from the same pipeline.

PRE-REGISTERED PREDICTION AND DECISION RULE PRINTED BEFORE ANY RESULT (see PREREG).
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

D = Path(r"E:\AIUB\THESIS\Resource\CIC-IDS2017")
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
ATK_BUDGETS = (0.002, 0.02, 0.20)
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
PASSES = 2
TARGET_FPR = 0.0168
KNN_N = 60_000
LOG = []

# CIC-UNSW reference, v7.3h arm A (same budget machinery, same matched fpr, same protocol)
REF = dict(name="CIC-UNSW full release", e_1nn=0.2084,
           clean_mf1=0.6718, clean_gate=0.9862,
           rnd={0.002: (0.5798, 0.9426), 0.02: (0.5059, 0.9081), 0.20: (0.2723, 0.4871)},
           evade={0.002: (0.5352, 0.7508), 0.02: (0.3584, 0.4695), 0.20: (0.1440, 0.0309)})


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "cicids_mechanism.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTION -- written before the run, printed before any result
==============================================================================
MECHANISM UNDER TEST: the v7.3 brittleness is caused by CLASS OVERLAP (rows sitting near
decision boundaries), not by the architecture. 1-NN error e_1NN measures that overlap
WITHOUT reference to any attack, so it is a clean independent variable.

  CIC-UNSW full release   e_1NN 0.2084   retained gate recall under white-box EVASION
                                         at budget 0.20 = 0.0313  (v7.3h arm A)
  CIC-IDS2017             e_1NN 0.0026   80x lower overlap

H1 (overlap causes brittleness): CIC-IDS2017 is dramatically more robust.
     H1a  retained gate recall under EVASION at budget 0.20 > 0.50   (vs 0.0313)
     H1b  retained macro-F1 under RANDOM noise at budget 0.20 > 0.80 (vs 0.4053)
     Both must hold. These are deliberately coarse thresholds: an 80x difference in the
     predictor should not need a delicate test to show up.

H0 (architectural): CIC-IDS2017 is comparably brittle -- retained values land near
     CIC-UNSW's. Then the Cover-Hart explanation of brittleness is REFUTED and must be
     withdrawn from the robustness chapter, which currently leans on it.

DECISION RULE, committed in advance:
  * H1 holds -> brittleness is a property of the REPRESENTATION's class overlap. The
                robustness chapter and the plateau diagnosis become ONE finding, and the
                practical implication is that robustness is bought by better features, not
                by better models -- consistent with every other result in this project.
  * H0 holds -> withdraw the Cover-Hart mechanism from the chapter and report brittleness
                as unexplained. Do NOT re-slice the arms to rescue it.
  * SPLIT (H1a xor H1b) -> report as partial: name which layer tracks overlap and which
                does not. Do not average them into a single verdict.

CONFOUND ACKNOWLEDGED IN ADVANCE, and it cuts against a clean reading either way:
CIC-IDS2017 is not merely lower-overlap, it is also a different testbed with a different
class mix and no protocol column (D1). A large effect in the predicted direction is
therefore SUGGESTIVE of the mechanism, not proof of it -- one confounded comparison cannot
isolate overlap from every other difference between two capture campaigns. Say that plainly
in the write-up; do not upgrade it to a controlled experiment.

TRIVIALITY GUARD: if the CIC-IDS2017 gate is already so strong that clean gate recall is
~1.0 AND the attack cannot move it at all even at the largest budget, check that the attack
is actually running (the `moved` fraction) before declaring robustness. A stalled search is
not a robust model -- the same trap v7.3e pre-registered and cleared.
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

FILES = ["Monday-WorkingHours.pcap_ISCX.csv",
         "Tuesday-WorkingHours.pcap_ISCX.csv",
         "Wednesday-workingHours.pcap_ISCX.csv",
         "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         "Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

fr = []
for f in FILES:
    d_ = pd.read_csv(D / f, low_memory=False, encoding="latin-1")
    d_.columns = [c.strip() for c in d_.columns]
    fr.append(d_)
df = pd.concat(fr, ignore_index=True); del fr; gc.collect()
df["Label"] = df["Label"].astype(str).str.strip()
df.loc[df["Label"].str.startswith("Web Attack"), "Label"] = "Web Attack"
FEAT = [c for c in df.columns if c != "Label"]
num = df[FEAT].to_numpy(dtype=np.float64)
bad = ~np.isfinite(num).all(axis=1)
df = df.loc[~bad].reset_index(drop=True); del num; gc.collect()
n0 = len(df); df = df.drop_duplicates().reset_index(drop=True)
say("\nloaded %d rows after inf/NaN drop and dedupe (%.2f%% dupes)" % (len(df), 100 * (1 - len(df) / n0)))

atk = sorted([l for l in df["Label"].unique() if l != "BENIGN"], key=str.lower)
CATM = {"BENIGN": 0}
for i, a in enumerate(atk): CATM[a] = i + 1
NAMES = ["BENIGN"] + atk; NCLS = len(NAMES)
y = df["Label"].map(CATM).astype(int).values
say("%d classes | attack share %.4f" % (NCLS, float((y != 0).mean())))

# ---------------------------------------------------------------- feature space
PORT_COL = "Destination Port"
Xdf = df[[c for c in FEAT if c != PORT_COL]]
nun = Xdf.nunique(); zero = nun[nun <= 1].index.tolist()
FULL = Xdf.drop(columns=zero)
corr = FULL.corr(numeric_only=True)
upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
BASE_COLS = [c for c in FULL.columns if c not in red]
say("perturbable space: %d raw cols -> BASE %d (zero-var %d, |r|>=0.95 %d)"
    % (FULL.shape[1], len(BASE_COLS), len(zero), len(red)))

REN = {"Total Fwd Packets": "Total Fwd Packet",
       "Total Backward Packets": "Total Bwd packets",
       "Total Length of Fwd Packets": "Total Length of Fwd Packet",
       "Total Length of Bwd Packets": "Total Length of Bwd Packet",
       "Init_Win_bytes_forward": "FWD Init Win Bytes",
       "Init_Win_bytes_backward": "Bwd Init Win Bytes"}


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def derive(F):
    F = F.rename(columns=REN)
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
    fpk = F["Total Fwd Packets"] if has("Total Fwd Packets") else None
    bpk = F["Total Backward Packets"] if has("Total Backward Packets") else None
    fby = F["Total Length of Fwd Packets"] if has("Total Length of Fwd Packets") else None
    bby = F["Total Length of Bwd Packets"] if has("Total Length of Bwd Packets") else None
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
        if fby is not None and has("Avg Fwd Segment Size"): F["Avg Fwd Segment Size"] = safe(fby, fpk)
        if bby is not None and has("Avg Bwd Segment Size"): F["Avg Bwd Segment Size"] = safe(bby, bpk)
    n_ = F.select_dtypes(include=[np.number]).columns
    F[n_] = F[n_].clip(lower=0.0)
    return F


DER_ALL = derive(FULL)
dz = DER_ALL.nunique(); DER_COLS = [c for c in DER_ALL.columns if c not in dz[dz <= 1].index]

# IMPOSSIBLE tier: destination port + service one-hots built from it (D1: no protocol column)
port = df[PORT_COL].to_numpy()
SVC_PORTS = [(20, "svc_ftpdata"), (21, "svc_ftp"), (22, "svc_ssh"), (23, "svc_telnet"),
             (25, "svc_smtp"), (53, "svc_dns"), (80, "svc_http"), (110, "svc_pop3"),
             (111, "svc_rpc"), (139, "svc_netbios"), (143, "svc_imap"), (179, "svc_bgp"),
             (443, "svc_https"), (445, "svc_smb"), (514, "svc_syslog"), (520, "svc_rip"),
             (1723, "svc_pptp"), (5190, "svc_aim"), (6881, "svc_bt"), (8080, "svc_httpalt")]
KNOWN = np.array([p for p, _ in SVC_PORTS])
sv = {nm: (port == p).astype(np.float32) for p, nm in SVC_PORTS}
sv["svc_other_low"] = ((port < 1024) & ~np.isin(port, KNOWN)).astype(np.float32)
sv["svc_registered"] = ((port >= 1024) & (port < 49152) & ~np.isin(port, KNOWN)).astype(np.float32)
sv["svc_ephemeral"] = (port >= 49152).astype(np.float32)
sv["dst_port_num"] = port.astype(np.float32)
SVC = pd.DataFrame(sv, index=df.index)
szz = SVC.nunique(); SVC = SVC.drop(columns=szz[szz <= 1].index.tolist())
SVC_COLS = list(SVC.columns)
FEAT_ORDER = BASE_COLS + DER_COLS + SVC_COLS
assert not (set(SVC_COLS) & set(BASE_COLS)) and not (set(SVC_COLS) & set(DER_COLS))


def assemble(F, S):
    return pd.concat([F[BASE_COLS].astype(np.float32), derive(F)[DER_COLS], S], axis=1)[FEAT_ORDER]


COSTLY_PREFIX = ("Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
                 "Total Length of Bwd Packets", "FIN Flag", "SYN Flag", "RST Flag", "PSH Flag",
                 "ACK Flag", "URG Flag", "CWE Flag", "ECE Flag", "Fwd PSH", "Bwd PSH",
                 "Fwd URG", "Bwd URG")
FREE_COLS = [c for c in FULL.columns if not c.startswith(COSTLY_PREFIX)]
COSTLY_COLS = [c for c in FULL.columns if c.startswith(COSTLY_PREFIX)]
assert len(FREE_COLS) + len(COSTLY_COLS) == FULL.shape[1]
say("tiers: FREE %d | COSTLY %d | IMPOSSIBLE %d (dst port + svc one-hots)"
    % (len(FREE_COLS), len(COSTLY_COLS), len(SVC_COLS)))

X = assemble(FULL, SVC)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smpl = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns
            if np.all(np.abs(smpl[c].to_numpy() - np.round(smpl[c].to_numpy())) < 1e-6)]
del X, corr, upper, DER_ALL; gc.collect()
say("train %d | test %d | %d integer cols | %d features"
    % (len(y_arr), len(y_te), len(INT_COLS), len(FEAT_ORDER)))
dump()

# ---------------------------------------------------------------- D5: e_1NN in this run
att_tr = np.where(y_arr != 0)[0]; att_te_i = np.where(y_te != 0)[0]
rng = np.random.RandomState(RS)
ref_i = rng.choice(att_tr, size=min(KNN_N, len(att_tr)), replace=False)
qry_i = rng.choice(att_te_i, size=min(KNN_N, len(att_te_i)), replace=False)
sc = RobustScaler().fit(X_trf.iloc[ref_i])
nn = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(sc.transform(X_trf.iloc[ref_i]))
X_qry = assemble(FULL_te.iloc[qry_i].reset_index(drop=True),
                 SVC_te.iloc[qry_i].reset_index(drop=True))
_, ii = nn.kneighbors(sc.transform(X_qry))
E1NN = float((y_arr[ref_i][ii[:, 0]] != y_te[qry_i]).mean())
say("\nD5: e_1NN measured IN THIS RUN = %.4f (ref %d, query %d)  [v6 run reported 0.0026]"
    % (E1NN, len(ref_i), len(qry_i)))
say("    CIC-UNSW reference e_1NN = %.4f -> predictor ratio %.0fx"
    % (REF["e_1nn"], REF["e_1nn"] / max(E1NN, 1e-9)))
del sc, nn; gc.collect(); dump()


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


fam_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(fam_labels)}
LABF = np.array(fam_labels)
t0 = time.time()
ytr_fam = np.array([fam_to_idx[v] for v in y_arr[att_tr]])
nb = np.bincount(ytr_fam, minlength=len(fam_labels))
wf = ((nb.sum() / np.maximum(nb, 1)) ** 0.25); wf = wf / wf.mean()
swf = wf[ytr_fam].astype(float); swf = swf * len(swf) / swf.sum()
SB = make_xgb().fit(X_trf.iloc[att_tr].reset_index(drop=True), ytr_fam, sample_weight=swf)
say("Stage B (%d families) trained %.1fm" % (len(fam_labels), (time.time() - t0) / 60))

sub, _ = train_test_split(np.arange(len(y_arr)), train_size=min(GATE_TRAIN, len(y_arr)),
                          random_state=RS, stratify=(y_arr != 0).astype(int))
yg = (y_arr[sub] != 0).astype(int)
GATE = make_xgb().fit(X_trf.iloc[sub].reset_index(drop=True), yg,
                      sample_weight=compute_sample_weight("balanced", yg))
say("gate trained on %d rows" % len(sub))
del X_trf; gc.collect()

i_att = np.where(y_te != 0)[0]; i_ben = np.where(y_te == 0)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
y_att = y_te[i_att]
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))
pb = GATE.predict_proba(X_ben)[:, 1]
THR = float(np.quantile(pb, 1.0 - TARGET_FPR))
gmb = pb > THR
YP_BEN = np.zeros(len(i_ben), int)
if gmb.any(): YP_BEN[gmb] = LABF[SB.predict_proba(X_ben[gmb]).argmax(1)]
say("test attack %d | benign %d | gate thr %.4f -> benign fpr %.4f"
    % (len(i_att), len(i_ben), THR, float(gmb.mean())))
dump()


def caps_of(b):
    c = {k: b * IQR[k] for k in FREE_COLS}
    c.update({k: b * COSTLY_RATIO * IQR[k] for k in COSTLY_COLS})
    return c


def build(off, caps):
    F_ = F_att0.copy()
    for c in caps: F_[c] = F_att0[c].to_numpy() + off[c]
    for c in INT_COLS: F_[c] = np.round(F_[c].to_numpy())
    F_ = repair(F_)
    for c in INT_COLS: F_[c] = np.round(F_[c].to_numpy())
    return F_


def craft(b, mode):
    caps = caps_of(b)
    off = {c: np.zeros(len(F_att0), np.float32) for c in caps}
    if mode == "random":
        r = np.random.RandomState(RS)
        for c in caps: off[c] = r.uniform(-caps[c], caps[c], len(F_att0)).astype(np.float32)
        return build(off, caps), float("nan")
    fn = lambda F_: GATE.predict_proba(assemble(F_, S_att0))[:, 1].astype(np.float32)
    p0 = fn(build(off, caps)); cur = p0.copy()
    for _ in range(PASSES):
        for c in caps:
            best_s, best_o = cur, off[c]
            for ds in (+1.0, -1.0):
                cand = np.clip(off[c] + ds * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                s_ = fn(build(o2, caps))
                take = s_ < best_s
                if take.any():
                    best_s = np.where(take, s_, best_s); best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_s
    return build(off, caps), float((cur < p0 - 1e-9).mean())


def evaluate(F_p):
    Xp = assemble(F_p, S_att0)
    gma = GATE.predict_proba(Xp)[:, 1] > THR
    yp_a = np.zeros(len(Xp), int)
    if gma.any(): yp_a[gma] = LABF[SB.predict_proba(Xp[gma]).argmax(1)]
    und = yp_a == 0; cor = yp_a == y_att; mis = (~und) & (~cor)
    yp = np.zeros(len(y_te), int); yp[i_ben] = YP_BEN; yp[i_att] = yp_a
    return dict(undetected=float(und.mean()), misattributed=float(mis.mean()),
                correct=float(cor.mean()), gate_recall=float(gma.mean()),
                macro_f1=float(f1_score(y_te, yp, average="macro", zero_division=0)),
                accuracy=float((yp == y_te).mean()))


CLEAN = evaluate(F_att0)
say("\nCLEAN: macro-F1 %.4f | accuracy %.4f | gate recall %.4f"
    % (CLEAN["macro_f1"], CLEAN["accuracy"], CLEAN["gate_recall"]))

rows = []
for mode, grid in (("RANDOM", BUDGETS), ("EVASION", (0.0,) + ATK_BUDGETS)):
    say("\n=== %s ===" % mode)
    say("  %8s %10s %12s %13s %9s %8s" % ("budget", "macro-F1", "gate-recall",
                                          "misattributed", "undet", "moved"))
    for b in grid:
        if b == 0.0:
            r, mv = dict(CLEAN), 0.0
        else:
            Fp, mv = craft(b, "random" if mode == "RANDOM" else "evade")
            r = evaluate(Fp)
        r.update(mode=mode, budget=b, moved=(None if np.isnan(mv) else round(mv, 4)),
                 ret_mf1=round(r["macro_f1"] / CLEAN["macro_f1"], 4),
                 ret_gate=round(r["gate_recall"] / CLEAN["gate_recall"], 4))
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        say("  %8.3f %10.4f %12.4f %13.4f %9.4f %8s"
            % (b, r["macro_f1"], r["gate_recall"], r["misattributed"], r["undetected"],
               "-" if np.isnan(mv) else "%.3f" % mv))
        dump()

R = pd.DataFrame(rows)
gv = lambda m, b, c: float(R[(R["mode"] == m) & (R["budget"] == b)][c].iloc[0])

say("\n" + "=" * 78); say("SIDE BY SIDE -- retained fraction of each dataset's own clean value")
say("=" * 78)
say("  %-10s %8s | %-22s | %-22s" % ("", "", "CIC-IDS2017 (e_1NN %.4f)" % E1NN,
                                     "CIC-UNSW (e_1NN %.4f)" % REF["e_1nn"]))
say("  %-10s %8s | %10s %11s | %10s %11s"
    % ("mode", "budget", "ret mF1", "ret gate", "ret mF1", "ret gate"))
for m, bs in (("RANDOM", (0.002, 0.02, 0.20)), ("EVASION", (0.002, 0.02, 0.20))):
    key = "rnd" if m == "RANDOM" else "evade"
    for b in bs:
        rm, rg = REF[key][b]
        say("  %-10s %8.3f | %10.4f %11.4f | %10.4f %11.4f"
            % (m, b, gv(m, b, "ret_mf1"), gv(m, b, "ret_gate"),
               rm / REF["clean_mf1"], rg / REF["clean_gate"]))

h1a = gv("EVASION", 0.20, "ret_gate") > 0.50
h1b = gv("RANDOM", 0.20, "ret_mf1") > 0.80
moved20 = gv("EVASION", 0.20, "moved")
say("\n" + "=" * 78); say("PRE-REGISTERED TEST"); say("=" * 78)
say("  H1a retained gate recall, EVASION @0.20 : %.4f  (>0.50, CIC-UNSW 0.0313)   %s"
    % (gv("EVASION", 0.20, "ret_gate"), "PASS" if h1a else "FAIL"))
say("  H1b retained macro-F1,    RANDOM  @0.20 : %.4f  (>0.80, CIC-UNSW 0.4053)   %s"
    % (gv("RANDOM", 0.20, "ret_mf1"), "PASS" if h1b else "FAIL"))
say("  TRIVIALITY GUARD -- attack `moved` fraction @0.20 = %.4f %s"
    % (moved20, "(search ran; robustness is real)" if moved20 > 0.5
       else "*** LOW -- search may have stalled; do NOT read this as robustness ***"))
if h1a and h1b:
    verdict = ("H1 SUPPORTED -- brittleness tracks class overlap. Robustness chapter and "
               "plateau diagnosis are ONE finding: robustness comes from better features, "
               "not better models.")
elif h1a or h1b:
    verdict = ("SPLIT -- one layer tracks overlap and the other does not. Report both, "
               "name which is which, do NOT average into a single verdict.")
else:
    verdict = ("H0 -- CIC-IDS2017 is comparably brittle. WITHDRAW the Cover-Hart mechanism "
               "from the robustness chapter and report brittleness as unexplained.")
say("\n  VERDICT: %s" % verdict)
say("\n  CONFOUND, as pre-registered: CIC-IDS2017 differs from CIC-UNSW in more than overlap")
say("  (different testbed, class mix, no protocol column). A large effect in the predicted")
say("  direction is SUGGESTIVE of the mechanism, not proof. Do not upgrade it to a")
say("  controlled experiment in the write-up.")

res = dict(experiment="v7.3i -- does brittleness track class overlap? CIC-IDS2017 mechanism test",
           prereg=dict(h1a_evasion_gate=bool(h1a), h1b_random_mf1=bool(h1b),
                       moved_at_020=(None if np.isnan(moved20) else round(moved20, 4)),
                       verdict=verdict),
           e_1nn_measured=round(E1NN, 4), e_1nn_reference_cic_unsw=REF["e_1nn"],
           clean={k: round(v, 4) for k, v in CLEAN.items()},
           deviations=["D1 no protocol column", "D2 no reject class", "D3 web-attack collapse "
                       "+ inf/NaN drop + latin-1", "D4 single-model Stage B",
                       "D5 e_1NN measured in-run"],
           n_classes=NCLS, features=len(FEAT_ORDER), threshold=THR, target_fpr=TARGET_FPR,
           budgets=list(BUDGETS), results=rows,
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "cicids_mechanism.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
