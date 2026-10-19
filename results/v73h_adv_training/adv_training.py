"""
v7.3h: controllability-aware ADVERSARIAL TRAINING -- the correction to v7.3g.

v7.3g applied controllability knowledge by RESTRICTION (delete what the attacker controls)
and it failed badly: clean macro-F1 -0.3354, and total collapse the moment the attacker
retargets the service. The lesson was that restriction CONCENTRATES risk.

Tafreshian & Zhang (IEEE AI+ TrustCom 2024, "A Defensive Framework Against Adversarial
Attacks on ML-Based NIDS") apply the same knowledge the OPPOSITE way: keep every feature and
AUGMENT training with perturbed examples, so the model learns that variation along
attacker-controllable directions is not evidence. Their framework is adversarial training +
SMOTE + feature engineering + ensembles, on NSL-KDD and UNSW-NB15. This project has already
tested balancing, feature engineering and ensembles; ADVERSARIAL TRAINING is the one
ingredient never tried here.

The insight that SURVIVED v7.3g's failure was stage-asymmetry -- what broke was the feature
mechanism, not the idea that the two stages face different threats. So this run applies
asymmetric TRAINING rather than asymmetric FEATURES:

    gate     <- trained on EVASION-adversarial examples      (v7.3c: the gate is the lever)
    Stage B  <- trained on MISATTRIBUTION-adversarial examples (v7.3f: that is what attacks it)

Four arms:

    A  baseline            no augmentation
    B  noise-augmented     random perturbation within the SAME budget      <- the cheap control
    C  adv gate            evasion-adversarial gate + baseline Stage B
    D  adv both            C + misattribution-adversarial Stage B

ARM B IS THE POINT OF THE EXPERIMENT, not a formality. v7.3 measured undirected random noise
doing about HALF the damage of a directed attack, so if plain noise augmentation captures most
of the benefit then the expensive adversarial machinery is unnecessary -- a positive practical
finding, and the question the source paper cannot ask because it has no noise control.

PRE-REGISTERED PREDICTIONS AND DECISION RULE PRINTED BEFORE ANY RESULT (see PREREG).

Same prep, splits, budget machinery, matched-fpr protocol and reject-class definition as
v7.3c/e/f/g. Stage B is a single XGB a=0.25 in every arm (v7: the 7-member ensemble is worth
~0.0038 at full scale), so training augmentation is the only moving part.
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
OUT = Path(__file__).parent
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
ATK_BUDGETS = (0.002, 0.02, 0.20)          # greedy attack grid at eval (cost control)
AUG_BUDGETS = (0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
AUG_ROWS = 30_000                          # train attack rows used for augmentation
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
PASSES = 2
TARGET_FPR = 0.0168
GATE_PENALTY = 10.0
NBOOT = 2000
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "adv_training.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTIONS -- written before the run, printed before any result
==============================================================================
CONTEXT. v7.3g refuted controllability-awareness applied as feature RESTRICTION. This run
tests the opposite application -- AUGMENTATION -- which is what the closest paper in the
literature (Tafreshian & Zhang, TrustCom 2024) actually does.

P1 THE DECISIVE TEST -- IS "ADVERSARIAL" EVEN NECESSARY?
    Random noise does about half the damage of a directed attack (v7.3), so noise
    augmentation may capture most of the benefit at a fraction of the cost.
      P1 holds if adversarial augmentation beats noise augmentation:
         retained gate recall at budget 0.20 under EVASION, arm C - arm B > +0.05.
      If P1 FAILS the finding is POSITIVE AND PRACTICAL, not a null: "noise augmentation is
      sufficient; the adversarial machinery is unnecessary." Report it that way. This is the
      question the source paper cannot ask, because it has no noise control arm.

P2 CLEAN COST. Adversarial training classically costs clean accuracy.
      P2 holds if |arm C clean macro-F1 - arm A clean macro-F1| < 0.02, judged by a PAIRED
      bootstrap (2000 resamples) rather than a point estimate, because v7.3d showed the
      paired interval for arms sharing machinery can be ~5x tighter than the marginal one.
      If P2 fails, augmentation is a trade-off, not a free defence -- say so.

P3 STAGE-ASYMMETRY, the surviving half of v7.3g's idea.
      Misattribution-adversarial training of Stage B should defend the LABEL specifically:
      arm D beats arm C on misattributed-rate at budget 0.20 by > 0.05, while neither
      differs much on evasion. If augmenting Stage B helps evasion equally, the
      stage-specific story is wrong and should be dropped.

EXPECTATION SET IN ADVANCE, so it cannot be claimed as a prediction afterwards: this project
has produced twelve nulls and three refuted pre-registrations. A null on P1 is the single
most likely outcome and is the one this design is built to detect cleanly.

MEASUREMENT RULES CARRIED FORWARD, all learned the hard way:
  * every gate is thresholded to the SAME benign fpr before any comparison (v7.3b -- at the
    default 0.5 cut the answer inverted completely)
  * `misattributed` is upper-bounded by gate recall (v7.3g), so gate recall is reported
    beside it in every table and cross-arm attribution claims are read with that in mind
  * attacks are white-box on ONE component while the deployed Stage B is a single model
    here; v7.3e measured surrogate inflation at +0.6773, so absolute attack strengths are
    understated -- which is conservative for a DEFENCE claim and must be stated with it
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

# ---------------------------------------------------------------- prep (v7.3c/e/f/g-identical)
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
SVC_COLS = list(SVC.columns)
FEAT_ORDER = BASE_COLS + DER_COLS + SVC_COLS


def assemble(F, S):
    return pd.concat([F[BASE_COLS].astype(np.float32), derive(F)[DER_COLS], S], axis=1)[FEAT_ORDER]


X = assemble(FULL, SVC)
idx_all = np.arange(len(y))
i_trf, i_test = train_test_split(idx_all, test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True)
y_arr = y[i_trf]; y_te = y[i_test]
FULL_trf = FULL.iloc[i_trf].reset_index(drop=True); SVC_trf = SVC.iloc[i_trf].reset_index(drop=True)
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
del X, corr, upper, DER_ALL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("\nprep done %.0fm | %d int cols" % ((time.time() - t_start) / 60, len(INT_COLS)))

FREE_PREFIX = ("Flow Duration", "Flow IAT", "Fwd IAT", "Bwd IAT", "Active", "Idle",
               "Fwd Packet Length", "Bwd Packet Length", "Packet Length", "Fwd Seg",
               "Bwd Seg", "Fwd Segment", "Bwd Segment", "Average Packet Size",
               "Fwd Header", "Bwd Header", "FWD Init Win", "Bwd Init Win", "Down/Up",
               "Flow Bytes", "Flow Packets", "Fwd Packets/s", "Bwd Packets/s",
               "Subflow", "Fwd Act Data", "Fwd Bytes", "Fwd Packet/Bulk", "Bwd Bytes",
               "Bwd Packet/Bulk", "Fwd Bulk", "Bwd Bulk")
COSTLY_PREFIX = ("Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                 "Total Length of Bwd Packet", "FIN Flag", "SYN Flag", "RST Flag",
                 "PSH Flag", "ACK Flag", "Fwd PSH", "Fwd URG", "Bwd URG", "URG Flag",
                 "CWR Flag", "ECE Flag", "Bwd PSH")
BK = {c: ("COSTLY" if c.startswith(COSTLY_PREFIX) else "FREE") for c in FULL.columns}
FREE_COLS = [c for c in FULL.columns if BK[c] == "FREE"]
COSTLY_COLS = [c for c in FULL.columns if BK[c] == "COSTLY"]
say("  FREE %d | COSTLY %d | IMPOSSIBLE %d" % (len(FREE_COLS), len(COSTLY_COLS), len(SVC_COLS)))

# ---------------------------------------------------------------- labels incl. reject
oof_gate = np.load(ROOT / "cache_oof_v7_full.npz")["gate"]
attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
NC = len(attack_labels); NCR = NC + 1; IREJ = NC
LAB10 = np.array(attack_labels + [0])
att_trf = y_arr != 0
fam_trf = np.full(len(y_arr), -1)
fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
yr10 = np.where((y_arr == 0) & (oof_gate == 1), IREJ, fam_trf)


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw); return XGBClassifier(**p)


def w_alpha(ya, n_cls, a=0.25):
    n = np.bincount(np.asarray(ya), minlength=n_cls)
    w = ((n.sum() / np.maximum(n, 1)) ** a); w = w / w.mean()
    sw = w[np.asarray(ya)].astype(float)
    return sw * len(sw) / sw.sum()


# ---------------------------------------------------------------- baseline models
m10 = np.where(yr10 >= 0)[0]
XB0 = X_trf.iloc[m10].reset_index(drop=True); yB0 = yr10[m10]
t0 = time.time()
SB_BASE = make_xgb().fit(XB0, yB0, sample_weight=w_alpha(yB0, NCR))
say("Stage-B baseline trained %.1fm" % ((time.time() - t0) / 60))

sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
XG0 = X_trf.iloc[sub].reset_index(drop=True); yG0 = (y_arr[sub] != 0).astype(int)
G_BASE = make_xgb().fit(XG0, yG0, sample_weight=compute_sample_weight("balanced", yG0))
say("gate baseline trained")

# ---------------------------------------------------------------- augmentation set
rng0 = np.random.RandomState(RS)
i_att_tr = np.where(att_trf)[0]
aug_idx = rng0.choice(i_att_tr, size=min(AUG_ROWS, len(i_att_tr)), replace=False)
say("augmentation pool: %d train attack rows (of %d), split across %d budgets"
    % (len(aug_idx), len(i_att_tr), len(AUG_BUDGETS)))
groups = np.array_split(rng0.permutation(aug_idx), len(AUG_BUDGETS))


def caps_of(budget):
    c = {k: budget * IQR[k] for k in FREE_COLS}
    c.update({k: budget * COSTLY_RATIO * IQR[k] for k in COSTLY_COLS})
    return c


def build_from(F0, off, caps):
    F_ = F0.copy()
    for c in caps: F_[c] = F0[c].values + off[c]
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    F_ = repair(F_)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    return F_


def perturb(F0, S0, budget, mode, model=None, fam=None, thr=None, seed=RS):
    """mode 'noise' = undirected; 'evade' = minimise P(attack); 'mislabel' = minimise
    P(true family) subject to staying above the gate threshold."""
    caps = caps_of(budget)
    off = {c: np.zeros(len(F0), np.float32) for c in caps}
    if mode == "noise":
        r = np.random.RandomState(seed)
        for c in caps: off[c] = r.uniform(-caps[c], caps[c], len(F0)).astype(np.float32)
        return build_from(F0, off, caps)

    def score(F_):
        Xp = assemble(F_, S0)
        if mode == "evade":
            return model.predict_proba(Xp)[:, 1].astype(np.float32)
        pB = model[0].predict_proba(Xp).astype(np.float32)
        pg = model[1].predict_proba(Xp)[:, 1].astype(np.float32)
        return pB[np.arange(len(Xp)), fam] + GATE_PENALTY * (pg <= thr)

    cur = score(build_from(F0, off, caps))
    for _ in range(PASSES):
        for c in caps:
            best_s, best_o = cur, off[c]
            for ds in (+1.0, -1.0):
                cand = np.clip(off[c] + ds * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                s_ = score(build_from(F0, o2, caps))
                take = s_ < best_s
                if take.any():
                    best_s = np.where(take, s_, best_s); best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_s
    return build_from(F0, off, caps)


THR0 = float(np.quantile(G_BASE.predict_proba(
    X_trf.iloc[sub][yG0 == 0])[:, 1], 1.0 - TARGET_FPR))

AUG = {"noise": [], "evade": [], "mislabel": []}
AUGY = []
t0 = time.time()
for b, grp in zip(AUG_BUDGETS, groups):
    F0 = FULL_trf.iloc[grp].reset_index(drop=True); S0 = SVC_trf.iloc[grp].reset_index(drop=True)
    fam = fam_trf[grp]
    AUG["noise"].append(assemble(perturb(F0, S0, b, "noise", seed=RS + int(b * 1000)), S0))
    AUG["evade"].append(assemble(perturb(F0, S0, b, "evade", model=G_BASE), S0))
    AUG["mislabel"].append(assemble(
        perturb(F0, S0, b, "mislabel", model=(SB_BASE, G_BASE), fam=fam, thr=THR0), S0))
    AUGY.append(grp)
    say("  augmented at budget %.3f : %d rows (%.1fm elapsed)"
        % (b, len(grp), (time.time() - t0) / 60))
    dump()
AUG = {k: pd.concat(v, ignore_index=True) for k, v in AUG.items()}
AUG_ROWIDX = np.concatenate(AUGY)
AUG_FAM = fam_trf[AUG_ROWIDX]
say("augmentation generated in %.1fm | %d rows each" % ((time.time() - t0) / 60, len(AUG_ROWIDX)))

# ---------------------------------------------------------------- augmented models
def fit_gate(extra):
    Xg = pd.concat([XG0, extra], ignore_index=True) if extra is not None else XG0
    yg = np.concatenate([yG0, np.ones(len(extra), int)]) if extra is not None else yG0
    return make_xgb().fit(Xg, yg, sample_weight=compute_sample_weight("balanced", yg))


def fit_sb(extra, extra_y):
    Xb = pd.concat([XB0, extra], ignore_index=True) if extra is not None else XB0
    yb = np.concatenate([yB0, extra_y]) if extra is not None else yB0
    return make_xgb().fit(Xb, yb, sample_weight=w_alpha(yb, NCR))


t0 = time.time()
G_NOISE = fit_gate(AUG["noise"]); G_ADV = fit_gate(AUG["evade"])
SB_NOISE = fit_sb(AUG["noise"], AUG_FAM); SB_ADV = fit_sb(AUG["mislabel"], AUG_FAM)
say("augmented models trained %.1fm" % ((time.time() - t0) / 60))
del X_trf, XB0, XG0, AUG; gc.collect()

ARMS = {"A baseline": (G_BASE, SB_BASE), "B noise-aug": (G_NOISE, SB_NOISE),
        "C adv-gate": (G_ADV, SB_BASE), "D adv-both": (G_ADV, SB_ADV)}

att_te = y_te != 0
i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
y_att = y_te[i_att]; FAM_TE = np.array([fam_to_idx[v] for v in y_att])
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))

THR = {}
for nm, (g, _) in ARMS.items():
    pb = g.predict_proba(X_ben)[:, 1]
    THR[nm] = float(np.quantile(pb, 1.0 - TARGET_FPR))
    say("  %-12s gate threshold %.4f -> benign fpr %.4f" % (nm, THR[nm], float((pb > THR[nm]).mean())))

BEN_PRED = {}
for nm, (g, sb) in ARMS.items():
    gmb = g.predict_proba(X_ben)[:, 1] > THR[nm]
    yp = np.zeros(len(i_ben), int)
    if gmb.any(): yp[gmb] = LAB10[sb.predict_proba(X_ben[gmb]).argmax(1)]
    BEN_PRED[nm] = yp
dump()

# ---------------------------------------------------------------- CRITICAL CONFOUND CHECK
# Augmentation added ONLY perturbed ATTACK rows, labelled attack=1. A gate can satisfy that
# by learning "perturbed => attack" rather than anything about intrusions. Thresholds above
# are matched on CLEAN benign, so that shortcut is invisible there -- and it would collapse
# precisely under measurement drift / a new exporter build, which is the scenario this whole
# chapter is motivated by (drift perturbs BENIGN traffic too; an adversary does not).
# So: perturb benign rows with the same undirected noise and re-measure fpr at fixed threshold.
F_ben0 = FULL_te.iloc[i_ben].reset_index(drop=True)
S_ben0 = SVC_te.iloc[i_ben].reset_index(drop=True)
say("\n" + "=" * 78)
say("CONFOUND CHECK -- benign fpr when NOISE HITS BENIGN TRAFFIC TOO (drift, not an attacker)")
say("=" * 78)
say("  a defence that only holds when noise is confined to attack rows is an ADVERSARY")
say("  defence, not a DRIFT defence. Thresholds fixed at their clean-matched values.")
say("  %8s %12s %12s %12s %12s" % ("budget", "A baseline", "B noise-aug", "C adv-gate", "D adv-both"))
FPR_DRIFT = []
for b in BUDGETS:
    Fb = F_ben0 if b == 0 else perturb(F_ben0, S_ben0, b, "noise", seed=RS + 7)
    Xb = assemble(Fb, S_ben0)
    row = dict(budget=b)
    for nm, (g, _) in ARMS.items():
        row[nm] = float((g.predict_proba(Xb)[:, 1] > THR[nm]).mean())
    FPR_DRIFT.append(row)
    say("  %8.3f %12.4f %12.4f %12.4f %12.4f"
        % (b, row["A baseline"], row["B noise-aug"], row["C adv-gate"], row["D adv-both"]))
    dump()
    del Fb, Xb; gc.collect()
fpr_b20 = FPR_DRIFT[-1]["B noise-aug"]; fpr_a20 = FPR_DRIFT[-1]["A baseline"]
say("  -> at budget 0.20: baseline fpr %.4f vs noise-aug fpr %.4f (ratio %.1fx)"
    % (fpr_a20, fpr_b20, fpr_b20 / max(fpr_a20, 1e-9)))
say("  If the noise-aug fpr explodes here, its apparent robustness is a 'noisy => attack'")
say("  shortcut and must NOT be reported as a drift defence.")


def full_pred(nm, F_p):
    g, sb = ARMS[nm]
    Xp = assemble(F_p, S_att0)
    gma = g.predict_proba(Xp)[:, 1] > THR[nm]
    yp_a = np.zeros(len(Xp), int)
    if gma.any(): yp_a[gma] = LAB10[sb.predict_proba(Xp[gma]).argmax(1)]
    yp = np.zeros(len(y_te), int); yp[i_ben] = BEN_PRED[nm]; yp[i_att] = yp_a
    return yp, yp_a, gma


def metrics(nm, F_p):
    yp, yp_a, gma = full_pred(nm, F_p)
    und = yp_a == 0; cor = yp_a == y_att; mis = (~und) & (~cor)
    return dict(undetected=float(und.mean()), misattributed=float(mis.mean()),
                correct=float(cor.mean()), gate_recall=float(gma.mean()),
                macro_f1=float(f1_score(y_te, yp, average="macro", zero_division=0)),
                accuracy=float((yp == y_te).mean()))


say("\n" + "=" * 78); say("CLEAN performance"); say("=" * 78)
say("  %-12s %10s %10s %12s" % ("arm", "macro-F1", "accuracy", "gate-recall"))
CLEAN = {}; CLEAN_PRED = {}
for nm in ARMS:
    CLEAN[nm] = metrics(nm, F_att0); CLEAN_PRED[nm] = full_pred(nm, F_att0)[0]
    say("  %-12s %10.4f %10.4f %12.4f"
        % (nm, CLEAN[nm]["macro_f1"], CLEAN[nm]["accuracy"], CLEAN[nm]["gate_recall"]))
dump()

rows = []
say("\n" + "=" * 78); say("RANDOM NOISE (undirected, full grid)"); say("=" * 78)
NOISE_CACHE = {b: perturb(F_att0, S_att0, b, "noise", seed=RS) for b in BUDGETS if b > 0}
for nm in ARMS:
    say("\n=== %s | RANDOM ===" % nm)
    say("  %8s %10s %12s %13s" % ("budget", "macro-F1", "gate-recall", "misattributed"))
    for b in BUDGETS:
        r = CLEAN[nm] if b == 0 else metrics(nm, NOISE_CACHE[b])
        r = dict(r); r.update(arm=nm, objective="RANDOM", budget=b)
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        say("  %8.3f %10.4f %12.4f %13.4f" % (b, r["macro_f1"], r["gate_recall"], r["misattributed"]))
    dump()

for obj in ("EVASION", "MISATTRIB"):
    for nm in ARMS:
        g, sb = ARMS[nm]
        say("\n=== %s | %s ===" % (nm, obj))
        say("  %8s %10s %12s %13s %10s" % ("budget", "macro-F1", "gate-recall", "misattributed", "undet"))
        for b in (0.0,) + ATK_BUDGETS:
            if b == 0.0:
                r = CLEAN[nm]
            else:
                Fp = (perturb(F_att0, S_att0, b, "evade", model=g) if obj == "EVASION"
                      else perturb(F_att0, S_att0, b, "mislabel", model=(sb, g),
                                   fam=FAM_TE, thr=THR[nm]))
                r = metrics(nm, Fp)
            r = dict(r); r.update(arm=nm, objective=obj, budget=b)
            rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
            say("  %8.3f %10.4f %12.4f %13.4f %10.4f"
                % (b, r["macro_f1"], r["gate_recall"], r["misattributed"], r["undetected"]))
            dump()

R = pd.DataFrame(rows)
gv = lambda a, o, c, b: float(R[(R["arm"] == a) & (R["objective"] == o) & (R["budget"] == b)][c].iloc[0])

# ---------------------------------------------------------------- paired bootstrap, clean cost
def fast_macro_f1(yt, yp):
    """bincount confusion matrix -> macro-F1; ~10x faster than sklearn on 521k rows, and
    verified against f1_score below before the bootstrap is trusted."""
    cm = np.bincount(yt * 10 + yp, minlength=100).reshape(10, 10)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


_chk = abs(fast_macro_f1(y_te, CLEAN_PRED["A baseline"])
           - f1_score(y_te, CLEAN_PRED["A baseline"], average="macro", zero_division=0))
assert _chk < 1e-9, "fast macro-F1 disagrees with sklearn (%.2e)" % _chk
say("\nfast macro-F1 verified against sklearn (diff %.1e)" % _chk)

rs = np.random.RandomState(RS)
BOOT = {}
for nm in ("B noise-aug", "C adv-gate", "D adv-both"):
    d = np.empty(NBOOT)
    pa, pb_ = CLEAN_PRED["A baseline"], CLEAN_PRED[nm]
    for k in range(NBOOT):
        ii = rs.randint(0, len(y_te), len(y_te))
        yt = y_te[ii]
        d[k] = fast_macro_f1(yt, pb_[ii]) - fast_macro_f1(yt, pa[ii])
    BOOT[nm] = (float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
                float(min((d <= 0).mean(), (d >= 0).mean()) * 2))
say("\n" + "=" * 78); say("CLEAN COST vs baseline -- paired bootstrap (%d resamples)" % NBOOT)
say("=" * 78)
for nm, (m, lo, hi, p) in BOOT.items():
    say("  %-12s %+.4f  [%+.4f, %+.4f]  p=%.3f" % (nm, m, lo, hi, p))

say("\n" + "=" * 78); say("PRE-REGISTERED TESTS"); say("=" * 78)
ret = lambda a, o, b: gv(a, o, "gate_recall", b) / CLEAN[a]["gate_recall"]
rB, rC = ret("B noise-aug", "EVASION", 0.20), ret("C adv-gate", "EVASION", 0.20)
rA = ret("A baseline", "EVASION", 0.20)
p1 = (rC - rB) > 0.05
say("  P1 IS 'ADVERSARIAL' NECESSARY? retained gate recall @0.20 under EVASION")
say("      A baseline %.4f | B noise-aug %.4f | C adv-gate %.4f" % (rA, rB, rC))
say("      C - B = %+.4f (threshold > +0.05)   %s" % (rC - rB, "PASS" if p1 else "FAIL"))
if not p1:
    say("      -> POSITIVE PRACTICAL FINDING: noise augmentation is sufficient;")
    say("         the adversarial machinery is unnecessary. Report as such, NOT as a null.")

mC, lC, hC, pC = BOOT["C adv-gate"]
p2 = abs(mC) < 0.02
say("  P2 CLEAN COST of arm C: %+.4f [%+.4f, %+.4f] (|.|<0.02)  %s"
    % (mC, lC, hC, "PASS" if p2 else "FAIL -- trade-off, not a free defence"))

mis_C = gv("C adv-gate", "MISATTRIB", "misattributed", 0.20)
mis_D = gv("D adv-both", "MISATTRIB", "misattributed", 0.20)
p3 = (mis_C - mis_D) > 0.05
say("  P3 STAGE-ASYMMETRY: misattributed @0.20  C %.4f vs D %.4f, D better by %+.4f (>0.05) %s"
    % (mis_C, mis_D, mis_C - mis_D, "PASS" if p3 else "FAIL -- drop the stage-specific story"))
say("      (gate recall beside it -- C %.4f | D %.4f -- misattributed is CAPPED by this)"
    % (gv("C adv-gate", "MISATTRIB", "gate_recall", 0.20),
       gv("D adv-both", "MISATTRIB", "gate_recall", 0.20)))

verdict = ("ADVERSARIAL TRAINING JUSTIFIED" if (p1 and p2) else
           "NOISE AUGMENTATION SUFFICIENT -- adversarial machinery unnecessary" if (not p1 and p2) else
           "AUGMENTATION IS A TRADE-OFF -- clean cost too high to recommend freely")
say("\n  VERDICT: %s" % verdict)
say("\n  CAVEAT to carry: attacks are white-box on ONE component and Stage B here is a single")
say("  model; v7.3e measured surrogate inflation at +0.6773, so attack strengths are")
say("  UNDERSTATED -- conservative for an attack claim, GENEROUS for a defence claim.")
say("  A defence that only works against understated attacks is not established.")

res = dict(experiment="v7.3h -- controllability-aware adversarial training (vs Tafreshian & Zhang)",
           prereg=dict(p1_adversarial_necessary=bool(p1), p1_gap=round(rC - rB, 4),
                       p2_clean_cost=round(mC, 4), p2_cheap=bool(p2),
                       p3_stage_asymmetry=bool(p3), p3_gap=round(mis_C - mis_D, 4),
                       verdict=verdict),
           clean={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in CLEAN.items()},
           clean_cost_paired_bootstrap={k: dict(mean=round(v[0], 4), lo=round(v[1], 4),
                                                hi=round(v[2], 4), p=round(v[3], 4))
                                        for k, v in BOOT.items()},
           aug_rows=int(len(AUG_ROWIDX)), aug_budgets=list(AUG_BUDGETS),
           thresholds=THR, target_fpr=TARGET_FPR, results=rows,
           fpr_under_benign_noise=FPR_DRIFT,
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "adv_training.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
