"""
v7.3f: the MISATTRIBUTION threat model -- what does it cost to make a NIDS lie about
WHICH attack it is, rather than to hide from it?

Every adversarial-NIDS attack in the literature optimises one thing: P(benign | x_hat).
Tafreshian & Zhang (TrustCom 2024) state that fitness literally; the surveys describe the
field the same way ("most commonly evasion, where malicious traffic is misclassified as
benign"). That objective is INEXPRESSIBLE in a binary detector, which is why a field working
mostly in binary has not asked the other question:

    what if the attacker does not hide, but makes you misidentify what they are?

Stay detected. Keep the alert firing. Flip Exploits -> Fuzzers. The SOC gets an alert, opens
the wrong playbook, applies the wrong containment, and burns analyst time.

This is measurable here because Stage A (detection) and Stage B (attribution) are separately
trained components, so an attack can be aimed at either one and the damage attributed to a
stage. It completes the 2x2 opened by v7.3c/e:

                        detection damage   attribution damage
    noise (no target)   low   (measured)   high  (measured)
    aimed at gate       high  (measured)   high  (measured)
    aimed at Stage B    ?                  ?                  <- this run

THE RESULT IS A PRICE, NOT A POSSIBILITY. "You break whatever you aim at" would be trivial.
The claim under test is that misattribution is much CHEAPER than evasion -- that the least
expensive attack on a modern NIDS is not hiding from it but lying to it.

Three attacker objectives, identical budget machinery, identical rows, all white-box on ONE
component (the gate is one model; Stage B's strongest single member stands in for Stage B --
stated, not hidden, and symmetric across arms so the price comparison is fair):

    EVASION        minimise P(attack) at the gate                      (the standard attack)
    MISATTRIBUTION minimise P(true family), subject to STAYING DETECTED
    ABSTENTION     maximise P(reject) -- get filed as a benign false alarm

Every arm is then evaluated END-TO-END through the real deployed pipeline (gate + 7-member
dynamic combiner), so the numbers describe the whole system, not the surrogate.

Outcomes per attack row are mutually exclusive and exhaustive:
    undetected     final prediction is Benign (gate-negative OR routed to Benign by reject)
    misattributed  detected, but the predicted family is wrong
    correct        detected and the family is right

PRE-REGISTERED PREDICTION AND DECISION RULE PRINTED BEFORE ANY RESULT (see PREREG).

Budget in train-IQR units; COSTLY at half; IMPOSSIBLE (svc/proto/dst port) frozen; integers
rounded; CICFlowMeter identities and all 31 derived columns recomputed after every move.
Only the test ATTACK rows move. ASCII-only prints; log written UTF-8.
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
RS, EPS = 42, 1e-6
CAT = {"Benign": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
       "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
NAMES = list(CAT.keys())
BUDGETS = (0.0, 0.002, 0.01, 0.02, 0.05, 0.10, 0.20)
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
PASSES = 2
TARGET_FPR = 0.0168
GATE_PENALTY = 10.0            # hard-ish constraint: never trade detection for misattribution
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "misattribution.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTION -- written before the run, printed before any result
==============================================================================
Claim under test: attribution is CHEAPER to corrupt than detection is to evade.

Rationale from existing evidence (v7.3c/e): under UNDIRECTED noise, with nothing aimed
anywhere, attribution already collapsed while detection held (RF retained 0.9931 of gate
recall but only 0.4834 of macro-F1). If attribution is intrinsically the more fragile
layer, a directed attack should corrupt it at a much smaller budget than evasion needs.

DEFINITION. b50(objective) = the smallest budget at which that objective's success
fraction reaches 0.50, linearly interpolated between grid points; reported as ">0.20"
if never reached. Outcomes are mutually exclusive: undetected / misattributed / correct.
  EVASION success        = final prediction is Benign
  MISATTRIBUTION success = still detected AND predicted family is wrong
  ABSTENTION success     = gate fired but Stage B routed the row to reject

H1 (misattribution is materially cheaper):  b50(MISATTRIB) < b50(EVASION) / 2
    i.e. the price ratio b50(EVASION)/b50(MISATTRIB) exceeds 2.

H0: the ratio is <= 2 -- corrupting the label costs about what evading costs, so the
    misattribution threat model adds no new operational exposure and should be reported
    as a null.

TRIVIALITY GUARD. "You break whatever you aim at" is not a finding. Two controls decide
whether anything here is non-trivial:
  (a) CROSS-EFFECT: the misattribution rate produced by the EVASION attack, and the
      evasion rate produced by the MISATTRIBUTION attack. If each objective only ever
      moves its own metric, the 2x2 is diagonal and the result is mechanical.
  (b) RANDOM control at the same budget. If undirected noise misattributes as effectively
      as the directed attack, then direction buys nothing and the correct claim is
      "attribution is fragile", NOT "attribution is cheap to target".

SCOPE, stated in advance: the attacker is white-box on ONE component (gate model, or
Stage B's strongest single member) while the DEPLOYED system is the full 7-member dynamic
combiner. v7.3e measured surrogate inflation at +0.6773, so attacks that must transfer
from a component to the ensemble are UNDERSTATED here. That works against H1, so an H1
pass is conservative; an H0 result must be reported with this caveat attached.
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

# ---------------------------------------------------------------- prep (v7.3c/e-identical)
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
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
del X, corr, upper, DER_ALL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("\nprep done %.0fm | %d integer columns" % ((time.time() - t_start) / 60, len(INT_COLS)))

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
assert len(FREE_COLS) + len(COSTLY_COLS) == FULL.shape[1]
say("  FREE %d | COSTLY %d | IMPOSSIBLE %d (svc/proto/dst port, frozen)"
    % (len(FREE_COLS), len(COSTLY_COLS), len(SVC_COLS)))

# ---------------------------------------------------------------- fixed Stage-B machinery
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
IDX_SURR = [s_[0] for s_ in SPECS].index("XGB a=0.25")     # Stage-B white-box surrogate
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


W = np.ones((M, NCR)) / M
def _sc(Wc):
    yp = LAB10[fuse(P_GPOS, Wc)]
    return macro_f1_from_counts(np.bincount(yt_gpos * 10 + yp, minlength=100).reshape(10, 10) + BG)
best = _sc(W)
t0 = time.time()
for _ in range(3):
    for c in range(NCR):
        for m_ in range(M):
            cur = W[m_, c]
            for gg in GRID_W:
                W[m_, c] = gg
                if W[:, c].sum() == 0: continue
                s_ = _sc(W)
                if s_ > best + 1e-7: best, cur = s_, gg
            W[m_, c] = cur
W_e2e = W
say("W_e2e fitted (%.0fs)" % (time.time() - t0))

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
say("Stage-B pool refit %.1fm (identical to v7.3c/e)" % ((time.time() - t0) / 60))

sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
Xg = X_trf.iloc[sub].reset_index(drop=True); yg = (y_arr[sub] != 0).astype(int)
GATE = make_xgb().fit(Xg, yg, sample_weight=compute_sample_weight("balanced", yg))
say("XGB gate trained")
del X_trf, Xg; gc.collect()

att_te = y_te != 0
i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
y_att = y_te[i_att]
FAM_TRUE = np.array([fam_to_idx[v] for v in y_att])          # index into the 10-class head
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))
pb_ben = GATE.predict_proba(X_ben)[:, 1]
THR = float(np.quantile(pb_ben, 1.0 - TARGET_FPR))
say("test attack rows %d | benign %d | gate threshold %.4f -> benign fpr %.4f"
    % (len(i_att), len(i_ben), THR, float((pb_ben > THR).mean())))

gmb = pb_ben > THR
SURR = POOL[IDX_SURR]


def stageB_full(Xp):
    """the deployed Stage B: 7-member pool + decision-space dynamic combiner."""
    Pg = [m.predict_proba(Xp).astype(np.float32) for m in POOL]
    _, ii = NN_D.kneighbors(prof(Pg))
    wn = WC_REF[y_ref[ii]]; den = wn.sum(1)
    Wd = np.empty((len(Xp), M))
    for m in range(M):
        Wd[:, m] = (CORRECT[m][ii] * wn).sum(1) / den
    return LAB10[fuse(Pg, W_e2e, Wd=Wd, p=3)]


YP_BEN = np.zeros(len(i_ben), int)
if gmb.any(): YP_BEN[gmb] = stageB_full(X_ben[gmb])
dump()

# ---------------------------------------------------------------- the three objectives
def _caps(budget):
    caps = {c: budget * IQR[c] for c in FREE_COLS}
    caps.update({c: budget * COSTLY_RATIO * IQR[c] for c in COSTLY_COLS})
    return caps


def _l2(off, caps):
    if not off: return 0.0
    Z = np.stack([off[c] / IQR[c] for c in caps], 1)
    return float(np.linalg.norm(Z, axis=1).mean())


def score_evasion(Xp):
    return GATE.predict_proba(Xp)[:, 1].astype(np.float32)


def score_misattrib(Xp):
    """minimise P(true family) but never at the cost of dropping below the gate threshold."""
    pg = GATE.predict_proba(Xp)[:, 1].astype(np.float32)
    pB = SURR.predict_proba(Xp).astype(np.float32)
    return pB[np.arange(len(Xp)), FAM_TRUE] + GATE_PENALTY * (pg <= THR)


def score_abstain(Xp):
    return -SURR.predict_proba(Xp).astype(np.float32)[:, IREJ]


SCORERS = {"EVASION": score_evasion, "MISATTRIB": score_misattrib, "ABSTAIN": score_abstain}


def craft(budget, objective):
    caps = _caps(budget)
    off = {c: np.zeros(len(F_att0), np.float32) for c in caps}

    def build(o):
        F_ = F_att0.copy()
        for c in caps: F_[c] = F_att0[c].values + o[c]
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        F_ = repair(F_)
        for c in INT_COLS: F_[c] = np.round(F_[c].values)
        return F_

    if objective == "RANDOM":
        rng = np.random.RandomState(RS)
        for c in caps:
            off[c] = rng.uniform(-caps[c], caps[c], len(F_att0)).astype(np.float32)
        return build(off), _l2(off, caps)

    fn = SCORERS[objective]
    cur = fn(assemble(build(off), S_att0))
    for _ in range(PASSES):
        for c in caps:
            best_s, best_o = cur, off[c]
            for dsign in (+1.0, -1.0):
                cand = np.clip(off[c] + dsign * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                s_ = fn(assemble(build(o2), S_att0))
                take = s_ < best_s
                if take.any():
                    best_s = np.where(take, s_, best_s)
                    best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_s
    return build(off), _l2(off, caps)


def evaluate(F_p):
    """end-to-end through the REAL pipeline; returns the three exclusive outcome rates."""
    Xp = assemble(F_p, S_att0)
    gma = GATE.predict_proba(Xp)[:, 1] > THR
    yp_a = np.zeros(len(Xp), int)
    if gma.any(): yp_a[gma] = stageB_full(Xp[gma])
    undetected = yp_a == 0
    correct = yp_a == y_att
    misattr = (~undetected) & (~correct)
    abstained = gma & (yp_a == 0)                       # gate fired, Stage B sent it back
    yp = np.zeros(len(y_te), int); yp[i_ben] = YP_BEN; yp[i_att] = yp_a
    return dict(undetected=float(undetected.mean()), misattributed=float(misattr.mean()),
                correct=float(correct.mean()), abstained=float(abstained.mean()),
                gate_recall=float(gma.mean()),
                macro_f1=float(f1_score(y_te, yp, average="macro", zero_division=0)),
                accuracy=float((yp == y_te).mean()))


ARMS = ["EVASION", "MISATTRIB", "ABSTAIN", "RANDOM"]
SUCCESS = {"EVASION": "undetected", "MISATTRIB": "misattributed",
           "ABSTAIN": "abstained", "RANDOM": "misattributed"}
rows = []
for arm in ARMS:
    say("\n=== %s ===" % arm)
    say("  %8s %11s %13s %9s %10s %11s %9s %8s"
        % ("budget", "undetected", "misattributed", "correct", "abstained", "gate-recall",
           "macro-F1", "L2"))
    for b in BUDGETS:
        t_b = time.time()
        F_p, l2 = (F_att0, 0.0) if b == 0.0 else craft(b, arm)
        r = evaluate(F_p)
        r.update(arm=arm, budget=b, l2_iqr=round(l2, 4), secs=round(time.time() - t_b, 1))
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        say("  %8.3f %11.4f %13.4f %9.4f %10.4f %11.4f %9.4f %8.3f"
            % (b, r["undetected"], r["misattributed"], r["correct"], r["abstained"],
               r["gate_recall"], r["macro_f1"], l2))
        dump()

R = pd.DataFrame(rows)


def b50(arm):
    """smallest budget at which the arm's own success metric reaches 0.50, interpolated."""
    col = SUCCESS[arm]
    d = R[R["arm"] == arm].sort_values("budget")
    xs = d["budget"].values; ys = d[col].values
    for i in range(1, len(xs)):
        if ys[i] >= 0.50:
            if ys[i] == ys[i - 1]: return float(xs[i])
            f = (0.50 - ys[i - 1]) / (ys[i] - ys[i - 1])
            return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
    return float("inf")


say("\n" + "=" * 78); say("OUTCOME RATES by budget (rows sum to 1 across the first three)")
say("=" * 78)
for col in ("undetected", "misattributed", "correct"):
    say("\n%s:" % col)
    say(R.pivot(index="budget", columns="arm", values=col).to_string())

B50 = {a: b50(a) for a in ARMS}
say("\n" + "=" * 78); say("PRE-REGISTERED TEST -- the PRICE of each objective"); say("=" * 78)
for a in ARMS:
    v = B50[a]
    say("  b50(%-10s) = %s   (success metric: %s)"
        % (a, (">0.20" if np.isinf(v) else "%.4f IQR" % v), SUCCESS[a]))
ratio = (B50["EVASION"] / B50["MISATTRIB"]) if B50["MISATTRIB"] > 0 and not np.isinf(B50["MISATTRIB"]) \
        else float("nan")
say("\n  PRICE RATIO b50(EVASION)/b50(MISATTRIB) = %s  (H1 threshold > 2)"
    % ("n/a" if np.isnan(ratio) else ("inf" if np.isinf(ratio) else "%.2f" % ratio)))
if np.isnan(ratio):
    verdict = "INCONCLUSIVE -- misattribution never reached 50% success within the budget grid"
elif ratio > 2:
    verdict = "H1 SUPPORTED -- corrupting the label is materially cheaper than evading"
else:
    verdict = "H0 -- misattribution costs about what evasion costs; report as a null"
say("  VERDICT: %s" % verdict)

say("\n" + "=" * 78); say("TRIVIALITY GUARDS"); say("=" * 78)
bmax = max(BUDGETS)
gv = lambda a, c, b: float(R[(R["arm"] == a) & (R["budget"] == b)][c].iloc[0])
say("  (a) CROSS-EFFECT at budget %.2f -- is the 2x2 diagonal?" % bmax)
say("      EVASION   attack -> misattributed %.4f | undetected %.4f"
    % (gv("EVASION", "misattributed", bmax), gv("EVASION", "undetected", bmax)))
say("      MISATTRIB attack -> misattributed %.4f | undetected %.4f"
    % (gv("MISATTRIB", "misattributed", bmax), gv("MISATTRIB", "undetected", bmax)))
say("      (MISATTRIB keeping 'undetected' LOW is the constraint working as designed)")
say("  (b) RANDOM control -- does direction buy anything?")
for b in BUDGETS:
    if b == 0.0: continue
    say("      budget %.3f : directed %.4f vs random %.4f  (delta %+.4f)"
        % (b, gv("MISATTRIB", "misattributed", b), gv("RANDOM", "misattributed", b),
           gv("MISATTRIB", "misattributed", b) - gv("RANDOM", "misattributed", b)))
say("      if delta ~ 0 the honest claim is 'attribution is fragile', NOT 'cheap to target'")

say("\nmacro-F1 by budget:")
say(R.pivot(index="budget", columns="arm", values="macro_f1").to_string())

res = dict(experiment="v7.3f -- misattribution threat model: the price of corrupting the label",
           prereg=dict(h1="b50(EVASION)/b50(MISATTRIB) > 2",
                       b50={a: (None if np.isinf(v) else round(v, 4)) for a, v in B50.items()},
                       price_ratio=(None if (np.isnan(ratio) or np.isinf(ratio)) else round(ratio, 3)),
                       verdict=verdict),
           surrogate_note="attacker white-box on ONE component (gate, or Stage-B member "
                          "'XGB a=0.25'); deployed system is the full 7-member dynamic combiner. "
                          "v7.3e measured surrogate inflation at +0.6773, so these attacks are "
                          "UNDERSTATED, which is conservative for H1.",
           gate_train_rows=GATE_TRAIN, target_fpr=TARGET_FPR, threshold=THR,
           passes=PASSES, costly_ratio=COSTLY_RATIO, gate_penalty=GATE_PENALTY,
           budgets=list(BUDGETS), results=rows,
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "misattribution.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
