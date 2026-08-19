"""
v7.3g: a CONTROLLABILITY-AWARE two-stage NIDS -- allocate feature access per stage by
threat exposure instead of by accuracy.

The v7.3 series established four things that point at one architecture:

  * the GATE is where robustness lives (v7.3c) -- and it collapses to 0.0309 retained recall
    under a targeted white-box attack (v7.3e)
  * the gate is otherwise essentially SOLVED (recall ~0.997; svc lifted its precision
    0.9267 -> 0.9411), while STAGE B is where the accuracy is scarce
  * feature IMPORTANCE does not predict robustness (v7.3, refuted): svc is 70.4% of
    importance and uncontrollable, and the model is evaded through the other 30% anyway.
    So the lever is RESTRICTING feature access, not reweighting it.
  * misattribution keeps the gate at 0.9969 while corrupting 63% of labels (v7.3f), i.e.
    hardening the gate should NOT defend the label.

Hence the design claim under test:

    Allocate feature access per stage by threat exposure, not by accuracy.
    The easy stage (detection) can afford to be robust; the hard stage (attribution) cannot.

2x2 over feature access. Stage B is a SINGLE XGB a=0.25 in every arm -- v7 established that
at full scale the 7-member ensemble is worth ~0.0038 over the best single model, so dropping
it costs almost nothing and removes the cached-OOF combiner dependency, leaving FEATURE
ACCESS as the only moving part:

    arm   gate features      Stage-B features
    A     full 99            full 99            baseline
    B     svc-only 25        full 99            <- the proposal: harden the cheap stage
    C     svc-only 25        svc-only 25        harden both
    D     full 99            svc-only 25        control (harden the wrong stage)

Two attacker objectives from v7.3f (EVASION, MISATTRIBUTION), and two svc mobility settings:

    FROZEN  svc/proto/port immovable -- changing them abandons the attack (the v7.3 model)
    MOBILE  the attacker RETARGETS THE SERVICE: dst port switched to any candidate, whole
            svc block recomputed coherently. This exists because with svc frozen an
            svc-only gate is unattackable BY CONSTRUCTION, which is a tautology, not a
            result. MOBILE prices what hardening actually buys against an attacker willing
            to abandon service targeting.

PRE-REGISTERED PREDICTIONS AND DECISION RULE PRINTED BEFORE ANY RESULT (see PREREG).

Same prep, splits, budget machinery, matched-fpr protocol and reject-class definition as
v7.3c/e/f. ASCII-only prints; log written UTF-8.
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
GATE_TRAIN = 800_000
COSTLY_RATIO = 0.5
PASSES = 2
TARGET_FPR = 0.0168
GATE_PENALTY = 10.0
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "ctrl_aware.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()

PREREG = """
==============================================================================
PRE-REGISTERED PREDICTIONS -- written before the run, printed before any result
==============================================================================
DESIGN CLAIM: allocate feature access per stage by THREAT EXPOSURE, not accuracy.
The easy stage (detection) can afford to be robust; the hard stage cannot.

P1 SANITY, NOT A FINDING. With svc FROZEN, an svc-only gate's recall must be EXACTLY
    constant across every budget -- none of its inputs can move. Asserted, not celebrated.
    If it varies at all, there is a leak in the feature partition and the run is void.

P2 THE REAL TEST -- ASYMMETRY. Hardening the GATE should defend EVASION and should NOT
    defend MISATTRIBUTION, because misattribution attacks Stage B while deliberately
    keeping the gate satisfied (v7.3f: gate recall 0.9969 while 63% of labels wrong).
      P2a  arm B evasion success at 0.20 << arm A evasion success  (gap > 0.30)
      P2b  arm B misattribution success at 0.20 ~ arm A's          (|gap| < 0.10)
    A defence that helped BOTH would mean the stage-allocation logic is wrong (or that
    something other than feature access is driving the result).

P3 THE LOAD-BEARING PRACTICAL CLAIM -- COST. Because detection is the easy sub-problem
    and svc already carries most of its signal, restricting the gate to uncontrollable
    features should be nearly free end-to-end:
      arm B clean macro-F1 within 0.02 of arm A clean macro-F1.
    IF P3 FAILS the architecture is NOT a recommendation regardless of how robust it is;
    report it as a trade-off curve and say so plainly.

P4 NON-TAUTOLOGY. Under svc MOBILE (attacker retargets the service, whole svc block
    recomputed) the hardened gate becomes attackable again. Reported as: how much of the
    protection survives an attacker willing to abandon service targeting. No threshold --
    this is a price, and it is the honest bound on the whole idea.

DECISION RULE, committed in advance:
  * P2 holds and P3 holds -> report as an architecture recommendation, scoped by P4.
  * P2 holds and P3 fails -> report as a trade-off curve only. No recommendation.
  * P2 fails              -> the stage-allocation logic is refuted; report as a null and
                             do NOT rescue it by re-slicing the arms after the fact.

STANDING HONESTY REQUIREMENT, to be repeated wherever this result appears: this defence
rests entirely on svc, and novelty.md section 6.3 already records that IXIA PerfectStorm
targets attacks at specific services, so svc may be partly a TESTBED ARTIFACT. A
controllability-aware architecture built on svc INHERITS that caveat and amplifies it,
because the whole defence now depends on the feature in question. State this in the same
breath as the result, never in a trailing limitations paragraph.
==============================================================================
"""
for _l in PREREG.strip().splitlines():
    say(_l)
dump()

# ---------------------------------------------------------------- prep (v7.3c/e/f-identical)
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


def svc_block(port_arr, proto_arr, index):
    """Build the svc feature block from raw port/proto -- the ONLY way svc is ever
    constructed, so a service switch stays internally coherent by construction."""
    s = {nm: (port_arr == p).astype(np.float32) for p, nm in SVC_PORTS}
    s["svc_other_low"] = ((port_arr < 1024) & ~np.isin(port_arr, KNOWN)).astype(np.float32)
    s["svc_registered"] = ((port_arr >= 1024) & (port_arr < 49152)
                           & ~np.isin(port_arr, KNOWN)).astype(np.float32)
    s["svc_ephemeral"] = (port_arr >= 49152).astype(np.float32)
    s["proto_tcp"] = (proto_arr == 6).astype(np.float32)
    s["proto_udp"] = (proto_arr == 17).astype(np.float32)
    s["dst_port_num"] = port_arr.astype(np.float32)
    return pd.DataFrame(s, index=index)


SVC = svc_block(port, proto, df.index)
sz = SVC.nunique(); DROP_SVC = sz[sz <= 1].index.tolist()
SVC = SVC.drop(columns=DROP_SVC)
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
say("\nprep done %.0fm | %d int cols | full %d feats | svc-only %d feats"
    % ((time.time() - t_start) / 60, len(INT_COLS), len(FEAT_ORDER), len(SVC_COLS)))

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
# the partition must be airtight or P1 is meaningless
assert not (set(SVC_COLS) & set(BASE_COLS)) and not (set(SVC_COLS) & set(DER_COLS)), \
    "svc columns leak into the perturbable space"
say("  FREE %d | COSTLY %d | IMPOSSIBLE %d (svc/proto/port)"
    % (len(FREE_COLS), len(COSTLY_COLS), len(SVC_COLS)))

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


VIEWS = {"full": FEAT_ORDER, "svc": SVC_COLS}
t0 = time.time()
m10 = yr10 >= 0
n10 = np.bincount(yr10[m10], minlength=NCR)
w10 = ((n10.sum() / np.maximum(n10, 1)) ** 0.25); w10 = w10 / w10.mean()
sw10 = w10[yr10[m10]].astype(float); sw10 = sw10 * len(sw10) / sw10.sum()
STAGEB = {}
for vn, cols in VIEWS.items():
    STAGEB[vn] = make_xgb().fit(X_trf.loc[m10, cols].reset_index(drop=True),
                                pd.Series(yr10[m10]), sample_weight=sw10)
    say("Stage-B (%s, %d feats) trained %.1fm" % (vn, len(cols), (time.time() - t0) / 60))

sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
yg = (y_arr[sub] != 0).astype(int); swg = compute_sample_weight("balanced", yg)
GATES = {}
for vn, cols in VIEWS.items():
    GATES[vn] = make_xgb().fit(X_trf.iloc[sub][cols].reset_index(drop=True), yg, sample_weight=swg)
    say("gate (%s, %d feats) trained" % (vn, len(cols)))
del X_trf; gc.collect()

att_te = y_te != 0
i_att = np.where(att_te)[0]; i_ben = np.where(~att_te)[0]
F_att0 = FULL_te.iloc[i_att].reset_index(drop=True); S_att0 = SVC_te.iloc[i_att].reset_index(drop=True)
y_att = y_te[i_att]
FAM_TRUE = np.array([fam_to_idx[v] for v in y_att])
PORT_ATT0 = port[i_test][i_att]; PROTO_ATT = proto[i_test][i_att]
X_ben = assemble(FULL_te.iloc[i_ben].reset_index(drop=True), SVC_te.iloc[i_ben].reset_index(drop=True))

THR = {}
for vn in VIEWS:
    pb = GATES[vn].predict_proba(X_ben[VIEWS[vn]])[:, 1]
    THR[vn] = float(np.quantile(pb, 1.0 - TARGET_FPR))
    say("  gate %-4s threshold %.4f -> benign fpr %.4f" % (vn, THR[vn], float((pb > THR[vn]).mean())))

BEN_PRED = {}
for gv in VIEWS:
    gmb = GATES[gv].predict_proba(X_ben[VIEWS[gv]])[:, 1] > THR[gv]
    for bv in VIEWS:
        yp = np.zeros(len(i_ben), int)
        if gmb.any(): yp[gmb] = LAB10[STAGEB[bv].predict_proba(X_ben.loc[gmb, VIEWS[bv]]).argmax(1)]
        BEN_PRED[(gv, bv)] = yp
dump()

# ---------------------------------------------------------------- attack machinery
CAND_PORTS = [p for p, _ in SVC_PORTS] + [3000, 50000]        # + registered + ephemeral


def _caps(budget):
    caps = {c: budget * IQR[c] for c in FREE_COLS}
    caps.update({c: budget * COSTLY_RATIO * IQR[c] for c in COSTLY_COLS})
    return caps


def build(off, caps):
    F_ = F_att0.copy()
    for c in caps: F_[c] = F_att0[c].values + off[c]
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    F_ = repair(F_)
    for c in INT_COLS: F_[c] = np.round(F_[c].values)
    return F_


def scorer(objective, gv, bv):
    """per-row score to MINIMISE; white-box on the deployed components of this arm."""
    gcols, bcols = VIEWS[gv], VIEWS[bv]

    def fn(F_, S_):
        Xp = assemble(F_, S_)
        if objective == "EVASION":
            return GATES[gv].predict_proba(Xp[gcols])[:, 1].astype(np.float32)
        pg = GATES[gv].predict_proba(Xp[gcols])[:, 1].astype(np.float32)
        pB = STAGEB[bv].predict_proba(Xp[bcols]).astype(np.float32)
        return pB[np.arange(len(Xp)), FAM_TRUE] + GATE_PENALTY * (pg <= THR[gv])
    return fn


def craft(budget, objective, gv, bv, svc_mobile):
    caps = _caps(budget)
    off = {c: np.zeros(len(F_att0), np.float32) for c in caps}
    S_ = S_att0.copy(); ports = PORT_ATT0.copy()
    fn = scorer(objective, gv, bv)
    cur = fn(build(off, caps), S_)
    for _ in range(PASSES):
        for c in caps:
            best_s, best_o = cur, off[c]
            for dsign in (+1.0, -1.0):
                cand = np.clip(off[c] + dsign * caps[c], -caps[c], caps[c])
                o2 = dict(off); o2[c] = cand
                s_ = fn(build(o2, caps), S_)
                take = s_ < best_s
                if take.any():
                    best_s = np.where(take, s_, best_s)
                    best_o = np.where(take, cand, best_o)
            off[c] = best_o.astype(np.float32); cur = best_s
        if svc_mobile:                       # retarget the service, whole block recomputed
            F_now = build(off, caps)
            for p_new in CAND_PORTS:
                trial_ports = np.full(len(ports), p_new, dtype=np.int64)
                S_try = svc_block(trial_ports, PROTO_ATT, S_att0.index).drop(columns=DROP_SVC)
                s_ = fn(F_now, S_try)
                take = s_ < cur
                if take.any():
                    ports = np.where(take, p_new, ports)
                    S_ = svc_block(ports, PROTO_ATT, S_att0.index).drop(columns=DROP_SVC)
                    cur = np.where(take, s_, cur)
    return build(off, caps), S_, float((ports != PORT_ATT0).mean())


def evaluate(F_p, S_p, gv, bv):
    Xp = assemble(F_p, S_p)
    gma = GATES[gv].predict_proba(Xp[VIEWS[gv]])[:, 1] > THR[gv]
    yp_a = np.zeros(len(Xp), int)
    if gma.any():
        yp_a[gma] = LAB10[STAGEB[bv].predict_proba(Xp.loc[gma, VIEWS[bv]]).argmax(1)]
    undet = yp_a == 0; corr_ = yp_a == y_att; mis = (~undet) & (~corr_)
    yp = np.zeros(len(y_te), int); yp[i_ben] = BEN_PRED[(gv, bv)]; yp[i_att] = yp_a
    return dict(undetected=float(undet.mean()), misattributed=float(mis.mean()),
                correct=float(corr_.mean()), gate_recall=float(gma.mean()),
                macro_f1=float(f1_score(y_te, yp, average="macro", zero_division=0)),
                accuracy=float((yp == y_te).mean()))


ARMS = {"A full/full": ("full", "full"), "B svcGate/full": ("svc", "full"),
        "C svcGate/svcB": ("svc", "svc"), "D full/svcB": ("full", "svc")}

say("\n" + "=" * 78); say("CLEAN performance -- the cost side of the trade"); say("=" * 78)
say("  %-16s %10s %10s %12s %11s" % ("arm", "macro-F1", "accuracy", "gate-recall", "correct"))
CLEAN = {}
for nm, (gv, bv) in ARMS.items():
    r = evaluate(F_att0, S_att0, gv, bv); CLEAN[nm] = r
    say("  %-16s %10.4f %10.4f %12.4f %11.4f"
        % (nm, r["macro_f1"], r["accuracy"], r["gate_recall"], r["correct"]))
dump()

RUNS = [("A full/full", "EVASION", False), ("B svcGate/full", "EVASION", False),
        ("A full/full", "MISATTRIB", False), ("B svcGate/full", "MISATTRIB", False),
        ("C svcGate/svcB", "MISATTRIB", False), ("B svcGate/full", "EVASION", True)]
rows = []
for nm, obj, mob in RUNS:
    gv, bv = ARMS[nm]
    tag = "%s | %s | svc=%s" % (nm, obj, "MOBILE" if mob else "FROZEN")
    say("\n=== %s ===" % tag)
    say("  %8s %11s %13s %9s %11s %9s %8s"
        % ("budget", "undetected", "misattributed", "correct", "gate-recall", "macro-F1", "retgt"))
    for b in BUDGETS:
        if b == 0.0:
            F_p, S_p, rt = F_att0, S_att0, 0.0
        else:
            F_p, S_p, rt = craft(b, obj, gv, bv, mob)
        r = evaluate(F_p, S_p, gv, bv)
        r.update(arm=nm, objective=obj, svc="MOBILE" if mob else "FROZEN", budget=b,
                 retargeted=round(rt, 4), tag=tag)
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        say("  %8.3f %11.4f %13.4f %9.4f %11.4f %9.4f %8.4f"
            % (b, r["undetected"], r["misattributed"], r["correct"], r["gate_recall"],
               r["macro_f1"], rt))
        dump()

R = pd.DataFrame(rows)
g = lambda tg, c, b: float(R[(R["tag"] == tg) & (R["budget"] == b)][c].iloc[0])
bmax = max(BUDGETS)
TA_E = "A full/full | EVASION | svc=FROZEN"; TB_E = "B svcGate/full | EVASION | svc=FROZEN"
TA_M = "A full/full | MISATTRIB | svc=FROZEN"; TB_M = "B svcGate/full | MISATTRIB | svc=FROZEN"
TC_M = "C svcGate/svcB | MISATTRIB | svc=FROZEN"; TB_MOB = "B svcGate/full | EVASION | svc=MOBILE"

say("\n" + "=" * 78); say("PRE-REGISTERED TESTS"); say("=" * 78)
gr_b = R[(R["tag"] == TB_E)]["gate_recall"].values
p1_ok = bool(np.ptp(gr_b) < 1e-9)
say("  P1 SANITY: svc-only gate recall constant under FROZEN svc?  %s (spread %.2e)"
    % ("PASS" if p1_ok else "*** FAIL -- PARTITION LEAK, RUN IS VOID ***", float(np.ptp(gr_b))))

ev_a, ev_b = g(TA_E, "undetected", bmax), g(TB_E, "undetected", bmax)
mi_a, mi_b = g(TA_M, "misattributed", bmax), g(TB_M, "misattributed", bmax)
p2a = (ev_a - ev_b) > 0.30; p2b = abs(mi_a - mi_b) < 0.10
say("  P2a EVASION defended by gate hardening: A %.4f vs B %.4f, gap %+.4f (>0.30)  %s"
    % (ev_a, ev_b, ev_a - ev_b, "PASS" if p2a else "FAIL"))
say("  P2b MISATTRIB NOT defended:             A %.4f vs B %.4f, gap %+.4f (|.|<0.10) %s"
    % (mi_a, mi_b, mi_a - mi_b, "PASS" if p2b else "FAIL"))
say("      -> ASYMMETRY %s" % ("CONFIRMED" if (p2a and p2b) else "NOT confirmed"))

cost = CLEAN["B svcGate/full"]["macro_f1"] - CLEAN["A full/full"]["macro_f1"]
p3 = abs(cost) < 0.02
say("  P3 COST of hardening the gate: %+.4f macro-F1 (|.|<0.02)  %s"
    % (cost, "PASS -- recommendable" if p3 else "FAIL -- trade-off curve only, NO recommendation"))

say("  P4 NON-TAUTOLOGY -- svc MOBILE (attacker retargets the service):")
say("      %8s %11s %11s %11s %9s" % ("budget", "B FROZEN", "B MOBILE", "protection", "retargeted"))
for b in BUDGETS:
    if b == 0.0: continue
    fz, mo = g(TB_E, "undetected", b), g(TB_MOB, "undetected", b)
    say("      %8.3f %11.4f %11.4f %11.4f %9.4f"
        % (b, fz, mo, mo - fz, g(TB_MOB, "retargeted", b)))
say("      (protection = extra evasion an attacker buys by abandoning service targeting)")

say("  C arm -- does hardening Stage B defend the LABEL, and at what clean cost?")
say("      misattributed at %.2f: A %.4f | B %.4f | C %.4f" % (bmax, mi_a, mi_b, g(TC_M, "misattributed", bmax)))
say("      clean macro-F1:       A %.4f | B %.4f | C %.4f"
    % (CLEAN["A full/full"]["macro_f1"], CLEAN["B svcGate/full"]["macro_f1"],
       CLEAN["C svcGate/svcB"]["macro_f1"]))

if not p1_ok:
    verdict = "VOID -- feature partition leaks; fix before interpreting anything"
elif not (p2a and p2b):
    verdict = "P2 FAILED -- stage-allocation logic refuted; report as a null, do not re-slice"
elif p3:
    verdict = "RECOMMENDABLE -- asymmetry confirmed and hardening the gate is nearly free"
else:
    verdict = "TRADE-OFF CURVE ONLY -- asymmetry confirmed but the clean cost is too high"
say("\n  VERDICT: %s" % verdict)
say("\n  STANDING CAVEAT: this defence rests entirely on svc, which novelty.md 6.3 records")
say("  may be partly a TESTBED ARTIFACT (IXIA targets attacks at specific services).")
say("  Quote that in the same breath as this result, never as a trailing limitation.")

res = dict(experiment="v7.3g -- controllability-aware stage-asymmetric feature allocation",
           prereg=dict(p1_constant_gate=p1_ok, p2a_evasion_defended=bool(p2a),
                       p2b_misattrib_not_defended=bool(p2b),
                       p3_cost=round(cost, 4), p3_cheap=bool(p3), verdict=verdict),
           clean={k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in CLEAN.items()},
           views={k: len(v) for k, v in VIEWS.items()}, thresholds=THR,
           stage_b="single XGB alpha=0.25 in every arm (v7: ensemble worth ~0.0038 at full scale)",
           caveat="defence rests on svc, which may be partly a testbed artifact (IXIA service targeting)",
           budgets=list(BUDGETS), results=rows,
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "ctrl_aware.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()
