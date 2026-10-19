"""
v7.3j: is the v7.3 "brittleness" partly an artifact of the repair step?

Every v7.3 script shortcuts budget 0 to the ORIGINAL flows but sends every budget > 0 through
repair(), which recomputes derived CICFlowMeter columns (Down/Up Ratio, Packet Length Mean, rates)
from the counts. On unperturbed flows repair() disagrees with CICFlowMeter's own values
(Down/Up Ratio > 50% off on ~19% of rows; Packet Length Mean > 1% off on ~44%). So a budget-0.002
arm differs from the clean arm by noise AND by repair.

Four arms on the test attack rows, identical models and matched-fpr gate threshold (v7.3f setup):
  ORIGINAL        the flows as recorded                         (= every v7.3 clean arm)
  REPAIR ONLY     repair() at budget 0, integers rounded         (the confound, isolated)
  NOISE+REPAIR    0.002 IQR noise then repair()                  (= the v7.3 arm)
  NOISE ONLY      0.002 IQR noise, no repair (derived re-computed, raw identities not enforced)

If REPAIR ONLY is close to NOISE+REPAIR, the headline 0.002 damage is mostly the artifact.
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
GATE_TRAIN, TARGET_FPR, BUDGET = 800_000, 0.0168, 0.002
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))
    io.open(OUT / "repair_confound.txt", "w", encoding="utf-8").write("\n".join(LOG))


t_start = time.time()
hdr = pd.read_csv(CFM, nrows=0).columns.tolist()
FEAT76 = hdr[7:-1]
dtypes = {c: np.float32 for c in FEAT76}
dtypes["Dst Port"] = np.int32; dtypes["Protocol"] = np.int16
df = pd.read_csv(CFM, usecols=["Dst Port", "Protocol"] + FEAT76 + ["Label"], dtype=dtypes)
df = df.drop_duplicates().reset_index(drop=True)
y = df["Label"].map(CAT).astype(int).values
Xraw = df[FEAT76]
nun = Xraw.nunique(); FULL = Xraw.drop(columns=nun[nun <= 1].index.tolist())
corr = FULL.corr(); upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
BASE_COLS = [c for c in FULL.columns if not (upper[c] >= 0.95).any()]


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def derive(F):                                   # verbatim from v7.3f
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


def repair(F):                                   # verbatim from v7.3f
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
FEAT_ORDER = BASE_COLS + DER_COLS + list(SVC.columns)
REPAIRED = ["Down/Up Ratio", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Average Packet Size",
            "Packet Length Mean", "Flow Bytes/s", "Fwd Segment Size Avg", "Bwd Segment Size Avg"]
say("repaired columns that are MODEL INPUTS (in BASE): %s"
    % [c for c in REPAIRED if c in BASE_COLS])


def assemble(F, S):
    return pd.concat([F[BASE_COLS].astype(np.float32), derive(F)[DER_COLS], S], axis=1)[FEAT_ORDER]


X = assemble(FULL, SVC)
i_trf, i_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=RS, stratify=y)
X_trf = X.iloc[i_trf].reset_index(drop=True); y_arr, y_te = y[i_trf], y[i_test]
FULL_te = FULL.iloc[i_test].reset_index(drop=True); SVC_te = SVC.iloc[i_test].reset_index(drop=True)
IQR = (FULL.iloc[i_trf].quantile(0.75) - FULL.iloc[i_trf].quantile(0.25)).replace(0.0, 1.0)
smp = FULL.iloc[i_trf[:300_000]]
INT_COLS = [c for c in FULL.columns if np.all(np.abs(smp[c].values - np.round(smp[c].values)) < 1e-6)]
COSTLY_PREFIX = ("Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
                 "Total Length of Bwd Packet", "FIN Flag", "SYN Flag", "RST Flag", "PSH Flag",
                 "ACK Flag", "Fwd PSH", "Fwd URG", "Bwd URG", "URG Flag", "CWR Flag", "ECE Flag", "Bwd PSH")
del X, corr, upper, DER_ALL, df, Xraw, FULL; gc.collect()
assert len(y_arr) == 2085442 and len(y_te) == 521361
say("prep done %.1fm" % ((time.time() - t_start) / 60))


def make_xgb():
    return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
                         colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
                         n_jobs=-1, random_state=RS)


sub, _ = train_test_split(np.arange(len(y_arr)), train_size=GATE_TRAIN, random_state=RS,
                          stratify=(y_arr != 0).astype(int))
yg = (y_arr[sub] != 0).astype(int)
GATE = make_xgb().fit(X_trf.iloc[sub], yg, sample_weight=compute_sample_weight("balanced", yg))
att = y_arr != 0
fam = y_arr[att] - 1
n = np.bincount(fam, minlength=9); w = (n.sum() / n) ** 0.25; w = w / w.mean()
FAMM = make_xgb().fit(X_trf[att], fam, sample_weight=w[fam])
say("gate + family model (XGB a=0.25, 9-class) trained %.1fm" % ((time.time() - t_start) / 60))
del X_trf; gc.collect()

i_att = np.where(y_te != 0)[0]; i_ben = np.where(y_te == 0)[0]
pb = GATE.predict_proba(assemble(FULL_te.iloc[i_ben], SVC_te.iloc[i_ben]))[:, 1]
THR = float(np.quantile(pb, 1 - TARGET_FPR))
F0 = FULL_te.iloc[i_att].reset_index(drop=True); S0 = SVC_te.iloc[i_att].reset_index(drop=True)
fam_te = y_te[i_att] - 1


def noise(F):
    rs = np.random.RandomState(RS); F_ = F.copy()
    for c in F.columns:
        cap = BUDGET * IQR[c] * (0.5 if c.startswith(COSTLY_PREFIX) else 1.0)
        F_[c] = F[c].values + rs.uniform(-cap, cap, len(F)).astype(np.float32)
    return F_


def rnd(F):
    F = F.copy()
    for c in INT_COLS: F[c] = np.round(F[c].values)
    return F


ARMS = {"ORIGINAL": F0,
        "REPAIR ONLY": rnd(repair(rnd(F0))),
        "NOISE+REPAIR (v7.3)": rnd(repair(rnd(noise(F0)))),
        "NOISE ONLY": rnd(noise(F0))}
rows = []
for nm, F_ in ARMS.items():
    Xa = assemble(F_, S0)
    gr = float((GATE.predict_proba(Xa)[:, 1] > THR).mean())
    pf = FAMM.predict(Xa)
    rows.append(dict(arm=nm, gate_recall=round(gr, 4),
                     family_acc=round(float((pf == fam_te).mean()), 4),
                     family_macroF1=round(f1_score(fam_te, pf, average="macro"), 4)))
R = pd.DataFrame(rows)
say("\nthreshold %.4f (matched benign fpr %.4f); %d test attack rows; budget %.3f IQR"
    % (THR, TARGET_FPR, len(i_att), BUDGET))
say(R.to_string(index=False))
o, r_, nr = (R.set_index("arm").loc[k] for k in ("ORIGINAL", "REPAIR ONLY", "NOISE+REPAIR (v7.3)"))
for col in ("gate_recall", "family_macroF1"):
    tot = o[col] - nr[col]
    say("%s: total v7.3 drop %.4f | repair alone %.4f (%.0f%% of it)"
        % (col, tot, o[col] - r_[col], 100 * (o[col] - r_[col]) / tot if tot else float("nan")))
(OUT / "repair_confound.json").write_text(json.dumps(dict(rows=rows, threshold=THR, budget=BUDGET),
                                                     indent=2), encoding="utf-8")
say("total %.1f min" % ((time.time() - t_start) / 60))
