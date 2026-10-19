"""
Run the final v7.1b + XAI pipeline (final_model_v7_1_xai.ipynb, via v71b_core.run) on one dataset.

    python run_dataset.py {full|curated|unsw|cicids2017|toniot}

Loaders follow each dataset's established prep in this repo:
  full        final_model_v7_1_xai.ipynb, cell for cell; reuses the notebook's OOF caches, so this
              run must reproduce the notebook (the port's correctness check)
  curated     results/cic_unsw_svc/ (DERIVED + svc via the feature-hash port join, 101 features)
  unsw        final_model_v6_unsw.ipynb, MINUS the TTL columns (sttl, dttl, ct_state_ttl and the
              derived ttl_diff / r_ttl) - rejected as a testbed label proxy in results/ttl_join/
  cicids2017  results/cicids2017_v6/ (Web Attack collapsed, inf/NaN rows dropped) + svc from
              Destination Port; k-NN reference capped at 40,000 as there
  toniot      results/toniot_audit/ column typing: endpoint identity (src_ip, src_port, dst_ip)
              and high-cardinality text excluded; binary `label` dropped (it is the gate target)
Every loader dedupes on features + label before the split, and applies the same label-free prune
(zero variance, |r| >= 0.95).
"""
import sys, gc
import numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v71b_core

ROOT = Path(r"E:\AIUB\THESIS")
OUT = Path(__file__).parent
EPS = 1e-6
SVC_PORTS = [(20, "svc_ftpdata"), (21, "svc_ftp"), (22, "svc_ssh"), (23, "svc_telnet"),
             (25, "svc_smtp"), (53, "svc_dns"), (80, "svc_http"), (110, "svc_pop3"),
             (111, "svc_rpc"), (139, "svc_netbios"), (143, "svc_imap"), (179, "svc_bgp"),
             (443, "svc_https"), (445, "svc_smb"), (514, "svc_syslog"), (520, "svc_rip"),
             (1723, "svc_pptp"), (5190, "svc_aim"), (6881, "svc_bt"), (8080, "svc_httpalt")]
CIC_NAMES = ["Benign", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
             "Reconnaissance", "Shellcode", "Worms"]


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def prune(F):
    nun = F.nunique(); F = F.drop(columns=nun[nun <= 1].index.tolist())
    corr = F.corr(numeric_only=True)
    upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
    red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
    return F, F.drop(columns=red)


def drop_const(D):
    nun = D.nunique()
    return D.drop(columns=nun[nun <= 1].index.tolist())


def derive_cic(F):
    """The 31 engineered CICFlowMeter columns (final_model_v7_1_xai.ipynb 2.7)."""
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


def svc_block(port, proto=None, ambiguous=False):
    known = np.array([p for p, _ in SVC_PORTS])
    s = {nm: (port == p).astype(np.float32) for p, nm in SVC_PORTS}
    lo = 0 if ambiguous else -10**9
    s["svc_other_low"] = ((port >= lo) & (port < 1024) & ~np.isin(port, known)).astype(np.float32)
    s["svc_registered"] = ((port >= 1024) & (port < 49152) & ~np.isin(port, known)).astype(np.float32)
    s["svc_ephemeral"] = (port >= 49152).astype(np.float32)
    if ambiguous:
        s["svc_ambiguous"] = (port < 0).astype(np.float32)
    if proto is not None:
        s["proto_tcp"] = (proto == 6).astype(np.float32)
        s["proto_udp"] = (proto == 17).astype(np.float32)
    s["dst_port_num"] = port.astype(np.float32)
    return drop_const(pd.DataFrame(s))


def cic_group_of(svc_cols):
    """v7.1e partition (final_model_v7_1_xai.ipynb 5.5)."""
    def view_of(c):
        if c in svc_cols:
            return "SVC"
        lc = c.lower()
        if "flag" in lc or lc.startswith("frac_") or lc == "r_syn_fin":
            return "FLAG"
        if any(t in lc for t in ("iat", "duration", "active", "idle", "/s", "dur_per_pkt")):
            return "TIME"
        return "SIZE"
    return view_of, ["SVC", "TIME", "SIZE", "FLAG"]


# ==================================================================== loaders
def load_full():
    CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
    hdr = pd.read_csv(CFM, nrows=0).columns.tolist(); FEAT76 = hdr[7:-1]
    dt = {c: np.float32 for c in FEAT76}; dt["Dst Port"] = np.int32; dt["Protocol"] = np.int16
    df = pd.read_csv(CFM, usecols=["Dst Port", "Protocol"] + FEAT76 + ["Label"], dtype=dt)
    df = df.drop_duplicates().reset_index(drop=True)
    y = df["Label"].map({n: i for i, n in enumerate(CIC_NAMES)}).astype(int).values
    FULL, BASE = prune(df[FEAT76])
    DER = drop_const(derive_cic(FULL))
    SVC = svc_block(df["Dst Port"].values, df["Protocol"].values); SVC.index = df.index
    X = pd.concat([BASE.astype(np.float32), DER, SVC], axis=1)
    assert X.shape[1] == 99 and len(y) == 2606803
    gof, groups = cic_group_of(set(SVC.columns))
    return dict(X=X, y=y, names=CIC_NAMES, group_of=gof, groups=groups,
                cache_gate=ROOT / "cache_oof_v71_gateproba.npz",
                cache_pool=ROOT / "cache_oof_v71_reject.npz")


def load_curated():
    Xraw = pd.read_csv(ROOT / "Resource" / "CIC-UNSW-Data.csv")
    yraw = pd.read_csv(ROOT / "Resource" / "CIC-UNSW-Label.csv")["Label"]
    z = np.load(ROOT / "results" / "cic_unsw_svc" / "cache_identity.npz")
    cols = list(Xraw.columns)
    df = pd.concat([Xraw, yraw], axis=1); df["_port"] = z["port"]; df["_proto"] = z["proto"]
    df = df.drop_duplicates(subset=cols + ["Label"]).reset_index(drop=True)
    y = df["Label"].astype(int).values
    FULL, BASE = prune(df[cols].astype(np.float64))
    DER = drop_const(derive_cic(FULL))
    SVC = svc_block(df["_port"].values, df["_proto"].values, ambiguous=True); SVC.index = df.index
    X = pd.concat([BASE.astype(np.float32), DER, SVC], axis=1)
    gof, groups = cic_group_of(set(SVC.columns))
    return dict(X=X, y=y, names=CIC_NAMES, group_of=gof, groups=groups,
                cache_gate=OUT / "cache_curated_gate.npz", cache_pool=OUT / "cache_curated_pool.npz")


TTL = {"sttl", "dttl", "ct_state_ttl"}


def load_unsw():
    D = ROOT / "Resource" / "UNSW-NB15"
    raw = pd.concat([pd.read_csv(D / "UNSW_NB15_training-set.csv"),
                     pd.read_csv(D / "UNSW_NB15_testing-set.csv")], ignore_index=True)
    raw.columns = [c.strip().lstrip("\ufeff") for c in raw.columns]
    CAT = {"Normal": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
           "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
    y_all = raw.attack_cat.map(CAT).astype(int)
    X_all = raw.drop(columns=["id", "attack_cat", "label"] + sorted(TTL))
    oh = pd.get_dummies(X_all[["service", "state"]], prefix=["svc", "st"]).astype(np.int8)
    X_all = X_all.drop(columns=["proto", "service", "state"]).join(oh)
    X_all["proto_code"] = raw.proto.astype("category").cat.codes.astype(np.int16)
    comb = pd.concat([X_all, y_all.rename("y")], axis=1).drop_duplicates().reset_index(drop=True)
    X_all, y = comb.drop(columns="y"), comb["y"].values
    FULL, BASE = prune(X_all)
    F = FULL; g = lambda c: F[c] if c in F.columns else pd.Series(0.0, index=F.index)
    sp, dp, sb, db = g("spkts"), g("dpkts"), g("sbytes"), g("dbytes")
    d = {}
    d["r_pkt_sd"] = safe(sp, dp); d["r_byte_sd"] = safe(sb, db)
    d["asym_pkt"] = safe(sp - dp, sp + dp); d["asym_byte"] = safe(sb - db, sb + db)
    d["bpp_flow"] = safe(sb + db, sp + dp)
    d["r_mean_sd"] = safe(g("smean"), g("dmean")); d["r_load_sd"] = safe(g("sload"), g("dload"))
    d["r_inpkt_sd"] = safe(g("sinpkt"), g("dinpkt")); d["r_jit_sd"] = safe(g("sjit"), g("djit"))
    d["loss_rate_s"] = safe(g("sloss"), sp); d["loss_rate_d"] = safe(g("dloss"), dp)
    d["cv_jit_s"] = safe(g("sjit"), g("sinpkt")); d["cv_jit_d"] = safe(g("djit"), g("dinpkt"))
    rtt = g("tcprtt")
    d["synack_share"] = safe(g("synack"), rtt); d["ackdat_share"] = safe(g("ackdat"), rtt)
    d["r_win_sd"] = safe(g("swin"), g("dwin"))
    dur = g("dur")
    d["dur_per_pkt"] = safe(dur, sp + dp); d["rate_per_pkt"] = safe(g("rate"), sp + dp)
    d["resp_share"] = safe(g("response_body_len"), db)
    d["r_srv_src_ltm"] = safe(g("ct_srv_src"), g("ct_src_ltm"))
    d["r_srv_dst_ltm"] = safe(g("ct_srv_dst"), g("ct_dst_ltm"))
    d["r_dst_src_ltm"] = safe(g("ct_dst_src_ltm"), g("ct_src_ltm"))
    d["ctx_total"] = g("ct_srv_src") + g("ct_dst_ltm") + g("ct_src_ltm") + g("ct_srv_dst")
    d["r_sport_dport"] = safe(g("ct_src_dport_ltm"), g("ct_dst_sport_ltm"))
    DER = drop_const(pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0))
    X = pd.concat([BASE, DER], axis=1).astype(np.float32)
    assert not (set(X.columns) & TTL)
    CTX = {"r_srv_src_ltm", "r_srv_dst_ltm", "r_dst_src_ltm", "ctx_total", "r_sport_dport"}
    TIMEC = {"dur", "rate", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit", "tcprtt",
             "synack", "ackdat", "dur_per_pkt", "rate_per_pkt", "r_load_sd", "r_inpkt_sd",
             "r_jit_sd", "cv_jit_s", "cv_jit_d", "synack_share", "ackdat_share"}

    def gof(c):
        if c.startswith("svc_") or c == "proto_code": return "SVC"
        if c.startswith("st_"): return "FLAG"
        if c.startswith("ct_") or c in CTX: return "CTX"
        if c in TIMEC: return "TIME"
        return "SIZE"
    names = ["Benign"] + CIC_NAMES[1:]
    return dict(X=X, y=y, names=names, group_of=gof, groups=["SVC", "TIME", "SIZE", "FLAG", "CTX"],
                cache_gate=OUT / "cache_unsw_gate.npz", cache_pool=OUT / "cache_unsw_pool.npz")


def load_cicids2017():
    D = ROOT / "Resource" / "CIC-IDS2017"
    fr = []
    for f in sorted(D.glob("*.pcap_ISCX.csv")):
        d_ = pd.read_csv(f, low_memory=False, encoding="latin-1")
        d_.columns = [c.strip() for c in d_.columns]; fr.append(d_)
    df = pd.concat(fr, ignore_index=True); del fr; gc.collect()
    df["Label"] = df["Label"].astype(str).str.strip()
    df.loc[df["Label"].str.startswith("Web Attack"), "Label"] = "Web Attack"
    FEAT = [c for c in df.columns if c != "Label"]
    bad = ~np.isfinite(df[FEAT].to_numpy(dtype=np.float64)).all(axis=1)
    df = df.loc[~bad].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    atk = sorted([l for l in df["Label"].unique() if l != "BENIGN"], key=str.lower)
    names = ["BENIGN"] + atk
    y = df["Label"].map({n: i for i, n in enumerate(names)}).astype(int).values
    port = df["Destination Port"].values.astype(np.int64)
    raw = [c for c in FEAT if c != "Destination Port"]
    FULL, BASE = prune(df[raw].astype(np.float32))
    REN = {"Total Fwd Packets": "Total Fwd Packet", "Total Backward Packets": "Total Bwd packets",
           "Total Length of Fwd Packets": "Total Length of Fwd Packet",
           "Total Length of Bwd Packets": "Total Length of Bwd Packet",
           "Init_Win_bytes_forward": "FWD Init Win Bytes", "Init_Win_bytes_backward": "Bwd Init Win Bytes"}
    DER = drop_const(derive_cic(FULL.rename(columns=REN)))
    SVC = svc_block(port); SVC.index = df.index
    X = pd.concat([BASE.astype(np.float32), DER, SVC], axis=1)
    del df, FULL, BASE, DER; gc.collect()
    gof, groups = cic_group_of(set(SVC.columns))
    return dict(X=X, y=y, names=names, group_of=gof, groups=groups, ref_cap=40000,
                cache_gate=OUT / "cache_cicids2017_gate.npz", cache_pool=OUT / "cache_cicids2017_pool.npz")


def load_toniot():
    NA = ["-", "(empty)", "", " ", "NaN", "nan", "None", "null"]
    df = pd.read_csv(ROOT / "Resource" / "TON-IoT" / "Train_Test_Network.csv", na_values=NA,
                     keep_default_na=True, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    ytxt = df["type"].astype(str).str.strip().str.lower()
    feat = [c for c in df.columns if c not in ("type", "label")]
    ENDPOINT = {"src_ip", "dst_ip", "src_port", "ts"}
    num, cat = [], []
    for c in feat:
        if c in ENDPOINT:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        elif df[c].nunique(dropna=True) <= 50:
            cat.append(c)                                   # >50 unique = free text, excluded
    miss = df[num + cat].isna().mean()
    num = [c for c in num if miss[c] <= 0.99]; cat = [c for c in cat if miss[c] <= 0.99]
    Xn = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xc = pd.get_dummies(df[cat].astype(str).fillna("__NA__"), prefix=cat, dtype=np.float32)
    X_all = pd.concat([Xn.astype(np.float32), Xc], axis=1)
    comb = pd.concat([X_all, ytxt.rename("_y")], axis=1).drop_duplicates().reset_index(drop=True)
    atk = sorted(l for l in comb["_y"].unique() if l != "normal")
    names = ["normal"] + atk
    y = comb["_y"].map({n: i for i, n in enumerate(names)}).astype(int).values
    FULL, BASE = prune(comb.drop(columns="_y"))
    F = FULL; g = lambda c: F[c] if c in F.columns else pd.Series(0.0, index=F.index)
    sp, dp, sb, db = g("src_pkts"), g("dst_pkts"), g("src_bytes"), g("dst_bytes")
    d = {"r_pkt_sd": safe(sp, dp), "r_byte_sd": safe(sb, db), "asym_pkt": safe(sp - dp, sp + dp),
         "asym_byte": safe(sb - db, sb + db), "bpp_src": safe(sb, sp), "bpp_dst": safe(db, dp),
         "dur_per_pkt": safe(g("duration"), sp + dp)}
    DER = drop_const(pd.DataFrame(d, index=F.index).replace([np.inf, -np.inf], 0.0).fillna(0.0))
    X = pd.concat([BASE, DER], axis=1).astype(np.float32)

    def gof(c):
        lc = c.lower()
        if lc.startswith(("proto", "service")) or lc == "dst_port": return "SVC"
        if lc.startswith("conn_state"): return "FLAG"
        if lc.startswith(("dns_", "ssl_", "http_", "weird_")): return "APP"
        if lc in ("duration", "dur_per_pkt"): return "TIME"
        return "SIZE"
    return dict(X=X, y=y, names=names, group_of=gof, groups=["SVC", "TIME", "SIZE", "FLAG", "APP"],
                cache_gate=OUT / "cache_toniot_gate.npz", cache_pool=OUT / "cache_toniot_pool.npz")


LOADERS = dict(full=load_full, curated=load_curated, unsw=load_unsw,
               cicids2017=load_cicids2017, toniot=load_toniot)

if __name__ == "__main__":
    name = sys.argv[1]
    a = LOADERS[name]()
    v71b_core.run(name, a.pop("X"), a.pop("y"), a.pop("names"), a.pop("group_of"), a.pop("groups"),
                  OUT, **a)
