"""
final_model_v6_unsw.ipynb, re-run with the dedupe step TOGGLED.

Purpose: quantify how much of a reported UNSW-NB15 number is duplicate leakage.
This is the empirical backing for literature_artifact_audit.md's claim that
deduplication moves reported accuracy by 10-15 points.

Arm A (DEDUPE=True)  = the published v6-on-UNSW result, 0.6387 / 0.8303.
Arm B (DEDUPE=False) = what a paper that skips dedupe would report.

Everything else -- seeds, splits, prune rule, derived features, 7-member pool,
combiner grid, SELV protocol -- is byte-identical to the notebook.

ASCII-only prints (terminal is cp1252); full report also written to UTF-8 file.
"""
import sys, io, json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
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

ROOT = Path(r"E:\AIUB\THESIS")
OUT = Path(__file__).parent
LABELS = {0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
          5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms"}
NAMES = [LABELS[i] for i in range(10)]
RS, EPS = 42, 1e-6
LOG = []


def say(s=""):
    print(s)
    sys.stdout.flush()
    LOG.append(s)


def safe(a, b):
    return (a / (b + EPS)).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def build_derived(F):
    d = {}
    g = lambda c: F[c] if c in F.columns else pd.Series(0.0, index=F.index)
    sp, dp = g("spkts"), g("dpkts")
    sb, db = g("sbytes"), g("dbytes")
    tot_p, tot_b = sp + dp, sb + db
    d["r_pkt_sd"] = safe(sp, dp);          d["r_byte_sd"] = safe(sb, db)
    d["asym_pkt"] = safe(sp - dp, tot_p);  d["asym_byte"] = safe(sb - db, tot_b)
    d["bpp_flow"] = safe(tot_b, tot_p)
    d["r_mean_sd"] = safe(g("smean"), g("dmean"))
    d["r_load_sd"] = safe(g("sload"), g("dload"))
    d["r_inpkt_sd"] = safe(g("sinpkt"), g("dinpkt"))
    d["r_jit_sd"] = safe(g("sjit"), g("djit"))
    d["loss_rate_s"] = safe(g("sloss"), sp); d["loss_rate_d"] = safe(g("dloss"), dp)
    d["cv_jit_s"] = safe(g("sjit"), g("sinpkt")); d["cv_jit_d"] = safe(g("djit"), g("dinpkt"))
    d["ttl_diff"] = g("sttl") - g("dttl"); d["r_ttl"] = safe(g("sttl"), g("dttl"))
    rtt = g("tcprtt")
    d["synack_share"] = safe(g("synack"), rtt); d["ackdat_share"] = safe(g("ackdat"), rtt)
    d["r_win_sd"] = safe(g("swin"), g("dwin"))
    dur = g("dur")
    d["dur_per_pkt"] = safe(dur, tot_p); d["rate_per_pkt"] = safe(g("rate"), tot_p)
    d["resp_share"] = safe(g("response_body_len"), db)
    d["r_srv_src_ltm"] = safe(g("ct_srv_src"), g("ct_src_ltm"))
    d["r_srv_dst_ltm"] = safe(g("ct_srv_dst"), g("ct_dst_ltm"))
    d["r_dst_src_ltm"] = safe(g("ct_dst_src_ltm"), g("ct_src_ltm"))
    d["ctx_total"] = (g("ct_srv_src") + g("ct_dst_ltm") + g("ct_src_ltm") + g("ct_srv_dst"))
    d["r_sport_dport"] = safe(g("ct_src_dport_ltm"), g("ct_dst_sport_ltm"))
    O = pd.DataFrame(d, index=F.index).astype(np.float64)
    return O.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def run_arm(DEDUPE):
    tag = "DEDUPE" if DEDUPE else "NO-DEDUPE"
    say("\n" + "=" * 78)
    say("ARM: %s" % tag)
    say("=" * 78)

    D = ROOT / "Resource" / "UNSW-NB15"
    raw = pd.concat([pd.read_csv(D / "UNSW_NB15_training-set.csv"),
                     pd.read_csv(D / "UNSW_NB15_testing-set.csv")], ignore_index=True)
    raw.columns = [c.strip().lstrip("\ufeff") for c in raw.columns]

    CAT = {"Normal": 0, "Analysis": 1, "Backdoor": 2, "DoS": 3, "Exploits": 4,
           "Fuzzers": 5, "Generic": 6, "Reconnaissance": 7, "Shellcode": 8, "Worms": 9}
    y_all = raw.attack_cat.map(CAT).astype(int)
    X_all = raw.drop(columns=["id", "attack_cat", "label"])
    CATCOLS = ["proto", "service", "state"]
    oh = pd.get_dummies(X_all[["service", "state"]], prefix=["svc", "st"]).astype(np.int8)
    X_all = X_all.drop(columns=CATCOLS).join(oh)
    X_all["proto_code"] = raw.proto.astype("category").cat.codes.astype(np.int16)

    comb = pd.concat([X_all, y_all.rename("y")], axis=1)
    n_pooled = len(comb)
    if DEDUPE:
        comb = comb.drop_duplicates()
    comb = comb.reset_index(drop=True)
    X_all, y = comb.drop(columns="y"), comb["y"]
    say("pooled %d -> %d rows (%s)" % (n_pooled, len(comb), tag))

    # --- unsupervised prune, identical rule ---
    nun = X_all.nunique(); zero = nun[nun <= 1].index.tolist()
    FULL = X_all.drop(columns=zero)
    corr = FULL.corr(numeric_only=True)
    upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
    red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
    BASE = FULL.drop(columns=red)

    DER = build_derived(FULL)
    dz = DER.nunique(); DER = DER.drop(columns=dz[dz <= 1].index.tolist())
    X = pd.concat([BASE, DER], axis=1)
    DERIVED_NAMES = set(DER.columns)
    CTX = [c for c in X.columns if c.startswith("ct_") or c in
           ("r_srv_src_ltm", "r_srv_dst_ltm", "r_dst_src_ltm", "ctx_total", "r_sport_dport")]
    say("prune: FULL %d -> BASE %d ; DERIVED space %d features" % (FULL.shape[1], BASE.shape[1], X.shape[1]))
    say("  zero-var dropped (%d): %s" % (len(zero), ", ".join(zero) or "none"))
    say("  |r|>=0.95 dropped (%d): %s" % (len(red), ", ".join(red) or "none"))

    X_trf, X_test, y_trf, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y)

    # --- LEAKAGE DIAGNOSTIC: how many test rows are exact copies of a train row? ---
    key_tr = pd.util.hash_pandas_object(
        pd.concat([X_trf, y_trf.rename("y")], axis=1), index=False)
    key_te = pd.util.hash_pandas_object(
        pd.concat([X_test, y_test.rename("y")], axis=1), index=False)
    seen = set(key_tr.values)
    leak_mask = np.array([k in seen for k in key_te.values])
    leak_rate = float(leak_mask.mean())
    say("LEAKAGE: %d of %d test rows (%.2f%%) have an exact (features+label) twin in train"
        % (leak_mask.sum(), len(leak_mask), 100 * leak_rate))
    per_cls_leak = pd.Series(leak_mask).groupby(y_test.values).mean()

    X_trf = X_trf.reset_index(drop=True); y_trf = y_trf.reset_index(drop=True)
    y_arr = y_trf.values

    attack_labels = sorted(l for l in y.unique() if l != 0)
    fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
    idx_to_fam = {i: f for f, i in fam_to_idx.items()}
    NC = len(attack_labels); FAM = [LABELS[idx_to_fam[i]] for i in range(NC)]
    att_trf = y_arr != 0
    fam_trf = np.full(len(y_arr), -1)
    fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
    y_te = y_test.values
    att_te = y_te != 0
    fam_te = pd.Series(y_te[att_te]).map(fam_to_idx).values
    say("train+val %d (%d attack) | test %d (%d attack)" % (len(y_arr), att_trf.sum(), len(y_te), att_te.sum()))

    # --- Cover-Hart bracket ---
    def cover_hart(e1nn, c):
        inner = 1 - (c / (c - 1)) * e1nn
        return np.nan if inner < 0 else (c - 1) / c * (1 - np.sqrt(inner))

    Xa_tr = X_trf[att_trf]; Xa_te = X_test[att_te]
    sc = RobustScaler().fit(Xa_tr)
    Ztr, Zte = sc.transform(Xa_tr), sc.transform(Xa_te)
    fam_tr_a = fam_trf[att_trf]
    nn1 = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=-1).fit(Ztr)
    _, ii1 = nn1.kneighbors(Zte)
    e1 = float((fam_tr_a[ii1[:, 0]] != fam_te).mean())
    say("e_1NN %.4f  ->  e* bracket [%.4f, %.4f]  (n_ref=%d)"
        % (e1, cover_hart(e1, NC), e1, len(Ztr)))

    # --- v6 architecture, unchanged ---
    def make_xgb(**kw):
        p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
                 colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
                 n_jobs=-1, random_state=RS)
        p.update(kw); return XGBClassifier(**p)

    wc_alpha = lambda n, a: ((n.sum() / n) ** a) / (((n.sum() / n) ** a).mean())

    def sample_w(ya, wc):
        w = wc[ya].astype(float); return w * len(w) / w.sum()

    SPECS = ([("XGB a=%s" % a, ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
             + [("XGB smote2000", ("smote", 2000)),
                ("RF a=0.25", ("rf", 0.25)), ("ET a=0.25", ("et", 0.25))])
    MODELS = [s[0] for s in SPECS]; M = len(MODELS); I25 = MODELS.index("XGB a=0.25")

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
        m = make_xgb(); m.fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb)); return m

    CACHE = OUT / ("cache_oof_unsw_%s.npz" % ("dedup" if DEDUPE else "nodedup"))
    if CACHE.exists():
        z = np.load(CACHE); OOF = [z["m%d" % m] for m in range(M)]; oof_gate = z["gate"]
        say("loaded %s" % CACHE.name)
    else:
        OOF = [np.zeros((len(y_arr), NC), np.float32) for _ in range(M)]
        oof_gate = np.zeros(len(y_arr), int)
        t0 = time.time()
        for f_, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr)):
            tf = time.time()
            oof_gate[va] = fit_gate(X_trf.iloc[tr], y_arr[tr]).predict(X_trf.iloc[va])
            ma = y_arr[tr] != 0
            Xa = X_trf.iloc[tr][ma].reset_index(drop=True); ya = pd.Series(fam_trf[tr][ma])
            for mi, (_, (k, p)) in enumerate(SPECS):
                OOF[mi][va] = fit_member(k, p, Xa, ya).predict_proba(X_trf.iloc[va]).astype(np.float32)
            say("  fold %d/5 (%.0fs)" % (f_ + 1, time.time() - tf))
        say("cross-fit %.0fs" % (time.time() - t0))
        np.savez_compressed(CACHE, gate=oof_gate, **{"m%d" % m: OOF[m] for m in range(M)})

    yb_tr = (y_arr != 0).astype(int)
    gate_r, gate_p = recall_score(yb_tr, oof_gate), precision_score(yb_tr, oof_gate)
    say("OOF gate: attack recall %.4f / precision %.4f" % (gate_r, gate_p))

    comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                          random_state=RS + 1, stratify=y_arr)
    ref_idx = comb_idx[att_trf[comb_idx]]
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
    say("W fitted in %.0fs | reference %d attack rows" % (time.time() - t0, len(ref_idx)))

    def score10(fam_g, ye, gmask, t):
        yp = np.zeros(len(ye), int)
        if len(fam_g): yp[gmask] = np.vectorize(idx_to_fam.get, otypes=[int])(fam_g)
        return dict(variant=t,
                    mF1=round(f1_score(ye, yp, average="macro", zero_division=0), 4),
                    mP=round(precision_score(ye, yp, average="macro", zero_division=0), 4),
                    mR=round(recall_score(ye, yp, average="macro", zero_division=0), 4),
                    acc=round(float((yp == ye).mean()), 4))

    def run_all(P_g, idx, ye, gmask):
        out = [score10(fuse_static(P_g, W_class), ye, gmask, "STATIC: per-class W"),
               score10(fuse_static(P_g, np.ones((M, NC)) / M), ye, gmask, "STATIC: equal average")]
        for m in range(M):
            out.append(score10(P_g[m].argmax(1), ye, gmask, "single: " + MODELS[m]))
        for k in (30, 100):
            for p in (1, 3, 8):
                out.append(score10(fuse_dyn(P_g, comp_ola(idx, k), p=p), ye, gmask,
                                   "DYN OLA-cw k=%d p=%d" % (k, p)))
        for k in (30, 100):
            out.append(score10(fuse_dyn(P_g, comp_ola(idx, k), p=8, Wclass=W_class), ye, gmask,
                               "HYBRID W x OLA-cw k=%d p=8" % k))
        return pd.DataFrame(out)

    gmask_selv = oof_gate[selv_idx] == 1
    gpos = selv_idx[gmask_selv]
    P_selv = [OOF[m][gpos] for m in range(M)]
    _, i_selv = NN_D.kneighbors(prof(P_selv))
    selv_tab = run_all(P_selv, i_selv, y_arr[selv_idx], gmask_selv).sort_values(
        "mF1", ascending=False).reset_index(drop=True)
    BESTV = selv_tab.iloc[0]["variant"]
    say("SELV winner: %s (%.4f)" % (BESTV, selv_tab.iloc[0]["mF1"]))

    t0 = time.time()
    gate_f = fit_gate(X_trf, y_arr)
    Xa_f = X_trf[att_trf].reset_index(drop=True); ya_f = pd.Series(fam_trf[att_trf])
    FINAL = [fit_member(k, p, Xa_f, ya_f) for _, (k, p) in SPECS]
    gmask_test = gate_f.predict(X_test) == 1
    P_test = [m.predict_proba(X_test.iloc[np.where(gmask_test)[0]]).astype(np.float32) for m in FINAL]
    _, i_test = NN_D.kneighbors(prof(P_test))
    say("refit + test in %.0fs" % (time.time() - t0))

    test_tab = run_all(P_test, i_test, y_te, gmask_test).sort_values(
        "mF1", ascending=False).reset_index(drop=True)
    say("\n--- TEST TABLE (%s) ---" % tag)
    say(test_tab.to_string(index=False))
    sel = test_tab[test_tab.variant == BESTV].iloc[0]
    rank = 1 + list(test_tab.variant).index(BESTV)
    say("\nSELV-selected '%s' ranks %d of %d on test" % (BESTV, rank, len(test_tab)))
    say("%s: macro-F1 %.4f | accuracy %.4f" % (tag, sel["mF1"], sel["acc"]))

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
    yp[gmask_test] = np.vectorize(idx_to_fam.get, otypes=[int])(fam_pred)
    say("\nper-class report (%s)" % tag)
    say(classification_report(y_te, yp, labels=list(range(10)), target_names=NAMES,
                              digits=3, zero_division=0))

    f1u = f1_score(y_te, yp, average=None, labels=range(10), zero_division=0)

    # accuracy split by whether the test row was leaked
    acc_leak = float((yp == y_te)[leak_mask].mean()) if leak_mask.any() else float("nan")
    acc_clean = float((yp == y_te)[~leak_mask].mean()) if (~leak_mask).any() else float("nan")
    say("accuracy on LEAKED test rows  : %.4f  (n=%d)" % (acc_leak, leak_mask.sum()))
    say("accuracy on CLEAN  test rows  : %.4f  (n=%d)" % (acc_clean, (~leak_mask).sum()))

    imp = pd.Series(FINAL[I25].feature_importances_, index=X.columns).sort_values(ascending=False)
    ctx_share = imp[[c for c in imp.index if c in CTX]].sum() / imp.sum()

    gate_test_r = recall_score((y_te != 0).astype(int), gmask_test.astype(int))
    gate_test_p = precision_score((y_te != 0).astype(int), gmask_test.astype(int))

    return dict(
        arm=tag, dedupe=DEDUPE, n_rows=int(len(comb)), n_pooled=int(n_pooled),
        n_features=int(X.shape[1]), n_base=int(BASE.shape[1]),
        zero_var=zero, corr_dropped=red,
        n_train=int(len(y_arr)), n_test=int(len(y_te)),
        leak_rate=leak_rate, n_leaked=int(leak_mask.sum()),
        per_class_leak={NAMES[k]: round(float(v), 4) for k, v in per_cls_leak.items()},
        e_1nn=round(e1, 4), e_star_lo=round(float(cover_hart(e1, NC)), 4),
        selv_winner=BESTV, test_rank=rank,
        mF1=float(sel["mF1"]), acc=float(sel["acc"]),
        mP=float(sel["mP"]), mR=float(sel["mR"]),
        per_class_f1={NAMES[i]: round(float(f1u[i]), 4) for i in range(10)},
        gate_oof_recall=round(float(gate_r), 4), gate_oof_precision=round(float(gate_p), 4),
        gate_test_recall=round(float(gate_test_r), 4), gate_test_precision=round(float(gate_test_p), 4),
        acc_leaked_rows=round(acc_leak, 4), acc_clean_rows=round(acc_clean, 4),
        ctx_importance_share=round(float(ctx_share), 4),
        test_table=test_tab.to_dict("records"),
    )


if __name__ == "__main__":
    res = {}
    for dd in (True, False):
        res["dedup" if dd else "nodedup"] = run_arm(dd)

    A, B = res["dedup"], res["nodedup"]
    say("\n" + "=" * 78)
    say("SIDE BY SIDE")
    say("=" * 78)
    say("%-28s %14s %14s %10s" % ("", "DEDUPE", "NO-DEDUPE", "delta"))
    for k, lab in (("n_rows", "rows used"), ("n_features", "features"),
                   ("n_test", "test rows"), ("mF1", "macro-F1"), ("acc", "accuracy"),
                   ("mP", "macro-P"), ("mR", "macro-R"), ("e_1nn", "e_1NN"),
                   ("gate_test_recall", "gate recall"), ("gate_test_precision", "gate precision"),
                   ("leak_rate", "test rows leaked")):
        a, b = A[k], B[k]
        d = b - a
        fmt = "%14.4f" if isinstance(a, float) else "%14d"
        say(("%-28s " + fmt + " " + fmt + " %+10.4f") % (lab, a, b, d))

    say("\nper-class F1")
    say("%-16s %10s %10s %10s" % ("family", "DEDUPE", "NO-DEDUP", "delta"))
    for n in NAMES:
        a, b = A["per_class_f1"][n], B["per_class_f1"][n]
        say("%-16s %10.4f %10.4f %+10.4f" % (n, a, b, b - a))

    say("\nleakage rate by class (NO-DEDUPE arm)")
    for n, v in B["per_class_leak"].items():
        say("  %-16s %6.2f%%" % (n, 100 * v))

    (OUT / "unsw_dedupe_ablation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    io.open(OUT / "unsw_dedupe_ablation.txt", "w", encoding="utf-8").write("\n".join(LOG))
    say("\nwrote unsw_dedupe_ablation.json / .txt")
