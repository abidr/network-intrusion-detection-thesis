"""
The v7.1b pipeline of final_model_v7_1_xai.ipynb as one dataset-agnostic function, plus the XAI
verification and group shares. Steps and seeds follow the notebook cell for cell:

  4.1 gate cross-fit (5-fold OOF)        4.2 reject class = benign OOF-gate false positives
  4.3 10-class pool cross-fit            4.4 W under Stage-B and end-to-end objectives
  4.5 21 rules on SELV + gate threshold  4.6 refit on train+val      5.1 test once
  5.5-5.7 XAI: additivity, 2nd implementation, linear rewrite, exact group Shapley, shares

Generalisations, each a no-op on the full CIC-UNSW release (verified there by reproducing the
notebook from its own caches):
  - K classes instead of 10 (label 0 = benign, attacks 1..K-1);
  - SMOTE k_neighbors clamped to min(5, smallest class - 1) (CIC-IDS2017 Heartbleed);
  - optional cap on the k-NN competence reference set (stratified), as cicids2017_v6 did;
  - G feature groups instead of 4 (2^G coalitions, still exact);
  - SHAP background = up to 40 Stage-B training rows, stratified over the K+1 head classes.
ASCII-only prints; log written UTF-8.
"""
import io, json, sys, time, gc
import numpy as np, pandas as pd
from pathlib import Path
from math import factorial
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

RS = 42
GRID_W = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SPECS = ([("XGB a=%s" % a, ("alpha", a)) for a in (0.0, 0.25, 0.5, 1.0)]
         + [("XGB smote2000", ("smote", 2000)),
            ("RF a=0.25", ("rf", 0.25)),
            ("ET a=0.25", ("et", 0.25))])
MODELS = [s_[0] for s_ in SPECS]
M = len(SPECS)
XGB_MEMBERS = [i for i, (_, (k, _)) in enumerate(SPECS) if k in ("alpha", "smote")]
IDX_A25 = MODELS.index("XGB a=0.25")


def make_xgb(**kw):
    p = dict(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9,
             colsample_bytree=0.9, tree_method="hist", eval_metric="mlogloss",
             n_jobs=-1, random_state=RS)
    p.update(kw)
    return XGBClassifier(**p)


def wc_alpha(n, a):
    w = (n.sum() / np.maximum(n, 1)) ** a
    return w / w.mean()


def sample_w(ya, wc):
    w = wc[np.asarray(ya)].astype(float)
    return w * len(w) / w.sum()


def fit_member(kind, param, Xa, ya, ncls):
    n = np.bincount(np.asarray(ya), minlength=ncls)
    if kind == "smote":
        strat = {c: param for c in range(ncls) if 1 < n[c] < param}
        if strat:
            kn = min(5, min(n[c] for c in strat) - 1)
            Xa, ya = SMOTE(random_state=RS, sampling_strategy=strat,
                           k_neighbors=kn).fit_resample(Xa, ya)
        m = make_xgb(); m.fit(Xa, ya); return m
    if kind == "alpha":  m = make_xgb()
    elif kind == "rf":   m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    elif kind == "et":   m = ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=RS)
    m.fit(Xa, ya, sample_weight=sample_w(ya, wc_alpha(n, param)))
    return m


def fit_gate(Xa, ya):
    yb = (np.asarray(ya) != 0).astype(int)
    gm = make_xgb()
    gm.fit(Xa, yb, sample_weight=compute_sample_weight("balanced", yb))
    return gm


def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)).mean()


prof = lambda ps: np.hstack(ps).astype(np.float32)


def fuse(P_g, Wd=None, p=1.0, Wclass=None):
    if Wd is not None:
        Wp = np.power(np.clip(Wd, 1e-6, None), p)
        Wp = Wp / Wp.sum(1, keepdims=True)
    S = np.zeros(P_g[0].shape, np.float64)
    for m in range(M):
        coef = 1.0 if Wd is None else Wp[:, m:m + 1]
        if Wclass is not None:
            coef = coef * Wclass[m][None, :]
        S += coef * P_g[m]
    return S


def rule_names():
    out = ["STATIC: equal average", "STATIC: per-class W"]
    out += ["single: " + m for m in MODELS]
    out += ["DYN OLA-cw k=%d p=%d" % (k, p) for k in (30, 100) for p in (1, 3, 8)]
    out += ["DYN LCA k=30"]
    out += ["HYBRID W x OLA-cw k=%d p=%d" % (k, p) for k in (30, 100) for p in (1, 3)]
    out += ["HYBRID W x LCA k=30 p=1"]
    return out


def run(tag, X, y, names, group_of, groups, out_dir, cache_gate=None, cache_pool=None,
        ref_cap=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    LOG = []

    def say(s=""):
        print(s); sys.stdout.flush(); LOG.append(str(s))
        io.open(out_dir / ("%s.txt" % tag), "w", encoding="utf-8").write("\n".join(LOG))

    t_start = time.time()
    y = np.asarray(y).astype(int)
    K = len(names)
    COLS = list(X.columns)
    assert set(np.unique(y)) == set(range(K)), "labels must be 0..K-1"
    say("=" * 78); say("v7.1b + XAI on %s" % tag); say("=" * 78)
    say("rows %d | features %d | classes %d" % (len(y), len(COLS), K))
    vc = np.bincount(y, minlength=K)
    for k in range(K):
        say("  %2d %-22s %9d  %7.3f%%" % (k, names[k], vc[k], 100 * vc[k] / len(y)))

    # ------------------------------------------------------------ 2.9 / 2.10 splits
    i_trf, i_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=RS, stratify=y)
    X_trf = X.iloc[i_trf].reset_index(drop=True)
    X_test = X.iloc[i_test].reset_index(drop=True)
    y_arr, y_te = y[i_trf], y[i_test]
    att_trf = y_arr != 0
    attack_labels = sorted(l for l in np.unique(y_arr) if l != 0)
    fam_to_idx = {f: i for i, f in enumerate(attack_labels)}
    NC = len(attack_labels); NCR, IREJ = NC + 1, NC
    fam_trf = np.full(len(y_arr), -1)
    fam_trf[att_trf] = pd.Series(y_arr[att_trf]).map(fam_to_idx).values
    comb_idx, selv_idx = train_test_split(np.arange(len(y_arr)), test_size=0.5,
                                          random_state=RS + 1, stratify=y_arr)
    in_comb = np.zeros(len(y_arr), bool); in_comb[comb_idx] = True
    folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X_trf, y_arr))
    say("\ntrain+val %d | test %d | COMB %d | SELV %d"
        % (len(y_arr), len(y_te), len(comb_idx), len(selv_idx)))

    # ------------------------------------------------------------ 4.1 gate cross-fit
    if cache_gate and Path(cache_gate).exists():
        p_gate = np.load(cache_gate)["p"]; say("loaded %s" % cache_gate)
    else:
        p_gate = np.zeros(len(y_arr), np.float32); t0 = time.time()
        for f_, (tr, va) in enumerate(folds):
            gm = fit_gate(X_trf.iloc[tr], y_arr[tr])
            p_gate[va] = gm.predict_proba(X_trf.iloc[va])[:, 1].astype(np.float32)
            del gm; gc.collect()
            say("  gate fold %d/5 (%.0fs)" % (f_ + 1, time.time() - t0))
        if cache_gate: np.savez_compressed(cache_gate, p=p_gate)
    oof_gate = (p_gate > 0.5).astype(np.int8)

    # ------------------------------------------------------------ 4.2 reject class
    REJECT = (y_arr == 0) & (oof_gate == 1)
    yr10 = np.where(REJECT, IREJ, fam_trf)
    LAB10 = np.array(attack_labels + [0])
    say("reject set %d | Stage-B rows %d" % (REJECT.sum(), (yr10 >= 0).sum()))

    # ------------------------------------------------------------ 4.3 pool cross-fit
    if cache_pool and Path(cache_pool).exists():
        zr = np.load(cache_pool); OOF10 = [zr["m%d" % m] for m in range(M)]
        say("loaded %s" % cache_pool)
    else:
        OOF10 = [np.zeros((len(y_arr), NCR), np.float32) for _ in range(M)]; t0 = time.time()
        for f_, (tr, va) in enumerate(folds):
            use = yr10[tr] >= 0
            Xa = X_trf.iloc[tr[use]].reset_index(drop=True)
            ya = pd.Series(yr10[tr[use]])
            for mi, (_, (kind, param)) in enumerate(SPECS):
                mdl = fit_member(kind, param, Xa, ya, NCR)
                OOF10[mi][va] = mdl.predict_proba(X_trf.iloc[va]).astype(np.float32)
                del mdl; gc.collect()
            say("  pool fold %d/5 (elapsed %.1fm)" % (f_ + 1, (time.time() - t0) / 60))
        if cache_pool: np.savez_compressed(cache_pool, **{"m%d" % m: OOF10[m] for m in range(M)})

    # ------------------------------------------------------------ 4.4 combiner
    ref = np.where(in_comb & (yr10 >= 0))[0]
    y_ref = yr10[ref]
    P_ref = [OOF10[m][ref] for m in range(M)]
    nn_sel = np.arange(len(ref))
    if ref_cap and len(ref) > ref_cap:
        try:
            nn_sel = train_test_split(nn_sel, train_size=ref_cap, random_state=RS, stratify=y_ref)[0]
        except ValueError:
            nn_sel = np.random.RandomState(RS).choice(nn_sel, ref_cap, replace=False)
        say("k-NN competence reference capped %d -> %d (stratified)" % (len(ref), len(nn_sel)))
    nn_y = y_ref[nn_sel]
    CORRECT = np.stack([(P_ref[m][nn_sel].argmax(1) == nn_y) for m in range(M)])
    WC = wc_alpha(np.bincount(nn_y, minlength=NCR), 0.25)
    NN = NearestNeighbors(n_neighbors=100, algorithm="brute", n_jobs=-1).fit(
        prof([P_ref[m][nn_sel] for m in range(M)]))

    def comp_ola(idx, k):
        nb = idx[:, :k]; wn = WC[nn_y[nb]]; den = wn.sum(1)
        Wd = np.empty((nb.shape[0], M))
        for m in range(M):
            Wd[:, m] = (CORRECT[m][nb] * wn).sum(1) / den
        return Wd

    def comp_lca(P_g, idx, k):
        nb = idx[:, :k]; yn = nn_y[nb]
        Wd = np.empty((nb.shape[0], M))
        for m in range(M):
            pq = P_g[m].argmax(1); mask = yn == pq[:, None]; cnt = mask.sum(1)
            Wd[:, m] = np.where(cnt > 0, (CORRECT[m][nb] & mask).sum(1) / np.maximum(cnt, 1),
                                CORRECT[m][nb].mean(1))
        return Wd

    def fit_W(score_fn, W0=None):
        W = (np.ones((M, NCR)) / M) if W0 is None else W0.copy()
        best = score_fn(W)
        for _ in range(3):
            for c in range(NCR):
                for m_ in range(M):
                    cur = W[m_, c]
                    for gg in GRID_W:
                        W[m_, c] = gg
                        if W[:, c].sum() == 0: continue
                        s_ = score_fn(W)
                        if s_ > best + 1e-7: best, cur = s_, gg
                    W[m_, c] = cur
        return W, best

    t0 = time.time()
    W_sb, _ = fit_W(lambda W: f1_score(y_ref, fuse(P_ref, Wclass=W).argmax(1),
                                       average="macro", zero_division=0))
    gpos_comb = comb_idx[p_gate[comb_idx] > 0.5]
    yt_gpos = y_arr[gpos_comb]
    P_GPOS = [OOF10[m][gpos_comb] for m in range(M)]
    BG = np.zeros((K, K), np.int64)
    for c, cnt in zip(*np.unique(y_arr[comb_idx[p_gate[comb_idx] <= 0.5]], return_counts=True)):
        BG[c, 0] = cnt

    def e2e_fn(W):
        yp = LAB10[fuse(P_GPOS, Wclass=W).argmax(1)]
        return macro_f1_from_cm(np.bincount(yt_gpos * K + yp, minlength=K * K).reshape(K, K) + BG)

    W_u, e_u = fit_W(e2e_fn)
    W_w, e_w = fit_W(e2e_fn, W0=W_sb)
    W_e2e = W_w if e_w >= e_u else W_u
    say("W fitted (%.0fs): COMB e2e mF1 %.4f" % (time.time() - t0, max(e_u, e_w)))

    def rule_scores(P_g, idx, Wclass, name):
        if name.startswith("single: "):
            return P_g[MODELS.index(name[8:])]
        if name == "STATIC: equal average":
            return fuse(P_g, Wclass=np.ones((M, NCR)) / M)
        if name == "STATIC: per-class W":
            return fuse(P_g, Wclass=Wclass)
        k = int(name.split("k=")[1].split()[0])
        p = float(name.split("p=")[1]) if "p=" in name else 1.0
        Wd = comp_lca(P_g, idx, k) if "LCA" in name else comp_ola(idx, k)
        return fuse(P_g, Wd=Wd, p=p, Wclass=(Wclass if name.startswith("HYBRID") else None))

    def rule_coef(P_g, idx, Wclass, name):
        n, ncl = P_g[0].shape
        if name.startswith("single: "):
            cf = np.zeros((n, M, ncl)); cf[:, MODELS.index(name[8:]), :] = 1.0; return cf
        if name == "STATIC: equal average":
            return np.full((n, M, ncl), 1.0 / M)
        if name == "STATIC: per-class W":
            return np.broadcast_to(Wclass[None], (n, M, ncl)).copy()
        k = int(name.split("k=")[1].split()[0])
        p = float(name.split("p=")[1]) if "p=" in name else 1.0
        Wd = comp_lca(P_g, idx, k) if "LCA" in name else comp_ola(idx, k)
        Wp = np.power(np.clip(Wd, 1e-6, None), p); Wp = Wp / Wp.sum(1, keepdims=True)
        cf = np.repeat(Wp[:, :, None], ncl, axis=2)
        return cf * Wclass[None] if name.startswith("HYBRID") else cf

    def score_e2e(fam_g, ye, gmask):
        yp = np.zeros(len(ye), int)
        if len(fam_g): yp[gmask] = LAB10[fam_g]
        return f1_score(ye, yp, average="macro", zero_division=0)

    def selv_at(t):
        gmask = p_gate[selv_idx] > t
        gpos = selv_idx[gmask]
        P = [OOF10[m][gpos] for m in range(M)]
        idx = NN.kneighbors(prof(P))[1] if len(gpos) else np.zeros((0, 100), int)
        return P, idx, gmask

    # ------------------------------------------------------------ 4.5 SELV selection
    P, idx, gmask = selv_at(0.5)
    tab = sorted(((score_e2e(rule_scores(P, idx, W_e2e, r).argmax(1), y_arr[selv_idx], gmask), r)
                  for r in rule_names()), reverse=True)
    WINNER = tab[0][1]
    say("\nSELV top 5 rules @ t=0.5:")
    for s_, r in tab[:5]:
        say("  %.4f  %s" % (s_, r))
    sweep = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        P, idx, gmask = selv_at(t)
        sweep.append((score_e2e(rule_scores(P, idx, W_e2e, WINNER).argmax(1),
                                y_arr[selv_idx], gmask), t))
    FINAL_T = max(sweep)[1]
    say("threshold sweep: %s -> t=%.1f" % (", ".join("%.1f:%.4f" % (t, s_) for s_, t in sweep), FINAL_T))
    del P, idx, OOF10, P_GPOS; gc.collect()

    # ------------------------------------------------------------ 4.6 refit, 5.1 test
    t0 = time.time()
    gate_f = fit_gate(X_trf, y_arr)
    m10 = yr10 >= 0
    Xa10 = X_trf[m10].reset_index(drop=True)
    FINAL10 = [fit_member(k, p, Xa10, pd.Series(yr10[m10]), NCR) for _, (k, p) in SPECS]
    del Xa10; gc.collect()
    say("refit %.1fm" % ((time.time() - t0) / 60))

    p_test = gate_f.predict_proba(X_test)[:, 1].astype(np.float32)
    gmask_t = p_test > FINAL_T
    gpos_t = np.where(gmask_t)[0]
    P_t = [m.predict_proba(X_test.iloc[gpos_t]).astype(np.float32) for m in FINAL10]
    idx_t = NN.kneighbors(prof(P_t))[1]
    S_t = rule_scores(P_t, idx_t, W_e2e, WINNER)
    y_pred = np.zeros(len(y_te), int)
    y_pred[gmask_t] = LAB10[S_t.argmax(1)]

    q = S_t / S_t.sum(1, keepdims=True)
    pa = p_test[gpos_t]
    y_proba = np.zeros((len(y_te), K)); y_proba[:, 0] = 1.0
    y_proba[gpos_t, 0] = (1 - pa) + pa * q[:, IREJ]
    for k in range(NC):
        y_proba[gpos_t, attack_labels[k]] = pa * q[:, k]
    y_proba = y_proba / y_proba.sum(1, keepdims=True)

    MET = dict(accuracy=accuracy_score(y_te, y_pred),
               macro_P=precision_score(y_te, y_pred, average="macro", zero_division=0),
               macro_R=recall_score(y_te, y_pred, average="macro", zero_division=0),
               macro_F1=f1_score(y_te, y_pred, average="macro", zero_division=0),
               wtd_P=precision_score(y_te, y_pred, average="weighted", zero_division=0),
               wtd_R=recall_score(y_te, y_pred, average="weighted", zero_division=0),
               wtd_F1=f1_score(y_te, y_pred, average="weighted", zero_division=0),
               roc_auc=roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro",
                                     labels=list(range(K))))
    MET = {k: round(float(v), 4) for k, v in MET.items()}
    say("\nTEST  %s @ t=%.1f" % (WINNER, FINAL_T))
    for k, v in MET.items():
        say("  %-9s %.4f" % (k, v))
    pcf = f1_score(y_te, y_pred, average=None, labels=list(range(K)), zero_division=0)
    for k in range(K):
        say("    F1 %-22s %.4f  (support %d)" % (names[k], pcf[k], (y_te == k).sum()))

    # ------------------------------------------------------------ XAI
    say("\n--- XAI ---")
    GOF = np.array([groups.index(group_of(c)) for c in COLS])
    GIDX = [np.where(GOF == g)[0] for g in range(len(groups))]
    keep = [g for g in range(len(groups)) if len(GIDX[g])]
    groups = [groups[g] for g in keep]; GIDX = [GIDX[g] for g in keep]; NG = len(groups)
    say("groups: %s" % ", ".join("%s %d" % (g, len(i)) for g, i in zip(groups, GIDX)))

    def contribs(model, Xdf):
        dm = xgb.DMatrix(Xdf); bst = model.get_booster()
        return bst.predict(dm, pred_contribs=True), bst.predict(dm, output_margin=True)

    rng = np.random.RandomState(RS)
    ben, att = np.where(y_te == 0)[0], np.where(y_te != 0)[0]
    G_rows = np.concatenate([rng.choice(ben, min(2000, len(ben)), replace=False),
                             rng.choice(att, min(2000, len(att)), replace=False)])
    Xg = X_test.iloc[G_rows].reset_index(drop=True)
    yt_g = y_te[gpos_t]
    E_pos = np.unique(np.concatenate([rng.choice(np.where(yt_g == c)[0], min(200, (yt_g == c).sum()),
                                                 replace=False) for c in range(K)]))
    Xe = X_test.iloc[gpos_t[E_pos]].reset_index(drop=True)
    ype = y_pred[gpos_t[E_pos]]

    PHI_G, MG_G = contribs(gate_f, Xg)
    ADD = {"gate": float(np.abs(PHI_G.sum(-1) - MG_G).max())}
    PHI_A = MG_A = None
    for m in XGB_MEMBERS:
        ph, mg = contribs(FINAL10[m], Xe)
        ADD[MODELS[m]] = float(np.abs(ph.sum(-1) - mg).max())
        if m == IDX_A25: PHI_A, MG_A = ph, mg
    sv_g = np.asarray(shap.TreeExplainer(gate_f).shap_values(Xg.iloc[:300]))
    sv_m = np.asarray(shap.TreeExplainer(FINAL10[IDX_A25]).shap_values(Xe.iloc[:300]))
    XIMPL = max(float(np.abs(sv_g - PHI_G[:300, :-1]).max()),
                float(np.abs(sv_m - np.transpose(PHI_A[:300, :, :-1], (0, 2, 1))).max()))
    say("additivity max |sum phi - margin| over gate + %d XGB members: %.2e"
        % (len(XGB_MEMBERS), max(ADD.values())))
    say("shap %s vs xgboost pred_contribs, max |diff|: %.2e" % (shap.__version__, XIMPL))

    COEF_t = rule_coef(P_t, idx_t, W_e2e, WINNER)
    fusedf = lambda P_, cf: np.einsum("nmc,mnc->nc", cf, np.stack(P_).astype(np.float64))
    lin_err = float(np.abs(fusedf(P_t, COEF_t) - S_t).max())
    say("linear rewrite of the rule vs rule_scores on %d rows: %.1e" % (len(gpos_t), lin_err))

    m10i, ya10 = np.where(m10)[0], yr10[m10]
    per = max(2, 40 // NCR)
    bg = np.concatenate([rng.choice(m10i[ya10 == c], min(per, (ya10 == c).sum()), replace=False)
                         for c in range(NCR)])
    BG_X = X_trf.iloc[bg].values.astype(np.float32); B = len(BG_X)
    BG_P = [m.predict_proba(pd.DataFrame(BG_X, columns=COLS)).astype(np.float32).mean(0)
            for m in FINAL10]
    n = len(Xe); Xn = Xe.values.astype(np.float32); full = (1 << NG) - 1
    V = {full: [P_t[m][E_pos] for m in range(M)], 0: [np.repeat(b[None], n, 0) for b in BG_P]}
    t0 = time.time()
    for mask in range(1, full):
        absent = np.concatenate([GIDX[g] for g in range(NG) if not (mask >> g) & 1])
        rep = np.repeat(Xn, B, axis=0)
        rep[:, absent] = np.tile(BG_X[:, absent], (n, 1))
        rdf = pd.DataFrame(rep, columns=COLS)
        V[mask] = [m.predict_proba(rdf).astype(np.float32).reshape(n, B, -1).mean(1) for m in FINAL10]
        del rep, rdf
    phi_m = np.zeros((M, n, NG, NCR))
    for g in range(NG):
        for mask in range(1 << NG):
            if (mask >> g) & 1: continue
            s_ = bin(mask).count("1")
            w_ = factorial(s_) * factorial(NG - s_ - 1) / factorial(NG)
            for m in range(M):
                phi_m[m, :, g, :] += w_ * (V[mask | (1 << g)][m] - V[mask][m])
    cf = COEF_t[E_pos]
    PHI = np.einsum("nmc,mngc->ngc", cf, phi_m)
    S_full = fusedf(V[full], cf)
    eff = float(np.abs(PHI.sum(1) + fusedf(V[0], cf) - S_full).max())
    head = S_full.argmax(1)
    repro = float((LAB10[head] == ype).mean())
    say("group Shapley: %d rows x %d coalitions x %d background (%.0fs)"
        % (n, 1 << NG, B, time.time() - t0))
    say("efficiency %.1e | deployed decision reproduced on %.4f%% of rows" % (eff, 100 * repro))

    share = lambda v: np.asarray(v, float) / np.sum(v)
    gate_sh = share(np.abs(np.stack([PHI_G[:, i].sum(1) for i in GIDX], 1)).mean(0))
    sb_sh = share(np.abs(PHI[np.arange(n), :, head]).mean(0))
    gain_gate = share([gate_f.feature_importances_[i].sum() for i in GIDX])
    gain_a25 = share([FINAL10[IDX_A25].feature_importances_[i].sum() for i in GIDX])
    say("\n%-26s %s" % ("share", "  ".join("%6s" % g for g in groups)))
    for nm, v in [("gate SHAP", gate_sh), ("gate gain", gain_gate),
                  ("Stage-B ensemble SHAP", sb_sh), ("Stage-B XGB a=0.25 gain", gain_a25)]:
        say("%-26s %s" % (nm, "  ".join("%6.3f" % x for x in v)))

    XAI = dict(groups=groups, group_sizes=[int(len(i)) for i in GIDX],
               additivity_max=max(ADD.values()), additivity=ADD, shap_vs_xgboost_max=XIMPL,
               linear_rewrite_max=lin_err, efficiency_max=eff, reproduction=repro,
               explain_rows=int(n), background_rows=int(B),
               gate_shap_share=dict(zip(groups, gate_sh.round(4).tolist())),
               stageB_shap_share=dict(zip(groups, sb_sh.round(4).tolist())),
               gate_gain_share=dict(zip(groups, gain_gate.round(4).tolist())),
               stageB_gain_share=dict(zip(groups, gain_a25.round(4).tolist())),
               gate_top=[groups[int(gate_sh.argmax())], round(float(gate_sh.max()), 4)],
               stageB_top=[groups[int(sb_sh.argmax())], round(float(sb_sh.max()), 4)])
    res = dict(dataset=tag, rows=int(len(y)), features=len(COLS), classes=K, names=list(names),
               test_rows=int(len(y_te)), selv_winner=WINNER, threshold=FINAL_T,
               metrics=MET, per_class_f1=dict(zip(names, pcf.round(4).tolist())),
               xai=XAI, ref_cap=ref_cap, minutes=round((time.time() - t_start) / 60, 1))
    (out_dir / ("%s.json" % tag)).write_text(json.dumps(res, indent=2), encoding="utf-8")
    say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
    return res
