"""
TON-IoT audit -- run this BEFORE any modelling, and treat its verdict as a gate.

Purpose. Two things must be settled before the v7.1b protocol is worth pointing at a
new dataset:

  (1) IS IT HONEST?  TON-IoT ships src_ip / dst_ip / src_port / ts. On a testbed,
      endpoint identity <-> label by construction. This is exactly the failure the
      TTL audit caught on our own data, where sttl in {62,254} alone recovered ~100%
      attack recall at fpr ~1.6% -- as good as the entire 99-feature learned gate.
      Any column that does that here must be excluded, or every downstream number is
      measuring the testbed rather than the traffic. The FINGERPRINT block below is a
      pre-registered go/no-go: a column is SUSPECT if, on its own, it reaches
      >= 0.95 attack recall at <= 0.05 fpr.

  (2) IS IT HARD?  e_1NN has predicted the architecture's value three times now
      (CIC-IDS2017 e_1NN 0.0026 -> all 17 combiner variants within 0.0007; CIC-UNSW
      e_1NN 0.3629 -> full stack worth +0.0502). If TON-IoT comes back near zero, the
      ensemble/weighting machinery will be worth nothing there and that is the finding,
      not a failure. Measuring it costs minutes; a pipeline run costs an evening.

Reference values to compare against (from this project):
      dataset            e_1NN     dup rate   architecture worth
      CIC-UNSW-2024      0.3629    32%        +0.0502
      UNSW-NB15          0.3043    36.8%      +0.045 (ensembling)
      CIC-IDS2017        0.0026    10.86%     +0.0007  <- too easy to test anything

Nothing here fits a classifier. It reports structure, leakage risk and difficulty.
ASCII-only prints (terminal is cp1252); log written UTF-8.

Usage:
    python toniot_audit.py [path_to_csv_or_dir] [--rows N]
Default search: Resource/TON-IoT/, Resource/TON_IoT/, Resource/Ton-IOT/ (*.csv).
"""
import io, json, sys, time
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score

ROOT = Path(r"E:\AIUB\THESIS")
OUT = Path(__file__).parent
RS = 42
MAX_ROWS = 3_000_000        # guard against the 22M-row full release
REF_CAP, QRY_CAP = 200_000, 50_000   # caps for the 1-NN / purity work
K_PURITY = 10
SUSPECT_RECALL, SUSPECT_FPR = 0.95, 0.05
MIN_SUPPORT = 20            # a value needs this many rows before a rule may use it
# Zeek/TON-IoT writes missing as a bare dash; pandas will not catch that by default.
NA = ["-", "(empty)", "", " ", "NaN", "nan", "None", "null"]
# ENDPOINT identity -- WHO sent the traffic. Excluded from the feature matrix: on a
# testbed this is a label proxy (the sttl failure, results/ttl_join/).
# NOTE the destination port is deliberately NOT here. It is SERVICE identity -- WHAT the
# attack targets -- which survived ablation on our own data at +0.0982 and is a native
# feature in v7. Matching must be exact-ish: a prefix rule on "src_ip" also swallows
# src_ip_bytes, which is a byte-volume feature, not an address.
ENDPOINT_EXACT = {"src_ip", "dst_ip", "source_ip", "dest_ip", "srcip", "dstip",
                  "ip_src", "ip_dst", "src_port", "sport", "source_port",
                  "ts", "timestamp", "time", "date", "datetime", "stime", "ltime",
                  "flow_id", "uid", "id"}


def is_endpoint(lc):
    if lc in ENDPOINT_EXACT or lc.endswith("_ip"):
        return True
    return lc.endswith("_port") and not lc.startswith(("dst", "dest", "d_"))
LABEL_HINTS = ("type", "attack_cat", "attack", "category", "label", "class")
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "toniot_audit.txt", "w", encoding="utf-8").write("\n".join(LOG))


t0 = time.time()
res = {}

# ---------------------------------------------------------------- locate the file
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--rows" in sys.argv:
    MAX_ROWS = int(sys.argv[sys.argv.index("--rows") + 1])
cands = []
if argv:
    p = Path(argv[0])
    cands = sorted(p.glob("*.csv")) if p.is_dir() else [p]
else:
    for d in ("TON-IoT", "TON_IoT", "Ton-IOT", "TONIoT", "ToN-IoT"):
        dd = ROOT / "Resource" / d
        if dd.exists():
            cands += sorted(dd.rglob("*.csv"))
if not cands:
    say("NO CSV FOUND. Put the TON-IoT network CSV in Resource/TON-IoT/ or pass a path.")
    say("Expected e.g. Resource/TON-IoT/Train_Test_Network.csv")
    dump(); sys.exit(1)
# prefer the network flow file if several are present
pref = [c for c in cands if "network" in c.name.lower()]
SRC = (pref or sorted(cands, key=lambda c: c.stat().st_size, reverse=True))[0]
say("=" * 78)
say("TON-IoT AUDIT -- %s" % SRC.name)
say("=" * 78)
if len(cands) > 1:
    say("other CSVs present (not audited): %s" % ", ".join(c.name for c in cands if c != SRC))

# ---------------------------------------------------------------- load
df = pd.read_csv(SRC, na_values=NA, keep_default_na=True, low_memory=False, nrows=MAX_ROWS)
truncated = len(df) == MAX_ROWS
df.columns = [c.strip() for c in df.columns]
say("\nrows %d%s | columns %d" % (len(df), "  (TRUNCATED at MAX_ROWS -- rerun with --rows)"
                                  if truncated else "", df.shape[1]))
res["file"] = SRC.name; res["rows"] = int(len(df)); res["cols"] = int(df.shape[1])
res["truncated"] = bool(truncated)

# ---------------------------------------------------------------- label columns
low = {c.lower(): c for c in df.columns}
multi = next((low[h] for h in ("type", "attack_cat", "category") if h in low), None)
binar = next((low[h] for h in ("label", "class") if h in low), None)
if multi is None and binar is None:
    say("\nNO LABEL COLUMN FOUND among %s -- inspect manually." % (LABEL_HINTS,))
    dump(); sys.exit(1)
YCOL = multi or binar
say("\nlabel column: '%s'%s" % (YCOL, "  (binary '%s' also present)" % binar if binar and multi else ""))
yraw = df[YCOL].astype(str).str.strip().str.lower()
NORMAL = {"normal", "benign", "0", "background"}
is_attack = (~yraw.isin(NORMAL)).values
res["label_col"] = YCOL

say("\n--- class distribution ---")
vc = yraw.value_counts()
for k, v in vc.items():
    say("  %-16s %9d  %6.3f%%" % (k, v, 100.0 * v / len(df)))
imb = float(vc.max() / max(vc.min(), 1))
say("  imbalance ratio %.0f:1 | attack share %.2f%%" % (imb, 100 * is_attack.mean()))
res["class_counts"] = {str(k): int(v) for k, v in vc.items()}
res["imbalance_ratio"] = round(imb, 1)
res["attack_share"] = round(float(is_attack.mean()), 4)
res["tiny_classes"] = [str(k) for k, v in vc.items() if v < 100]
if res["tiny_classes"]:
    say("  !! classes under 100 rows (cannot support a stratified split): %s"
        % ", ".join(res["tiny_classes"]))
dump()

# ---------------------------------------------------------------- column typing
feat_cols = [c for c in df.columns if c not in {YCOL, binar, multi} - {None}]
ident, cat, txt, num = [], [], [], []
for c in feat_cols:
    lc = c.lower()
    nun = df[c].nunique(dropna=True)
    if is_endpoint(lc):
        ident.append(c)
    elif pd.api.types.is_numeric_dtype(df[c]):
        num.append(c)
    elif nun <= 50:
        cat.append(c)
    else:
        txt.append(c)
say("\n--- column types ---")
say("  endpoint identity%2d : %s   (EXCLUDED from the matrix -- 'who sent it')" % (len(ident), ", ".join(ident)))
say("  numeric          %2d : %s%s" % (len(num), ", ".join(num[:8]), " ..." if len(num) > 8 else ""))
say("  categorical(<=50)%2d : %s" % (len(cat), ", ".join(cat)))
say("  high-card/text   %2d : %s" % (len(txt), ", ".join(txt)))
say("  NOTE: endpoint-identity and high-card/text columns are EXCLUDED from the feature")
say("        matrix; they are audited for leakage below, not modelled. The DESTINATION")
say("        port is kept -- it is service identity ('what is targeted'), which survived")
say("        ablation at +0.0982 on our data and is a native v7 feature. Its contribution")
say("        is isolated by a separate no-dst-port arm below.")
res["cols_identity"] = ident; res["cols_numeric"] = num
res["cols_categorical"] = cat; res["cols_text"] = txt

# ---------------------------------------------------------------- data quality
say("\n--- data quality ---")
nan_ct = df[feat_cols].isna().sum()
nan_ct = nan_ct[nan_ct > 0].sort_values(ascending=False)
if len(nan_ct):
    say("  columns with missing values (after treating %s as NA):" % NA[:2])
    for c, v in nan_ct.head(15).items():
        say("    %-28s %8d  %6.2f%%" % (c, v, 100.0 * v / len(df)))
    if len(nan_ct) > 15: say("    ... and %d more" % (len(nan_ct) - 15))
else:
    say("  no missing values")
numdf = df[num]
inf_ct = int(np.isinf(numdf.select_dtypes(include=[np.number]).to_numpy(np.float64,
                                                                       na_value=0.0)).sum())
say("  infinite values in numeric columns: %d" % inf_ct)
obj_num = [c for c in feat_cols if c not in ident + txt and not pd.api.types.is_numeric_dtype(df[c])
           and c not in cat]
if obj_num:
    say("  !! non-numeric columns that look numeric (dash-as-missing?): %s" % ", ".join(obj_num))
res["nan_columns"] = {str(k): int(v) for k, v in nan_ct.head(30).items()}
res["inf_count"] = inf_ct

model_cols = num + cat
dup_fl = int(df.duplicated(subset=model_cols + [YCOL]).sum())
dup_f = int(df.duplicated(subset=model_cols).sum())
say("  duplicates on features+label : %d  (%.2f%%)" % (dup_fl, 100.0 * dup_fl / len(df)))
say("  duplicates on features only  : %d  (%.2f%%)" % (dup_f, 100.0 * dup_f / len(df)))
say("  -> rows with a feature twin carrying a CONFLICTING label: %d" % (dup_f - dup_fl))
say("     (CIC-UNSW 32% / UNSW-NB15 36.8% / CIC-IDS2017 10.86%)")
res["dup_features_label"] = dup_fl; res["dup_features_only"] = dup_f
res["dup_rate"] = round(dup_fl / len(df), 4)
dump()

# ---------------------------------------------------------------- FINGERPRINT AUDIT (the gate)
say("\n" + "=" * 78)
say("FINGERPRINT AUDIT -- pre-registered go/no-go")
say("A column is SUSPECT if, ALONE, it reaches >= %.2f attack recall at <= %.2f fpr."
    % (SUSPECT_RECALL, SUSPECT_FPR))
say("Reference: on our own data sttl in {62,254} did exactly this (recall ~1.00, fpr ~0.016)")
say("and was rejected as an endpoint label proxy -- see results/ttl_join/.")
say("=" * 78)


def fingerprint(col):
    """Best single-column rule, evaluated OUT OF SAMPLE.

    The value->attack map is learned on half the rows and scored on the other half,
    and a value must carry >= MIN_SUPPORT training rows to be usable. Both guards are
    essential: scored in-sample, ANY high-cardinality column (ts, src_port, a float
    duration) trivially reaches ~1.0 recall at ~0 fpr because almost every value is
    unique to one row. That is memorisation inside the audit, not a fingerprint. The
    in-sample number is reported alongside so the gap is visible -- a large gap means
    'high cardinality', a small gap at high recall means 'genuine label proxy'.
    """
    v = df[col].astype(str).fillna("__NA__").values
    ia, ib = train_test_split(np.arange(len(v)), test_size=0.5, random_state=RS,
                              stratify=is_attack)
    tr = pd.DataFrame({"v": v[ia], "a": is_attack[ia]})
    grp = tr.groupby("v")["a"].agg(["sum", "count"])
    usable = grp[grp["count"] >= MIN_SUPPORT]
    att_vals = set(usable.index[(usable["sum"] / usable["count"]) > 0.5])
    pred = np.isin(v[ib], list(att_vals)) if att_vals else np.zeros(len(ib), bool)
    ya = is_attack[ib]
    recall = float(pred[ya].mean()) if ya.any() else 0.0
    fpr = float(pred[~ya].mean()) if (~ya).any() else 0.0
    # in-sample counterpart, for the cardinality gap
    tab = pd.crosstab(v, is_attack)
    for c_ in (False, True):
        if c_ not in tab.columns: tab[c_] = 0
    n_att, n_ben = tab[True].values, tab[False].values
    rate = n_att / np.maximum(n_att + n_ben, 1)
    sel = rate > 0.5
    rec_in = float(n_att[sel].sum() / max(n_att.sum(), 1))
    pure = ((rate >= 0.99) | (rate <= 0.01)) & ((n_att + n_ben) >= MIN_SUPPORT)
    pure_share = float((n_att + n_ben)[pure].sum() / len(df))
    nmi = float(mutual_info_score(v, is_attack)
                / max(mutual_info_score(is_attack, is_attack), 1e-12))
    return dict(col=col, n_values=int(len(tab)), usable_values=int(len(att_vals)),
                recall=recall, fpr=fpr, recall_insample=rec_in,
                pure_row_share=pure_share, nmi=nmi,
                suspect=bool(recall >= SUSPECT_RECALL and fpr <= SUSPECT_FPR))


fp_cols = ident + cat + [c for c in num if df[c].nunique() <= 200]
FP = [fingerprint(c) for c in fp_cols]
FP.sort(key=lambda r: (-r["suspect"], -(r["recall"] - r["fpr"])))
say("\n  recall/fpr are OUT-OF-SAMPLE (map learned on half the rows, scored on the other")
say("  half, values needing >= %d support). rec_in is the in-sample counterpart: a big" % MIN_SUPPORT)
say("  rec_in - recall gap means high cardinality, not a fingerprint. NMI inflates with")
say("  cardinality too -- read it only alongside n_values.")
say("\n  %-24s %7s %7s %8s %7s %8s %6s"
    % ("column", "values", "usable", "recall", "fpr", "rec_in", "NMI"))
for r in FP:
    say("  %-24s %7d %7d %8.4f %7.4f %8.4f %6.3f%s"
        % (r["col"], r["n_values"], r["usable_values"], r["recall"], r["fpr"],
           r["recall_insample"], r["nmi"], "   <== SUSPECT" if r["suspect"] else ""))
SUSPECT = [r["col"] for r in FP if r["suspect"]]
res["fingerprint"] = FP; res["suspect_columns"] = SUSPECT
say("")
if SUSPECT:
    say("VERDICT: %d SUSPECT column(s): %s" % (len(SUSPECT), ", ".join(SUSPECT)))
    say("  -> EXCLUDE these before modelling, or every downstream number measures the")
    say("     testbed rather than the traffic. If one is a service/port column, see the")
    say("     TTL-vs-svc distinction in CLAUDE.md: services are what attacks TARGET and")
    say("     survived ablation (+0.098); endpoint identity is WHO SENT it and did not.")
else:
    say("VERDICT: no single column solves attack-vs-benign on its own. Proceed --")
    say("  but still exclude raw IPs/timestamps from the feature matrix on principle.")
dump()

# ---------------------------------------------------------------- feature matrix + prune
say("\n" + "=" * 78)
say("DIFFICULTY -- feature matrix, prune, e_1NN, purity")
say("=" * 78)
# columns that are >99% missing carry almost nothing but one-hot into a lot of noise
miss = df[num + cat].isna().mean()
dead = miss[miss > 0.99].index.tolist()
if dead:
    say("\ndropping %d column(s) that are >99%% missing: %s" % (len(dead), ", ".join(dead)))
num_k = [c for c in num if c not in dead]
cat_k = [c for c in cat if c not in dead]
Xn = df[num_k].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
Xc = pd.get_dummies(df[cat_k].astype(str).fillna("__NA__"), prefix=cat_k, dtype=np.float32) if cat_k else pd.DataFrame(index=df.index)
X = pd.concat([Xn.astype(np.float32), Xc], axis=1)
res["dropped_mostly_missing"] = dead
n0 = X.shape[1]
nun = X.nunique(); X = X.drop(columns=nun[nun <= 1].index.tolist())
n1 = X.shape[1]
cm = X.corr().abs()
upper = cm.where(np.triu(np.ones(cm.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
X = X.drop(columns=red)
say("\nprune: %d -> %d (zero-variance) -> %d (|r| >= 0.95)" % (n0, n1, X.shape[1]))
say("  reference widths: CIC-UNSW 76->67->44 | CIC-IDS2017 78->70->47")
res["prune"] = [int(n0), int(n1), int(X.shape[1])]

keep = ~df.duplicated(subset=model_cols + [YCOL]).values
Xd = X[keep].reset_index(drop=True)
yd = yraw[keep].reset_index(drop=True).values
att_d = is_attack[keep]
say("dedupe: %d -> %d rows (%.2f%% removed)" % (len(X), len(Xd), 100 * (1 - len(Xd) / len(X))))


def bracket(e1, C):
    a = C / max(C - 1, 1)
    lo = (1 - np.sqrt(max(1 - a * e1, 0.0))) / a
    return float(lo), float(e1)


def e1nn(Xm, ym, tag):
    """Leakage-safe 1-NN error: reference = train split, query = test split."""
    cnt = pd.Series(ym).value_counts()
    ok = np.isin(ym, cnt[cnt >= 2].index)
    Xm, ym = Xm[ok], ym[ok]
    if len(np.unique(ym)) < 2 or len(ym) < 200:
        say("  %s: too few rows/classes to measure" % tag); return None
    itr, ite = train_test_split(np.arange(len(ym)), test_size=0.2, random_state=RS, stratify=ym)
    rng = np.random.RandomState(RS)
    if len(itr) > REF_CAP: itr = rng.choice(itr, REF_CAP, replace=False)
    if len(ite) > QRY_CAP: ite = rng.choice(ite, QRY_CAP, replace=False)
    sc = RobustScaler().fit(Xm[itr])
    A, B = sc.transform(Xm[itr]).astype(np.float32), sc.transform(Xm[ite]).astype(np.float32)
    A = np.nan_to_num(A, posinf=0, neginf=0); B = np.nan_to_num(B, posinf=0, neginf=0)
    nn = NearestNeighbors(n_neighbors=K_PURITY, algorithm="brute", n_jobs=-1).fit(A)
    _, ii = nn.kneighbors(B)
    ytr, yte = ym[itr], ym[ite]
    err = float((ytr[ii[:, 0]] != yte).mean())
    C = len(np.unique(ym))
    lo, hi = bracket(err, C)
    pur = {}
    for c in np.unique(yte):
        s = yte == c
        pur[str(c)] = round(float((ytr[ii[s]] == c).mean()), 4)
    say("  %s: e_1NN %.4f | Cover-Hart e* in [%.4f, %.4f] -> accuracy ceiling [%.4f, %.4f]"
        % (tag, err, lo, hi, 1 - hi, 1 - lo))
    say("     (ref %d rows, query %d rows, %d classes, %d features, metric=RobustScaler Euclidean)"
        % (len(itr), len(ite), C, Xm.shape[1]))
    say("     k=%d purity by class:" % K_PURITY)
    for c in sorted(pur, key=lambda k_: -pur[k_]):
        say("       %-16s %.4f" % (c, pur[c]))
    return dict(e_1nn=round(err, 4), bracket=[round(lo, 4), round(hi, 4)],
                classes=int(C), features=int(Xm.shape[1]), purity=pur,
                ref_rows=int(len(itr)), query_rows=int(len(ite)))


Xv = Xd.to_numpy(np.float32)
say("\nALL-CLASS (gate + families together):")
res["e1nn_all"] = e1nn(Xv, yd, "all classes")
say("\nATTACK-ONLY (Stage-B analogue -- this is the number comparable to ours):")
res["e1nn_attack"] = e1nn(Xv[att_d], yd[att_d], "attack families")

# what does service identity buy? -- the TON-IoT analogue of our +0.0982 svc ablation
port_cols = [c for c in Xd.columns if "port" in c.lower()]
if port_cols:
    say("\nNO-DST-PORT control (drops %s) -- isolates what service identity contributes:"
        % ", ".join(port_cols))
    Xnp = Xd.drop(columns=port_cols).to_numpy(np.float32)
    res["e1nn_attack_no_port"] = e1nn(Xnp[att_d], yd[att_d], "attack families, no port")
    a1 = (res.get("e1nn_attack") or {}).get("e_1nn")
    a2 = (res.get("e1nn_attack_no_port") or {}).get("e_1nn")
    if a1 is not None and a2 is not None:
        say("  -> service identity moves e_1NN by %+.4f (ours: -0.153 on CIC-UNSW)" % (a1 - a2))
dump()

# ---------------------------------------------------------------- verdict
say("\n" + "=" * 78)
say("SUMMARY")
say("=" * 78)
ea = (res.get("e1nn_attack") or {}).get("e_1nn")
say("  leakage gate    : %s" % ("FAIL -- %s" % ", ".join(SUSPECT) if SUSPECT else "PASS"))
say("  duplicate rate  : %.2f%%" % (100 * res["dup_rate"]))
say("  imbalance       : %.0f:1%s" % (imb, "  (tiny classes: %s)" % ", ".join(res["tiny_classes"])
                                      if res["tiny_classes"] else ""))
if ea is not None:
    if ea < 0.05:
        say("  difficulty      : e_1NN %.4f -- TOO EASY. Expect every combiner variant to tie" % ea)
        say("                    (CIC-IDS2017 precedent: 0.0026 -> all 17 within 0.0007).")
        say("                    Report it as a conditional-value result, not a benchmark win.")
    elif ea < 0.15:
        say("  difficulty      : e_1NN %.4f -- easy-ish; expect small architecture effects." % ea)
    elif ea <= 0.45:
        say("  difficulty      : e_1NN %.4f -- comparable to our data (0.30-0.36)." % ea)
        say("                    The weighting/ensemble stack should pay here.")
    else:
        say("  difficulty      : e_1NN %.4f -- FAR harder than our data (0.30-0.36)." % ea)
        say("                    Before reading this as 'a hard dataset', check the feature")
        say("                    matrix: %d columns survived the prune, and the high-card /" % X.shape[1])
        say("                    text columns excluded above may be where the family signal")
        say("                    actually lives. A weak matrix and a hard problem look alike here.")
say("\n  NEXT: if the gate PASSES, run the v7.1b protocol with SELV re-selecting from")
say("  scratch -- do NOT port 'HYBRID W x OLA-cw k=30 p=3'. SELV picked dynamic on")
say("  CIC-UNSW and static on UNSW-NB15; the configuration is a property of the data.")
say("  Levels are NOT comparable across datasets -- report the paired same-pipeline gap.")

res["minutes"] = round((time.time() - t0) / 60, 1)
(OUT / "toniot_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
say("\ntotal %.1f min" % res["minutes"])
dump()
