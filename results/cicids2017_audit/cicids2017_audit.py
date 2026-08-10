"""
CIC-IDS2017 data audit -- the equivalent of CLAUDE.md's "Verified data facts"
section for CIC-UNSW-Data.csv. Nothing is assumed; every claim is measured.

Checks: shapes, column agreement across the 8 day-files, duplicate column names,
dtypes, NaN/inf, duplicate ROWS (the leakage trap), class distribution,
zero-variance columns, and |r|>=0.95 redundancy.

ASCII-only prints (terminal is cp1252); full report also written UTF-8.
"""
import io, json, sys
import numpy as np, pandas as pd
from pathlib import Path

D = Path(r"E:\AIUB\THESIS\Resource\CIC-IDS2017")
OUT = Path(__file__).parent
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


FILES = ["Monday-WorkingHours.pcap_ISCX.csv",
         "Tuesday-WorkingHours.pcap_ISCX.csv",
         "Wednesday-workingHours.pcap_ISCX.csv",
         "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
         "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
         "Friday-WorkingHours-Morning.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
         "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

say("=" * 78)
say("1. PER-FILE SHAPE AND COLUMN AGREEMENT")
say("=" * 78)

frames, colsets = [], {}
for f in FILES:
    df = pd.read_csv(D / f, low_memory=False, encoding="latin-1")
    raw_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    dupcols = [c for c in pd.Index(df.columns) if list(df.columns).count(c) > 1]
    lead_ws = sum(1 for c in raw_cols if c != c.strip())
    lab = df.columns[-1]
    say("%-58s %8d x %3d  ws-cols=%2d  dup-cols=%s"
        % (f[:58], df.shape[0], df.shape[1], lead_ws, sorted(set(dupcols)) or "none"))
    colsets[f] = tuple(df.columns)
    frames.append(df)

ref = colsets[FILES[0]]
same = all(colsets[f] == ref for f in FILES)
say("\nall 8 files share an identical column list (after strip): %s" % same)
if not same:
    for f in FILES:
        if colsets[f] != ref:
            say("  MISMATCH %s" % f)
            say("    only here : %s" % (set(colsets[f]) - set(ref)))
            say("    missing   : %s" % (set(ref) - set(colsets[f])))

df = pd.concat(frames, ignore_index=True)
del frames
say("\npooled: %d rows x %d cols" % df.shape)

LABEL = "Label"
say("label column: '%s'" % LABEL)

say("\n" + "=" * 78)
say("2. DTYPES AND NON-NUMERIC COLUMNS")
say("=" * 78)
feat = [c for c in df.columns if c != LABEL]
nonnum = [c for c in feat if not pd.api.types.is_numeric_dtype(df[c])]
say("features: %d | non-numeric: %s" % (len(feat), nonnum or "none"))
for c in nonnum:
    say("  %s -> sample: %s" % (c, df[c].dropna().unique()[:5]))
    df[c] = pd.to_numeric(df[c], errors="coerce")
say("dtype counts: %s" % dict(df[feat].dtypes.astype(str).value_counts()))

say("\n" + "=" * 78)
say("3. NaN / inf  (CIC-UNSW had ZERO of both -- expect otherwise here)")
say("=" * 78)
nan_ct = df[feat].isna().sum()
nan_ct = nan_ct[nan_ct > 0].sort_values(ascending=False)
say("columns with NaN: %d" % len(nan_ct))
for c, v in nan_ct.items():
    say("  %-30s %8d  (%.4f%%)" % (c, v, 100 * v / len(df)))

num = df[feat].select_dtypes(include=[np.number])
inf_ct = np.isinf(num.to_numpy(dtype=np.float64, na_value=0.0)).sum(0)
inf_s = pd.Series(inf_ct, index=num.columns)
inf_s = inf_s[inf_s > 0].sort_values(ascending=False)
say("columns with inf: %d" % len(inf_s))
for c, v in inf_s.items():
    say("  %-30s %8d  (%.4f%%)" % (c, v, 100 * v / len(df)))

bad_rows = int((df[feat].isna().any(axis=1).to_numpy() |
                np.isinf(num.to_numpy(dtype=np.float64, na_value=0.0)).any(axis=1)).sum())
say("rows affected by NaN or inf: %d (%.4f%%)" % (bad_rows, 100 * bad_rows / len(df)))

say("\n" + "=" * 78)
say("4. CLASS DISTRIBUTION")
say("=" * 78)
df[LABEL] = df[LABEL].astype(str).str.strip()
vc = df[LABEL].value_counts()
say("%d distinct labels" % len(vc))
for k, v in vc.items():
    say("  %-30s %9d  %7.4f%%" % (k, v, 100 * v / len(df)))
say("\nimbalance ratio (largest:smallest) = %.0f : 1" % (vc.iloc[0] / vc.iloc[-1]))

say("\n" + "=" * 78)
say("5. DUPLICATE ROWS -- the leakage trap")
say("=" * 78)
n_all = len(df)
dup_fl = int(df.duplicated().sum())
say("exact duplicates on features+label : %d  (%.2f%%)" % (dup_fl, 100 * dup_fl / n_all))
dup_f = int(df.duplicated(subset=feat).sum())
say("exact duplicates on features only  : %d  (%.2f%%)" % (dup_f, 100 * dup_f / n_all))
say("=> feature-vectors with CONFLICTING labels account for %d rows" % (dup_f - dup_fl))
say("CIC-UNSW-2024 reference: 447,915 -> 306,173 (32%% duplicates)")
say("UNSW-NB15 reference    : 257,673 -> 162,745 (36.8%% duplicates)")

ded = df.drop_duplicates()
say("\nafter dedupe: %d rows" % len(ded))
vc2 = ded[LABEL].value_counts()
say("  %-28s %9s %9s %8s" % ("label", "before", "after", "kept%"))
for k in vc.index:
    a, b = vc[k], vc2.get(k, 0)
    say("  %-28s %9d %9d %7.1f%%" % (k, a, b, 100 * b / a))

say("\n" + "=" * 78)
say("6. ZERO-VARIANCE COLUMNS (unsupervised prune, step 1)")
say("=" * 78)
nun = ded[feat].nunique()
zero = nun[nun <= 1].index.tolist()
say("zero-variance: %d" % len(zero))
for c in zero:
    say("  %-30s constant = %s" % (c, ded[c].iloc[0]))
say("(CIC-UNSW had 9 such columns)")

say("\n" + "=" * 78)
say("7. |r| >= 0.95 REDUNDANCY (unsupervised prune, step 2)")
say("=" * 78)
work = ded[feat].drop(columns=zero)
work = work.replace([np.inf, -np.inf], np.nan)
corr = work.corr(numeric_only=True)
upper = corr.abs().where(np.triu(np.ones(corr.shape, bool), k=1))
red = [c for c in upper.columns if (upper[c] >= 0.95).any()]
say("redundant (dropped): %d" % len(red))
for c in red:
    partner = upper[c].idxmax()
    say("  %-30s r=%.3f with %s" % (c, upper[c].max(), partner))
say("\nFULL %d -> after zero-var %d -> BASE %d features"
    % (len(feat), len(feat) - len(zero), len(feat) - len(zero) - len(red)))
say("(CIC-UNSW: 76 -> 67 -> 44)")

summary = dict(
    files=len(FILES), pooled_rows=int(n_all), n_features=len(feat),
    columns_identical=bool(same), nonnumeric=nonnum,
    n_labels=int(len(vc)), labels={str(k): int(v) for k, v in vc.items()},
    nan_cols={str(k): int(v) for k, v in nan_ct.items()},
    inf_cols={str(k): int(v) for k, v in inf_s.items()},
    rows_nan_or_inf=bad_rows,
    dup_feat_label=dup_fl, dup_feat_only=dup_f,
    conflicting_label_rows=int(dup_f - dup_fl),
    rows_after_dedupe=int(len(ded)),
    labels_after_dedupe={str(k): int(v) for k, v in vc2.items()},
    zero_variance=zero, corr_redundant=red,
    base_features=int(len(feat) - len(zero) - len(red)),
)
(OUT / "cicids2017_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
io.open(OUT / "cicids2017_audit.txt", "w", encoding="utf-8").write("\n".join(LOG))
say("\nwrote cicids2017_audit.json / .txt")
