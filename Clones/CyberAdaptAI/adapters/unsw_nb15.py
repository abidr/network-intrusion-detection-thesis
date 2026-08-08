"""Adapter: the ORIGINAL UNSW-NB15 dataset (Moustafa & Slay, 2015) -- the one the paper uses.

This is NOT the same as Resource/CIC-UNSW-NB15-2024. That is a 2024 CICFlowMeter re-extraction
of the same attack taxonomy with 76 different flow features. This adapter targets the original
Argus/Bro-derived feature set, so results here are comparable to the paper's Table 4.

Two distributions of UNSW-NB15 exist. Both are supported:

  "presplit"  UNSW_NB15_training-set.csv (175,341 rows) + UNSW_NB15_testing-set.csv (82,332)
              45 columns with a header, including `attack_cat` and `label`. This is the
              commonly used form and what most papers mean by "UNSW-NB15".
  "full"      UNSW-NB15_1.csv .. UNSW-NB15_4.csv (2,540,044 rows total), NO header, 49 columns.
              Needs NUSW-NB15_features.csv for column names.

Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Targets:
  task="binary"      -> `label` column, 0 = Normal, 1 = Attack.
  task="multiclass"  -> `attack_cat`, encoded 0-9 with Normal = 0 and the nine attack families
                        alphabetical, matching the CIC-UNSW-2024 convention so the two adapters
                        are directly comparable.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Normal = 0, then the nine attack families alphabetical -- same convention as cic_unsw_2024.
LABEL_NAMES = {
    0: "Normal", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
}
_NAME_TO_CODE = {v.lower(): k for k, v in LABEL_NAMES.items()}
# The raw files are inconsistent about these two.
_ALIASES = {"backdoors": "backdoor", "normal": "normal", "": "normal"}

# Identifier-like columns: leak-prone or meaningless as features.
_DROP_ALWAYS = ["id", "srcip", "dstip", "sport", "dsport", "stime", "ltime"]

PRESPLIT_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
FULL_FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
FEATURES_FILE = "NUSW-NB15_features.csv"

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Resource" / "UNSW-NB15"


def _normalise_attack_cat(s: pd.Series) -> pd.Series:
    """attack_cat is dirty in the raw files: NaN for Normal, stray whitespace, 'Backdoors'."""
    s = s.fillna("Normal").astype(str).str.strip().str.lower()
    s = s.replace(_ALIASES)
    unknown = sorted(set(s.unique()) - set(_NAME_TO_CODE))
    if unknown:
        raise ValueError(f"Unrecognised attack_cat values: {unknown}")
    return s.map(_NAME_TO_CODE).astype(np.int64)


def _discover(root: Path) -> str:
    if all((root / f).exists() for f in PRESPLIT_FILES):
        return "presplit"
    if (root / PRESPLIT_FILES[0]).exists():
        return "presplit"
    if any((root / f).exists() for f in FULL_FILES):
        return "full"
    raise FileNotFoundError(
        f"No UNSW-NB15 files found in {root}.\n"
        f"  Expected either {PRESPLIT_FILES}\n"
        f"  or {FULL_FILES} (+ {FEATURES_FILE}).\n"
        f"  Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
    )


def _load_presplit(root: Path, combine: bool) -> pd.DataFrame:
    """Load the 45-column pre-split form.

    The paper builds its own 80:20 sequential stream split (section 3.2), so by default both
    official files are concatenated and re-split by the framework. Set combine=False to use the
    training file alone.
    """
    frames = []
    for f in PRESPLIT_FILES if combine else PRESPLIT_FILES[:1]:
        p = root / f
        if p.exists():
            frames.append(pd.read_csv(p))
        elif combine and f == PRESPLIT_FILES[1]:
            pass  # testing file is optional
        else:
            raise FileNotFoundError(p)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _load_full(root: Path) -> pd.DataFrame:
    """Load the 49-column headerless form, naming columns from NUSW-NB15_features.csv."""
    feat_path = root / FEATURES_FILE
    if not feat_path.exists():
        raise FileNotFoundError(
            f"{feat_path} is required to name the columns of the headerless "
            f"UNSW-NB15_1..4.csv files."
        )
    feats = pd.read_csv(feat_path, encoding="latin-1")
    name_col = next(c for c in feats.columns if c.strip().lower() == "name")
    columns = [str(c).strip() for c in feats[name_col].tolist()]

    frames = []
    for f in FULL_FILES:
        p = root / f
        if p.exists():
            frames.append(pd.read_csv(p, header=None, names=columns, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"None of {FULL_FILES} found in {root}")
    return pd.concat(frames, ignore_index=True)


def load(
    root: Optional[Path] = None,
    task: str = "binary",
    form: Optional[str] = None,
    combine_presplit: bool = True,
    subsample: Optional[int] = None,
    seed: int = 42,
    keep_id: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load UNSW-NB15 into (X, y).

    Args:
        root: directory holding the CSVs. Defaults to <repo>/Resource/UNSW-NB15.
        task: "binary" (label 0/1) or "multiclass" (attack_cat 0-9).
        form: "presplit" or "full". Auto-detected when omitted.
        combine_presplit: concatenate the official train+test files (the paper does its own
            80:20 stream split, so the official split is not used as-is).
        subsample: keep only the first N rows -- head, not random: the stream simulation is
            defined on row order.

    Returns:
        (X, y) with X all-numeric (categoricals label-encoded per section 3.2) and y aligned.
    """
    root = Path(root) if root else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Create it and place the UNSW-NB15 CSVs there, or pass "
            f"--data-root. Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
        )

    form = form or _discover(root)
    df = _load_presplit(root, combine_presplit) if form == "presplit" else _load_full(root)
    df.columns = [str(c).strip() for c in df.columns]

    # --- target ---
    if task == "binary":
        if "label" not in df.columns:
            raise ValueError(f"No 'label' column; found {list(df.columns)[:10]}...")
        y = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    elif task == "multiclass":
        if "attack_cat" not in df.columns:
            raise ValueError("No 'attack_cat' column -- multiclass target unavailable.")
        y = _normalise_attack_cat(df["attack_cat"]).to_numpy()
    else:
        raise ValueError(f"task must be 'binary' or 'multiclass', got {task!r}")

    # --- features ---
    X = df.drop(columns=[c for c in ["label", "attack_cat"] if c in df.columns])
    drop_cols = list(_DROP_ALWAYS)
    if keep_id:
        # LEAKAGE. `id` is a row index that tracks class ordering: RF on `id` alone reaches
        # 0.79 binary accuracy, and it restarts from 1 in the testing file. Exposed only to
        # test whether a published score depends on having left it in.
        drop_cols = [c for c in drop_cols if c != "id"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # Section 3.2: "Symbolic attributes are converted to numeric form ... by label encoding".
    # proto / service / state are the categorical columns in this dataset.
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.factorize(X[col].astype(str).str.strip())[0].astype(np.float64)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop columns that are entirely missing; mean-impute the rest (section 3.2).
    X = X.drop(columns=X.columns[X.isna().all()].tolist())
    X = X.fillna(X.mean(numeric_only=True)).fillna(0.0)

    if subsample is not None and subsample < len(X):
        X, y = X.iloc[:subsample].reset_index(drop=True), y[:subsample]

    return X.reset_index(drop=True), y


def class_summary(y: np.ndarray, task: str = "binary") -> pd.DataFrame:
    values, counts = np.unique(y, return_counts=True)
    if task == "binary":
        names = ["Normal" if int(v) == 0 else "Attack" for v in values]
    else:
        names = [LABEL_NAMES.get(int(v), "?") for v in values]
    return pd.DataFrame({
        "label": values, "name": names, "count": counts,
        "pct": (counts / len(y) * 100).round(3),
    })
