"""Adapter: load this thesis's CIC-UNSW-NB15-2024 dataset into CyberAdaptAI's stream format.

The paper uses CIC-IDS2017 and UNSW-NB15. Neither is in this repo, so this adapter targets the
local Resource/ dataset, which is the 2024 CICFlowMeter re-extraction of the same UNSW-NB15
attack taxonomy -- close enough for a like-for-like comparison against the thesis baselines.

IMPORTANT CAVEAT for any drift claim made on this data: the two CSVs are row-aligned by
position with no timestamp column, so "chronological" batching is really row-order batching.
Whether that ordering carries temporal structure is unverified. The same caveat applies to the
paper's own use of CIC-IDS2017/UNSW-NB15.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Confirmed by the project owner: attack families are alphabetical, Benign = 0.
LABEL_NAMES = {
    0: "Benign", 1: "Analysis", 2: "Backdoor", 3: "DoS", 4: "Exploits",
    5: "Fuzzers", 6: "Generic", 7: "Reconnaissance", 8: "Shellcode", 9: "Worms",
}

# Verified zero-variance columns (see CLAUDE.md). Dropped before anything else: they carry no
# information, and a zero-variance column makes Eq 1's (max - min) denominator zero.
ZERO_VARIANCE_COLUMNS = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
]

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "Resource"


def load(
    root: Optional[Path] = None,
    drop_zero_variance: bool = True,
    subsample: Optional[int] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load CIC-UNSW-Data.csv + CIC-UNSW-Label.csv.

    Args:
        root: directory holding the two CSVs. Defaults to <repo>/Resource.
        drop_zero_variance: drop the 9 verified constant columns.
        subsample: keep only the first N rows (row order preserved -- required, since the
            stream simulation depends on it). Useful for smoke tests.

    Returns:
        (X, y) with X numeric and row-aligned to y.
    """
    root = Path(root) if root else DEFAULT_ROOT
    data_path, label_path = root / "CIC-UNSW-Data.csv", root / "CIC-UNSW-Label.csv"

    for p in (data_path, label_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. The dataset is git-excluded (187MB); see the repo README "
                f"for setup, or pass --data-root."
            )

    X = pd.read_csv(data_path)
    y_df = pd.read_csv(label_path)

    if len(X) != len(y_df):
        raise ValueError(
            f"Row misalignment: data has {len(X):,} rows, labels {len(y_df):,}. "
            f"These files are aligned by position only -- never sort one without the other."
        )

    y = y_df["Label"].to_numpy()

    if drop_zero_variance:
        present = [c for c in ZERO_VARIANCE_COLUMNS if c in X.columns]
        X = X.drop(columns=present)

    if subsample is not None and subsample < len(X):
        # Head, not a random sample: the stream simulation is defined on row order.
        X, y = X.iloc[:subsample].reset_index(drop=True), y[:subsample]

    return X, y


def class_summary(y: np.ndarray) -> pd.DataFrame:
    values, counts = np.unique(y, return_counts=True)
    return pd.DataFrame({
        "label": values,
        "name": [LABEL_NAMES.get(int(v), "?") for v in values],
        "count": counts,
        "pct": (counts / len(y) * 100).round(3),
    })
