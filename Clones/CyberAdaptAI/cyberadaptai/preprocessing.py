"""Section 3.2 -- Dataset preparation. Implements Eq 1 (min-max normalisation).

    x'_ij = (x_ij - min(x_j)) / (max(x_j) - min(x_j))          (Eq 1)

The paper never states the scope of min/max in a streaming setting. See README ambiguity 3.
"""

import numpy as np
import pandas as pd

from .config import Config, ScalerFit


class StreamPreprocessor:
    """Cleaning, label encoding and Eq-1 min-max scaling for a simulated stream."""

    def __init__(self, config: Config):
        self.config = config
        self._min = None
        self._max = None
        self._fitted = False
        self._impute_values = None
        self._cat_maps = {}

    # --- Eq 1 ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "StreamPreprocessor":
        """Fit min/max and imputation statistics."""
        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        self._impute_values = X.mean(axis=0) if self.config.impute_strategy == "mean" else X.median(axis=0)
        self._fitted = True
        return self

    def _update_running(self, X: pd.DataFrame) -> None:
        self._min = X.min(axis=0) if self._min is None else np.minimum(self._min, X.min(axis=0))
        self._max = X.max(axis=0) if self._max is None else np.maximum(self._max, X.max(axis=0))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("StreamPreprocessor.fit() must be called before transform()")

        X = X.copy()

        # "mean imputation of missing values" (section 3.2)
        X = X.fillna(self._impute_values)
        # Eq 1 is undefined on non-finite values; treat them as missing.
        X = X.replace([np.inf, -np.inf], np.nan).fillna(self._impute_values).fillna(0.0)

        if self.config.scaler_fit == ScalerFit.RUNNING:
            self._update_running(X)

        span = (self._max - self._min).replace(0, 1.0)  # constant column -> maps to 0
        scaled = (X - self._min) / span
        # A running/first-batch fit can see values outside the fitted range in later batches.
        return scaled.clip(0.0, 1.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # --- Section 3.2 cleaning ---------------------------------------------------
    @staticmethod
    def clean(X: pd.DataFrame, y: pd.Series, drop_duplicates: bool = True):
        """'data cleaning includes mean imputation of missing values and eliminating
        duplicate records' (section 3.2).

        NOTE: the paper deduplicates the whole dataset before constructing the stream, so
        duplicates are removed across the train/test boundary rather than within it. That is
        the leakage-safe ordering only if dedup happens before the split, which it does here.
        The paper reports no duplicate counts; this function returns them.
        """
        n_before = len(X)
        if drop_duplicates:
            combined = X.copy()
            combined["__label__"] = y.values
            combined = combined.drop_duplicates()
            y = combined.pop("__label__")
            X = combined
        return X.reset_index(drop=True), y.reset_index(drop=True), n_before - len(X)

    @staticmethod
    def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
        """'Symbolic attributes are converted to numeric form either by label encoding or
        one-hot encoding' (section 3.2). Section 4.1 settles on label encoding.
        """
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object or str(X[col].dtype) == "category":
                X[col] = pd.factorize(X[col])[0].astype(np.float64)
        return X
