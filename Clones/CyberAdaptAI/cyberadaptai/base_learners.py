"""Section 4.1 -- the base learner pool.

'The base learner pool in AdaptEnsembleNet consisted of four classifiers: Random Forest,
 XGBoost, Extra Trees, and a Hoeffding Tree.'

Each wrapper exposes a uniform interface so `ensemble.py` can treat online and batch learners
identically:

    fit(X, y, classes)      full (re)fit
    partial_fit(X, y)       incremental update; no-op for batch-only learners
    predict(X)              hard labels
    predict_proba(X)        probabilities, or None if the learner has none
    supports_online         whether partial_fit does anything
"""

from typing import List, Optional

import numpy as np

from .config import Config


class BaseLearner:
    """Uniform wrapper interface."""

    name = "base"
    supports_online = False

    def fit(self, X, y, classes: Optional[List[int]] = None):
        raise NotImplementedError

    def partial_fit(self, X, y, classes: Optional[List[int]] = None):
        """Algorithm 1 line 12: 'Partially fit h_m on B_t if online-capable'."""
        return self

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X) -> Optional[np.ndarray]:
        return None

    def reset(self):
        """Algorithm 1 line 11: 'Reset or retrain underperforming classifiers'."""
        raise NotImplementedError


class SklearnLearner(BaseLearner):
    """Batch learner (Random Forest / Extra Trees). Retrained wholesale on drift."""

    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory
        self.model = factory()

    def fit(self, X, y, classes=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        try:
            return self.model.predict_proba(X)
        except Exception:
            return None

    def reset(self):
        self.model = self._factory()
        return self


class XGBoostLearner(BaseLearner):
    """Section 4.1: learning rate 0.1, 150 estimators, max depth 5.

    Labels are remapped to a contiguous 0..K-1 range because XGBoost requires it; the mapping
    is inverted on predict so callers always see original label values.
    """

    name = "xgboost"

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._classes = None
        self._constant = None

    def _new_model(self):
        from xgboost import XGBClassifier

        return XGBClassifier(
            learning_rate=self.config.xgb_learning_rate,
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            random_state=self.config.seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            # NOTE: num_class is deliberately NOT passed -- with the sklearn wrapper, setting it
            # manually makes a binary model's .predict return a 2-D array.
        )

    def fit(self, X, y, classes=None):
        """Fit on the classes actually PRESENT in y, remapped to a contiguous 0..K-1 range.

        XGBoost validates that the label set is exactly {0, .., K-1} and rejects anything else.
        Keying the remap off the global class list breaks on any subset fit -- which is exactly
        what a drift retrain on a partial window is (`Expected: [0 1 2 3 4 5], got [0 3 4 5 6 7]`).
        `_classes` therefore records what THIS fit can emit, and predict() maps indices back
        through it. The `classes` argument is accepted for interface symmetry but not used to
        build the mapping.
        """
        y = np.asarray(y)
        present = np.array(sorted({int(v) for v in y.tolist()}))
        self._classes = present

        if len(present) < 2:
            # XGBoost cannot train on a single class. Degenerate to a constant predictor rather
            # than crashing -- a retrain window can legitimately be pure Benign.
            self.model = None
            self._constant = int(present[0])
            return self

        self._constant = None
        remap = {c: i for i, c in enumerate(present)}
        y_enc = np.array([remap[int(v)] for v in y])
        self.model = self._new_model()
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        if getattr(self, "_constant", None) is not None:
            return np.full(len(X), self._constant)
        idx = np.asarray(self.model.predict(X)).ravel().astype(int)
        return self._classes[idx]

    def predict_proba(self, X):
        """Probabilities over the classes THIS fit saw -- not necessarily the global set.

        The ensemble votes on hard labels (Eq 2), so this is not used in the pipeline; it is
        here for inspection. Check `classes_` before aligning columns to anything.
        """
        if getattr(self, "_constant", None) is not None:
            return np.ones((len(X), 1))
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return self._classes

    def reset(self):
        self.model = None
        self._constant = None
        return self


class HoeffdingTreeLearner(BaseLearner):
    """Section 4.1: 'sourced from the River library and used default confidence parameters'.

    `river` is an optional dependency. If it is absent, this falls back to an SGD-based online
    learner -- a substitution the paper itself sanctions in section 3.4, which names
    'Hoeffding Trees, adaptive SGD-based models' as interchangeable online-capable learners.
    The fallback is reported in `self.name` so it cannot be silently mistaken for the real thing.
    """

    supports_online = True

    def __init__(self, config: Config):
        self.config = config
        self._impl = None
        self._is_river = False
        self._classes = None
        self._init_impl()

    def _init_impl(self):
        try:
            from river.tree import HoeffdingTreeClassifier

            self._impl = HoeffdingTreeClassifier()
            self._is_river = True
            self.name = "hoeffding_tree"
        except ImportError:
            from sklearn.linear_model import SGDClassifier

            self._impl = SGDClassifier(
                loss="log_loss",
                learning_rate="optimal",
                random_state=self.config.seed,
            )
            self._is_river = False
            self.name = "hoeffding_tree[SGD-fallback]"

    @property
    def is_fallback(self) -> bool:
        return not self._is_river

    def _to_dicts(self, X):
        cols = list(X.columns) if hasattr(X, "columns") else range(np.asarray(X).shape[1])
        arr = X.values if hasattr(X, "values") else np.asarray(X)
        return [dict(zip(cols, row)) for row in arr]

    def fit(self, X, y, classes=None):
        y = np.asarray(y)
        self._classes = np.array(sorted(set(classes) if classes is not None else set(y.tolist())))
        if self._is_river:
            for xi, yi in zip(self._to_dicts(X), y):
                self._impl.learn_one(xi, yi)
        else:
            self._impl.partial_fit(X, y, classes=self._classes)
        return self

    def partial_fit(self, X, y, classes=None):
        y = np.asarray(y)
        if self._classes is None:
            return self.fit(X, y, classes)
        if self._is_river:
            for xi, yi in zip(self._to_dicts(X), y):
                self._impl.learn_one(xi, yi)
        else:
            self._impl.partial_fit(X, y, classes=self._classes)
        return self

    def predict(self, X):
        if self._is_river:
            out = [self._impl.predict_one(xi) for xi in self._to_dicts(X)]
            default = self._classes[0] if self._classes is not None else 0
            return np.array([default if o is None else o for o in out])
        return self._impl.predict(X)

    def predict_proba(self, X):
        if self._is_river:
            rows = []
            for xi in self._to_dicts(X):
                d = self._impl.predict_proba_one(xi) or {}
                rows.append([d.get(c, 0.0) for c in self._classes])
            return np.asarray(rows)
        try:
            return self._impl.predict_proba(X)
        except Exception:
            return None

    def reset(self):
        self._init_impl()
        self._classes = None
        return self


def build_pool(config: Config) -> List[BaseLearner]:
    """Construct the four-classifier pool exactly as specified in section 4.1."""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    pool: List[BaseLearner] = [
        SklearnLearner(
            "random_forest",
            lambda: RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                random_state=config.seed,
                n_jobs=-1,
            ),
        ),
        XGBoostLearner(config),
        SklearnLearner(
            "extra_trees",
            lambda: ExtraTreesClassifier(
                n_estimators=config.et_n_estimators,
                criterion=config.et_criterion,
                max_features=config.et_max_features,
                random_state=config.seed,
                n_jobs=-1,
            ),
        ),
    ]
    if config.use_hoeffding_tree:
        pool.append(HoeffdingTreeLearner(config))
    return pool
