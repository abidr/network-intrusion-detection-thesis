"""Section 3.3 -- AdaptEnsembleNet. Implements Eq 2, Eq 3 and Eq 8.

    y_hat_i = argmax_c  sum_m  w_m * I[h_m(x_i) = c]                   (Eq 2)
    w_m^t   = alpha * A_m^t + (1 - alpha) * D_m^t                      (Eq 3)
    w_m^t+1 = gamma * w_m^t + (1 - gamma) * (1 - E_m^t)                (Eq 8)

Eq 3 and Eq 8 are mutually inconsistent -- Eq 3 has no temporal smoothing, Eq 8 has no
diversity term, and Algorithm 1 line 9 applies only Eq 8, which would leave alpha=0.6 and the
whole of Eq 4 unused. See README ambiguity 1. All three readings are selectable via
`Config.weight_rule`; the default HYBRID uses the Eq-3 expression as the target of the Eq-8
exponential smoother, which is the only reading under which every stated hyperparameter is live.
"""

from typing import Dict, List, Optional

import numpy as np

from .base_learners import BaseLearner
from .config import Config, WeightRule
from .diversity import diversity_scores


class AdaptEnsembleNet:
    """Adaptive ensemble with performance- and diversity-driven weighted majority voting."""

    def __init__(self, pool: List[BaseLearner], config: Config, classes: List[int]):
        self.pool = pool
        self.config = config
        self.classes = np.array(sorted(classes))
        self._class_index = {c: i for i, c in enumerate(self.classes)}

        # Section 4.1: "Adaptive weights were initialized equally"
        m = len(pool)
        self.weights = np.full(m, 1.0 / m, dtype=float)

        # E_m^t, the exponentially weighted moving average error per classifier (Eq 8)
        self.ewma_error = np.zeros(m, dtype=float)
        self._ewma_initialised = False

        self.history: List[Dict] = []

    # --- training ---------------------------------------------------------------
    def fit(self, X, y):
        """Initial batch fit of every learner in the pool."""
        for learner in self.pool:
            learner.fit(X, y, classes=self.classes.tolist())
        return self

    def partial_fit(self, X, y):
        """Algorithm 1 line 12 -- update online-capable learners only."""
        for learner in self.pool:
            if learner.supports_online:
                learner.partial_fit(X, y, classes=self.classes.tolist())
        return self

    # --- Eq 2 -------------------------------------------------------------------
    def member_predictions(self, X) -> List[np.ndarray]:
        return [learner.predict(X) for learner in self.pool]

    def vote(self, member_preds: List[np.ndarray]) -> np.ndarray:
        """Eq 2: weighted majority vote over hard labels."""
        n = len(member_preds[0])
        scores = np.zeros((n, len(self.classes)), dtype=float)
        for w, preds in zip(self.weights, member_preds):
            for ci, c in enumerate(self.classes):
                scores[:, ci] += w * (np.asarray(preds) == c)  # w_m * I[h_m(x_i) = c]
        self._last_vote_scores = scores
        return self.classes[np.argmax(scores, axis=1)]

    def predict(self, X) -> np.ndarray:
        return self.vote(self.member_predictions(X))

    def vote_scores(self, X) -> np.ndarray:
        """Normalised vote mass per class -- the ensemble-agreement confidence of section 3.5."""
        self.vote(self.member_predictions(X))
        total = self._last_vote_scores.sum(axis=1, keepdims=True)
        return self._last_vote_scores / np.where(total == 0, 1.0, total)

    # --- Eq 3 / Eq 8 ------------------------------------------------------------
    def update_weights(self, member_preds: List[np.ndarray], y: np.ndarray) -> Dict:
        """Recompute w_m after observing the true labels of a batch.

        Algorithm 1 lines 8-9. Returns a diagnostic record for the trajectory log.
        """
        y = np.asarray(y)
        accuracies = np.array([np.mean(np.asarray(p) == y) for p in member_preds], dtype=float)
        errors = 1.0 - accuracies

        # E_m^t: EWMA of the per-classifier error.
        if not self._ewma_initialised:
            self.ewma_error = errors.copy()
            self._ewma_initialised = True
        else:
            g = self.config.gamma
            self.ewma_error = g * self.ewma_error + (1.0 - g) * errors

        divers = diversity_scores(member_preds, y, self.config.diversity_measure)

        rule = self.config.weight_rule
        alpha, gamma = self.config.alpha, self.config.gamma

        if rule == WeightRule.EQ3:
            raw = alpha * accuracies + (1.0 - alpha) * divers
        elif rule == WeightRule.EQ8:
            raw = gamma * self.weights + (1.0 - gamma) * (1.0 - self.ewma_error)
        elif rule == WeightRule.HYBRID:
            target = alpha * accuracies + (1.0 - alpha) * divers
            raw = gamma * self.weights + (1.0 - gamma) * target
        else:
            raise ValueError(f"unknown weight rule: {rule}")

        raw = np.clip(raw, 1e-9, None)
        # Eq 2 is scale-invariant, but normalising keeps the trajectory readable and stops
        # the Eq-8 recursion from drifting in magnitude.
        self.weights = raw / raw.sum()

        record = {
            "accuracies": accuracies.tolist(),
            "diversity": divers.tolist(),
            "ewma_error": self.ewma_error.tolist(),
            "weights": self.weights.tolist(),
        }
        self.history.append(record)
        return record

    # --- Algorithm 1 line 11 ----------------------------------------------------
    def retrain_underperformers(self, X, y, accuracies: Optional[np.ndarray] = None) -> List[str]:
        """'Reset or retrain underperforming classifiers in H'.

        The paper does not define "underperforming". Policy here: any learner whose accuracy on
        the triggering batch is below the pool mean is refit on the current window.
        """
        if accuracies is None:
            preds = self.member_predictions(X)
            accuracies = np.array([np.mean(np.asarray(p) == np.asarray(y)) for p in preds])

        threshold = float(np.mean(accuracies))
        retrained = []
        for learner, acc in zip(self.pool, accuracies):
            if acc < threshold:
                learner.reset()
                learner.fit(X, y, classes=self.classes.tolist())
                retrained.append(learner.name)
        return retrained

    @property
    def learner_names(self) -> List[str]:
        return [l.name for l in self.pool]
