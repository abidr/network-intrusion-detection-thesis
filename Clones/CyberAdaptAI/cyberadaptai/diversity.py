"""Section 3.3 -- Classifier diversity. Implements Eq 4.

    Q(h_m, h_n) = (N11*N00 - N10*N01) / (N11*N00 + N10*N01)          (Eq 4)

where N_ab counts instances on which h_m is correct/incorrect (a) and h_n is correct/incorrect
(b) -- the standard oracle-output formulation of Yule's Q.

WARNING (README ambiguity 2). Q is +1 when two classifiers agree perfectly and negative when
they disagree. Eq 3 adds the diversity term D_m with a POSITIVE coefficient, and section 3.3
says "If the average disagreement score is high, it indicates that the classifier brings new
decision boundaries, thus strengthening the Ensemble." Substituting raw Q for D_m therefore
rewards redundancy -- the opposite of the stated intent. This module provides Eq 4 verbatim
(`q_statistic`) plus two sign-correct alternatives; the default is `disagreement`.
"""

from typing import Sequence

import numpy as np

from .config import DiversityMeasure


def _oracle(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Oracle output: 1 where the classifier is correct, 0 otherwise."""
    return (np.asarray(pred) == np.asarray(y)).astype(np.int8)


def pairwise_counts(oracle_m: np.ndarray, oracle_n: np.ndarray):
    """Return (N11, N10, N01, N00) for a pair of oracle-output vectors."""
    n11 = int(np.sum((oracle_m == 1) & (oracle_n == 1)))
    n10 = int(np.sum((oracle_m == 1) & (oracle_n == 0)))
    n01 = int(np.sum((oracle_m == 0) & (oracle_n == 1)))
    n00 = int(np.sum((oracle_m == 0) & (oracle_n == 0)))
    return n11, n10, n01, n00


def q_statistic(oracle_m: np.ndarray, oracle_n: np.ndarray) -> float:
    """Eq 4, verbatim. Range [-1, 1]; HIGH means the two classifiers agree."""
    n11, n10, n01, n00 = pairwise_counts(oracle_m, oracle_n)
    num = n11 * n00 - n10 * n01
    den = n11 * n00 + n10 * n01
    if den == 0:
        # Degenerate batch (e.g. every classifier correct on every row). Q is undefined;
        # 0.0 is the neutral choice and keeps Eq 3 finite.
        return 0.0
    return num / den


def disagreement(oracle_m: np.ndarray, oracle_n: np.ndarray) -> float:
    """Disagreement measure: fraction of rows where exactly one of the pair is correct.

    Range [0, 1]; HIGH means diverse. This is what section 3.3's prose describes.
    """
    n11, n10, n01, n00 = pairwise_counts(oracle_m, oracle_n)
    total = n11 + n10 + n01 + n00
    return (n10 + n01) / total if total else 0.0


def neg_q(oracle_m: np.ndarray, oracle_n: np.ndarray) -> float:
    """Sign-corrected Eq 4, rescaled to [0, 1]: (1 - Q) / 2. HIGH means diverse."""
    return (1.0 - q_statistic(oracle_m, oracle_n)) / 2.0


_MEASURES = {
    DiversityMeasure.Q_STATISTIC: q_statistic,
    DiversityMeasure.DISAGREEMENT: disagreement,
    DiversityMeasure.NEG_Q: neg_q,
}


def diversity_scores(
    predictions: Sequence[np.ndarray],
    y: np.ndarray,
    measure: DiversityMeasure = DiversityMeasure.DISAGREEMENT,
) -> np.ndarray:
    """D_m for every classifier: its average pairwise diversity against the rest of the pool.

    Section 3.3: 'D_m^t [is] the average disagreement with the remaining classifiers'.

    Args:
        predictions: one predicted-label array per base classifier.
        y: true labels for the batch.
        measure: which pairwise measure to average.

    Returns:
        Array of shape (M,) with D_m for each classifier.
    """
    fn = _MEASURES[measure]
    oracles = [_oracle(p, y) for p in predictions]
    m = len(oracles)
    if m < 2:
        return np.zeros(m, dtype=float)

    scores = np.zeros(m, dtype=float)
    for i in range(m):
        vals = [fn(oracles[i], oracles[j]) for j in range(m) if j != i]
        scores[i] = float(np.mean(vals))
    return scores
