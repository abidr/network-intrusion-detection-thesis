"""Section 3.4 -- Concept drift detection. Implements Eq 6-7 (ADWIN) and the section-4.1
accuracy-drop trigger.

    |mu_1 - mu_2| > eps_cut                                            (Eq 6)
    eps_cut = sqrt( (1/2) * ln(4/delta) * (1/n_1 + 1/n_2) )            (Eq 7)

The paper sources ADWIN from `river`. This module implements Eq 6-7 directly so drift detection
has no optional dependency, and so the cut condition matches the paper's stated formula rather
than river's internal variant (river uses a different, tighter bound with a variance term).
Set `use_river=True` to delegate to river.drift.ADWIN if it is installed.
"""

import math
from collections import deque
from typing import List, Optional

import numpy as np


class ADWIN:
    """Adaptive Windowing over a stream of errors, using the paper's Eq 6-7 cut condition.

    Maintains a window of recent error indicators and tests every split point. When a split
    shows a mean difference exceeding eps_cut, drift is flagged and the older sub-window is
    dropped.

    This is the classic "check all cut points" ADWIN0 rather than the exponential-histogram
    ADWIN2. It is O(W) per update, bounded by `max_window`.
    """

    def __init__(self, delta: float = 0.002, max_window: int = 2000, min_subwindow: int = 30):
        self.delta = delta
        self.max_window = max_window
        self.min_subwindow = min_subwindow
        self.window: deque = deque(maxlen=max_window)
        self.n_detections = 0

    def eps_cut(self, n1: int, n2: int) -> float:
        """Eq 7."""
        if n1 <= 0 or n2 <= 0:
            return float("inf")
        return math.sqrt(0.5 * math.log(4.0 / self.delta) * (1.0 / n1 + 1.0 / n2))

    def update(self, error: float) -> bool:
        """Add one error observation (1.0 = misclassified). Returns True if drift detected."""
        self.window.append(float(error))
        if len(self.window) < 2 * self.min_subwindow:
            return False

        arr = np.fromiter(self.window, dtype=float)
        n = len(arr)
        # Prefix sums make every candidate split O(1).
        prefix = np.cumsum(arr)
        total = prefix[-1]

        for cut in range(self.min_subwindow, n - self.min_subwindow + 1):
            n1, n2 = cut, n - cut
            mu1 = prefix[cut - 1] / n1
            mu2 = (total - prefix[cut - 1]) / n2
            if abs(mu1 - mu2) > self.eps_cut(n1, n2):  # Eq 6
                # Section 3.4: shrink the window by discarding the stale sub-window.
                for _ in range(cut):
                    self.window.popleft()
                self.n_detections += 1
                return True
        return False

    def update_batch(self, errors) -> bool:
        """Feed a batch of error indicators. Returns True if drift fired at any point."""
        detected = False
        for e in np.asarray(errors).ravel():
            detected |= self.update(e)
        return detected

    def reset(self) -> None:
        self.window.clear()


class AccuracyDropMonitor:
    """Section 4.1: 'A sliding window of size five was used to observe a drift in the accuracy
    of the model. If accuracy decreases consistently by more than 8% over three consecutive
    batches, weaker classifiers are retrained with new data from the current window.'

    Interpretation: fire when accuracy has fallen monotonically for `n_batches` consecutive
    batches AND the cumulative drop across that run exceeds `threshold`. The paper's wording
    ("decreases consistently ... by more than 8% over three") supports reading the 8% as the
    total drop over the run rather than per-batch; per-batch would be a far harsher trigger.
    """

    def __init__(self, window: int = 5, threshold: float = 0.08, n_batches: int = 3):
        self.window = window
        self.threshold = threshold
        self.n_batches = n_batches
        self.history: deque = deque(maxlen=max(window, n_batches + 1))

    def update(self, accuracy: float) -> bool:
        self.history.append(float(accuracy))
        if len(self.history) < self.n_batches + 1:
            return False

        recent: List[float] = list(self.history)[-(self.n_batches + 1):]
        monotone_decline = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
        cumulative_drop = recent[0] - recent[-1]
        return monotone_decline and cumulative_drop > self.threshold

    def reset(self) -> None:
        self.history.clear()


class DriftDetector:
    """Combines the two triggers described in the paper.

    Section 3.4 uses ADWIN on the error stream; section 4.1 adds the accuracy-drop rule.
    Algorithm 1 line 10 says "If drift is detected (e.g., via ADWIN on error stream)" -- either
    trigger is treated as drift.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 2000,
        accuracy_window: int = 5,
        accuracy_drop_threshold: float = 0.08,
        accuracy_drop_batches: int = 3,
        use_river: bool = False,
    ):
        self.accuracy_monitor = AccuracyDropMonitor(
            window=accuracy_window,
            threshold=accuracy_drop_threshold,
            n_batches=accuracy_drop_batches,
        )
        self._river = None
        if use_river:
            try:
                from river.drift import ADWIN as RiverADWIN

                self._river = RiverADWIN(delta=delta)
            except ImportError:
                use_river = False
        self.adwin = None if self._river else ADWIN(delta=delta, max_window=max_window)
        self.last_reason: Optional[str] = None

    def update(self, errors, batch_accuracy: float) -> bool:
        if self._river is not None:
            adwin_fired = False
            for e in np.asarray(errors).ravel():
                self._river.update(float(e))
                adwin_fired |= bool(self._river.drift_detected)
        else:
            adwin_fired = self.adwin.update_batch(errors)

        accuracy_fired = self.accuracy_monitor.update(batch_accuracy)

        if adwin_fired and accuracy_fired:
            self.last_reason = "adwin+accuracy"
        elif adwin_fired:
            self.last_reason = "adwin"
        elif accuracy_fired:
            self.last_reason = "accuracy_drop"
        else:
            self.last_reason = None

        return adwin_fired or accuracy_fired
