"""Section 3.2 -- Stream simulation.

'the preprocessed dataset D is split into several consecutive mini-batches
 B_t = {(x_i, y_i)}, where t is the batch index'

Section 4.1 fixes the batch size at 1,500-2,000 samples.

CAVEAT worth carrying into any write-up: neither CIC-IDS2017, UNSW-NB15, nor CIC-UNSW-NB15-2024
is a true stream. Batches are row-order chunks. Whether the row order encodes real temporal
drift is an assumption the paper does not test.
"""

from dataclasses import dataclass
from typing import Iterator, List

import numpy as np
import pandas as pd


@dataclass
class Batch:
    """A mini-batch B_t."""

    index: int
    X: pd.DataFrame
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def class_distribution(self) -> dict:
        """Section 3.2: 'We also keep track of the class distribution of each batch.'"""
        values, counts = np.unique(self.y, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}


class BatchStream:
    """Chunks a dataset into consecutive mini-batches, preserving row order."""

    def __init__(self, X: pd.DataFrame, y: np.ndarray, batch_size: int = 2000):
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
        self.X = X.reset_index(drop=True)
        self.y = np.asarray(y)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.y) / self.batch_size))

    def __iter__(self) -> Iterator[Batch]:
        for i in range(len(self)):
            lo, hi = i * self.batch_size, min((i + 1) * self.batch_size, len(self.y))
            yield Batch(index=i, X=self.X.iloc[lo:hi], y=self.y[lo:hi])

    def split_sequential_holdout(self, train_ratio: float = 0.8):
        """Section 3.2: 'a sequential holdout approach, in which prior batches are used for
        training and following batches are used for evaluation', at a fixed 80:20 ratio.

        Returns (train_stream, test_stream). The split is on batch boundaries, so no batch
        straddles the boundary.
        """
        n_batches = len(self)
        n_train = max(1, int(round(n_batches * train_ratio)))
        cut = n_train * self.batch_size
        return (
            BatchStream(self.X.iloc[:cut], self.y[:cut], self.batch_size),
            BatchStream(self.X.iloc[cut:], self.y[cut:], self.batch_size),
        )

    def all_classes(self) -> List[int]:
        return sorted(int(c) for c in np.unique(self.y))
