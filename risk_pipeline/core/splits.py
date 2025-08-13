from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, List


@dataclass(frozen=True)
class SplitParams:
    n: int
    train_size: int
    val_size: int
    step: int


def generate_sliding_splits(n: int, train_size: int, val_size: int, step: int) -> Iterator[Tuple[slice, slice]]:
    """Yield deterministic sliding windows as (train_slice, val_slice).

    No leakage: validation window always starts after the train window.
    Bounds-safe: yields only when full train and val windows fit in [0, n).
    """
    if train_size <= 0 or val_size <= 0 or step <= 0 or n <= 0:
        return
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        val_start = train_end
        val_end = val_start + val_size
        if val_end > n:
            break
        yield slice(train_start, train_end), slice(val_start, val_end)
        start += step
