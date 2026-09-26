"""Pure functions for balanced hashing and AB / AA / tail statistics."""

from __future__ import annotations

import numpy as np


def balanced_assign(n: int, n_buckets: int, rng: np.random.Generator) -> np.ndarray:
    """Balanced uniform assignment into ``0 .. n_buckets-1``."""
    if n_buckets < 1:
        raise ValueError("n_buckets must be positive")
    if n_buckets > n:
        raise ValueError("n_buckets cannot exceed the number of users")
    labels = (np.arange(n) % n_buckets).astype(np.int64)
    rng.shuffle(labels)
    return labels


def balanced_into(
    n: int,
    bucket_ids: tuple[int, ...] | list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Balanced assignment into an explicit list of bucket ids."""
    ids = np.asarray(list(bucket_ids), dtype=np.int64)
    if n == 0:
        return np.array([], dtype=np.int64)
    if ids.size == 0:
        raise ValueError("no destination buckets")
    order = ids.copy()
    rng.shuffle(order)
    counts = np.full(order.size, n // order.size, dtype=np.int64)
    counts[: n % order.size] += 1
    out = np.repeat(order, counts)
    rng.shuffle(out)
    return out


def tail_count(n: int, fraction: float) -> int:
    if n < 2:
        raise ValueError("need at least two users")
    if not 0 < fraction <= 0.5:
        raise ValueError("arm fraction must be in (0, 0.5]")
    k = int(round(n * fraction))
    return min(max(k, 1), n // 2)


def tail_bounds(values: np.ndarray, fraction: float) -> tuple[float, float]:
    """Return (best b% mean - worst b% mean, worst b% mean - best b% mean).

    The two groups are disjoint tails of the current values. This is the
    largest and smallest AB contrast attainable by any grouping whose arms
    each contain that fraction of ``values``, ignoring hash constraints.
    """
    x = np.sort(np.asarray(values, dtype=float))
    k = tail_count(x.size, fraction)
    worst = float(x[:k].mean())
    best = float(x[-k:].mean())
    return best - worst, worst - best


def assign_tails(values: np.ndarray, fraction: float) -> np.ndarray:
    """Bucket 0 = worst tail, bucket 1 = best tail, bucket 2 = everyone else."""
    n = int(values.shape[0])
    k = tail_count(n, fraction)
    order = np.argsort(values, kind="mergesort")
    assignment = np.full(n, 2, dtype=np.int64)
    assignment[order[:k]] = 0
    assignment[order[-k:]] = 1
    return assignment


def bucket_means(
    values: np.ndarray,
    assignment: np.ndarray,
    mask: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    means = np.full(n_buckets, np.nan, dtype=float)
    usable = mask & ~np.isnan(values)
    for bucket in range(n_buckets):
        selected = usable & (assignment == bucket)
        if np.any(selected):
            means[bucket] = float(values[selected].mean())
    return means


def arm_delta(
    values: np.ndarray,
    assignment: np.ndarray,
    mask: np.ndarray,
    control_bucket: int,
    treatment_bucket: int,
) -> float:
    """Mean(treatment) - mean(control) on ``mask``."""
    usable = mask & ~np.isnan(values)
    treated = usable & (assignment == treatment_bucket)
    control = usable & (assignment == control_bucket)
    if not np.any(treated) or not np.any(control):
        return float("nan")
    return float(values[treated].mean() - values[control].mean())


def mask_from_order(order: np.ndarray, n: int, rate: float) -> np.ndarray:
    """Nested exposure: a higher rate adds users, it does not replace them."""
    if not 0 < rate <= 1:
        raise ValueError("exposure rate must be in (0, 1]")
    k = n if rate >= 1 else int(round(n * rate))
    k = min(max(k, 1), n)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask
