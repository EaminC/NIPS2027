"""Gain definitions.

AB: mean(B) - mean(A) on the current sample.
AA: population mean now - population mean at the start.

Two theory bounds:
  selection_bounds -- mean(top b%) - mean(bottom b%) by score.
                     Assumes the best people can all sit in B.
  group_bounds     -- best group - worst group on an already assigned partition.
"""

import numpy as np


def ab_gain(scores: np.ndarray, arm: np.ndarray, mask: np.ndarray) -> float:
    """arm 1 is B, 0 is A. Other labels are out of this AB."""
    scores = np.asarray(scores, dtype=float)
    arm = np.asarray(arm)
    use = np.asarray(mask, dtype=bool)
    treated = scores[use & (arm == 1)]
    control = scores[use & (arm == 0)]
    if treated.size == 0 or control.size == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def aa_gain(before: np.ndarray, after: np.ndarray) -> float:
    """Change in the population mean."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.size == 0 or after.size == 0:
        return float("nan")
    return float(after.mean() - before.mean())


def selection_bounds(
    scores: np.ndarray, mask: np.ndarray, fraction: float
) -> tuple[float, float]:
    """Theory (upper, lower): best fraction minus worst fraction by score."""
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    scores = np.asarray(scores, dtype=float)
    use = np.asarray(mask, dtype=bool)
    values = scores[use]
    if values.size < 2:
        return float("nan"), float("nan")
    k = max(1, int(round(values.size * fraction)))
    k = min(k, values.size // 2)
    ordered = np.sort(values)
    best = float(ordered[-k:].mean())
    worst = float(ordered[:k].mean())
    return best - worst, worst - best


def group_bounds(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Best group mean minus worst group mean on a fixed partition (and reverse)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    use = np.asarray(mask, dtype=bool)
    means = []
    for label in np.unique(labels):
        seen = scores[use & (labels == label)]
        if seen.size:
            means.append(float(seen.mean()))
    if len(means) < 2:
        return float("nan"), float("nan")
    best = max(means)
    worst = min(means)
    return best - worst, worst - best
