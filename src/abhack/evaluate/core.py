"""收益的定义。

AB 是当前样本里 B 组均值减去 A 组均值。
AA 是全体用户现在的均值减去起始均值。
上界是当前分组里最好的一组减去最差的一组，下界是反过来。
"""

import numpy as np


def ab_gain(scores: np.ndarray, arm: np.ndarray, mask: np.ndarray) -> float:
    """arm 里 1 是 B，0 是 A，其余不进这次 AB。"""
    scores = np.asarray(scores, dtype=float)
    arm = np.asarray(arm)
    use = np.asarray(mask, dtype=bool)
    treated = scores[use & (arm == 1)]
    control = scores[use & (arm == 0)]
    if treated.size == 0 or control.size == 0:
        return float("nan")
    return float(treated.mean() - control.mean())


def aa_gain(before: np.ndarray, after: np.ndarray) -> float:
    """全体用户的均值变化。"""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.size == 0 or after.size == 0:
        return float("nan")
    return float(after.mean() - before.mean())


def group_bounds(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """返回 (上界, 下界)。组内没有落在 mask 里的人不参与。"""
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
