"""分配规则。

hackable：层内哈希一开始就定了，后面不再变。
unhackable：每次进入这一层都重新随机分桶。
honest：随机分成两组，不按分数挑哪一组当 B。
extremes：留住最好和最差的一组，中间几组重分；更好就换掉最好，更差就换掉最差。
"""

import numpy as np

from abhack.utils.hash import hash_buckets


def traffic_mask(user_ids: np.ndarray, fraction: float, layer: str) -> np.ndarray:
    """小流量里的人。fraction 为 1 时是全体。同一个人每次都在或不在。"""
    if not 0 < fraction <= 1:
        raise ValueError("traffic must be in (0, 1]")
    ids = np.asarray(user_ids)
    if fraction == 1:
        return np.ones(ids.shape, dtype=bool)
    slots = hash_buckets(ids, 10_000, f"{layer}#traffic")
    return slots < int(round(fraction * 10_000))


def sticky_buckets(user_ids: np.ndarray, n_buckets: int, layer: str) -> np.ndarray:
    """哈希一次就固定。放大到全量时，同一个人还在同一个桶。"""
    return hash_buckets(user_ids, n_buckets, layer)


def unhackable_buckets(n_users: int, n_buckets: int, rng: np.random.Generator) -> np.ndarray:
    """这一轮重新随机分桶，不记住上一轮。"""
    if n_buckets < 2:
        raise ValueError("bucket count must be at least 2")
    return rng.integers(0, n_buckets, size=n_users)


def honest_arm(n_users: int, rng: np.random.Generator) -> np.ndarray:
    """随机一半是 B。不看分数，所以不会把更好的那一半叫成 B。"""
    if n_users < 2:
        raise ValueError("need at least 2 users")
    arm = np.zeros(n_users, dtype=np.int64)
    arm[: n_users // 2] = 1
    rng.shuffle(arm)
    return arm


def label_best_worst(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按 mask 里看到的均值，最好的一组标成 B，最差的一组标成 A。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    means: dict[int, float] = {}
    for label in np.unique(labels):
        seen = scores[mask & (labels == label)]
        if seen.size:
            means[int(label)] = float(seen.mean())
    arm = np.full(labels.shape, -1, dtype=np.int64)
    if len(means) < 2:
        return arm
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    arm[labels == best] = 1
    if best != worst:
        arm[labels == worst] = 0
    return arm


def keep_extremes(
    scores: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
    mask: np.ndarray,
) -> np.ndarray:
    """留住最好和最差，把中间重分成同样多的组，更好或更差就替换。"""
    if n_groups < 3:
        raise ValueError("the extremes hack needs at least 3 groups")
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    members = [np.flatnonzero(groups == label) for label in range(n_groups)]

    def mean_of(index: np.ndarray) -> float | None:
        if index.size == 0:
            return None
        seen = index[mask[index]]
        if seen.size == 0:
            return None
        return float(scores[seen].mean())

    means = [mean_of(index) for index in members]
    ranked = [i for i, value in enumerate(means) if value is not None]
    if len(ranked) < 2:
        return groups
    best = max(ranked, key=lambda i: means[i])
    worst = min(ranked, key=lambda i: means[i])
    if best == worst:
        return groups
    middle_labels = [i for i in range(n_groups) if i != best and i != worst]
    middle = np.concatenate([members[i] for i in middle_labels])
    if middle.size == 0:
        return groups
    rng.shuffle(middle)
    parts = list(np.array_split(middle, len(middle_labels)))
    part_means = [mean_of(part) for part in parts]

    better = [
        (part_means[j], j)
        for j in range(len(parts))
        if part_means[j] is not None and part_means[j] > means[best]
    ]
    if better:
        _, chosen = max(better)
        members[best], parts[chosen] = parts[chosen], members[best]
        part_means[chosen] = means[best]

    worse = [
        (part_means[j], j)
        for j in range(len(parts))
        if part_means[j] is not None and part_means[j] < means[worst]
    ]
    if worse:
        _, chosen = min(worse)
        members[worst], parts[chosen] = parts[chosen], members[worst]

    updated = np.empty_like(groups)
    updated[members[best]] = best
    updated[members[worst]] = worst
    for label, part in zip(middle_labels, parts):
        updated[part] = label
    return updated
