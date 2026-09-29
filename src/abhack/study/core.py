"""一轮一轮地记 AB、AA 和当前分组的上下界。"""

from dataclasses import dataclass

import numpy as np

from abhack.evaluate import aa_gain, ab_gain, group_bounds
from abhack.process import advance
from abhack.strategies import (
    honest_arm,
    keep_extremes,
    label_best_worst,
    sticky_buckets,
    traffic_mask,
    unhackable_buckets,
)
from abhack.users import generate_users
from abhack.utils.distributions import Distribution

METHODS = ("honest", "hackable", "unhackable", "extremes")


@dataclass(frozen=True)
class RoundRow:
    round: int
    method: str
    ab: float
    ab_full: float
    aa: float
    upper: float
    lower: float


def simulate(
    users: int,
    dist: Distribution,
    process: str,
    rounds: int,
    seed: int,
    traffic: float,
    layer: str,
    buckets: int,
    groups: int,
    phi: float = 0.0,
    shock: float = 0.0,
) -> list[RoundRow]:
    """同一批用户、同一条分数路径，四种做法各记一行。"""
    if users < groups or users < buckets:
        raise ValueError("need at least as many users as groups")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if buckets < 2:
        raise ValueError("bucket count must be at least 2")
    if groups < 3:
        raise ValueError("extremes needs at least 3 groups")

    rng = np.random.default_rng(seed)
    ids = generate_users(users)
    start = np.asarray(dist.sample(users, seed), dtype=float)
    scores = start.copy()
    seen = traffic_mask(ids, traffic, layer)
    everyone = np.ones(users, dtype=bool)
    sticky = sticky_buckets(ids, buckets, layer)
    extreme_groups = sticky_buckets(ids, groups, f"{layer}#extremes")
    rows: list[RoundRow] = []

    for step in range(rounds):
        if step:
            scores = advance(scores, dist, process, rng, phi=phi, shock=shock)
        aa = aa_gain(start, scores)
        rows.append(_honest(step, scores, seen, everyone, aa, rng))
        rows.append(_peeking(step, "hackable", sticky, scores, seen, everyone, aa))
        fresh = unhackable_buckets(users, buckets, rng)
        rows.append(_peeking(step, "unhackable", fresh, scores, seen, everyone, aa))
        if step:
            extreme_groups = keep_extremes(scores, extreme_groups, groups, rng, seen)
        rows.append(_peeking(step, "extremes", extreme_groups, scores, seen, everyone, aa))
    return rows


def _honest(step, scores, seen, everyone, aa, rng) -> RoundRow:
    arm = honest_arm(scores.size, rng)
    return RoundRow(
        step,
        "honest",
        ab_gain(scores, arm, seen),
        ab_gain(scores, arm, everyone),
        aa,
        *_bounds(scores, arm, seen),
    )


def _peeking(step, name, labels, scores, seen, everyone, aa) -> RoundRow:
    arm = label_best_worst(labels, scores, seen)
    return RoundRow(
        step,
        name,
        ab_gain(scores, arm, seen),
        ab_gain(scores, arm, everyone),
        aa,
        *_bounds(scores, labels, seen),
    )


def _bounds(scores, labels, seen) -> tuple[float, float]:
    return group_bounds(scores, labels, seen)
