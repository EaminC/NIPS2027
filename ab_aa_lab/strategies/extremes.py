"""Keep the current best and worst buckets; redraw only the middle.

This is the in-layer hack:

1. Draw ``n_buckets`` equal buckets once.
2. Publish AB as the best bucket mean minus the worst bucket mean, using only
   exposed users.
3. Freeze those two buckets. Redraw the remaining users into the other
   ``n_buckets - 2`` buckets.
4. If a new bucket beats the current best, it becomes the best. If one falls
   below the current worst, it becomes the worst. The displaced bucket returns
   to the middle pool on the next round.

On a sticky layer the frozen users stay, so the exposed-sample gap ratchets
up. On a reshuffle layer the freeze is ignored and the gap cannot accumulate.
On a large fully exposed population a single 20% bucket mean sits near the
grand mean, so this whole-bucket search does not approach the oracle tail gap.
The inflation shows up in the small-traffic sample and collapses when exposure
ramps to 100%.
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import strategies
from ab_aa_lab.stats import bucket_means
from ab_aa_lab.strategies.base import Strategy
from ab_aa_lab.types import Action, Claim, Observation


def best_worst(means: np.ndarray) -> tuple[int, int] | None:
    if np.all(np.isnan(means)):
        return None
    best = int(np.nanargmax(means))
    worst = int(np.nanargmin(means))
    if best == worst:
        return None
    return best, worst


@strategies.register("extremes_keeper")
class ExtremesKeeper(Strategy):
    def __init__(self, n_buckets: int = 5) -> None:
        if n_buckets < 4:
            raise ValueError(
                "extremes_keeper needs at least 4 buckets so the middle pool "
                "can be split into 2 or more new groups"
            )
        self.n_buckets = n_buckets
        self.best_id: int | None = None
        self.worst_id: int | None = None

    def act(self, obs: Observation) -> Action:
        del obs
        if self.best_id is None or self.worst_id is None or self.best_id == self.worst_id:
            return Action(kind="assign_all", n_buckets=self.n_buckets)
        return Action(
            kind="freeze_reshuffle",
            n_buckets=self.n_buckets,
            freeze_buckets=(self.best_id, self.worst_id),
        )

    def observe(self, obs: Observation) -> None:
        means = bucket_means(obs.values, obs.assignment, obs.exposure, self.n_buckets)
        picked = best_worst(means)
        if picked is None:
            return
        self.best_id, self.worst_id = picked

    def claim(self, obs: Observation) -> Claim:
        del obs
        if self.best_id is None or self.worst_id is None:
            return Claim(control_bucket=0, treatment_bucket=1)
        return Claim(control_bucket=self.worst_id, treatment_bucket=self.best_id)
