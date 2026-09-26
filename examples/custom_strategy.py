"""一次偷看的策略：只随机分一次桶，把最好桶和最差桶当作 AB，然后保持不动。

这是热插拔示例。把它登记进实验不需要改 runner：

    {"plugins": ["../examples/custom_strategy.py"],
     "strategy": {"name": "peek_once", "n_buckets": 5}}

``plugins`` 路径相对于配置文件所在目录。
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import strategies
from ab_aa_lab.stats import bucket_means
from ab_aa_lab.strategies.base import Strategy
from ab_aa_lab.types import Action, Claim, Observation


@strategies.register("peek_once")
class PeekOnce(Strategy):
    def __init__(self, n_buckets: int = 5) -> None:
        if n_buckets < 2:
            raise ValueError("peek_once needs at least 2 buckets")
        self.n_buckets = n_buckets
        self.best_id = 1
        self.worst_id = 0
        self._opened = False
        self._locked = False

    def act(self, obs: Observation) -> Action:
        del obs
        if not self._opened:
            self._opened = True
            return Action(kind="assign_all", n_buckets=self.n_buckets)
        return Action(kind="hold", n_buckets=self.n_buckets)

    def observe(self, obs: Observation) -> None:
        if self._locked:
            return
        means = bucket_means(obs.values, obs.assignment, obs.exposure, self.n_buckets)
        if np.all(np.isnan(means)):
            return
        best = int(np.nanargmax(means))
        worst = int(np.nanargmin(means))
        if best == worst:
            return
        self.best_id = best
        self.worst_id = worst
        self._locked = True

    def claim(self, obs: Observation) -> Claim:
        del obs
        return Claim(control_bucket=self.worst_id, treatment_bucket=self.best_id)
