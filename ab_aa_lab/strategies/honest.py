"""Honest baseline: a fresh pre-registered randomization every active round.

Buckets 0 and 1 are declared before looking at outcomes. The strategy never
keeps a lucky split and never relabels the better bucket as treatment.
"""

from __future__ import annotations

from ab_aa_lab.registry import strategies
from ab_aa_lab.strategies.base import Strategy
from ab_aa_lab.types import Action, Claim, Observation


@strategies.register("honest")
class Honest(Strategy):
    def __init__(
        self,
        n_buckets: int = 2,
        control_bucket: int = 0,
        treatment_bucket: int = 1,
    ) -> None:
        if n_buckets < 2:
            raise ValueError("honest needs at least 2 buckets")
        if control_bucket == treatment_bucket:
            raise ValueError("control and treatment buckets must differ")
        if not 0 <= control_bucket < n_buckets or not 0 <= treatment_bucket < n_buckets:
            raise ValueError("claim buckets must lie inside the assignment")
        self.n_buckets = n_buckets
        self.control_bucket = control_bucket
        self.treatment_bucket = treatment_bucket

    def act(self, obs: Observation) -> Action:
        del obs
        return Action(kind="new_layer", n_buckets=self.n_buckets)

    def claim(self, obs: Observation) -> Claim:
        del obs
        return Claim(
            control_bucket=self.control_bucket,
            treatment_bucket=self.treatment_bucket,
        )
