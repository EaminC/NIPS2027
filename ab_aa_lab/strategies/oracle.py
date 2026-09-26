"""Infeasible reference: assign the best and worst tails of the current S.

This ignores the hash layer. It is the theoretical AB ceiling (or floor) for
a two-arm report whose arms each contain ``arm_fraction`` of the population.
"""

from __future__ import annotations

from ab_aa_lab.registry import strategies
from ab_aa_lab.stats import assign_tails
from ab_aa_lab.strategies.base import Strategy
from ab_aa_lab.types import Action, Claim, Observation


@strategies.register("oracle")
class Oracle(Strategy):
    infeasible = True

    def __init__(self, direction: str = "upper", arm_fraction: float = 0.5) -> None:
        if direction not in ("upper", "lower"):
            raise ValueError("direction must be 'upper' or 'lower'")
        if not 0 < arm_fraction <= 0.5:
            raise ValueError("arm_fraction must be in (0, 0.5]")
        self.direction = direction
        self._arm_fraction = float(arm_fraction)
        self.n_buckets = 2 if arm_fraction >= 0.5 - 1e-15 else 3

    @property
    def arm_fraction(self) -> float:
        return self._arm_fraction

    def act(self, obs: Observation) -> Action:
        assignment = assign_tails(obs.values, self._arm_fraction)
        return Action(
            kind="explicit",
            n_buckets=self.n_buckets,
            assignment=assignment,
            force=True,
        )

    def claim(self, obs: Observation) -> Claim:
        del obs
        if self.direction == "upper":
            return Claim(control_bucket=0, treatment_bucket=1)
        return Claim(control_bucket=1, treatment_bucket=0)
