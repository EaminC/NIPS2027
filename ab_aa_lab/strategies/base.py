"""Strategy contract.

A strategy is a stateful object with three methods:

- ``act(obs) -> Action`` decides how to touch the layer before seeing this
  round's post-action metric.
- ``observe(obs)`` is called after the action has been applied, on the same
  latent values. Selection strategies update their best/worst buckets here.
- ``claim(obs) -> Claim`` chooses the published A/B contrast.

The runner calls ``observe`` and ``act`` only while the phase is
``strategy_active``. During a later hold/ramp phase the last claim and the
last assignment stay put, which is what makes fade and ramp measurable.

New strategies belong in their own module and register themselves. See
``examples/custom_strategy.py``.
"""

from __future__ import annotations

from ab_aa_lab.types import Action, Claim, Observation


class Strategy:
    name = "strategy"
    n_buckets = 2
    infeasible = False

    @property
    def arm_fraction(self) -> float:
        return 1.0 / self.n_buckets

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def observe(self, obs: Observation) -> None:
        del obs

    def claim(self, obs: Observation) -> Claim:
        raise NotImplementedError
