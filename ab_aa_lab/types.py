"""Shared data objects passed between the runner, policies, and strategies.

Actions are requests. Whether a request is honored depends on the assignment
policy (the traffic-layer system), except ``force=True``, which bypasses the
policy and is reserved for infeasible reference strategies such as the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ActionKind = Literal["assign_all", "new_layer", "freeze_reshuffle", "hold", "explicit"]


@dataclass(eq=False)
class Action:
    """A strategy's request for one round.

    ``assign_all``
        Draw the layer hash for users who do not have one yet. On a sticky
        layer this does not move users who are already assigned.
    ``new_layer``
        Open a fresh layer salt and redraw every user. Both systems honor it.
        This is a new experiment, not an in-layer edit.
    ``freeze_reshuffle``
        Keep users currently in ``freeze_buckets`` and redraw everyone else
        into the remaining bucket ids. A sticky layer honors the freeze. A
        reshuffle layer treats any enter/exit as a full redraw and ignores it.
    ``hold``
        Leave the current hash untouched.
    ``explicit``
        Install ``assignment`` directly. Requires ``force=True``.
    """

    kind: ActionKind
    n_buckets: int = 2
    freeze_buckets: tuple[int, ...] = ()
    assignment: np.ndarray | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if self.n_buckets < 1:
            raise ValueError("n_buckets must be positive")
        if self.kind == "explicit":
            if self.assignment is None:
                raise ValueError("explicit action requires assignment")
            self.force = True


@dataclass(frozen=True)
class Claim:
    """Pre-registered or selected AB contrast: mean(treatment) - mean(control)."""

    control_bucket: int
    treatment_bucket: int


@dataclass(eq=False)
class Observation:
    """What a strategy can read at decision time.

    ``values`` is the full population metric, before any measurement-only
    effect is applied. Feasible strategies must restrict themselves to
    ``exposure``; ``visible_values`` does that for them. ``values`` itself is
    available because oracle bounds are defined on the current population.
    """

    round_index: int
    phase: str
    values: np.ndarray
    assignment: np.ndarray
    exposure: np.ndarray
    n_buckets: int

    def __post_init__(self) -> None:
        if self.values.shape != self.assignment.shape or self.values.shape != self.exposure.shape:
            raise ValueError("values, assignment, and exposure must share a shape")

    @property
    def n_users(self) -> int:
        return int(self.values.shape[0])

    @property
    def visible_values(self) -> np.ndarray:
        out = np.array(self.values, dtype=float, copy=True)
        out[~self.exposure] = np.nan
        return out


@dataclass(frozen=True)
class Phase:
    name: str
    rounds: int
    evolve: bool = False
    strategy_active: bool = True
    exposure: float | None = None

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f"phase {self.name!r} must have at least one round")
        if self.exposure is not None and not 0 < self.exposure <= 1:
            raise ValueError(f"phase {self.name!r} exposure must be in (0, 1]")
