"""Traffic-layer assignment policies.

``sticky`` is the hackable system: a user's bucket is fixed when the layer
hash is first drawn, and a later in-layer toggle does not move them unless
the strategy explicitly releases their bucket.

``reshuffle`` is the unhackable system: every enter/exit of the layer draws
a fresh bucket, so a request to keep the best and worst buckets is ignored.
"""

from __future__ import annotations

import numpy as np

from ab_aa_lab.registry import policies
from ab_aa_lab.stats import balanced_assign, balanced_into
from ab_aa_lab.types import Action


def freeze_reshuffle(
    assignment: np.ndarray,
    action: Action,
    rng: np.random.Generator,
) -> np.ndarray:
    freeze = tuple(action.freeze_buckets)
    if len(set(freeze)) != len(freeze):
        raise ValueError("freeze_buckets contains duplicates")
    for bucket in freeze:
        if not 0 <= bucket < action.n_buckets:
            raise ValueError(f"freeze bucket {bucket} is outside 0..{action.n_buckets - 1}")
    free_ids = [bucket for bucket in range(action.n_buckets) if bucket not in set(freeze)]
    out = np.array(assignment, dtype=np.int64, copy=True)
    if freeze:
        movable = ~np.isin(out, freeze)
    else:
        movable = np.ones(out.shape[0], dtype=bool)
    n_movable = int(movable.sum())
    if n_movable == 0 or not free_ids:
        return out
    out[movable] = balanced_into(n_movable, free_ids, rng)
    return out


class Policy:
    name = "policy"

    def apply(
        self,
        assignment: np.ndarray,
        action: Action,
        rng: np.random.Generator,
    ) -> np.ndarray:
        raise NotImplementedError


def _full_draw(assignment: np.ndarray, action: Action, rng: np.random.Generator) -> np.ndarray:
    return balanced_assign(assignment.shape[0], action.n_buckets, rng)


@policies.register("sticky")
class StickyHash(Policy):
    """Hackable layer: hash sticks unless the user is explicitly released."""

    def apply(
        self,
        assignment: np.ndarray,
        action: Action,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if action.kind == "hold":
            return np.array(assignment, copy=True)
        if action.kind == "new_layer":
            return _full_draw(assignment, action, rng)
        if action.kind == "assign_all":
            if np.any(assignment >= 0):
                return np.array(assignment, copy=True)
            return _full_draw(assignment, action, rng)
        if action.kind == "freeze_reshuffle":
            return freeze_reshuffle(assignment, action, rng)
        if action.kind == "explicit":
            raise PermissionError("sticky hash refuses explicit reassignment; use force on the action")
        raise ValueError(f"unsupported action {action.kind}")


@policies.register("reshuffle")
class ReshuffleOnToggle(Policy):
    """Unhackable layer: every enter/exit redraws every user."""

    def apply(
        self,
        assignment: np.ndarray,
        action: Action,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if action.kind == "hold":
            return np.array(assignment, copy=True)
        if action.kind in ("assign_all", "new_layer", "freeze_reshuffle"):
            return _full_draw(assignment, action, rng)
        if action.kind == "explicit":
            raise PermissionError("reshuffle layer refuses explicit reassignment; use force on the action")
        raise ValueError(f"unsupported action {action.kind}")


def apply_action(
    policy: Policy,
    assignment: np.ndarray,
    action: Action,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply ``action``, bypassing ``policy`` only when ``action.force`` is set."""
    if action.force:
        if action.assignment is None:
            raise ValueError("a forced action requires an explicit assignment")
        if action.assignment.shape != assignment.shape:
            raise ValueError("explicit assignment length does not match the population")
        return np.array(action.assignment, dtype=np.int64, copy=True)
    return policy.apply(assignment, action, rng)
