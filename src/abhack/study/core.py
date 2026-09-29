"""System property vs experimenter behavior.

System:
  hackable   -- layer hash is fixed once and never changes
  unhackable -- every entry into the layer draws a fresh random split

Agent:
  honest -- no peeking; A/B is random each round
  hacker -- keep the best and worst groups, reshuffle the middle

Theory bounds (not experiment arms):

  select   -- mean(top b%) - mean(bottom b%) by score.
              Absolute ceiling. Moves only if scores move.
  assigned -- best - worst among groups the *system* made this round.
              Hackable + static scores: flat.
              Unhackable: redrawn every round, so this bound floats.
              A remapping hack can beat assigned, but not select.
"""

from dataclasses import dataclass

import numpy as np

from abhack.evaluate import aa_gain, ab_gain, group_bounds, selection_bounds
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

SYSTEMS = ("hackable", "unhackable")
AGENTS = ("honest", "hacker")
CELLS = tuple((system, agent) for system in SYSTEMS for agent in AGENTS)


@dataclass(frozen=True)
class RoundRow:
    round: int
    system: str
    agent: str
    ab: float
    ab_full: float
    aa: float
    select_upper: float
    select_lower: float
    assigned_upper: float
    assigned_lower: float


def simulate(
    users: int,
    dist: Distribution,
    process: str,
    rounds: int,
    seed: int,
    traffic: float,
    layer: str,
    groups: int,
    phi: float = 0.0,
    shock: float = 0.0,
) -> list[RoundRow]:
    """One score path. Cells are system x agent. Theory is select + assigned."""
    if users < groups:
        raise ValueError("need at least as many users as groups")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if groups < 3:
        raise ValueError("groups must be at least 3 so the extremes hack has a middle")

    rng = np.random.default_rng(seed)
    ids = generate_users(users)
    start = np.asarray(dist.sample(users, seed), dtype=float)
    scores = start.copy()
    seen = traffic_mask(ids, traffic, layer)
    everyone = np.ones(users, dtype=bool)
    fraction = 1.0 / groups

    sticky = sticky_buckets(ids, groups, layer)
    hacker_sticky = sticky.copy()
    rows: list[RoundRow] = []

    for step in range(rounds):
        if step:
            scores = advance(scores, dist, process, rng, phi=phi, shock=shock)

        aa = aa_gain(start, scores)
        select_upper, select_lower = selection_bounds(scores, seen, fraction)
        # Unhackable: new system groups every round → assigned bound floats.
        fresh = unhackable_buckets(users, groups, rng)
        system_labels = {"hackable": sticky, "unhackable": fresh}

        for system in SYSTEMS:
            labels = system_labels[system]
            assigned_upper, assigned_lower = group_bounds(scores, labels, seen)

            arm = honest_arm(users, rng)
            rows.append(
                RoundRow(
                    step,
                    system,
                    "honest",
                    ab_gain(scores, arm, seen),
                    ab_gain(scores, arm, everyone),
                    aa,
                    select_upper,
                    select_lower,
                    assigned_upper,
                    assigned_lower,
                )
            )

            if system == "hackable":
                if step:
                    hacker_sticky = keep_extremes(
                        scores, hacker_sticky, groups, rng, seen
                    )
                working = hacker_sticky
            else:
                # Fresh system split each round; one keep-extremes step on top.
                working = keep_extremes(scores, labels.copy(), groups, rng, seen)
            arm = label_best_worst(working, scores, seen)
            rows.append(
                RoundRow(
                    step,
                    system,
                    "hacker",
                    ab_gain(scores, arm, seen),
                    ab_gain(scores, arm, everyone),
                    aa,
                    select_upper,
                    select_lower,
                    assigned_upper,
                    assigned_lower,
                )
            )
    return rows
