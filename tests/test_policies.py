import numpy as np

from ab_aa_lab.policies import ReshuffleOnToggle, StickyHash
from ab_aa_lab.stats import balanced_assign, balanced_into, mask_from_order, tail_bounds
from ab_aa_lab.types import Action


def test_tail_bounds_are_negatives_and_dominate_a_random_split():
    rng = np.random.default_rng(0)
    values = rng.normal(size=4000)
    upper, lower = tail_bounds(values, 0.2)
    assert lower == -upper
    assert upper > 0
    assignment = balanced_assign(4000, 5, rng)
    gap = values[assignment == assignment.max()].mean() - values[assignment == 0].mean()
    assert abs(gap) <= upper + 1e-12


def test_exposure_is_nested():
    order = np.arange(1000)
    small = mask_from_order(order, 1000, 0.1)
    mid = mask_from_order(order, 1000, 0.5)
    full = mask_from_order(order, 1000, 1.0)
    assert small.sum() == 100
    assert np.all(mid[small])
    assert full.sum() == 1000


def test_balanced_into_uses_only_requested_buckets():
    rng = np.random.default_rng(1)
    labels = balanced_into(300, [1, 3, 4], rng)
    assert set(np.unique(labels)) == {1, 3, 4}
    _, counts = np.unique(labels, return_counts=True)
    assert counts.max() - counts.min() <= 1


def test_sticky_freeze_keeps_users_and_second_assign_is_a_noop():
    rng = np.random.default_rng(2)
    assignment = balanced_assign(1000, 5, rng)
    frozen = np.isin(assignment, [0, 1])
    policy = StickyHash()
    held = policy.apply(
        assignment,
        Action(kind="freeze_reshuffle", n_buckets=5, freeze_buckets=(0, 1)),
        rng,
    )
    assert np.array_equal(held[frozen], assignment[frozen])
    assert set(np.unique(held[~frozen])).issubset({2, 3, 4})
    again = policy.apply(held, Action(kind="assign_all", n_buckets=5), rng)
    assert np.array_equal(again, held)
    fresh = policy.apply(held, Action(kind="new_layer", n_buckets=5), rng)
    assert not np.array_equal(fresh, held)


def test_reshuffle_ignores_freeze():
    rng = np.random.default_rng(3)
    assignment = balanced_assign(2000, 5, rng)
    members = np.where(assignment == 0)[0]
    out = ReshuffleOnToggle().apply(
        assignment,
        Action(kind="freeze_reshuffle", n_buckets=5, freeze_buckets=(0, 1)),
        rng,
    )
    _, counts = np.unique(out[members], return_counts=True)
    assert counts.max() / members.size < 0.5
    held = ReshuffleOnToggle().apply(out, Action(kind="hold", n_buckets=5), rng)
    assert np.array_equal(held, out)
