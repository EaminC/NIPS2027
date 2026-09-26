import pytest

from ab_aa_lab.experiment import run_suite


def _base(**overrides):
    config = {
        "seed": 0,
        "n_users": 6000,
        "exposure": 1.0,
        "metric": {"name": "normal", "mu": 0.0, "sigma": 1.0},
        "process": {"name": "static"},
        "phases": [
            {"name": "search", "rounds": 12, "evolve": False, "strategy_active": True, "exposure": 1.0}
        ],
        "system": "sticky",
        "strategy": {"name": "extremes_keeper", "n_buckets": 5},
    }
    config.update(overrides)
    return config


def test_sticky_extremes_gap_is_monotone_and_aa_stays_flat():
    result = run_suite(_base())[0]
    gaps = [row["ab_claim"] for row in result.rows]
    assert all(gaps[i + 1] + 1e-9 >= gaps[i] for i in range(len(gaps) - 1))
    assert gaps[-1] == pytest.approx(max(gaps))
    assert gaps[-1] <= result.rows[-1]["oracle_upper"] + 1e-8
    for row in result.rows:
        assert row["ab_claim"] == pytest.approx(row["partition_gap"])
        assert row["aa_latent"] == pytest.approx(0.0, abs=1e-12)
        assert row["ab_claim"] <= row["oracle_upper"] + 1e-8


def test_only_the_sticky_layer_keeps_the_extreme_buckets():
    results = run_suite(
        _base(
            n_users=8000,
            phases=[
                {
                    "name": "search",
                    "rounds": 15,
                    "evolve": False,
                    "strategy_active": True,
                    "exposure": 0.1,
                }
            ],
            runs=[
                {
                    "name": "sticky",
                    "system": "sticky",
                    "strategy": {"name": "extremes_keeper", "n_buckets": 5},
                },
                {
                    "name": "reshuffle",
                    "system": "reshuffle",
                    "strategy": {"name": "extremes_keeper", "n_buckets": 5},
                },
            ],
        )
    )
    by_name = {result.name: result for result in results}
    sticky = [row["ab_claim"] for row in by_name["sticky"].rows]
    reshuffle = [row["ab_claim"] for row in by_name["reshuffle"].rows]
    assert all(sticky[i + 1] + 1e-9 >= sticky[i] for i in range(len(sticky) - 1))
    assert sticky[-1] == pytest.approx(max(sticky))
    assert any(reshuffle[i + 1] < reshuffle[i] - 1e-4 for i in range(len(reshuffle) - 1))
    assert by_name["sticky"].summary["sample_optimism"] > 0.08
    assert by_name["sticky"].summary["ab_population_search_end"] < 0.12
    assert by_name["sticky"].summary["aa_latent_final"] == pytest.approx(0.0, abs=1e-12)


def test_oracle_matches_the_tail_bound():
    results = run_suite(
        _base(
            phases=[
                {"name": "search", "rounds": 1, "evolve": False, "strategy_active": True, "exposure": 1.0}
            ],
            runs=[
                {
                    "name": "upper",
                    "system": "sticky",
                    "strategy": {"name": "oracle", "direction": "upper", "arm_fraction": 0.2},
                },
                {
                    "name": "lower",
                    "system": "sticky",
                    "strategy": {"name": "oracle", "direction": "lower", "arm_fraction": 0.2},
                },
            ],
        )
    )
    by_name = {result.name: result.rows[0] for result in results}
    assert by_name["upper"]["ab_population"] == pytest.approx(by_name["upper"]["oracle_upper"])
    assert by_name["lower"]["ab_population"] == pytest.approx(by_name["lower"]["oracle_lower"])
    assert by_name["upper"]["oracle_lower"] == pytest.approx(-by_name["upper"]["oracle_upper"])


def test_honest_labels_stay_preregistered_and_inside_the_bound():
    result = run_suite(
        _base(
            n_users=4000,
            strategy={"name": "honest", "n_buckets": 2},
            phases=[
                {"name": "search", "rounds": 4, "evolve": False, "strategy_active": True, "exposure": 1.0}
            ],
        )
    )[0]
    claims = [(row["control_bucket"], row["treatment_bucket"]) for row in result.rows]
    assert claims == [(0, 1)] * 4
    assert result.rows[0]["ab_claim"] != result.rows[1]["ab_claim"]
    for row in result.rows:
        assert abs(row["ab_population"]) <= row["oracle_upper"] + 1e-8


def test_ramp_publishes_the_full_population_contrast():
    result = run_suite(
        _base(
            n_users=10000,
            phases=[
                {
                    "name": "search",
                    "rounds": 10,
                    "evolve": False,
                    "strategy_active": True,
                    "exposure": 0.1,
                },
                {
                    "name": "ramp",
                    "rounds": 1,
                    "evolve": False,
                    "strategy_active": False,
                    "exposure": 1.0,
                },
            ],
        )
    )[0]
    search = [row for row in result.rows if row["phase"] == "search"]
    ramp = result.rows[-1]
    assert search[-1]["exposure_rate"] == pytest.approx(0.1)
    assert ramp["ab_claim"] == pytest.approx(ramp["ab_population"])
    assert ramp["n_exposed"] == 10000
    assert search[-1]["n_exposed"] == 1000
    assert result.summary["sample_optimism"] > 0


def test_frozen_gap_fades_under_mean_reversion():
    result = run_suite(
        _base(
            n_users=8000,
            process={"name": "ar1", "phi": 0.5, "sigma": 0.05, "mu": 0.0},
            phases=[
                {"name": "search", "rounds": 1, "evolve": False, "strategy_active": True, "exposure": 1.0},
                {"name": "hold", "rounds": 8, "evolve": True, "strategy_active": False, "exposure": 1.0},
            ],
            strategy={"name": "oracle", "direction": "upper", "arm_fraction": 0.2},
        )
    )[0]
    search_gap = result.summary["ab_population_search_end"]
    assert search_gap > 2.0
    assert result.summary["ab_population_final"] < 0.05 * search_gap
    assert abs(result.summary["aa_latent_final"]) < 0.05


def test_constant_lift_moves_ab_and_aa_without_touching_the_latent_path():
    result = run_suite(
        _base(
            n_users=5000,
            phases=[
                {"name": "search", "rounds": 1, "evolve": False, "strategy_active": True, "exposure": 1.0}
            ],
            strategy={"name": "oracle", "direction": "upper", "arm_fraction": 0.2},
            effect={"name": "constant_lift", "delta": 0.4},
        )
    )[0]
    row = result.rows[0]
    assert row["ab_population"] == pytest.approx(row["oracle_upper"] + 0.4)
    assert row["aa"] - row["aa_latent"] == pytest.approx(0.4 * 0.2)
    assert row["aa_latent"] == pytest.approx(0.0, abs=1e-12)


def test_suite_shares_one_latent_path():
    results = run_suite(
        _base(
            process={"name": "ar1", "phi": 0.8, "sigma": 0.2, "mu": 0.0},
            phases=[
                {"name": "search", "rounds": 1, "evolve": False, "strategy_active": True},
                {"name": "hold", "rounds": 5, "evolve": True, "strategy_active": False},
            ],
            runs=[
                {"name": "a", "system": "sticky", "strategy": {"name": "honest", "n_buckets": 2}},
                {"name": "b", "system": "reshuffle", "strategy": {"name": "honest", "n_buckets": 2}},
            ],
        )
    )
    latent_a = [row["aa_latent"] for row in results[0].rows]
    latent_b = [row["aa_latent"] for row in results[1].rows]
    assert latent_a == latent_b
    assert results[0].rows[-1]["oracle_upper"] < 0.7 * results[0].rows[0]["oracle_upper"]
    assert results[0].rows[0]["oracle_upper"] == results[1].rows[0]["oracle_upper"]


def test_run_cannot_override_the_shared_world():
    with pytest.raises(ValueError):
        run_suite(_base(runs=[{"name": "bad", "system": "sticky", "n_users": 10}]))
