from pathlib import Path

from ab_aa_lab.experiment import run_suite
from ab_aa_lab.plugins import load_plugin
from ab_aa_lab.registry import strategies


ROOT = Path(__file__).resolve().parents[1]


def test_plugin_file_registers_and_runs():
    load_plugin(ROOT / "examples" / "custom_strategy.py")
    assert "peek_once" in strategies.names()
    result = run_suite(
        {
            "seed": 4,
            "n_users": 2000,
            "exposure": 1.0,
            "metric": {"name": "normal"},
            "process": {"name": "static"},
            "phases": [
                {"name": "search", "rounds": 3, "evolve": False, "strategy_active": True, "exposure": 1.0}
            ],
            "system": "sticky",
            "strategy": {"name": "peek_once", "n_buckets": 5},
        }
    )[0]
    assert result.strategy == "peek_once"
    gaps = [row["ab_claim"] for row in result.rows]
    assert gaps[0] == gaps[1] == gaps[2]
    assert gaps[0] == result.rows[0]["partition_gap"]
    assert gaps[0] > 0
