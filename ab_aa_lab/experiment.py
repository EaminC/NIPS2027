"""Run one shared latent world through one or more strategy/system pairs.

A suite draws S(i), its later path, and the exposure order once. Each run
then has its own assignment stream. Differences between runs are therefore
the strategy and the layer policy, not a different population.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import ab_aa_lab.effects  # noqa: F401  (register builtins)
import ab_aa_lab.metrics  # noqa: F401
import ab_aa_lab.policies  # noqa: F401
import ab_aa_lab.processes  # noqa: F401
import ab_aa_lab.strategies  # noqa: F401
from ab_aa_lab.plugins import load_plugin
from ab_aa_lab.policies import apply_action
from ab_aa_lab.registry import effects, metrics, policies, processes, strategies
from ab_aa_lab.stats import arm_delta, bucket_means, mask_from_order, tail_bounds
from ab_aa_lab.types import Claim, Observation, Phase

_RUN_KEYS = {"name", "system", "strategy", "effect"}


@dataclass
class World:
    n_users: int
    snapshots: list[np.ndarray]
    baseline_mean: float
    exposure_order: np.ndarray
    phases: list[Phase]
    default_exposure: float


@dataclass
class RunResult:
    name: str
    system: str
    strategy: str
    arm_fraction: float
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def parse_phases(raw: list[dict], default_exposure: float) -> list[Phase]:
    if not raw:
        raise ValueError("config needs at least one phase")
    phases: list[Phase] = []
    for item in raw:
        exposure = item.get("exposure", default_exposure)
        phases.append(
            Phase(
                name=str(item["name"]),
                rounds=int(item["rounds"]),
                evolve=bool(item.get("evolve", False)),
                strategy_active=bool(item.get("strategy_active", True)),
                exposure=float(exposure) if exposure is not None else None,
            )
        )
    return phases


def build_world(config: dict) -> World:
    n_users = int(config["n_users"])
    if n_users < 2:
        raise ValueError("n_users must be at least 2")
    default_exposure = float(config.get("exposure", 1.0))
    if not 0 < default_exposure <= 1:
        raise ValueError("exposure must be in (0, 1]")
    phases = parse_phases(config["phases"], default_exposure)
    seed = int(config.get("seed", 0))
    stream = np.random.default_rng(seed)
    metric_rng = np.random.default_rng(int(stream.integers(0, 2**63 - 1)))
    process_rng = np.random.default_rng(int(stream.integers(0, 2**63 - 1)))
    exposure_rng = np.random.default_rng(int(stream.integers(0, 2**63 - 1)))

    metric = _build(metrics, config.get("metric", {"name": "normal"}))
    process = _build(processes, config.get("process", {"name": "static"}))
    values = np.asarray(metric.initial(n_users, metric_rng), dtype=float)
    if values.shape != (n_users,):
        raise ValueError("metric.initial must return shape (n_users,)")
    baseline_mean = float(values.mean())

    snapshots: list[np.ndarray] = []
    for phase in phases:
        for _ in range(phase.rounds):
            if phase.evolve:
                values = np.asarray(process.step(np.array(values, copy=True), process_rng), dtype=float)
            shot = np.array(values, dtype=float, copy=True)
            shot.setflags(write=False)
            snapshots.append(shot)

    order = np.arange(n_users)
    exposure_rng.shuffle(order)
    return World(
        n_users=n_users,
        snapshots=snapshots,
        baseline_mean=baseline_mean,
        exposure_order=order,
        phases=phases,
        default_exposure=default_exposure,
    )


def run_suite(config: dict, *, config_dir: Path | None = None) -> list[RunResult]:
    root = config_dir or Path.cwd()
    for plugin in config.get("plugins") or []:
        load_plugin(_resolve_plugin(plugin, root))
    world = build_world(config)
    results = []
    base_seed = int(config.get("seed", 0))
    for index, run_cfg in enumerate(expand_runs(config)):
        assignment_seed = base_seed + 10007 * (index + 1)
        results.append(run_one(world, run_cfg, assignment_seed=assignment_seed))
    return results


def run_one(world: World, config: dict, *, assignment_seed: int) -> RunResult:
    policy = policies.create(str(config["system"]))
    strategy = _build(strategies, config["strategy"])
    effect = _build(effects, config.get("effect", {"name": "null"}))
    arm_fraction = float(strategy.arm_fraction)
    if not 0 < arm_fraction <= 0.5:
        raise ValueError("strategy.arm_fraction must be in (0, 0.5]")
    rng = np.random.default_rng(assignment_seed)
    assignment = np.full(world.n_users, -1, dtype=np.int64)
    rows: list[dict] = []
    snapshot_index = 0

    for phase in world.phases:
        rate = world.default_exposure if phase.exposure is None else phase.exposure
        exposure = mask_from_order(world.exposure_order, world.n_users, float(rate))
        for _ in range(phase.rounds):
            latent = world.snapshots[snapshot_index]
            snapshot_index += 1
            if phase.strategy_active:
                obs = _observation(snapshot_index - 1, phase.name, latent, assignment, exposure, strategy)
                action = strategy.act(obs)
                assignment = apply_action(policy, assignment, action, rng)
                assignment.setflags(write=False)
                obs = _observation(snapshot_index - 1, phase.name, latent, assignment, exposure, strategy)
                strategy.observe(obs)
            claim = strategy.claim(
                _observation(snapshot_index - 1, phase.name, latent, assignment, exposure, strategy)
            )
            measured = np.asarray(effect.apply(latent, assignment, claim), dtype=float)
            rows.append(
                _row(
                    round_index=snapshot_index - 1,
                    phase=phase,
                    exposure_rate=float(rate),
                    exposure=exposure,
                    latent=latent,
                    measured=measured,
                    assignment=assignment,
                    claim=claim,
                    baseline_mean=world.baseline_mean,
                    arm_fraction=arm_fraction,
                    n_buckets=int(strategy.n_buckets),
                    run_name=str(config.get("name", strategy.name)),
                    system=str(config["system"]),
                    strategy_name=str(strategy.name),
                )
            )

    result = RunResult(
        name=str(config.get("name", strategy.name)),
        system=str(config["system"]),
        strategy=str(strategy.name),
        arm_fraction=arm_fraction,
        rows=rows,
    )
    result.summary = summarize(result)
    return result


def expand_runs(config: dict) -> list[dict]:
    if "runs" not in config:
        _require_run_fields(config)
        return [config]
    expanded = []
    for spec in config["runs"]:
        unknown = set(spec) - _RUN_KEYS
        if unknown:
            raise ValueError(
                "each run may only override "
                f"{sorted(_RUN_KEYS)}; got {sorted(unknown)}. "
                "Metric, process, phases, and seed are shared by the suite."
            )
        merged = {key: value for key, value in config.items() if key not in {"runs", "plugins"}}
        merged.update(spec)
        _require_run_fields(merged)
        expanded.append(merged)
    if not expanded:
        raise ValueError("runs is empty")
    return expanded


def summarize(result: RunResult) -> dict:
    rows = result.rows
    if not rows:
        return {}
    active = [row for row in rows if row["strategy_active"]]
    search_end = active[-1] if active else rows[-1]
    final = rows[-1]
    partial = [row for row in rows if row["exposure_rate"] < 1 - 1e-12]
    full = [row for row in rows if row["exposure_rate"] >= 1 - 1e-12]
    summary = {
        "name": result.name,
        "system": result.system,
        "strategy": result.strategy,
        "arm_fraction": result.arm_fraction,
        "ab_claim_search_end": search_end["ab_claim"],
        "ab_population_search_end": search_end["ab_population"],
        "ab_claim_final": final["ab_claim"],
        "ab_population_final": final["ab_population"],
        "aa_final": final["aa"],
        "aa_latent_final": final["aa_latent"],
        "oracle_upper_final": final["oracle_upper"],
        "oracle_lower_final": final["oracle_lower"],
        "claim_fade": search_end["ab_population"] - final["ab_population"],
        "sample_optimism": search_end["ab_claim"] - search_end["ab_population"],
    }
    if partial and full:
        summary["ramp_drop"] = partial[-1]["ab_claim"] - full[0]["ab_population"]
    else:
        summary["ramp_drop"] = None
    return summary


def _build(registry, spec):
    if isinstance(spec, str):
        return registry.create(spec)
    if isinstance(spec, dict):
        payload = dict(spec)
        try:
            name = payload.pop("name")
        except KeyError as exc:
            raise ValueError(f"{registry.kind} spec needs a name") from exc
        return registry.create(str(name), **payload)
    raise TypeError(f"{registry.kind} spec must be a string or an object")


def _require_run_fields(config: dict) -> None:
    for key in ("system", "strategy"):
        if key not in config:
            raise ValueError(f"config needs {key!r}")


def _resolve_plugin(plugin: str, config_dir: Path) -> Path:
    path = Path(plugin)
    if path.is_file():
        return path
    candidate = config_dir / plugin
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(plugin)


def _observation(round_index, phase, latent, assignment, exposure, strategy) -> Observation:
    return Observation(
        round_index=round_index,
        phase=phase,
        values=latent,
        assignment=assignment,
        exposure=exposure,
        n_buckets=int(strategy.n_buckets),
    )


def _row(
    *,
    round_index: int,
    phase: Phase,
    exposure_rate: float,
    exposure: np.ndarray,
    latent: np.ndarray,
    measured: np.ndarray,
    assignment: np.ndarray,
    claim: Claim,
    baseline_mean: float,
    arm_fraction: float,
    n_buckets: int,
    run_name: str,
    system: str,
    strategy_name: str,
) -> dict:
    population = np.ones(latent.shape[0], dtype=bool)
    upper, lower = tail_bounds(latent, arm_fraction)
    exposed_values = latent[exposure]
    if exposed_values.size >= 2 and arm_fraction <= 0.5:
        upper_s, lower_s = tail_bounds(exposed_values, arm_fraction)
    else:
        upper_s, lower_s = float("nan"), float("nan")
    means = bucket_means(latent, assignment, exposure, n_buckets)
    finite = means[~np.isnan(means)]
    partition_gap = float(finite.max() - finite.min()) if finite.size >= 2 else float("nan")
    return {
        "run": run_name,
        "system": system,
        "strategy": strategy_name,
        "round": round_index,
        "phase": phase.name,
        "strategy_active": phase.strategy_active,
        "exposure_rate": exposure_rate,
        "n_exposed": int(exposure.sum()),
        "control_bucket": claim.control_bucket,
        "treatment_bucket": claim.treatment_bucket,
        "ab_claim": arm_delta(
            measured, assignment, exposure, claim.control_bucket, claim.treatment_bucket
        ),
        "ab_population": arm_delta(
            measured, assignment, population, claim.control_bucket, claim.treatment_bucket
        ),
        "aa": float(measured.mean() - baseline_mean),
        "aa_latent": float(latent.mean() - baseline_mean),
        "oracle_upper": upper,
        "oracle_lower": lower,
        "oracle_upper_sample": upper_s,
        "oracle_lower_sample": lower_s,
        "partition_gap": partition_gap,
    }
