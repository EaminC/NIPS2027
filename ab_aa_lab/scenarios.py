"""Turn a demo form into one shared-world suite.

The web lab speaks in hackable / unhackable and in Chinese labels. The runner
still receives ``sticky`` / ``reshuffle`` and the registered strategy names.
Oracle arms use 20% so the bound lines up with the five-bucket hack, not with
the honest 50/50 split.
"""

from __future__ import annotations

from ab_aa_lab.experiment import run_suite
from ab_aa_lab.reporting import to_builtin

RUN_ORDER = (
    "honest",
    "extremes_hackable",
    "extremes_unhackable",
    "oracle_upper",
    "oracle_lower",
)

RUNS = {
    "honest": {
        "name": "honest",
        "system": "sticky",
        "strategy": {"name": "honest", "n_buckets": 2},
        "label": "诚实 baseline",
        "role": "feasible",
        "blurb": "每轮重新随机。A/B 事先指定，不把更好的桶改叫实验组。",
    },
    "extremes_hackable": {
        "name": "extremes_hackable",
        "system": "sticky",
        "strategy": {"name": "extremes_keeper", "n_buckets": 5},
        "label": "hackable · 极值保留",
        "role": "feasible",
        "blurb": "五组各 20%。留下最好和最差，中间三组重分；更好或更差就替换。hash 保持不变。",
    },
    "extremes_unhackable": {
        "name": "extremes_unhackable",
        "system": "reshuffle",
        "strategy": {"name": "extremes_keeper", "n_buckets": 5},
        "label": "unhackable · 极值保留",
        "role": "feasible",
        "blurb": "同一套保留请求。每次进出流量层都整层重抽，最好和最差留不住。",
    },
    "oracle_upper": {
        "name": "oracle_upper",
        "system": "sticky",
        "strategy": {"name": "oracle", "direction": "upper", "arm_fraction": 0.2},
        "label": "理论上界",
        "role": "oracle",
        "blurb": "当前 S 上最好的 20% 减最差的 20%。不经过 hash，只作参照。",
    },
    "oracle_lower": {
        "name": "oracle_lower",
        "system": "sticky",
        "strategy": {"name": "oracle", "direction": "lower", "arm_fraction": 0.2},
        "label": "理论下界",
        "role": "oracle",
        "blurb": "当前 S 上最差的 20% 减最好的 20%。",
    },
}

METRICS = {
    "normal": {"name": "normal", "mu": 0.0, "sigma": 1.0},
    "lognormal": {"name": "lognormal", "mu": 0.0, "sigma": 0.5},
    "uniform": {"name": "uniform", "low": 0.0, "high": 1.0},
}

METRIC_LABELS = {
    "normal": "正态 Normal(0, 1)",
    "lognormal": "对数正态 LogNormal(0, 0.5)",
    "uniform": "均匀 Uniform(0, 1)",
}

SYSTEM_LABELS = {"sticky": "hackable", "reshuffle": "unhackable"}

SERIES_FIELDS = (
    "round",
    "phase",
    "strategy_active",
    "exposure_rate",
    "n_exposed",
    "ab_claim",
    "ab_population",
    "aa",
    "aa_latent",
    "oracle_upper",
    "oracle_lower",
    "partition_gap",
)


def presets() -> dict:
    shared = {
        "n_users": 3000,
        "seed": 7,
        "metric": "normal",
        "exposure_percent": 10,
        "search_rounds": 30,
        "hold_rounds": 16,
        "phi": 0.85,
        "sigma": 0.15,
        "compare": list(RUN_ORDER),
    }
    selection = dict(shared, process="static", afterward="ramp")
    fade = dict(shared, seed=11, process="ar1", afterward="hold", search_rounds=24)
    return {"selection": selection, "fade": fade}


def catalog() -> dict:
    return {
        "presets": presets(),
        "metrics": [{"id": key, "label": METRIC_LABELS[key]} for key in METRICS],
        "processes": [
            {"id": "static", "label": "不随时间变化"},
            {"id": "ar1", "label": "AR(1) 均值回归"},
            {"id": "random_walk", "label": "随机游走"},
        ],
        "afterwards": [
            {"id": "stop", "label": "只做搜索"},
            {"id": "ramp", "label": "搜索后推全到 100%"},
            {"id": "hold", "label": "保持分组，指标继续演化"},
        ],
        "runs": [
            {"id": key, "label": RUNS[key]["label"], "role": RUNS[key]["role"], "blurb": RUNS[key]["blurb"]}
            for key in RUN_ORDER
        ],
    }


def config_from_form(form: dict) -> dict:
    if not isinstance(form, dict):
        raise ValueError("请求体必须是对象")
    n_users = _int(form.get("n_users"), "用户数", 1000, 50000)
    seed = _int(form.get("seed", 0), "随机种子", 0, 1_000_000_000)
    exposure_percent = _int(form.get("exposure_percent"), "小流量比例", 1, 100)
    search_rounds = _int(form.get("search_rounds"), "搜索轮数", 1, 40)
    hold_rounds = _int(form.get("hold_rounds", 1), "演化轮数", 1, 40)
    metric_name = form.get("metric", "normal")
    process_name = form.get("process", "static")
    afterward = form.get("afterward", "stop")
    if metric_name not in METRICS:
        raise ValueError("请选择正态、对数正态或均匀分布")
    if process_name not in ("static", "ar1", "random_walk"):
        raise ValueError("请选择一种时间过程")
    if afterward not in ("stop", "ramp", "hold"):
        raise ValueError("请选择搜索之后的阶段")
    phi = _float(form.get("phi", 0.85), "phi", -1, 1)
    sigma = _float(form.get("sigma", 0.15), "sigma", 0, 2)
    chosen = form.get("compare", list(RUN_ORDER))
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("至少选择一种对比")
    unknown = [item for item in chosen if item not in RUNS]
    if unknown:
        raise ValueError("包含未知的对比策略")
    exposure = exposure_percent / 100
    phases = [
        {
            "name": "search",
            "rounds": search_rounds,
            "evolve": False,
            "strategy_active": True,
            "exposure": exposure,
        }
    ]
    if afterward == "ramp":
        phases.append(
            {
                "name": "ramp",
                "rounds": 1,
                "evolve": False,
                "strategy_active": False,
                "exposure": 1.0,
            }
        )
    elif afterward == "hold":
        phases.append(
            {
                "name": "hold",
                "rounds": hold_rounds,
                "evolve": True,
                "strategy_active": False,
                "exposure": exposure,
            }
        )
    return {
        "seed": seed,
        "n_users": n_users,
        "exposure": exposure,
        "metric": dict(METRICS[metric_name]),
        "process": _process(process_name, phi, sigma),
        "phases": phases,
        "runs": [{key: spec[key] for key in ("name", "system", "strategy")} for spec in _selected(chosen)],
    }


def run_form(form: dict) -> dict:
    config = config_from_form(form)
    results = run_suite(config)
    payload = _present(config, results)
    return to_builtin(payload)


def _selected(chosen: list) -> list[dict]:
    picked = set(chosen)
    return [RUNS[key] for key in RUN_ORDER if key in picked]


def _process(name: str, phi: float, sigma: float) -> dict:
    if name == "ar1":
        return {"name": "ar1", "phi": phi, "sigma": sigma, "mu": 0.0}
    if name == "random_walk":
        return {"name": "random_walk", "sigma": sigma, "drift": 0.0}
    return {"name": "static"}


def _present(config: dict, results) -> dict:
    by_name = {result.name: result for result in results}
    first = next(iter(by_name.values())).rows
    return {
        "n_users": config["n_users"],
        "seed": config["seed"],
        "metric": config["metric"]["name"],
        "process": config["process"]["name"],
        "phases": _phase_spans(first),
        "runs": [_run_payload(RUNS[name], by_name[name]) for name in RUN_ORDER if name in by_name],
    }


def _run_payload(spec: dict, result) -> dict:
    active = [row for row in result.rows if row["strategy_active"]]
    search_end = active[-1] if active else result.rows[-1]
    final = result.rows[-1]
    summary = dict(result.summary)
    summary["n_exposed_search"] = search_end["n_exposed"]
    summary["n_exposed_final"] = final["n_exposed"]
    summary["exposure_search"] = search_end["exposure_rate"]
    system = result.system
    return {
        "id": result.name,
        "label": spec["label"],
        "role": spec["role"],
        "system": SYSTEM_LABELS.get(system, system) if spec["role"] != "oracle" else "参照",
        "strategy": result.strategy,
        "arm_fraction": result.arm_fraction,
        "summary": summary,
        "series": [{key: row[key] for key in SERIES_FIELDS} for row in result.rows],
    }


def _phase_spans(rows: list[dict]) -> list[dict]:
    spans: list[dict] = []
    for row in rows:
        if not spans or spans[-1]["name"] != row["phase"]:
            spans.append({"name": row["phase"], "start": row["round"], "end": row["round"]})
        else:
            spans[-1]["end"] = row["round"]
    return spans


def _int(value, key: str, lo: int, hi: int) -> int:
    number = _number(value, key)
    if not float(number).is_integer():
        raise ValueError(f"{key}必须为整数")
    number = int(number)
    if not lo <= number <= hi:
        raise ValueError(f"{key}范围为 {lo}–{hi}")
    return number


def _float(value, key: str, lo: float, hi: float) -> float:
    number = float(_number(value, key))
    if not lo <= number <= hi:
        raise ValueError(f"{key}范围为 {lo:g}–{hi:g}")
    return number


def _number(value, key: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}必须为数字")
    return value
