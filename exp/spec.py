"""读取一次实验的 toml。预设写在同一个文件里，用名字引用。"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from abhack.utils.distributions import Distribution, make

_PARAMS = ("mu", "sigma", "low", "high", "mean", "p")


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    preset: str | None
    seed: int
    dist: Distribution


@dataclass(frozen=True)
class RunSpec:
    name: str
    users: int
    buckets: int
    layer: str
    scores: list[ScoreSpec]


def load_run(path: Path) -> RunSpec:
    """读实验文件。分数可以点名预设，再在自己这行改参数。"""
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    presets = data.get("presets") or {}
    if not isinstance(presets, dict):
        raise ValueError("presets must be a table of named presets")
    raw_scores = data.get("scores") or []
    if not raw_scores:
        raise ValueError(f"{path} has no scores")

    default_seed = int(data.get("seed", 0))
    scores: list[ScoreSpec] = []
    seen: set[str] = set()
    for item in raw_scores:
        name = str(item.get("name", "")).strip()
        if name == "":
            raise ValueError("each score needs a name")
        if name in seen:
            raise ValueError(f"duplicate score name: {name}")
        seen.add(name)
        preset_name = item.get("preset")
        dist_name, params = _resolve(item, presets, preset_name)
        seed = int(item["seed"]) if "seed" in item else default_seed
        try:
            dist = make(dist_name, **params)
        except ValueError as exc:
            raise ValueError(f"score {name}: {exc}") from exc
        scores.append(ScoreSpec(name, None if preset_name is None else str(preset_name), seed, dist))

    if "users" not in data or "buckets" not in data:
        raise ValueError(f"{path} needs users and buckets")
    return RunSpec(
        name=str(data.get("name") or path.stem),
        users=int(data["users"]),
        buckets=int(data["buckets"]),
        layer=str(data.get("layer") or "layer0"),
        scores=scores,
    )


def _resolve(item: dict, presets: dict, preset_name) -> tuple[str, dict]:
    params: dict = {}
    dist_name = None
    if preset_name is not None:
        try:
            preset = presets[preset_name]
        except KeyError as exc:
            known = ", ".join(presets) or "(none)"
            raise ValueError(f"unknown preset {preset_name}. known: {known}") from exc
        if "dist" not in preset:
            raise ValueError(f"preset {preset_name} is missing dist")
        dist_name = preset["dist"]
        params.update(_params_of(preset))
    if "dist" in item:
        dist_name = item["dist"]
    if dist_name is None:
        raise ValueError(f"score {item.get('name')} needs a preset or a dist")
    params.update(_params_of(item))
    return str(dist_name), params


def _params_of(block: dict) -> dict:
    return {key: block[key] for key in _PARAMS if key in block}


@dataclass(frozen=True)
class StudySpec:
    name: str
    users: int
    rounds: int
    seed: int
    traffic: float
    layer: str
    buckets: int
    groups: int
    dist: Distribution
    process: str
    phi: float
    shock: float


def load_study(path: Path) -> StudySpec:
    """读 AB/AA 比较实验。分数分布和随时间的过程写在同一个文件里。"""
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    score = data.get("score") or {}
    if "dist" not in score:
        raise ValueError(f"{path} needs [score].dist")
    process = data.get("process") or {}
    kind = str(process.get("kind") or "static")
    if kind not in {"static", "redraw", "ar1"}:
        raise ValueError("process.kind must be static, redraw, or ar1")
    for key in ("users", "rounds"):
        if key not in data:
            raise ValueError(f"{path} needs {key}")
    try:
        dist = make(str(score["dist"]), **_params_of(score))
    except ValueError as exc:
        raise ValueError(f"score: {exc}") from exc
    return StudySpec(
        name=str(data.get("name") or path.stem),
        users=int(data["users"]),
        rounds=int(data["rounds"]),
        seed=int(data.get("seed", 0)),
        traffic=float(data.get("traffic", 1)),
        layer=str(data.get("layer") or "layer0"),
        buckets=int(data.get("buckets", 2)),
        groups=int(data.get("groups", 5)),
        dist=dist,
        process=kind,
        phi=float(process.get("phi", 0.9)),
        shock=float(process.get("shock", 0.25)),
    )
