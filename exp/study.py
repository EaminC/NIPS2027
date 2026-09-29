"""比较诚实对照、固定哈希、每次重分，以及保留极端组。

    python exp/study.py
    python exp/study.py exp/runs/hack.toml
    python exp/study.py exp/runs/hack.toml -M 20000
"""

import argparse
import csv
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abhack.study import simulate  # noqa: E402
from abhack.utils.term import render_study  # noqa: E402
from spec import load_study  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def _argv(argv):
    """把全角或长破折号收成普通减号，这样 —M 和 -M 都能用。"""
    source = sys.argv[1:] if argv is None else argv
    fixed = []
    for arg in source:
        for dash in ("—", "–", "－"):
            if arg.startswith(dash):
                arg = "-" + arg[len(dash) :]
                break
        fixed.append(arg)
    return fixed


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare AB claims, AA change, and the group bounds")
    ap.add_argument("config", nargs="?", type=Path, default=EXP_DIR / "runs" / "hack.toml")
    ap.add_argument("-M", "--num-users", type=int, default=None, help="override the user count")
    ap.add_argument("--rounds", type=int, default=None, help="override the number of rounds")
    ap.add_argument("--traffic", type=float, default=None, help="override the sample fraction, e.g. 0.1")
    ap.add_argument("--out-dir", type=Path, default=None, help="default is exp/out/<experiment name>")
    args = ap.parse_args(_argv(argv))
    try:
        study = load_study(args.config)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        ap.error(str(exc))
    users = study.users if args.num_users is None else args.num_users
    rounds = study.rounds if args.rounds is None else args.rounds
    traffic = study.traffic if args.traffic is None else args.traffic
    if users < 2:
        ap.error("user count must be >= 2")
    try:
        rows = simulate(
            users=users,
            dist=study.dist,
            process=study.process,
            rounds=rounds,
            seed=study.seed,
            traffic=traffic,
            layer=study.layer,
            buckets=study.buckets,
            groups=study.groups,
            phi=study.phi,
            shock=study.shock,
        )
    except ValueError as exc:
        ap.error(str(exc))

    out = args.out_dir or (EXP_DIR / "out" / study.name)
    out.mkdir(parents=True, exist_ok=True)
    table = out / "study.csv"
    with table.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "method", "ab", "ab_full", "aa", "upper", "lower"])
        for row in rows:
            writer.writerow(
                [
                    row.round,
                    row.method,
                    _cell(row.ab),
                    _cell(row.ab_full),
                    _cell(row.aa),
                    _cell(row.upper),
                    _cell(row.lower),
                ]
            )
    print(
        render_study(
            study.name,
            study.dist,
            study.process,
            users,
            traffic,
            rounds,
            study.groups,
            str(table),
            rows,
        )
    )
    return 0


def _cell(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.8g}"


if __name__ == "__main__":
    raise SystemExit(main())
