"""按一个 toml 跑完用户、分桶和打分。

    python exp/run.py
    python exp/run.py exp/runs/demo.toml
    python exp/run.py exp/runs/demo.toml -M 10000 -N 3
"""

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abhack.layer import check_assignment_file, write_assignment  # noqa: E402
from abhack.metric import check_metric_file, read_metric, write_metric, write_score_table  # noqa: E402
from abhack.users import read_csv, write_csv  # noqa: E402
from abhack.utils.term import (  # noqa: E402
    render_buckets,
    render_run_done,
    render_run_header,
    render_scores,
    render_users,
)
from spec import load_run  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def _argv(argv):
    """把全角或长破折号收成普通减号，这样 —N 和 -N 都能用。"""
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
    ap = argparse.ArgumentParser(description="Run users, hashing, and scores from one experiment file")
    ap.add_argument("config", nargs="?", type=Path, default=EXP_DIR / "runs" / "demo.toml")
    ap.add_argument("-M", "--num-users", type=int, default=None, help="override the user count in the file")
    ap.add_argument("-N", "--buckets", type=int, default=None, help="override the bucket count in the file")
    ap.add_argument("--seed", type=int, default=None, help="override the file default seed for scores that omit seed")
    ap.add_argument("--out-dir", type=Path, default=None, help="default is exp/out/<experiment name>")
    args = ap.parse_args(_argv(argv))
    try:
        run = load_run(args.config, seed=args.seed)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        ap.error(str(exc))
    users = run.users if args.num_users is None else args.num_users
    buckets = run.buckets if args.buckets is None else args.buckets
    if users < 0:
        ap.error("user count must be >= 0")
    if buckets < 2:
        ap.error("bucket count must be at least 2")

    out = args.out_dir or (EXP_DIR / "out" / run.name)
    out.mkdir(parents=True, exist_ok=True)
    print(render_run_header(run.name, str(args.config), str(out)))
    print()
    users_path = out / "users.csv"
    buckets_path = out / f"{run.layer}.csv"

    write_csv(users, str(users_path))
    print(render_users(users, str(users_path)))
    print()

    write_assignment(str(users_path), str(buckets_path), run.layer, buckets)
    errors, counts, _checked = check_assignment_file(
        str(users_path), str(buckets_path), run.layer, buckets
    )
    print(render_buckets(counts, run.layer, str(buckets_path), errors))
    if errors:
        return 1
    print()

    user_ids = read_csv(str(users_path))
    columns = []
    failed = False
    for spec in run.scores:
        score_path = out / f"{spec.name}.csv"
        write_metric(str(users_path), str(score_path), spec.name, spec.dist, spec.seed)
        _ids, values, found = read_metric(str(score_path))
        result, found = check_metric_file(
            str(score_path), spec.dist, spec.seed, name=spec.name
        )
        print(
            render_scores(
                found,
                spec.dist,
                values,
                path=str(score_path),
                result=result,
                seed=spec.seed,
                preset=spec.preset,
            )
        )
        print()
        columns.append((spec.name, values, spec.dist.discrete))
        failed = failed or not result["ok"]
    table_path = out / "scores.csv"
    write_score_table(str(table_path), user_ids, columns)
    print(render_run_done(run.name, len(run.scores), str(out), str(table_path)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
