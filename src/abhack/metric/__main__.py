"""按指定分布打分，或对分数 CSV 做逆检验。"""

import argparse

from abhack.utils.distributions import NAMES, make
from abhack.utils.term import render_scores

from .io import check_metric_file, read_metric, write_metric


def _dist(args):
    return make(
        args.dist,
        mu=args.mu,
        sigma=args.sigma,
        low=args.low,
        high=args.high,
        mean=args.mean,
        p=args.p,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw a named score for each user")
    ap.add_argument("-i", "--input", required=True, help="user CSV when drawing, score CSV when checking")
    ap.add_argument("-o", "--output", help="score CSV to write")
    ap.add_argument("--name", required=True, help="score name; becomes the column name")
    ap.add_argument("--dist", required=True, choices=NAMES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float)
    ap.add_argument("--sigma", type=float)
    ap.add_argument("--low", type=float)
    ap.add_argument("--high", type=float)
    ap.add_argument("--mean", type=float, help="mean of an exponential distribution")
    ap.add_argument("--p", type=float, help="success probability of a bernoulli distribution")
    ap.add_argument("--check", action="store_true", help="check a score file")
    ap.add_argument("--no-replay", action="store_true", help="check the distribution only, skip the seed replay")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    args = ap.parse_args(argv)
    try:
        dist = _dist(args)
    except ValueError as exc:
        ap.error(str(exc))

    if args.check:
        _ids, values, found = read_metric(args.input)
        result, found = check_metric_file(
            args.input,
            dist,
            args.seed,
            replay=not args.no_replay,
            alpha=args.alpha,
            name=args.name,
        )
        print(
            render_scores(
                found,
                dist,
                values,
                path=args.input,
                result=result,
                seed=None if args.no_replay else args.seed,
            )
        )
        return 0 if result["ok"] else 1

    if not args.output:
        ap.error("drawing scores needs -o")
    write_metric(
        args.input,
        args.output,
        args.name,
        dist,
        args.seed,
        args.chunk_size,
    )
    _ids, values, found = read_metric(args.output)
    print(render_scores(found, dist, values, path=args.output, seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
