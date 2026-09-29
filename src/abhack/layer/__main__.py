"""对用户 CSV 做层内哈希分桶，或检验一份分桶 CSV。"""

import argparse

from abhack.utils.term import render_buckets

from .io import check_assignment_file, count_assignment, write_assignment


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hash users into buckets on one layer")
    ap.add_argument("-i", "--input", required=True, help="user CSV")
    ap.add_argument("-o", "--output", help="bucket CSV to write")
    ap.add_argument("-b", "--buckets-csv", help="bucket CSV to check")
    ap.add_argument("--layer", required=True, help="traffic layer name")
    ap.add_argument("-k", "--buckets", type=int, required=True, help="number of buckets; 2 for a two-arm test")
    ap.add_argument("--check", action="store_true", help="check a bucket file instead of writing one")
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    if args.check:
        if not args.buckets_csv:
            ap.error("checking needs -b, the bucket CSV")
        errors, counts, _n = check_assignment_file(
            args.input, args.buckets_csv, args.layer, args.buckets, args.chunk_size
        )
        print(render_buckets(counts, args.layer, args.buckets_csv, errors))
        return 1 if errors else 0

    if not args.output:
        ap.error("writing buckets needs -o")
    write_assignment(args.input, args.output, args.layer, args.buckets, args.chunk_size)
    counts, _total = count_assignment(args.output, args.buckets, args.chunk_size)
    print(render_buckets(counts, args.layer, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
