"""生成 M 个用户并写入 CSV。"""

import argparse

from abhack.utils.term import render_users

from .io import write_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write M users to a CSV")
    ap.add_argument("-M", "--num-users", type=int, required=True, help="number of users")
    ap.add_argument("-o", "--output", default="users.csv")
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    args = ap.parse_args(argv)
    write_csv(args.num_users, args.output, args.chunk_size)
    print(render_users(args.num_users, args.output))


if __name__ == "__main__":
    main()
