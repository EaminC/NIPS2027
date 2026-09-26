"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ab_aa_lab.experiment import run_suite
from ab_aa_lab.registry import effects, metrics, policies, processes, strategies
from ab_aa_lab.reporting import format_summary, plot_results, write_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ab-aa-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered plugins")

    run_parser = sub.add_parser("run", help="run a JSON config")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--out", type=Path, default=None, help="output directory")
    run_parser.add_argument("--plot", action="store_true", help="write ab_aa.png next to the CSVs")

    args = parser.parse_args(argv)
    if args.command == "list":
        _print_registry("metric", metrics)
        _print_registry("process", processes)
        _print_registry("policy", policies)
        _print_registry("strategy", strategies)
        _print_registry("effect", effects)
        return
    if args.command == "run":
        config = json.loads(args.config.read_text())
        results = run_suite(config, config_dir=args.config.parent)
        out = args.out or Path("outputs") / args.config.stem
        write_results(results, out)
        if args.plot:
            plot_results(results, out / "ab_aa.png")
        print(format_summary(results))
        print(f"\nwrote {out}")
        return
    raise AssertionError(args.command)


def _print_registry(title: str, registry) -> None:
    print(f"{title}: {', '.join(registry.names())}")
