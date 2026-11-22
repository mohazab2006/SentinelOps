"""Command-line entry point for SentinelOps.

Subcommands:

* ``run`` — run an agent on seeded scenarios of one family and report the
  share of failed legitimate queries per seed.
* ``report-detection`` — measure the detector and the diagnosis network on
  labelled tune-split runs.
* ``serve`` — start the operations console API (and the built web console).

More subcommands (experiments, replay) are added as they are built.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

from sentinelops import __version__
from sentinelops.detect import IsolationDetector, healthy_windows
from sentinelops.diagnose import fit_network
from sentinelops.eval.dataset import Dataset, build_dataset
from sentinelops.eval.harness import AGENTS, RunResult, make_agent, simulate, summarize
from sentinelops.eval.report import (
    detection_report,
    diagnosis_report,
    format_detection,
    format_diagnosis,
)
from sentinelops.eval.splits import Split, split_of
from sentinelops.scenarios import FAMILIES, Scenario, generate
from sentinelops.scenarios.families import DEFAULT_HORIZON
from sentinelops.sim.topology import default_topology


def parse_seeds(text: str) -> list[int]:
    """Parse a seed list such as ``"0-4"``, ``"7"`` or ``"0,2,10-12"``.

    Ranges are inclusive. Order is kept and duplicates are dropped.

    Raises:
        argparse.ArgumentTypeError: If the text is not a valid seed list.
    """
    seeds: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        low, dash, high = part.strip().partition("-")
        try:
            first = int(low)
            last = int(high) if dash else first
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid seed list {text!r}") from None
        if first < 0 or last < first:
            raise argparse.ArgumentTypeError(f"invalid seed range {part.strip()!r} in {text!r}")
        for seed in range(first, last + 1):
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
    return seeds


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinelops",
        description="AI-assisted diagnosis and self-healing for simulated anycast DNS.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="<command>")

    run = commands.add_parser(
        "run",
        help="run an agent on seeded scenarios and report failed legitimate queries",
        description=(
            "Run an agent on seeded scenarios of one family. Each seed fixes the "
            "fault, the demand and the monitoring noise, so results are reproducible."
        ),
    )
    run.add_argument("--family", required=True, choices=FAMILIES, help="scenario family")
    run.add_argument("--agent", required=True, choices=tuple(AGENTS), help="agent to run")
    run.add_argument(
        "--seeds",
        type=parse_seeds,
        default=[0],
        metavar="LIST",
        help='seeds to run, e.g. "0-4" or "0,3,7-9" (default: 0)',
    )
    run.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"ticks per run (default: {DEFAULT_HORIZON})",
    )
    run.add_argument(
        "--events",
        type=Path,
        metavar="DIR",
        help="also write each run's event log to DIR as JSON Lines",
    )
    run.set_defaults(subparser=run)

    report = commands.add_parser(
        "report-detection",
        help="measure detection and diagnosis on labelled tune-split runs",
        description=(
            "Fit the detector on healthy runs and the diagnosis network on train-split "
            "runs, then score both on tune-split runs. The test split is never used."
        ),
    )
    report.add_argument(
        "--tune-seeds",
        type=parse_seeds,
        default=parse_seeds("100-105"),
        metavar="LIST",
        help="tune-split seeds to score, per family (default: 100-105)",
    )
    report.add_argument(
        "--train-seeds",
        type=parse_seeds,
        default=parse_seeds("0-9"),
        metavar="LIST",
        help="train-split seeds to fit the network on, per family (default: 0-9)",
    )
    report.add_argument(
        "--healthy-seeds",
        type=parse_seeds,
        default=parse_seeds("0-4"),
        metavar="LIST",
        help="fault-free runs to fit the detector on (default: 0-4)",
    )
    report.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
        metavar="FAMILY",
        help="scenario families to include (default: all)",
    )
    report.set_defaults(subparser=report)

    serve = commands.add_parser(
        "serve",
        help="start the operations console API",
        description="Serve the console API and, once built, the web console.",
    )
    serve.add_argument("--host", default="127.0.0.1", help="interface (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    serve.add_argument(
        "--console",
        type=Path,
        default=Path("web/dist"),
        metavar="DIR",
        help="built web console to serve at / (default: web/dist, if it exists)",
    )
    serve.set_defaults(subparser=serve)
    return parser


def _describe(scenario: Scenario) -> str:
    """Summarise a scenario's faults, e.g. ``"crash lon-1 @412"``."""
    return ", ".join(f"{f.kind} {f.target} @{f.onset}" for f in scenario.faults)


def _header(fault_width: int) -> str:
    """Column titles and a rule for the results table."""
    title = f"{'seed':>6}  {'faults':<{fault_width}}  {'failed':>8}  {'actions':>13}  digest"
    return f"{title}\n{'-' * len(title)}"


def _row(result: RunResult, scenario: Scenario, fault_width: int) -> str:
    """One table row: seed, faults, failed share, actions and event-log digest."""
    actions = f"{result.actions_issued} ({result.actions_failed} failed)"
    return (
        f"{result.seed:>6}  {_describe(scenario):<{fault_width}}  "
        f"{result.legit_failure_share:>8.3%}  {actions:>13}  {result.event_digest[:12]}"
    )


def _summary(results: Sequence[RunResult]) -> str:
    """Mean failed share over all runs, with the standard deviation if there are several."""
    shares = [r.legit_failure_share for r in results]
    spread = f" (sd {statistics.stdev(shares):.3%})" if len(shares) > 1 else ""
    return f"mean failed share over {len(shares)} run(s): {statistics.fmean(shares):.3%}{spread}"


def _run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand."""
    parser: argparse.ArgumentParser = args.subparser
    try:
        make_agent(args.agent)  # fail fast, before any simulation, if it is unavailable
        topology = default_topology()
        scenarios = [generate(args.family, seed, topology, args.horizon) for seed in args.seeds]
        if args.events is not None:
            args.events.mkdir(parents=True, exist_ok=True)
    except ValueError as error:
        parser.error(str(error))
    except OSError as error:
        parser.error(f"cannot write event logs to {args.events}: {error.strerror}")

    fault_width = max(len("faults"), *(len(_describe(s)) for s in scenarios))
    print(f"family={args.family} agent={args.agent} horizon={args.horizon} ticks")
    print(_header(fault_width))
    results = []
    for scenario in scenarios:
        agent = make_agent(args.agent)  # a fresh agent per run: nothing carries over
        env = simulate(scenario, agent, topology)
        result = summarize(scenario, agent.name, env)
        results.append(result)
        print(_row(result, scenario, fault_width), flush=True)  # rows appear as runs finish
        if args.events is not None:
            path = args.events / f"{args.family}-{args.agent}-seed{scenario.seed}.jsonl"
            path.write_text(env.events.to_jsonl(), encoding="utf-8", newline="\n")
    print(_summary(results))
    return 0


def _seeds_in(split: Split, seeds: Sequence[int]) -> str | None:
    """An error message if any seed is outside ``split``, else ``None``."""
    outside = [seed for seed in seeds if split_of(seed) is not split]
    return f"seeds {outside} are not in the {split} split" if outside else None


def _report_detection(args: argparse.Namespace) -> int:
    """Execute the ``report-detection`` subcommand."""
    parser: argparse.ArgumentParser = args.subparser
    for split, seeds in ((Split.TUNE, args.tune_seeds), (Split.TRAIN, args.train_seeds)):
        if (problem := _seeds_in(split, seeds)) is not None:
            parser.error(problem)
    topology = default_topology()

    def labelled(seeds: Sequence[int], what: str) -> Dataset:
        scenarios = [generate(f, seed, topology) for f in args.families for seed in seeds]
        print(f"labelling {len(scenarios)} {what} runs ...", file=sys.stderr, flush=True)
        return build_dataset(scenarios, topology)

    tune = labelled(args.tune_seeds, "tune")
    print("fitting the detector on healthy runs ...", file=sys.stderr, flush=True)
    detector = IsolationDetector(seed=0).fit(healthy_windows(topology, args.healthy_seeds))
    windows = tune.windows
    detection = detection_report(detector.score(windows.features), windows.fault, windows.kind)
    network = fit_network(labelled(args.train_seeds, "train"))
    print("diagnosing ...", file=sys.stderr, flush=True)
    diagnosis = diagnosis_report(network, tune, topology)

    seeds = ",".join(map(str, args.tune_seeds))
    print(f"tune seeds {seeds} | families: {' '.join(args.families)}")
    print(
        f"detector fitted on {len(args.healthy_seeds)} healthy runs, network on "
        f"{len(args.train_seeds)} train seeds per family"
    )
    print()
    print(format_detection(detection))
    print()
    print(format_diagnosis(diagnosis))
    return 0


def _serve(args: argparse.Namespace) -> int:
    """Execute the ``serve`` subcommand."""
    import uvicorn

    from sentinelops.api.app import create_app

    if not args.console.is_dir():
        print(
            f"note: no built console at {args.console}; serving the API only "
            "(build it with `npm run build` in web/, or pass --console DIR)",
            file=sys.stderr,
        )
    uvicorn.run(create_app(static_dir=args.console), host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success. Invalid arguments exit with status 2.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "report-detection":
        return _report_detection(args)
    if args.command == "serve":
        return _serve(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
