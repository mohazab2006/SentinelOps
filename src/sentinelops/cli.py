"""Command-line entry point for SentinelOps.

Subcommands:

* ``run`` — run an agent on seeded scenarios of one family and report the
  share of failed legitimate queries per seed.

More subcommands (experiments, the console server) are added as they are built.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path

from sentinelops import __version__
from sentinelops.eval.harness import AGENTS, RunResult, make_agent, simulate, summarize
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
    except ValueError as error:
        parser.error(str(error))
    if args.events is not None:
        args.events.mkdir(parents=True, exist_ok=True)

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
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
