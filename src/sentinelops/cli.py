"""Command-line entry point for SentinelOps.

Subcommands (simulation runs, experiments, the console server) are registered
here as they are implemented.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sentinelops import __version__


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="sentinelops",
        description="AI-assisted diagnosis and self-healing for simulated anycast DNS.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
