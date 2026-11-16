"""Reproducibility: the same command and seed give byte-identical event logs.

Each run happens in a fresh Python process with a different hash seed, so the
test would catch any dependence on set or dict iteration order, object ids or
global random state — not just on the seeds we pass in.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cli(out_dir: Path, hash_seed: str) -> bytes:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    command = [
        sys.executable,
        "-m",
        "sentinelops.cli",
        "run",
        "--family",
        "mixed",
        "--agent",
        "noop",
        "--seeds",
        "3",
        "--horizon",
        "1601",
        "--events",
        str(out_dir),
    ]
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return (out_dir / "mixed-noop-seed3.jsonl").read_bytes()


def test_event_log_is_identical_across_fresh_processes(tmp_path: Path) -> None:
    first = _run_cli(tmp_path / "a", hash_seed="0")
    second = _run_cli(tmp_path / "b", hash_seed="12345")
    assert first == second
    assert first.count(b"\n") > 1601  # at least one event per tick
