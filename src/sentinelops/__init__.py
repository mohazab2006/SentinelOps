"""SentinelOps: AI-assisted diagnosis and self-healing for simulated anycast DNS.

The package is organised around the agent loop: telemetry from the simulator
(``sim``) is screened for anomalies (``detect``), explained with probabilistic
diagnosis (``diagnose``), turned into a recovery plan (``plan``), authorised by
operator policy (``policy``), and checked after execution (``verify``). The
``agent`` package wires these together; ``baselines`` and ``eval`` provide the
comparison agents and the experimental harness.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sentinelops")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

__all__ = ["__version__"]
