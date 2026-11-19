"""Labelled data from simulated incidents, for fitting and scoring the models.

Each scenario is run with no remediation, so every fault plays out exactly as
injected. Two tables come out of a run:

* **windows** - one row per site every ``stride`` ticks after the warm-up: the
  21 sliding-window features of :mod:`sentinelops.detect.features` and whether
  a fault affecting that site is active at the window's last tick. These train
  and score the anomaly detector.
* **probes** - every ``probe_every`` ticks each probe is run on every target;
  one row per result with the fault active at its target when it completed.
  These fit the probe outcome tables of the Bayesian network.

Probes are passive, so the probing does not change the run: the windows are
exactly those of an unprobed run.

A fault *affects* a target when it hits it directly or hits what serves it:

* a server fault affects its server and that server's site;
* a site fault affects the site and both of its servers;
* a flood or surge affects the sites that receive its region's traffic (its
  *catchment*: the region's first site, as no site is ever withdrawn here) and
  their servers.

Labels come from the injected faults, which agents never see; only fitting and
scoring code reads them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sentinelops.detect.features import FEATURE_NAMES, WARMUP_TICKS, WindowFeaturizer
from sentinelops.scenarios import Scenario
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.probes import ProbeKind, ProbeRequest, ProbeResult
from sentinelops.sim.topology import RegionId, ServerId, SiteId, Topology
from sentinelops.sim.traffic import catchment_shares

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
StrArray = npt.NDArray[np.str_]

HEALTHY = "healthy"
"""Kind label of a row with no fault affecting its target."""

DEFAULT_STRIDE = 10
"""Keep one window per site every this many ticks (windows overlap: they are 30 ticks long)."""

DEFAULT_PROBE_EVERY = 60
"""Run every probe on every target once per this many ticks."""

_SERVER_PROBES = frozenset({ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK})
_SERVER_KINDS = frozenset({FaultKind.CRASH, FaultKind.EXHAUSTION, FaultKind.HARDWARE})
_SITE_KINDS = frozenset({FaultKind.PATH_LOSS, FaultKind.SITE_OUTAGE})


def _frozen(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    array.flags.writeable = False
    return array


def _concat(tables: Sequence[Any], cls: type[Any]) -> Any:
    """Join tables of one type column by column; an empty sequence gives an empty table."""
    if not tables:
        return cls.empty()
    return cls(
        **{
            f.name: _frozen(np.concatenate([getattr(t, f.name) for t in tables]))
            for f in fields(cls)
        }
    )


@dataclass(frozen=True, slots=True)
class WindowTable:
    """Feature windows with labels, one row per (run, site, tick). Columns are read-only.

    Attributes:
        family: Scenario family of the run.
        seed: Scenario seed of the run.
        site: Site the window describes.
        tick: Last tick of the window.
        features: Shape ``(n, 21)``, columns in ``FEATURE_NAMES`` order.
        fault: Whether a fault affecting the site is active at ``tick``.
        kind: The affecting fault's kind, or ``"healthy"``. If several faults
            affect the site, the one that started first.
        n_faults: How many faults affect the site at ``tick``.
    """

    family: StrArray
    seed: IntArray
    site: StrArray
    tick: IntArray
    features: FloatArray
    fault: BoolArray
    kind: StrArray
    n_faults: IntArray

    @classmethod
    def empty(cls) -> WindowTable:
        """A table with no rows."""
        return cls(
            family=_frozen(np.array([], dtype=np.str_)),
            seed=_frozen(np.array([], dtype=np.int64)),
            site=_frozen(np.array([], dtype=np.str_)),
            tick=_frozen(np.array([], dtype=np.int64)),
            features=_frozen(np.empty((0, len(FEATURE_NAMES)))),
            fault=_frozen(np.array([], dtype=np.bool_)),
            kind=_frozen(np.array([], dtype=np.str_)),
            n_faults=_frozen(np.array([], dtype=np.int64)),
        )

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.tick)


@dataclass(frozen=True, slots=True)
class ProbeTable:
    """Probe results with labels, one row per result. Columns are read-only.

    Attributes:
        family: Scenario family of the run.
        seed: Scenario seed of the run.
        probe: Probe kind (a ``ProbeKind`` value).
        target: Server or site probed.
        site: Site of the target (the target itself for site probes).
        tick: Tick the result describes (the probe's completion tick).
        outcome: What the probe reported (a ``ProbeOutcome`` value).
        fault: Whether a fault affecting the target is active at ``tick``.
        kind: The affecting fault's kind, or ``"healthy"`` (first started wins).
        n_faults: How many faults affect the target at ``tick``.
    """

    family: StrArray
    seed: IntArray
    probe: StrArray
    target: StrArray
    site: StrArray
    tick: IntArray
    outcome: StrArray
    fault: BoolArray
    kind: StrArray
    n_faults: IntArray

    @classmethod
    def empty(cls) -> ProbeTable:
        """A table with no rows."""
        text = np.array([], dtype=np.str_)
        number = np.array([], dtype=np.int64)
        return cls(
            family=_frozen(text.copy()),
            seed=_frozen(number.copy()),
            probe=_frozen(text.copy()),
            target=_frozen(text.copy()),
            site=_frozen(text.copy()),
            tick=_frozen(number.copy()),
            outcome=_frozen(text.copy()),
            fault=_frozen(np.array([], dtype=np.bool_)),
            kind=_frozen(text.copy()),
            n_faults=_frozen(number.copy()),
        )

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.tick)


@dataclass(frozen=True, slots=True)
class Dataset:
    """Labelled windows and probe results from one or more runs."""

    windows: WindowTable
    probes: ProbeTable

    @classmethod
    def concat(cls, parts: Sequence[Dataset]) -> Dataset:
        """Join datasets, keeping row order."""
        return cls(
            windows=_concat([p.windows for p in parts], WindowTable),
            probes=_concat([p.probes for p in parts], ProbeTable),
        )

    def save(self, path: str | Path) -> None:
        """Write the dataset to a compressed ``.npz`` file (no pickling)."""
        columns = {f"windows.{f.name}": getattr(self.windows, f.name) for f in fields(WindowTable)}
        columns |= {f"probes.{f.name}": getattr(self.probes, f.name) for f in fields(ProbeTable)}
        with Path(path).open("wb") as file:
            np.savez_compressed(file, **columns)

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Read a dataset written by :meth:`save`.

        Raises:
            ValueError: If the file does not hold every column of a dataset.
        """
        with np.load(Path(path), allow_pickle=False) as data:
            try:
                windows = WindowTable(
                    **{f.name: _frozen(data[f"windows.{f.name}"]) for f in fields(WindowTable)}
                )
                probes = ProbeTable(
                    **{f.name: _frozen(data[f"probes.{f.name}"]) for f in fields(ProbeTable)}
                )
            except KeyError as error:
                raise ValueError(f"{path} is not a saved dataset: {error}") from error
        return cls(windows=windows, probes=probes)


def catchment(region: str, topology: Topology) -> frozenset[SiteId]:
    """Sites that receive ``region``'s traffic when every site is announced."""
    return frozenset(catchment_shares(topology.preferences[RegionId(region)], {}))


def site_of(target: str, topology: Topology) -> SiteId:
    """The site of a server, or the site itself.

    Raises:
        KeyError: If ``target`` is neither a server nor a site.
    """
    if target in topology.servers:
        return topology.site_of(ServerId(target))
    if target in topology.sites:
        return SiteId(target)
    raise KeyError(f"unknown target {target!r}")


def affected_sites(fault: Fault, topology: Topology) -> frozenset[SiteId]:
    """Sites a fault affects (see the module docstring)."""
    if fault.kind in _SERVER_KINDS or fault.kind in _SITE_KINDS:
        return frozenset({site_of(fault.target, topology)})
    return catchment(fault.target, topology)


def affects(fault: Fault, target: str, topology: Topology) -> bool:
    """Whether ``fault`` affects ``target``, a server or site id.

    Raises:
        KeyError: If ``target`` is neither a server nor a site.
    """
    site = site_of(target, topology)
    if target in topology.servers and fault.kind in _SERVER_KINDS:
        return fault.target == target
    return site in affected_sites(fault, topology)


def label(faults: Sequence[Fault], target: str, tick: int, topology: Topology) -> tuple[str, int]:
    """Return ``(kind, count)`` of the faults affecting ``target`` at ``tick``.

    ``kind`` is that of the affecting fault that started first (the earlier one
    in ``faults`` on a tie), or ``"healthy"`` if none does.
    """
    hits = [f for f in faults if f.active_at(tick) and affects(f, target, topology)]
    if not hits:
        return HEALTHY, 0
    first = min(hits, key=lambda f: f.onset)  # min keeps the earliest index on ties
    return str(first.kind), len(hits)


def _sweep(topology: Topology) -> tuple[ProbeRequest, ...]:
    """Every probe on every valid target, in a fixed order."""
    requests: list[ProbeRequest] = []
    for kind in ProbeKind:
        targets = topology.servers if kind in _SERVER_PROBES else topology.sites
        requests.extend(ProbeRequest(kind, target) for target in sorted(targets))
    return tuple(requests)


def _check_settings(stride: int, probe_every: int | None) -> None:
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    if probe_every is not None and probe_every < 1:
        raise ValueError(f"probe_every must be at least 1 or None, got {probe_every}")


def label_run(
    scenario: Scenario,
    topology: Topology,
    *,
    stride: int = DEFAULT_STRIDE,
    probe_every: int | None = DEFAULT_PROBE_EVERY,
) -> Dataset:
    """Run one scenario with no remediation and label its windows and probe results.

    Args:
        scenario: The incident to run (its seed seeds the simulator).
        topology: The network.
        stride: Keep windows whose last tick is ``WARMUP_TICKS - 1 + k * stride``.
        probe_every: Start a sweep of every probe over every target at ticks
            ``probe_every, 2 * probe_every, ...``; ``None`` runs no probes.

    Returns:
        The run's windows (sites in sorted order within a tick) and its probe
        results (in completion order). Deterministic in the arguments.

    Raises:
        ValueError: If ``stride`` < 1 or ``probe_every`` < 1.
    """
    _check_settings(stride, probe_every)
    env = DnsEnv(topology, scenario.faults, scenario.horizon, seed=scenario.seed)
    featurizer = WindowFeaturizer(topology)
    sweep = _sweep(topology) if probe_every is not None else ()

    sites: list[str] = []
    ticks: list[int] = []
    rows: list[FloatArray] = []
    results: list[ProbeResult] = []
    obs = env.reset()
    while True:
        results.extend(obs.probe_results)
        for site, vector in featurizer.update(obs).items():
            if (vector.tick - (WARMUP_TICKS - 1)) % stride == 0:
                sites.append(site)
                ticks.append(vector.tick)
                rows.append(vector.values)
        if env.done:
            break
        due = probe_every is not None and (obs.tick + 1) % probe_every == 0
        obs = env.step(sweep if due else ())  # never an action: faults run their course

    faults = scenario.faults
    window_labels = [label(faults, s, t, topology) for s, t in zip(sites, ticks, strict=True)]
    probe_targets = [r.request.target for r in results]
    probe_labels = [label(faults, r.request.target, r.completed, topology) for r in results]
    n, m = len(ticks), len(results)
    windows = WindowTable(
        family=_frozen(np.full(n, scenario.family)),
        seed=_frozen(np.full(n, scenario.seed, dtype=np.int64)),
        site=_frozen(np.array(sites, dtype=np.str_)),
        tick=_frozen(np.array(ticks, dtype=np.int64)),
        features=_frozen(np.array(rows).reshape(n, len(FEATURE_NAMES))),
        fault=_frozen(np.array([c > 0 for _, c in window_labels], dtype=np.bool_)),
        kind=_frozen(np.array([k for k, _ in window_labels], dtype=np.str_)),
        n_faults=_frozen(np.array([c for _, c in window_labels], dtype=np.int64)),
    )
    probes = ProbeTable(
        family=_frozen(np.full(m, scenario.family)),
        seed=_frozen(np.full(m, scenario.seed, dtype=np.int64)),
        probe=_frozen(np.array([str(r.request.kind) for r in results], dtype=np.str_)),
        target=_frozen(np.array(probe_targets, dtype=np.str_)),
        site=_frozen(np.array([site_of(t, topology) for t in probe_targets], dtype=np.str_)),
        tick=_frozen(np.array([r.completed for r in results], dtype=np.int64)),
        outcome=_frozen(np.array([str(r.outcome) for r in results], dtype=np.str_)),
        fault=_frozen(np.array([c > 0 for _, c in probe_labels], dtype=np.bool_)),
        kind=_frozen(np.array([k for k, _ in probe_labels], dtype=np.str_)),
        n_faults=_frozen(np.array([c for _, c in probe_labels], dtype=np.int64)),
    )
    return Dataset(windows=windows, probes=probes)


def build_dataset(
    scenarios: Iterable[Scenario],
    topology: Topology,
    *,
    stride: int = DEFAULT_STRIDE,
    probe_every: int | None = DEFAULT_PROBE_EVERY,
    progress: Callable[[Scenario], None] | None = None,
) -> Dataset:
    """Label every scenario and join the results, in scenario order.

    Typical use, with :mod:`sentinelops.eval.splits`::

        train = build_dataset(split_scenarios("train", topology), topology)

    Args:
        scenarios: Incidents to run.
        topology: The network.
        stride: See :func:`label_run`.
        probe_every: See :func:`label_run`.
        progress: Called with each scenario once it is labelled.

    Raises:
        ValueError: If ``stride`` < 1 or ``probe_every`` < 1.
    """
    _check_settings(stride, probe_every)
    parts: list[Dataset] = []
    for scenario in scenarios:
        parts.append(label_run(scenario, topology, stride=stride, probe_every=probe_every))
        if progress is not None:
            progress(scenario)
    return Dataset.concat(parts)
