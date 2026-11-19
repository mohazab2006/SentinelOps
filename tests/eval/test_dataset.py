"""Tests for labelled dataset generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sentinelops.agent.base import NoOpAgent
from sentinelops.detect.features import FEATURE_NAMES, WARMUP_TICKS, featurize_run
from sentinelops.eval.dataset import (
    DEFAULT_PROBE_EVERY,
    HEALTHY,
    Dataset,
    ProbeTable,
    WindowTable,
    affected_sites,
    affects,
    build_dataset,
    catchment,
    label,
    label_run,
    site_of,
)
from sentinelops.scenarios import Scenario
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.probes import PROBE_TICKS, ProbeKind, ProbeOutcome
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import Topology, default_topology

HORIZON = 700
CRASH = Fault(FaultKind.CRASH, "ott-1", onset=400, duration=None, severity=1.0)
FLOOD = Fault(FaultKind.FLOOD, "eu-west", onset=300, duration=200, severity=1.0)


@pytest.fixture(scope="module")
def topology() -> Topology:
    return default_topology()


@pytest.fixture(scope="module")
def crash_run(topology: Topology) -> Dataset:
    return label_run(Scenario("crash", 7, HORIZON, (CRASH,)), topology)


def _unprobed_features(scenario: Scenario, topology: Topology) -> dict[tuple[str, int], object]:
    env = DnsEnv(topology, scenario.faults, scenario.horizon, seed=scenario.seed)
    agent = NoOpAgent()
    observations: list[Observation] = [env.reset()]
    while not env.done:
        observations.append(env.step(agent.act(observations[-1])))
    return {(v.site, v.tick): v.values for v in featurize_run(topology, observations)}


# ---------------------------------------------------------------- what a fault affects


def test_catchment_is_the_first_preference(topology: Topology) -> None:
    for region, preferences in topology.preferences.items():
        assert catchment(region, topology) == {preferences[0]}


@pytest.mark.parametrize(
    ("fault", "sites"),
    [
        (CRASH, {"ott"}),
        (Fault(FaultKind("hardware"), "sin-2", 1, None, 0.5), {"sin"}),
        (Fault(FaultKind("path_loss"), "par", 1, 10, 0.3), {"par"}),
        (Fault(FaultKind("site_outage"), "fra", 1, None, 1.0), {"fra"}),
        (FLOOD, {"lon"}),
        (Fault(FaultKind("benign_surge"), "na-west", 1, 10, 0.5), {"yyz"}),
    ],
)
def test_affected_sites(topology: Topology, fault: Fault, sites: set[str]) -> None:
    assert affected_sites(fault, topology) == sites


def test_server_fault_affects_its_server_and_site_only(topology: Topology) -> None:
    assert affects(CRASH, "ott-1", topology)
    assert affects(CRASH, "ott", topology)
    assert not affects(CRASH, "ott-2", topology)  # the healthy sibling
    assert not affects(CRASH, "yyz", topology)


def test_site_and_region_faults_affect_the_servers_there(topology: Topology) -> None:
    outage = Fault(FaultKind("site_outage"), "fra", 1, None, 1.0)
    assert affects(outage, "fra-1", topology) and affects(outage, "fra-2", topology)
    assert affects(FLOOD, "lon-2", topology)
    assert not affects(FLOOD, "par", topology)


def test_site_of(topology: Topology) -> None:
    assert site_of("ott-2", topology) == "ott"
    assert site_of("sin", topology) == "sin"


def test_unknown_target(topology: Topology) -> None:
    with pytest.raises(KeyError, match="mars"):
        affects(CRASH, "mars", topology)
    with pytest.raises(KeyError, match="mars"):
        site_of("mars", topology)


def test_label_follows_the_active_window(topology: Topology) -> None:
    assert label((FLOOD,), "lon", 299, topology) == (HEALTHY, 0)
    assert label((FLOOD,), "lon", 300, topology) == ("flood", 1)
    assert label((FLOOD,), "lon", 499, topology) == ("flood", 1)
    assert label((FLOOD,), "lon", 500, topology) == (HEALTHY, 0)
    assert label((FLOOD,), "par", 400, topology) == (HEALTHY, 0)


def test_label_with_overlapping_faults_names_the_first_started(topology: Topology) -> None:
    leak = Fault(FaultKind("exhaustion"), "ott-2", 350, None, 0.5)
    assert label((CRASH, leak), "ott", 500, topology) == ("exhaustion", 2)
    assert label((CRASH, leak), "ott-1", 500, topology) == ("crash", 1)
    twin = Fault(FaultKind("hardware"), "ott-2", 400, None, 0.5)
    assert label((CRASH, twin), "ott", 500, topology) == ("crash", 2)  # tie → listed first
    assert label((twin, CRASH), "ott", 500, topology) == ("hardware", 2)


# ---------------------------------------------------------------- windows


def test_windows_cover_every_site_every_stride(topology: Topology, crash_run: Dataset) -> None:
    w = crash_run.windows
    ticks = list(range(WARMUP_TICKS - 1, HORIZON, 10))
    assert len(w) == len(ticks) * len(topology.sites)
    assert list(w.tick) == [t for t in ticks for _ in topology.sites]
    assert list(w.site) == sorted(topology.sites) * len(ticks)
    assert w.features.shape == (len(w), len(FEATURE_NAMES))
    assert set(w.family) == {"crash"} and set(w.seed) == {7}


def test_window_labels(crash_run: Dataset) -> None:
    w = crash_run.windows
    expected = (w.site == "ott") & (w.tick >= CRASH.onset)
    assert np.array_equal(w.fault, expected)
    assert set(w.kind[w.fault]) == {"crash"}
    assert set(w.kind[~w.fault]) == {HEALTHY}
    assert np.array_equal(w.n_faults, expected.astype(np.int64))


def test_probing_does_not_change_the_windows(topology: Topology, crash_run: Dataset) -> None:
    unprobed = _unprobed_features(Scenario("crash", 7, HORIZON, (CRASH,)), topology)
    w = crash_run.windows
    for site, tick, row in zip(w.site, w.tick, w.features, strict=True):
        assert np.array_equal(row, unprobed[(str(site), int(tick))])


def test_stride_one_keeps_every_window(topology: Topology) -> None:
    scenario = Scenario("crash", 1, 200, (CRASH,))
    data = label_run(scenario, topology, stride=1, probe_every=None)
    assert len(data.windows) == (200 - (WARMUP_TICKS - 1)) * len(topology.sites)
    assert len(_unprobed_features(scenario, topology)) == len(data.windows)


def test_region_fault_marks_its_catchment_site(topology: Topology) -> None:
    data = label_run(Scenario("flood", 3, HORIZON, (FLOOD,)), topology, probe_every=None)
    w = data.windows
    expected = (w.site == "lon") & (w.tick >= 300) & (w.tick < 500)
    assert np.array_equal(w.fault, expected)
    assert set(w.kind[w.fault]) == {"flood"}


def test_a_run_shorter_than_the_warm_up_has_no_windows(topology: Topology) -> None:
    data = label_run(Scenario("crash", 1, 50, ()), topology, probe_every=None)
    assert len(data.windows) == 0
    assert data.windows.features.shape == (0, len(FEATURE_NAMES))


# ---------------------------------------------------------------- probes


def test_every_probe_runs_on_every_target_each_sweep(
    topology: Topology, crash_run: Dataset
) -> None:
    p = crash_run.probes
    issued = p.tick - np.array([PROBE_TICKS[ProbeKind(k)] for k in p.probe])
    sweeps = range(DEFAULT_PROBE_EVERY, HORIZON, DEFAULT_PROBE_EVERY)
    assert set(issued) == set(sweeps)  # every sweep finishes before the run ends here
    per_sweep = 2 * len(topology.servers) + 3 * len(topology.sites)
    for start in sweeps:
        pairs = Counter(zip(p.probe[issued == start], p.target[issued == start], strict=True))
        assert len(pairs) == per_sweep and set(pairs.values()) == {1}


def test_probes_still_running_at_the_end_are_dropped(topology: Topology) -> None:
    data = label_run(Scenario("crash", 1, 130, ()), topology, probe_every=120)
    # issued at 120: local 2, neighbour 3, remote 5 finish; traffic 10 and hardware 15 would not
    assert set(data.probes.tick) == {122, 123, 125}
    assert (data.probes.tick <= 129).all()


def test_probe_columns_are_consistent(topology: Topology, crash_run: Dataset) -> None:
    p = crash_run.probes
    assert len(p) > 0
    for probe, target, site, tick, outcome, fault, kind, n in zip(
        p.probe, p.target, p.site, p.tick, p.outcome, p.fault, p.kind, p.n_faults, strict=True
    ):
        server_probe = ProbeKind(probe) in (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK)
        assert (target in topology.servers) == server_probe
        assert site == (topology.site_of(target) if server_probe else target)
        assert ProbeOutcome(outcome)
        assert (kind, n) == label((CRASH,), str(target), int(tick), topology)
        assert fault == (n > 0)
    assert list(p.tick) == sorted(p.tick)  # completion order


def test_probe_labels_reflect_the_fault(crash_run: Dataset) -> None:
    p = crash_run.probes
    on_crashed = (p.probe == "local_dns") & (p.target == "ott-1") & p.fault
    assert on_crashed.sum() >= 3
    assert Counter(p.outcome[on_crashed]).most_common(1)[0][0] == "refused"
    sibling = (p.probe == "local_dns") & (p.target == "ott-2")
    assert not p.fault[sibling].any()


def test_no_probes_when_disabled(topology: Topology, crash_run: Dataset) -> None:
    data = label_run(Scenario("crash", 7, HORIZON, (CRASH,)), topology, probe_every=None)
    assert len(data.probes) == 0
    for name in ("site", "tick", "features", "fault", "kind", "n_faults"):
        assert np.array_equal(getattr(data.windows, name), getattr(crash_run.windows, name))


@pytest.mark.parametrize(("stride", "probe_every"), [(0, 60), (10, 0), (-1, None)])
def test_bad_settings(topology: Topology, stride: int, probe_every: int | None) -> None:
    scenario = Scenario("crash", 1, 200, ())
    with pytest.raises(ValueError, match="at least 1"):
        label_run(scenario, topology, stride=stride, probe_every=probe_every)
    with pytest.raises(ValueError, match="at least 1"):
        build_dataset([], topology, stride=stride, probe_every=probe_every)


# ---------------------------------------------------------------- tables and datasets


def test_runs_are_deterministic(topology: Topology, crash_run: Dataset) -> None:
    again = label_run(Scenario("crash", 7, HORIZON, (CRASH,)), topology)
    _assert_same(again, crash_run)


def test_columns_are_read_only(crash_run: Dataset) -> None:
    with pytest.raises(ValueError, match="read-only"):
        crash_run.windows.features[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        crash_run.probes.outcome[0] = "ok"


def test_build_dataset_joins_runs_in_order(topology: Topology) -> None:
    scenarios = [Scenario("crash", 1, 300, (CRASH,)), Scenario("flood", 2, 300, (FLOOD,))]
    seen: list[str] = []
    data = build_dataset(
        scenarios, topology, stride=20, probe_every=100, progress=lambda s: seen.append(s.family)
    )
    assert seen == ["crash", "flood"]
    parts = [label_run(s, topology, stride=20, probe_every=100) for s in scenarios]
    _assert_same(data, Dataset.concat(parts))
    assert list(dict.fromkeys(data.windows.family)) == ["crash", "flood"]
    assert len(data.windows) == sum(len(p.windows) for p in parts)
    assert len(data.probes) == sum(len(p.probes) for p in parts)
    quiet = build_dataset(scenarios, topology, stride=20, probe_every=100)
    _assert_same(quiet, data)


def test_empty_dataset(topology: Topology) -> None:
    data = build_dataset([], topology)
    assert len(data.windows) == 0 and len(data.probes) == 0
    assert data.windows.features.shape == (0, len(FEATURE_NAMES))
    _assert_same(data, Dataset(WindowTable.empty(), ProbeTable.empty()))


def test_save_and_load_round_trip(tmp_path: Path, crash_run: Dataset) -> None:
    path = tmp_path / "train.npz"
    crash_run.save(path)
    loaded = Dataset.load(path)
    _assert_same(loaded, crash_run)
    assert not loaded.windows.tick.flags.writeable


def test_load_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, x=np.zeros(3))
    with pytest.raises(ValueError, match="not a saved dataset"):
        Dataset.load(path)


def _assert_same(a: Dataset, b: Dataset) -> None:
    for table_a, table_b in ((a.windows, b.windows), (a.probes, b.probes)):
        for name in table_a.__dataclass_fields__:
            assert np.array_equal(getattr(table_a, name), getattr(table_b, name)), name
