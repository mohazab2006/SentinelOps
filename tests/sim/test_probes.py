"""Tests for diagnostic probes: the outcome model and how the engine runs probes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.probes import (
    FLAKE_PROB,
    OUTCOMES,
    PROBE_TICKS,
    ProbeKind,
    ProbeOutcome,
    ProbeRequest,
    ProbeResult,
    ServerState,
    SiteState,
    probe_outcome,
    validate_probe,
)
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import default_topology

TOPO = default_topology()
DRAWS = 4000

HEALTHY_SERVER = ServerState(
    managed=True, host_up=True, process_up=True, hardware_fault=False, slowness=0.0
)
HEALTHY_SITE = SiteState(
    path_loss=0.0,
    powered=True,
    serving=True,
    slowness=0.0,
    reaching_qps=5000.0,
    attack_share=0.0,
    surge_share=0.0,
)


def _shares(kind: ProbeKind, state: ServerState | SiteState) -> dict[ProbeOutcome, float]:
    """Share of each outcome over many draws from a fixed generator."""
    rng = np.random.default_rng(7)
    counts = Counter(probe_outcome(kind, state, rng) for _ in range(DRAWS))
    return {outcome: n / DRAWS for outcome, n in counts.items()}


# ---------------------------------------------------------------- outcome model


def test_every_kind_has_outcomes_and_a_duration_and_every_outcome_is_used() -> None:
    assert set(OUTCOMES) == set(ProbeKind) == set(PROBE_TICKS)
    assert all(ticks >= 1 for ticks in PROBE_TICKS.values())
    used = {outcome for outcomes in OUTCOMES.values() for outcome in outcomes}
    assert used == set(ProbeOutcome)


def test_request_accepts_a_kind_name_and_rejects_unknown_kinds() -> None:
    assert ProbeRequest("hw_check", "ott-1").kind is ProbeKind.HW_CHECK  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ping"):
        ProbeRequest("ping", "ott")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "good", "bad"),
    [
        (ProbeKind.LOCAL_DNS, "ott-1", "ott"),
        (ProbeKind.HW_CHECK, "sin-2", "sin"),
        (ProbeKind.REMOTE_DNS, "lon", "lon-1"),
        (ProbeKind.NEIGHBOUR_DNS, "fra", "eu-west"),
        (ProbeKind.TRAFFIC_SAMPLE, "par", "nowhere"),
    ],
)
def test_probe_targets_are_checked_against_the_topology(
    kind: ProbeKind, good: str, bad: str
) -> None:
    validate_probe(ProbeRequest(kind, good), TOPO)
    with pytest.raises(ValueError, match="which is not a"):
        validate_probe(ProbeRequest(kind, bad), TOPO)


def test_healthy_targets_mostly_answer_ok_but_probes_sometimes_flake() -> None:
    for kind in (ProbeKind.LOCAL_DNS, ProbeKind.REMOTE_DNS, ProbeKind.NEIGHBOUR_DNS):
        state = HEALTHY_SERVER if kind == ProbeKind.LOCAL_DNS else HEALTHY_SITE
        shares = _shares(kind, state)
        assert set(shares) == {ProbeOutcome.OK, ProbeOutcome.TIMEOUT}
        assert shares[ProbeOutcome.TIMEOUT] == pytest.approx(FLAKE_PROB, abs=0.01)


def test_local_probe_tells_a_crashed_process_from_a_dead_host() -> None:
    crashed = replace(HEALTHY_SERVER, process_up=False)
    dead = replace(HEALTHY_SERVER, host_up=False, process_up=False)
    assert _shares(ProbeKind.LOCAL_DNS, crashed)[ProbeOutcome.REFUSED] > 0.95
    assert _shares(ProbeKind.LOCAL_DNS, dead) == {ProbeOutcome.TIMEOUT: 1.0}


def test_slowness_reads_as_slow_with_noise_near_the_threshold() -> None:
    assert (
        _shares(ProbeKind.LOCAL_DNS, replace(HEALTHY_SERVER, slowness=1.0))[ProbeOutcome.SLOW]
        > 0.95
    )
    borderline = _shares(ProbeKind.NEIGHBOUR_DNS, replace(HEALTHY_SITE, slowness=0.5))
    assert 0.4 < borderline[ProbeOutcome.SLOW] < 0.6
    assert 0.4 < borderline[ProbeOutcome.OK] < 0.6


def test_hardware_check_finds_faults_with_some_misses_and_false_alarms() -> None:
    broken = _shares(ProbeKind.HW_CHECK, replace(HEALTHY_SERVER, hardware_fault=True))
    healthy = _shares(ProbeKind.HW_CHECK, HEALTHY_SERVER)
    assert 0.90 < broken[ProbeOutcome.HARDWARE_FAULT] < 0.96
    assert 0.0 < broken[ProbeOutcome.HEALTHY] < 0.08
    assert 0.0 < healthy[ProbeOutcome.HARDWARE_FAULT] < 0.03
    unmanaged = replace(HEALTHY_SERVER, managed=False, hardware_fault=True)
    assert _shares(ProbeKind.HW_CHECK, unmanaged) == {ProbeOutcome.UNREACHABLE: 1.0}


def test_path_loss_shows_remotely_but_not_over_the_backbone() -> None:
    lossy = replace(HEALTHY_SITE, path_loss=0.3)
    assert _shares(ProbeKind.REMOTE_DNS, lossy)[ProbeOutcome.LOSSY] > 0.9
    assert _shares(ProbeKind.NEIGHBOUR_DNS, lossy)[ProbeOutcome.OK] > 0.95
    cut_off = replace(HEALTHY_SITE, path_loss=1.0)
    assert _shares(ProbeKind.REMOTE_DNS, cut_off) == {ProbeOutcome.TIMEOUT: 1.0}


def test_a_site_with_no_working_server_answers_servfail() -> None:
    empty = replace(HEALTHY_SITE, serving=False)
    for kind in (ProbeKind.REMOTE_DNS, ProbeKind.NEIGHBOUR_DNS):
        assert _shares(kind, empty)[ProbeOutcome.SERVFAIL] > 0.95


def test_a_site_that_is_down_times_out_everywhere() -> None:
    down = replace(HEALTHY_SITE, powered=False, serving=False, reaching_qps=0.0)
    for kind in (ProbeKind.REMOTE_DNS, ProbeKind.NEIGHBOUR_DNS):
        assert _shares(kind, down) == {ProbeOutcome.TIMEOUT: 1.0}
    assert _shares(ProbeKind.TRAFFIC_SAMPLE, down) == {ProbeOutcome.NO_TRAFFIC: 1.0}


def test_traffic_sample_reads_floods_and_surges_with_shared_ambiguity() -> None:
    flood = _shares(ProbeKind.TRAFFIC_SAMPLE, replace(HEALTHY_SITE, attack_share=0.8))
    surge = _shares(ProbeKind.TRAFFIC_SAMPLE, replace(HEALTHY_SITE, surge_share=0.8))
    normal = _shares(ProbeKind.TRAFFIC_SAMPLE, HEALTHY_SITE)
    assert flood[ProbeOutcome.ATTACK_PATTERN] == pytest.approx(0.65, abs=0.03)
    assert surge[ProbeOutcome.BROAD_SURGE] == pytest.approx(0.70, abs=0.03)
    assert normal[ProbeOutcome.NORMAL_MIX] == pytest.approx(0.94, abs=0.02)
    # A known resolver's burst of normal names is consistent with both: the agent
    # cannot settle flood vs surge from it and should escalate, not rate limit.
    assert flood[ProbeOutcome.RESOLVER_BURST] > 0.25
    assert surge[ProbeOutcome.RESOLVER_BURST] > 0.2
    assert ProbeOutcome.NORMAL_MIX not in flood and ProbeOutcome.NORMAL_MIX not in surge


def test_a_small_share_of_unusual_traffic_still_looks_normal() -> None:
    mild = replace(HEALTHY_SITE, attack_share=0.1, surge_share=0.1)
    assert _shares(ProbeKind.TRAFFIC_SAMPLE, mild)[ProbeOutcome.NORMAL_MIX] > 0.9


def test_wrong_state_type_for_the_probe_is_rejected() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="ServerState"):
        probe_outcome(ProbeKind.LOCAL_DNS, HEALTHY_SITE, rng)
    with pytest.raises(TypeError, match="SiteState"):
        probe_outcome(ProbeKind.REMOTE_DNS, HEALTHY_SERVER, rng)


_fraction = st.floats(0.0, 1.0)


@given(
    kind=st.sampled_from(list(ProbeKind)),
    flags=st.tuples(st.booleans(), st.booleans(), st.booleans(), st.booleans()),
    numbers=st.tuples(_fraction, _fraction, _fraction, _fraction),
    reaching=st.floats(0.0, 1e5),
    seed=st.integers(0, 2**32 - 1),
)
def test_outcome_is_always_one_the_kind_can_give(
    kind: ProbeKind,
    flags: tuple[bool, bool, bool, bool],
    numbers: tuple[float, float, float, float],
    reaching: float,
    seed: int,
) -> None:
    a, b, c, d = flags
    loss, slowness, attack, surge = numbers
    state: ServerState | SiteState
    if kind in (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK):
        state = ServerState(a, b, c, d, slowness)
    else:
        state = SiteState(loss, a, b, slowness, reaching, attack, surge)
    assert probe_outcome(kind, state, np.random.default_rng(seed)) in OUTCOMES[kind]


# ---------------------------------------------------------------- probes in the engine

Requests = Sequence[Action | ProbeRequest]


def _run(
    faults: Sequence[Fault],
    plan: Callable[[Observation], Requests],
    horizon: int,
    seed: int = 0,
) -> tuple[DnsEnv, list[Observation]]:
    env = DnsEnv(TOPO, faults, horizon, seed)
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(plan(observations[-1])))
    return env, observations


def _at(tick: int, *requests: Action | ProbeRequest) -> Callable[[Observation], Requests]:
    """Issue ``requests`` so the engine starts them at ``tick``."""
    return lambda obs: requests if obs.tick == tick - 1 else ()


def _target(kind: ProbeKind, site: str) -> str:
    """The site itself, or its first server for the server probes."""
    return f"{site}-1" if kind in (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK) else site


def _results(observations: list[Observation]) -> list[ProbeResult]:
    return [r for obs in observations for r in obs.probe_results]


def _probe_outcomes(
    faults: Sequence[Fault], tick: int, request: ProbeRequest, repeats: int = 5
) -> Counter[ProbeOutcome]:
    """Outcomes of ``repeats`` copies of one probe started at ``tick``."""
    horizon = tick + PROBE_TICKS[request.kind] + 1
    _, observations = _run(faults, _at(tick, *[request] * repeats), horizon)
    return Counter(r.outcome for r in _results(observations))


def test_observations_carry_no_probe_results_by_default() -> None:
    obs = Observation(tick=0, sites={}, servers={}, action_updates=())
    assert obs.probe_results == ()


def test_probe_result_arrives_after_its_duration_and_is_logged() -> None:
    requests = [ProbeRequest(kind, _target(kind, "ott")) for kind in ProbeKind]
    env, observations = _run((), _at(10, *requests), horizon=40)
    results = _results(observations)
    assert [r.request for r in results] == sorted(requests, key=lambda r: PROBE_TICKS[r.kind])
    for result in results:
        assert result.issued == 10
        assert result.completed == 10 + PROBE_TICKS[result.request.kind]
        assert result in observations[result.completed].probe_results
    started = env.events.named("probe_started")
    finished = env.events.named("probe_result")
    assert [e.data["probe_id"] for e in started] == list(range(len(requests)))
    assert sorted(int(str(e.data["probe_id"])) for e in finished) == list(range(len(requests)))
    assert {e.data["outcome"] for e in finished} <= {str(o) for o in ProbeOutcome}


def test_probes_finishing_together_are_reported_in_request_order() -> None:
    first, second = (
        ProbeRequest(ProbeKind.REMOTE_DNS, "lon"),
        ProbeRequest(ProbeKind.REMOTE_DNS, "fra"),
    )
    _, observations = _run((), _at(3, first, second), horizon=12)
    assert [r.request for r in _results(observations)] == [first, second]


def test_probes_do_not_change_the_network_or_its_telemetry() -> None:
    faults = [Fault(FaultKind.CRASH, "yyz-1", 30, None, 1.0)]
    probes = [ProbeRequest(kind, _target(kind, "yyz")) for kind in ProbeKind]
    plain, plain_obs = _run(faults, lambda obs: (), horizon=120)
    probed, probed_obs = _run(faults, lambda obs: probes if obs.tick % 7 == 0 else (), 120)
    assert np.array_equal(plain.truth.legit_failed, probed.truth.legit_failed)
    assert [o.sites for o in plain_obs] == [o.sites for o in probed_obs]
    assert [o.servers for o in plain_obs] == [o.servers for o in probed_obs]


def test_same_seed_and_probes_give_identical_event_logs() -> None:
    faults = [Fault(FaultKind.PATH_LOSS, "lon", 20, None, 0.4)]
    plan = _at(
        40, ProbeRequest(ProbeKind.REMOTE_DNS, "lon"), ProbeRequest(ProbeKind.HW_CHECK, "lon-2")
    )
    first, _ = _run(faults, plan, horizon=80, seed=3)
    second, _ = _run(faults, plan, horizon=80, seed=3)
    assert first.events.to_jsonl() == second.events.to_jsonl()


def test_invalid_probe_target_raises() -> None:
    env = DnsEnv(TOPO, (), 10, 0)
    env.reset()
    with pytest.raises(ValueError, match="not a server"):
        env.step([ProbeRequest(ProbeKind.LOCAL_DNS, "ott")])


def test_a_bad_request_leaves_the_run_untouched() -> None:
    env = DnsEnv(TOPO, (), 10, 0)
    env.reset()
    restart = Action(ActionKind.RESTART, "ott-1")
    with pytest.raises(ValueError, match="not a server"):
        env.step([restart, ProbeRequest(ProbeKind.HW_CHECK, "ott")])
    with pytest.raises(TypeError, match="Action or ProbeRequest"):
        env.step([restart, "restart ott-2"])  # type: ignore[list-item]
    assert env.tick == 0
    assert env.events.named("action") == []
    assert env.step([restart]).tick == 1


def test_probes_work_at_a_site_without_a_queue() -> None:
    no_queue = {sid: replace(site, queue_limit=0) for sid, site in TOPO.sites.items()}
    topology = replace(TOPO, sites=no_queue)
    env = DnsEnv(topology, (), 20, 0)
    obs = env.reset()
    requests = [ProbeRequest(kind, _target(kind, "ott")) for kind in ProbeKind]
    results: list[ProbeResult] = []
    while not env.done:
        obs = env.step(requests if obs.tick == 0 else ())
        results.extend(obs.probe_results)
    assert len(results) == len(ProbeKind)


def test_probe_still_running_at_the_end_of_the_run_is_simply_dropped() -> None:
    _, observations = _run((), _at(8, ProbeRequest(ProbeKind.HW_CHECK, "par-1")), horizon=10)
    assert _results(observations) == []


def _mostly(outcomes: Counter[ProbeOutcome], expected: ProbeOutcome) -> bool:
    return outcomes[expected] >= sum(outcomes.values()) - 1  # allow one flake


def test_crash_is_refused_locally_while_the_host_stays_healthy() -> None:
    crash = [Fault(FaultKind.CRASH, "yyz-1", 20, None, 1.0)]
    assert _mostly(
        _probe_outcomes(crash, 40, ProbeRequest(ProbeKind.LOCAL_DNS, "yyz-1")), ProbeOutcome.REFUSED
    )
    assert _mostly(
        _probe_outcomes(crash, 40, ProbeRequest(ProbeKind.HW_CHECK, "yyz-1")), ProbeOutcome.HEALTHY
    )
    assert _mostly(
        _probe_outcomes(crash, 40, ProbeRequest(ProbeKind.LOCAL_DNS, "yyz-2")), ProbeOutcome.OK
    )


def test_hardware_fault_times_out_locally_and_shows_on_the_hardware_check() -> None:
    hardware = [Fault(FaultKind.HARDWARE, "fra-2", 20, None, 1.0)]
    assert _mostly(
        _probe_outcomes(hardware, 40, ProbeRequest(ProbeKind.LOCAL_DNS, "fra-2")),
        ProbeOutcome.TIMEOUT,
    )
    assert _mostly(
        _probe_outcomes(hardware, 40, ProbeRequest(ProbeKind.HW_CHECK, "fra-2")),
        ProbeOutcome.HARDWARE_FAULT,
    )


def test_path_loss_is_lossy_remotely_and_clean_from_the_neighbour() -> None:
    loss = [Fault(FaultKind.PATH_LOSS, "lon", 20, None, 0.5)]
    assert _mostly(
        _probe_outcomes(loss, 40, ProbeRequest(ProbeKind.REMOTE_DNS, "lon")), ProbeOutcome.LOSSY
    )
    assert _mostly(
        _probe_outcomes(loss, 40, ProbeRequest(ProbeKind.NEIGHBOUR_DNS, "lon")), ProbeOutcome.OK
    )


def test_site_outage_looks_dead_from_every_side() -> None:
    outage = [Fault(FaultKind.SITE_OUTAGE, "par", 20, None, 1.0)]
    assert _probe_outcomes(outage, 40, ProbeRequest(ProbeKind.NEIGHBOUR_DNS, "par")) == {
        ProbeOutcome.TIMEOUT: 5
    }
    assert _probe_outcomes(outage, 40, ProbeRequest(ProbeKind.TRAFFIC_SAMPLE, "par")) == {
        ProbeOutcome.NO_TRAFFIC: 5
    }
    assert _probe_outcomes(outage, 40, ProbeRequest(ProbeKind.HW_CHECK, "par-1")) == {
        ProbeOutcome.UNREACHABLE: 5
    }


def test_a_server_under_repair_cannot_be_checked() -> None:
    plan = _at(
        5, Action(ActionKind.REQUEST_REPAIR, "sin-1"), ProbeRequest(ProbeKind.HW_CHECK, "sin-1")
    )
    _, observations = _run((), plan, horizon=30)
    assert [r.outcome for r in _results(observations)] == [ProbeOutcome.UNREACHABLE]


def test_flood_traffic_is_never_sampled_as_normal() -> None:
    flood = [Fault(FaultKind.FLOOD, "eu-west", 20, None, 1.0)]
    outcomes = _probe_outcomes(flood, 40, ProbeRequest(ProbeKind.TRAFFIC_SAMPLE, "lon"), 20)
    assert outcomes[ProbeOutcome.NORMAL_MIX] == 0
    assert outcomes[ProbeOutcome.ATTACK_PATTERN] > outcomes[ProbeOutcome.BROAD_SURGE]


def test_benign_surge_is_sampled_mostly_as_real_users() -> None:
    surge = [Fault(FaultKind.BENIGN_SURGE, "eu-west", 20, None, 1.0)]
    outcomes = _probe_outcomes(surge, 40, ProbeRequest(ProbeKind.TRAFFIC_SAMPLE, "lon"), 20)
    assert outcomes[ProbeOutcome.NORMAL_MIX] == 0
    assert outcomes[ProbeOutcome.BROAD_SURGE] > outcomes[ProbeOutcome.ATTACK_PATTERN]
