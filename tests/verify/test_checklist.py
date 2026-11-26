"""Tests for the recovery verification checklist."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from sentinelops.scenarios import generate
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.telemetry import SiteTelemetry
from sentinelops.sim.topology import SiteId, default_topology
from sentinelops.verify import CheckResult, HealthLimits, SiteVerifier, VerifyStatus, check_site

TOPOLOGY = default_topology()
LON = SiteId("lon")
QUEUE_LIMIT = float(TOPOLOGY.sites[LON].queue_limit)


def reading(**changes: Any) -> SiteTelemetry:
    """A healthy site reading, with ``changes`` applied."""
    healthy: dict[str, Any] = {
        "qps": 100.0,
        "p95_latency_ms": 40.0,
        "loss": 0.0,
        "queue_depth": 0.0,
        "ok_replies": 100.0,
        "failed_replies": 0.0,
        "announced": True,
        "weight": 1.0,
    }
    return SiteTelemetry(**{**healthy, **changes})


HEALTHY = reading()
BAD = reading(p95_latency_ms=2700.0)
MISSING = reading(p95_latency_ms=None)


def verifier(start_tick: int = 0, **kwargs: Any) -> SiteVerifier:
    return SiteVerifier(LON, TOPOLOGY, start_tick, **kwargs)


def feed(v: SiteVerifier, readings: list[SiteTelemetry], first_tick: int = 1) -> list[VerifyStatus]:
    """Feed ``readings`` on consecutive ticks; return the status after each."""
    return [v.update(first_tick + i, r) for i, r in enumerate(readings)]


# ------------------------------------------------------------------ check_site


def test_healthy_reading_passes_all_four_checks_in_order() -> None:
    checks = check_site(HEALTHY, QUEUE_LIMIT)
    assert [c.name for c in checks] == ["fail_ratio", "p95_latency", "loss", "queue"]
    assert all(c.ok is True for c in checks)
    assert [c.limit for c in checks] == [0.01, 250.0, 0.05, 0.1]


@pytest.mark.parametrize(
    ("index", "at_limit", "above"),
    [
        (
            0,
            {"ok_replies": 99.0, "failed_replies": 1.0},
            {"ok_replies": 98.0, "failed_replies": 2.0},
        ),
        (1, {"p95_latency_ms": 250.0}, {"p95_latency_ms": 250.5}),
        (2, {"loss": 0.05}, {"loss": 0.051}),
        (3, {"queue_depth": 0.1 * QUEUE_LIMIT}, {"queue_depth": 0.1 * QUEUE_LIMIT + 1}),
    ],
)
def test_each_limit_on_its_own(
    index: int, at_limit: dict[str, float], above: dict[str, float]
) -> None:
    assert all(c.ok for c in check_site(reading(**at_limit), QUEUE_LIMIT))  # equal passes
    oks = [c.ok for c in check_site(reading(**above), QUEUE_LIMIT)]
    assert oks == [i != index for i in range(4)]  # only that check fails


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("ok_replies", 0),
        ("failed_replies", 0),
        ("p95_latency_ms", 1),
        ("loss", 2),
        ("queue_depth", 3),
    ],
)
def test_missing_reading_is_unknown_not_failed(field: str, index: int) -> None:
    check = check_site(reading(**{field: None}), QUEUE_LIMIT)[index]
    assert (check.value, check.ok) == (None, None)


def test_zero_traffic_counts_as_zero_fail_ratio() -> None:
    check = check_site(reading(ok_replies=0.0, failed_replies=0.0), QUEUE_LIMIT)[0]
    assert (check.value, check.ok) == (0.0, True)


def test_fail_ratio_value_is_failed_over_total() -> None:
    check = check_site(reading(ok_replies=75.0, failed_replies=25.0), QUEUE_LIMIT)[0]
    assert check.value == 0.25
    assert check.ok is False


@pytest.mark.parametrize(("depth", "ok"), [(0.0, True), (1.0, False)])
def test_site_without_a_queue_accepts_only_an_empty_queue(depth: float, ok: bool) -> None:
    assert check_site(reading(queue_depth=depth), 0)[3].ok is ok


def test_custom_limits_are_used() -> None:
    strict = HealthLimits(max_p95_ms=30.0)
    assert check_site(HEALTHY, QUEUE_LIMIT, strict)[1].ok is False


# ------------------------------------------------------------------ SiteVerifier streaks


def test_passes_exactly_on_the_thirtieth_healthy_tick() -> None:
    v = verifier()
    statuses = feed(v, [HEALTHY] * 30)
    assert statuses[28] == VerifyStatus.PENDING
    assert statuses[29] == VerifyStatus.PASSED
    assert v.streak == 30


def test_a_bad_tick_resets_the_streak() -> None:
    v = verifier()
    statuses = feed(v, [HEALTHY] * 29 + [BAD] + [HEALTHY] * 30)
    assert v.status == VerifyStatus.PASSED
    assert statuses.index(VerifyStatus.PASSED) == 59  # only at the very end
    assert statuses[29] == VerifyStatus.PENDING


def test_missing_readings_neither_reset_nor_extend_the_streak() -> None:
    v = verifier()
    statuses = feed(v, [HEALTHY] * 15 + [MISSING] * 5 + [HEALTHY] * 15)
    assert v.streak == 30
    assert statuses[-2] == VerifyStatus.PENDING
    assert statuses[-1] == VerifyStatus.PASSED


def test_missing_readings_do_not_reset_the_bad_streak_either() -> None:
    v = verifier()
    feed(v, [BAD] * 5 + [MISSING] * 3 + [BAD] * 5)
    assert v.status == VerifyStatus.FAILED


def test_ten_bad_ticks_in_a_row_fail() -> None:
    v = verifier()
    statuses = feed(v, [BAD] * 10)
    assert statuses[8] == VerifyStatus.PENDING
    assert statuses[9] == VerifyStatus.FAILED


def test_nine_bad_then_one_good_is_still_pending() -> None:
    v = verifier()
    feed(v, [BAD] * 9 + [HEALTHY] + [BAD] * 9)
    assert v.status == VerifyStatus.PENDING


def test_deadline_fails_a_site_that_never_settles() -> None:
    v = verifier(start_tick=100)
    flapping = ([HEALTHY] * 20 + [BAD]) * 20  # never 30 good or 10 bad in a row
    statuses = feed(v, flapping, first_tick=101)
    assert statuses[298] == VerifyStatus.PENDING  # tick 399
    assert statuses[299] == VerifyStatus.FAILED  # tick 400 = start 100 + deadline 300


def test_status_is_final() -> None:
    v = verifier()
    feed(v, [BAD] * 10)
    checks_at_failure = v.last_checks
    assert feed(v, [HEALTHY] * 40, first_tick=11)[-1] == VerifyStatus.FAILED
    assert v.last_checks == checks_at_failure  # later ticks are ignored
    passed = verifier()
    feed(passed, [HEALTHY] * 30)
    assert feed(passed, [BAD] * 20, first_tick=31)[-1] == VerifyStatus.PASSED


def test_last_checks_are_empty_before_the_first_update() -> None:
    v = verifier()
    assert v.last_checks == ()
    assert (v.status, v.streak) == (VerifyStatus.PENDING, 0)
    v.update(1, BAD)
    assert v.last_checks[1] == CheckResult("p95_latency", 2700.0, 250.0, False)


def test_properties_are_read_only() -> None:
    v = verifier()
    for name in ("status", "streak", "last_checks"):
        with pytest.raises(AttributeError):
            setattr(v, name, None)


def test_custom_settings_are_used() -> None:
    v = verifier(required_ticks=3, fail_after=2, deadline_ticks=5)
    assert feed(v, [HEALTHY] * 3)[-1] == VerifyStatus.PASSED
    v = verifier(required_ticks=3, fail_after=2, limits=HealthLimits(max_p95_ms=30.0))
    assert feed(v, [HEALTHY] * 2)[-1] == VerifyStatus.FAILED  # 40 ms is bad under this limit


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "field", ["max_fail_ratio", "max_p95_ms", "max_loss", "max_queue_fraction"]
)
def test_negative_limit_is_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        HealthLimits(**{field: -0.1})


def test_limits_and_results_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        HealthLimits().max_loss = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        check_site(HEALTHY, QUEUE_LIMIT)[0].ok = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"required_ticks": 0}, "required_ticks"),
        ({"fail_after": 0}, "fail_after"),
        ({"required_ticks": 30, "deadline_ticks": 29}, "deadline_ticks"),
    ],
)
def test_bad_settings_are_rejected(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        verifier(**kwargs)


def test_unknown_site_is_rejected() -> None:
    with pytest.raises(ValueError, match="tor"):
        SiteVerifier(SiteId("tor"), TOPOLOGY, 0)


# ------------------------------------------------------------------ real runs


def test_real_healthy_run_passes() -> None:
    env = DnsEnv(TOPOLOGY, faults=(), horizon=61, seed=0)  # ticks 0-60: 60 steps after reset
    obs = env.reset()
    v = SiteVerifier(LON, TOPOLOGY, start_tick=0)
    for _ in range(60):
        obs = env.step([])
        v.update(obs.tick, obs.sites[LON])
    assert v.status == VerifyStatus.PASSED


def test_real_crash_without_an_agent_fails_via_bad_streak() -> None:
    scenario = generate("crash", 100, TOPOLOGY)
    fault = scenario.faults[0]
    site = TOPOLOGY.servers[fault.target].site  # type: ignore[index]
    start = fault.onset + 30
    env = DnsEnv(TOPOLOGY, scenario.faults, horizon=start + 60, seed=100)
    obs = env.reset()
    while obs.tick < start:
        obs = env.step([])
    v = SiteVerifier(site, TOPOLOGY, start_tick=start)
    statuses = []
    for _ in range(20):
        obs = env.step([])
        statuses.append(v.update(obs.tick, obs.sites[site]))
    assert v.status == VerifyStatus.FAILED
    assert (
        statuses.index(VerifyStatus.FAILED) == 9
    )  # ten bad ticks in a row, well before the deadline
