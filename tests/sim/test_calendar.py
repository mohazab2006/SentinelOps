"""Tests for the simulated calendar and its use in the environment."""

from __future__ import annotations

import pytest

from sentinelops.scenarios.families import generate
from sentinelops.sim.calendar import (
    DEFAULT_START,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    WEEKDAY_NAMES,
    Calendar,
    SimTime,
)
from sentinelops.sim.env import DnsEnv, EnvConfig
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import default_topology

TOPO = default_topology()

# ------------------------------------------------------------------ SimTime


def test_fields_default_to_the_top_of_the_hour() -> None:
    assert SimTime(2, 14) == SimTime(weekday=2, hour=14, minute=0, second=0)


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("weekday", {"weekday": 7, "hour": 0}),
        ("weekday", {"weekday": -1, "hour": 0}),
        ("hour", {"weekday": 0, "hour": 24}),
        ("minute", {"weekday": 0, "hour": 0, "minute": 60}),
        ("second", {"weekday": 0, "hour": 0, "second": -1}),
    ],
)
def test_out_of_range_fields_are_rejected(field: str, values: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=field):
        SimTime(**values)


def test_extremes_are_accepted() -> None:
    assert str(SimTime(6, 23, 59, 59)) == "Sun 23:59:59"
    assert str(SimTime(0, 0)) == "Mon 00:00:00"


def test_week_seconds_round_trip() -> None:
    for seconds in (0, 1, 3_599, 36_000, SECONDS_PER_DAY + 61, SECONDS_PER_WEEK - 1):
        assert SimTime.from_week_seconds(seconds).week_seconds == seconds


def test_from_week_seconds_wraps_every_week() -> None:
    assert SimTime.from_week_seconds(SECONDS_PER_WEEK) == SimTime(0, 0)
    assert SimTime.from_week_seconds(SECONDS_PER_WEEK + 90) == SimTime(0, 0, 1, 30)


def test_weekend() -> None:
    assert [SimTime(day, 12).is_weekend for day in range(7)] == [False] * 5 + [True] * 2


def test_ordering_follows_the_week() -> None:
    times = [SimTime(4, 9), SimTime(0, 23, 59), SimTime(0, 0, 0, 1), SimTime(6, 0)]
    assert sorted(times) == [SimTime(0, 0, 0, 1), SimTime(0, 23, 59), SimTime(4, 9), SimTime(6, 0)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tue 14:05", SimTime(1, 14, 5)),
        ("tue 14:05:30", SimTime(1, 14, 5, 30)),
        ("  SUN 3:00 ", SimTime(6, 3)),
    ],
)
def test_parse(text: str, expected: SimTime) -> None:
    assert SimTime.parse(text) == expected


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("Tuesday 14:05", "expected a time"),
        ("Tue 1405", "expected a time"),
        ("", "expected a time"),
        ("Xyz 10:00", "unknown weekday"),
        ("Tue 25:00", "hour"),
        ("Tue 10:61", "minute"),
    ],
)
def test_parse_rejects_bad_text(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SimTime.parse(text)


def test_str_and_parse_agree_for_every_weekday() -> None:
    for day, name in enumerate(WEEKDAY_NAMES):
        moment = SimTime(day, 8, 30, 15)
        assert str(moment).startswith(name)
        assert SimTime.parse(str(moment)) == moment


# ------------------------------------------------------------------ Calendar


def test_default_calendar_starts_monday_ten() -> None:
    assert Calendar().start == DEFAULT_START == SimTime(0, 10)
    assert Calendar().at(0) == DEFAULT_START


def test_one_tick_is_one_second() -> None:
    calendar = Calendar(SimTime(1, 16, 59, 58))
    assert [str(calendar.at(t)) for t in range(3)] == [
        "Tue 16:59:58",
        "Tue 16:59:59",
        "Tue 17:00:00",
    ]


def test_calendar_crosses_midnight_and_the_week_boundary() -> None:
    calendar = Calendar(SimTime(6, 23, 59, 30))
    assert calendar.at(30) == SimTime(0, 0)
    assert calendar.at(30 + SECONDS_PER_DAY) == SimTime(1, 0)


def test_negative_tick_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Calendar().at(-1)


# ------------------------------------------------------------------ environment


def _run(config: EnvConfig | None = None, horizon: int = 120) -> tuple[DnsEnv, list[Observation]]:
    scenario = generate("crash", 100, TOPO)
    env = DnsEnv(TOPO, scenario.faults, horizon, scenario.seed, config)
    observations = [env.reset()]
    while not env.done:
        observations.append(env.step(()))
    return env, observations


def test_observations_carry_the_moment_of_their_tick() -> None:
    start = SimTime(5, 2, 59, 0)  # Saturday, just before 03:00
    _, observations = _run(EnvConfig(calendar=Calendar(start)))
    assert observations[0].calendar == start
    assert observations[60].calendar == SimTime(5, 3)
    assert all(obs.calendar == Calendar(start).at(obs.tick) for obs in observations)


def test_default_run_starts_monday_ten_and_logs_it() -> None:
    env, observations = _run()
    assert observations[0].calendar == DEFAULT_START
    started = next(e for e in env.events if e.name == "run_started")
    assert started.data["start"] == "Mon 10:00:00"


def test_start_time_does_not_change_the_incident() -> None:
    """Same seed, different start: telemetry and ground truth are identical."""
    env_a, obs_a = _run(horizon=400)
    env_b, obs_b = _run(EnvConfig(calendar=Calendar(SimTime(6, 3))), horizon=400)
    assert [o.sites for o in obs_a] == [o.sites for o in obs_b]
    assert [o.servers for o in obs_a] == [o.servers for o in obs_b]
    assert env_a.truth.legit_failure_share() == env_b.truth.legit_failure_share()


def test_hand_made_observation_defaults_to_monday_ten() -> None:
    obs = Observation(tick=5, sites={}, servers={}, action_updates=())
    assert obs.calendar == DEFAULT_START
