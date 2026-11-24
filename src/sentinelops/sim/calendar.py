"""Simulated wall-clock time: which weekday and hour each tick falls on.

Operations policies depend on the time: a restart that is fine on a Tuesday
afternoon may need approval at 3 a.m. or during a weekend change freeze. The
simulator counts ticks (one tick is one second), so a :class:`Calendar` maps
tick numbers to a :class:`SimTime` in the operator's time zone, starting from
a chosen moment of the week.

Time only labels the run: demand and faults do not depend on it, so the same
seed gives the same incident whatever the start time. That lets an experiment
replay one incident at different times of the week and compare only what the
policy allows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_WEEK = 7 * SECONDS_PER_DAY

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
"""Short weekday names, indexed like :attr:`SimTime.weekday` (0 is Monday)."""

_TIME_PATTERN = re.compile(r"^\s*([A-Za-z]{3})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")


@dataclass(frozen=True, slots=True, order=True)
class SimTime:
    """A moment of the week, in the operator's time zone.

    Ordering compares moments within one week (Monday 00:00:00 first).

    Attributes:
        weekday: Day of the week, 0 (Monday) to 6 (Sunday), as in
            :meth:`datetime.date.weekday`.
        hour: Hour of the day, 0 to 23.
        minute: Minute of the hour, 0 to 59.
        second: Second of the minute, 0 to 59.
    """

    weekday: int
    hour: int
    minute: int = 0
    second: int = 0

    def __post_init__(self) -> None:
        """Reject a field outside its range.

        Raises:
            ValueError: If any field is out of range.
        """
        for name, value, top in (
            ("weekday", self.weekday, 6),
            ("hour", self.hour, 23),
            ("minute", self.minute, 59),
            ("second", self.second, 59),
        ):
            if not 0 <= value <= top:
                raise ValueError(f"{name} must be in [0, {top}], got {value}")

    @classmethod
    def from_week_seconds(cls, seconds: int) -> SimTime:
        """The moment ``seconds`` after Monday 00:00:00, wrapping every week."""
        seconds %= SECONDS_PER_WEEK
        day, rest = divmod(seconds, SECONDS_PER_DAY)
        hour, rest = divmod(rest, SECONDS_PER_HOUR)
        minute, second = divmod(rest, SECONDS_PER_MINUTE)
        return cls(day, hour, minute, second)

    @classmethod
    def parse(cls, text: str) -> SimTime:
        """Read a moment written as ``"Tue 14:05"`` or ``"Tue 14:05:30"``.

        The weekday name is case-insensitive.

        Raises:
            ValueError: If ``text`` is not in that format or a field is out of range.
        """
        match = _TIME_PATTERN.match(text)
        if match is None:
            raise ValueError(f"expected a time like 'Tue 14:05', got {text!r}")
        name, hour, minute, second = match.groups()
        names = [n.lower() for n in WEEKDAY_NAMES]
        if name.lower() not in names:
            raise ValueError(f"unknown weekday {name!r}; use one of {', '.join(WEEKDAY_NAMES)}")
        return cls(names.index(name.lower()), int(hour), int(minute), int(second or 0))

    @property
    def week_seconds(self) -> int:
        """Seconds since Monday 00:00:00 of the same week."""
        return (
            self.weekday * SECONDS_PER_DAY
            + self.hour * SECONDS_PER_HOUR
            + self.minute * SECONDS_PER_MINUTE
            + self.second
        )

    @property
    def is_weekend(self) -> bool:
        """Whether the moment falls on a Saturday or Sunday."""
        return self.weekday >= 5

    def __str__(self) -> str:
        """The moment as ``"Tue 14:05:30"``."""
        return f"{WEEKDAY_NAMES[self.weekday]} {self.hour:02d}:{self.minute:02d}:{self.second:02d}"


DEFAULT_START = SimTime(weekday=0, hour=10)
"""Default start of a run: Monday 10:00, inside business hours."""


@dataclass(frozen=True, slots=True)
class Calendar:
    """Maps simulation ticks to moments of the week.

    Tick 0 is :attr:`start`; each tick adds one second, and time wraps from
    Sunday 23:59:59 to Monday 00:00:00.

    Attributes:
        start: The moment of tick 0.
    """

    start: SimTime = DEFAULT_START

    def at(self, tick: int) -> SimTime:
        """The moment of ``tick``.

        Raises:
            ValueError: If ``tick`` is negative.
        """
        if tick < 0:
            raise ValueError(f"tick must be non-negative, got {tick}")
        return SimTime.from_week_seconds(self.start.week_seconds + tick)
