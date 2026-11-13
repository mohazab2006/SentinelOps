"""Append-only event log of a simulation run.

The event log records everything notable that happens in a run: faults starting
and ending, actions changing status, and a per-tick summary of traffic. It is
the single source for replays, the operations console, incident reports and
scoring. It contains ground truth, so it is never shown to agents.

Events serialise to JSON Lines with sorted keys, so two runs with the same seed
and the same agent decisions produce byte-identical logs.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

JsonScalar = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened in a run.

    Attributes:
        tick: Tick at which it happened.
        name: Event type, e.g. ``"fault_started"`` or ``"action"``.
        data: Event details; values are JSON scalars.
    """

    tick: int
    name: str
    data: Mapping[str, JsonScalar]

    def to_json(self) -> str:
        """Serialise the event as one compact JSON object with sorted keys."""
        record = {"tick": self.tick, "event": self.name, **self.data}
        return json.dumps(record, sort_keys=True, separators=(",", ":"))


class EventLog:
    """An append-only, ordered list of :class:`Event` records."""

    def __init__(self) -> None:
        """Create an empty log."""
        self._events: list[Event] = []

    def append(self, tick: int, name: str, /, **data: JsonScalar) -> Event:
        """Record an event called ``name`` with details ``data`` and return it."""
        if self._events and tick < self._events[-1].tick:
            raise ValueError("events must be appended in tick order")
        event = Event(tick, name, MappingProxyType(dict(data)))
        self._events.append(event)
        return event

    def __len__(self) -> int:
        """Return the number of events."""
        return len(self._events)

    def __iter__(self) -> Iterator[Event]:
        """Iterate over events in the order they were recorded."""
        return iter(self._events)

    def named(self, name: str) -> list[Event]:
        """Return all events called ``name``, in order."""
        return [event for event in self._events if event.name == name]

    def to_jsonl(self) -> str:
        """Serialise the whole log as JSON Lines (one event per line)."""
        return "".join(event.to_json() + "\n" for event in self._events)
