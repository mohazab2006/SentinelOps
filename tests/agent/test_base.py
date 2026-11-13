"""Tests for the agent interface."""

from __future__ import annotations

from types import MappingProxyType

from sentinelops.agent.base import Agent, NoOpAgent
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import default_topology


def test_noop_agent_satisfies_the_protocol_and_never_acts() -> None:
    agent = NoOpAgent()
    assert isinstance(agent, Agent)
    assert agent.name == "noop"
    agent.reset(default_topology())
    obs = Observation(
        tick=0, sites=MappingProxyType({}), servers=MappingProxyType({}), action_updates=()
    )
    assert agent.act(obs) == ()


def test_objects_without_act_are_not_agents() -> None:
    class NotAnAgent:
        name = "nope"

        def reset(self, topology: object) -> None:
            pass

    assert not isinstance(NotAnAgent(), Agent)
