"""Run agents against seeded scenarios and measure failed legitimate queries.

This is the single place where an agent meets the simulator. Every experiment,
the command line and the tests go through :func:`simulate`, so all agents are
driven by exactly the same loop::

    obs = env.reset()
    while not env.done:
        obs = env.step(agent.act(obs))

The primary metric is the share of legitimate queries that failed (lost,
rejected by a rate limit, overflowed a queue, timed out or found no announcing
site), computed from the simulator's ground truth, which agents never see.

Agents are looked up by name in :data:`AGENTS` and imported only when used, so
an agent that is not installed yet does not break the others.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from sentinelops.agent.base import Agent
from sentinelops.scenarios import Scenario
from sentinelops.sim.actions import ActionStatus
from sentinelops.sim.env import FAILURE_CAUSES, DnsEnv
from sentinelops.sim.topology import Topology

AGENTS: Mapping[str, str] = MappingProxyType(
    {
        "noop": "sentinelops.agent.base:NoOpAgent",
        "rules": "sentinelops.baselines.rules_agent:RuleBasedAgent",
    }
)
"""Agents available by name, as ``"module:Class"``; each class takes no arguments."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one agent on one scenario.

    Attributes:
        family: Scenario family.
        seed: Scenario seed (also the simulator seed).
        agent: Name of the agent.
        horizon: Ticks simulated.
        legit_offered: Legitimate queries sent by clients over the run.
        legit_failed: Legitimate queries that failed over the run.
        legit_failure_share: ``legit_failed / legit_offered`` (0.0 if nothing
            was offered) — the primary metric.
        failed_by_cause: Legitimate failures per cause, in
            :data:`~sentinelops.sim.env.FAILURE_CAUSES` order.
        actions_issued: Actions the agent asked for.
        actions_succeeded: Actions that completed successfully.
        actions_failed: Actions that failed (conflicts, or a restart that
            could not fix a hardware fault).
        event_digest: SHA-256 of the event log; equal digests mean the two
            runs were identical event by event.
    """

    family: str
    seed: int
    agent: str
    horizon: int
    legit_offered: float
    legit_failed: float
    legit_failure_share: float
    failed_by_cause: Mapping[str, float] = field(hash=False)
    actions_issued: int
    actions_succeeded: int
    actions_failed: int
    event_digest: str


def make_agent(name: str) -> Agent:
    """Create a fresh agent by name.

    Args:
        name: A key of :data:`AGENTS`.

    Returns:
        A new instance of the agent.

    Raises:
        ValueError: If the name is unknown, the agent is not part of this
            installation yet, it cannot be created without arguments, or it
            does not implement :class:`Agent`.
    """
    if name not in AGENTS:
        raise ValueError(f"unknown agent {name!r}; valid agents: {', '.join(AGENTS)}")
    module_name, _, class_name = AGENTS[name].partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if module_name != missing and not module_name.startswith(missing + "."):
            raise  # the agent module exists but one of its own imports is missing
        raise ValueError(f"agent {name!r} is not available ({module_name} not found)") from error
    agent_class = getattr(module, class_name, None)
    if agent_class is None:
        raise ValueError(f"agent {name!r} is not available ({module_name} has no {class_name})")
    try:
        agent = agent_class()
    except TypeError as error:
        raise ValueError(f"agent {name!r} cannot be created without arguments: {error}") from error
    if not isinstance(agent, Agent):
        raise ValueError(f"{AGENTS[name]} does not implement the Agent protocol")
    return agent


def simulate(scenario: Scenario, agent: Agent, topology: Topology) -> DnsEnv:
    """Run ``agent`` on ``scenario`` from the first tick to the last.

    The simulator is seeded with the scenario's seed, so every agent faces the
    same demand, noise and faults for a given scenario.

    Returns:
        The finished environment, for its event log and ground truth.
    """
    env = DnsEnv(topology, scenario.faults, scenario.horizon, seed=scenario.seed)
    agent.reset(topology)
    obs = env.reset()
    while not env.done:
        obs = env.step(agent.act(obs))
    return env


def summarize(scenario: Scenario, agent_name: str, env: DnsEnv) -> RunResult:
    """Score a finished run.

    Args:
        scenario: The scenario that was run.
        agent_name: Name of the agent that ran it.
        env: The finished environment returned by :func:`simulate`.

    Raises:
        ValueError: If the run has not finished, or ``env`` was not built from
            ``scenario``.
    """
    if not env.done:
        raise ValueError("the run has not finished")
    if (env.seed, env.horizon, env.faults) != (scenario.seed, scenario.horizon, scenario.faults):
        raise ValueError("env was not run from this scenario")
    truth = env.truth
    final_status: dict[object, str] = {}
    for event in env.events.named("action"):
        final_status[event.data["action_id"]] = str(event.data["status"])
    statuses = list(final_status.values())
    return RunResult(
        family=scenario.family,
        seed=scenario.seed,
        agent=agent_name,
        horizon=env.horizon,
        legit_offered=float(truth.legit_offered.sum()),
        legit_failed=float(truth.legit_failed.sum()),
        legit_failure_share=truth.legit_failure_share(),
        failed_by_cause=MappingProxyType(
            {cause: float(truth.legit_failed_by_cause[cause].sum()) for cause in FAILURE_CAUSES}
        ),
        actions_issued=len(statuses),
        actions_succeeded=statuses.count(ActionStatus.SUCCEEDED),
        actions_failed=statuses.count(ActionStatus.FAILED),
        event_digest=hashlib.sha256(env.events.to_jsonl().encode()).hexdigest(),
    )


def run_episode(scenario: Scenario, agent: Agent, topology: Topology) -> RunResult:
    """Run ``agent`` on ``scenario`` and return its scored result."""
    return summarize(scenario, agent.name, simulate(scenario, agent, topology))
