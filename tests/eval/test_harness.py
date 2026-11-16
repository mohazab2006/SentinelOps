"""Tests for the run harness: agent registry, episode loop and scoring."""

from __future__ import annotations

import importlib.util
import textwrap
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import pytest

from sentinelops.agent.base import NoOpAgent
from sentinelops.eval import harness
from sentinelops.eval.harness import AGENTS, make_agent, run_episode, simulate, summarize
from sentinelops.scenarios import Scenario, generate
from sentinelops.sim.actions import Action, ActionKind
from sentinelops.sim.env import FAILURE_CAUSES, DnsEnv
from sentinelops.sim.faults import Fault, FaultKind
from sentinelops.sim.telemetry import Observation
from sentinelops.sim.topology import Topology, default_topology

TOPO = default_topology()
CRASH = Scenario(
    family="crash",
    seed=5,
    horizon=700,
    faults=(Fault(FaultKind.CRASH, "lon-1", 300, None, 0.8),),
)
RULES_INSTALLED = importlib.util.find_spec("sentinelops.baselines.rules_agent") is not None


class ScriptedAgent:
    """Issues fixed actions at fixed ticks and records what it was given."""

    name = "scripted"

    def __init__(self, plan: dict[int, Sequence[Action]]) -> None:
        self.plan = plan
        self.resets: list[Topology] = []
        self.ticks: list[int] = []

    def reset(self, topology: Topology) -> None:
        self.resets.append(topology)
        self.ticks.clear()

    def act(self, obs: Observation) -> Sequence[Action]:
        self.ticks.append(obs.tick)
        return self.plan.get(obs.tick, ())


def restart_lon1_at(tick: int) -> ScriptedAgent:
    return ScriptedAgent({tick: (Action(ActionKind.RESTART, "lon-1"),)})


# --------------------------------------------------------------------- registry


def test_registry_names_the_baselines() -> None:
    assert set(AGENTS) == {"noop", "rules"}


def test_make_agent_returns_a_fresh_instance_each_time() -> None:
    first, second = make_agent("noop"), make_agent("noop")
    assert isinstance(first, NoOpAgent)
    assert first is not second


def test_unknown_agent_lists_the_valid_names() -> None:
    with pytest.raises(ValueError, match="valid agents: noop, rules"):
        make_agent("oracle")


def test_missing_agent_module_is_reported_as_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(harness, "AGENTS", MappingProxyType({"ghost": "sentinelops.ghost:Ghost"}))
    with pytest.raises(ValueError, match="not available"):
        make_agent("ghost")


def test_broken_import_inside_an_agent_module_is_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "broken_agent.py").write_text(
        textwrap.dedent(
            """
            import package_that_does_not_exist

            class Broken:
                name = "broken"
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(harness, "AGENTS", MappingProxyType({"broken": "broken_agent:Broken"}))
    with pytest.raises(ModuleNotFoundError, match="package_that_does_not_exist"):
        make_agent("broken")


def test_registered_class_must_implement_the_agent_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness, "AGENTS", MappingProxyType({"dict": "collections:OrderedDict"}))
    with pytest.raises(ValueError, match="Agent protocol"):
        make_agent("dict")


def test_missing_class_in_an_existing_module_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness, "AGENTS", MappingProxyType({"typo": "sentinelops.agent.base:Nope"})
    )
    with pytest.raises(ValueError, match="has no Nope"):
        make_agent("typo")


def test_agent_needing_arguments_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        harness, "AGENTS", MappingProxyType({"args": "sentinelops.sim.site:SiteQueue"})
    )
    with pytest.raises(ValueError, match="without arguments"):
        make_agent("args")


@pytest.mark.skipif(not RULES_INSTALLED, reason="rule-based baseline not added yet (Y1.4)")
def test_rules_agent_is_available_by_name() -> None:
    assert make_agent("rules").name == "rules"


# ---------------------------------------------------------------------- episodes


def test_simulate_runs_every_tick_and_resets_the_agent_once() -> None:
    agent = ScriptedAgent({})
    env = simulate(CRASH, agent, TOPO)
    assert env.done
    assert env.seed == CRASH.seed
    assert agent.resets == [TOPO]
    # The agent sees every observation except the last one (nothing follows it).
    assert agent.ticks == list(range(CRASH.horizon - 1))


def test_noop_result_scores_the_unattended_incident() -> None:
    result = run_episode(CRASH, NoOpAgent(), TOPO)
    assert (result.family, result.seed, result.agent, result.horizon) == ("crash", 5, "noop", 700)
    assert result.legit_failure_share > 0.0
    assert result.legit_failure_share == pytest.approx(result.legit_failed / result.legit_offered)
    assert tuple(result.failed_by_cause) == FAILURE_CAUSES
    assert sum(result.failed_by_cause.values()) == pytest.approx(result.legit_failed)
    assert (result.actions_issued, result.actions_succeeded, result.actions_failed) == (0, 0, 0)


def test_result_matches_the_simulator_ground_truth() -> None:
    env = simulate(CRASH, NoOpAgent(), TOPO)
    result = summarize(CRASH, "noop", env)
    assert result.legit_failure_share == env.truth.legit_failure_share()
    assert result.legit_offered == pytest.approx(float(env.truth.legit_offered.sum()))


def test_a_timely_restart_reduces_failures_and_is_counted() -> None:
    unattended = run_episode(CRASH, NoOpAgent(), TOPO)
    repaired = run_episode(CRASH, restart_lon1_at(320), TOPO)
    assert repaired.legit_failure_share < unattended.legit_failure_share / 2
    assert (repaired.actions_issued, repaired.actions_succeeded, repaired.actions_failed) == (
        1,
        1,
        0,
    )


def test_conflicting_actions_are_counted_as_failed() -> None:
    restart = Action(ActionKind.RESTART, "lon-1")
    agent = ScriptedAgent({320: (restart,), 321: (restart,)})  # second hits a busy server
    result = run_episode(CRASH, agent, TOPO)
    assert (result.actions_issued, result.actions_succeeded, result.actions_failed) == (2, 1, 1)


def test_summarize_rejects_a_scenario_that_does_not_match_the_run() -> None:
    env = simulate(CRASH, NoOpAgent(), TOPO)
    other = Scenario(family="crash", seed=6, horizon=CRASH.horizon, faults=CRASH.faults)
    with pytest.raises(ValueError, match="not run from this scenario"):
        summarize(other, "noop", env)


def test_results_are_hashable_value_objects() -> None:
    result = run_episode(CRASH, NoOpAgent(), TOPO)
    assert len({result, result}) == 1


def test_summarize_rejects_an_unfinished_run() -> None:
    env = DnsEnv(TOPO, CRASH.faults, CRASH.horizon, CRASH.seed)
    env.reset()
    with pytest.raises(ValueError, match="not finished"):
        summarize(CRASH, "noop", env)


# ------------------------------------------------------------------- determinism


def test_same_scenario_and_agent_give_identical_runs() -> None:
    # Restart durations are random, so this also covers the action stream.
    first = run_episode(CRASH, restart_lon1_at(320), TOPO)
    second = run_episode(CRASH, restart_lon1_at(320), TOPO)
    assert first == second


def test_different_seeds_give_different_runs() -> None:
    other = Scenario(family="crash", seed=6, horizon=CRASH.horizon, faults=CRASH.faults)
    assert run_episode(CRASH, NoOpAgent(), TOPO).event_digest != (
        run_episode(other, NoOpAgent(), TOPO).event_digest
    )


def test_generated_scenarios_run_end_to_end() -> None:
    scenario = generate("mixed", 2, TOPO, horizon=1601)
    result = run_episode(scenario, NoOpAgent(), TOPO)
    assert result.horizon == 1601
    assert 0.0 < result.legit_failure_share < 1.0
