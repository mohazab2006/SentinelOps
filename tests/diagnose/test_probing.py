"""Tests for probe selection by expected information gain per second."""

from __future__ import annotations

from functools import cache
from types import MappingProxyType

import numpy as np
import pytest

from sentinelops.diagnose import FAULT_STATES, LEVELS, SYMPTOMS, DiagnosisNetwork, fit_network
from sentinelops.diagnose.inference import Belief, Hypothesis, hypotheses, posterior
from sentinelops.diagnose.probing import (
    DEFAULT_CONFIDENCE,
    SERVER_PROBES,
    SITE_PROBES,
    ProbeScore,
    StopReason,
    candidate_probes,
    expected_information_gain,
    next_probe,
    outcome_likelihoods,
    score_probes,
)
from sentinelops.eval.dataset import HEALTHY, build_dataset
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.sim.env import DnsEnv
from sentinelops.sim.probes import OUTCOMES, PROBE_TICKS, ProbeKind, ProbeRequest, ProbeResult
from sentinelops.sim.topology import SiteId, default_topology

TOPO = default_topology()
SITE = sorted(TOPO.sites)[0]
A, B = sorted(TOPO.sites[SITE].servers)
N = len(FAULT_STATES)


def _network(**tables: np.ndarray) -> DiagnosisNetwork:
    """Uniform tables (no probe tells anything) except the ones given by kind name."""
    probes = {k: np.full((N, len(o)), 1 / len(o)) for k, o in OUTCOMES.items()}
    probes.update({ProbeKind(k): v for k, v in tables.items()})
    return DiagnosisNetwork(
        prior=np.full(N, 1 / N),
        bounds=np.zeros((len(SYMPTOMS), 2)),
        symptoms=np.full((len(SYMPTOMS), N, len(LEVELS)), 1 / len(LEVELS)),
        probes=MappingProxyType(probes),
    )


def _perfect_hw_check() -> np.ndarray:
    """hw_check reports hardware_fault exactly when the probed server has a hardware fault."""
    table = np.zeros((N, len(OUTCOMES[ProbeKind.HW_CHECK])))
    table[:, 0] = 1.0  # healthy
    table[FAULT_STATES.index("hardware")] = [0.0, 1.0, 0.0]
    return table


def _belief(weights: dict[str, float]) -> Belief:
    """A belief about SITE with the given weights on hypotheses (by str), zero elsewhere."""
    hs = hypotheses(SITE, TOPO)
    probs = np.array([weights.get(str(h), 0.0) for h in hs])
    return Belief(SiteId(SITE), hs, probs / probs.sum())


def _uniform_belief() -> Belief:
    return _belief({str(h): 1.0 for h in hypotheses(SITE, TOPO)})


# ------------------------------------------------------------------ candidates


def test_candidates_are_site_probes_then_each_servers_probes() -> None:
    assert candidate_probes(SITE, TOPO) == (
        *(ProbeRequest(k, SITE) for k in SITE_PROBES),
        *(ProbeRequest(k, s) for s in (A, B) for k in SERVER_PROBES),
    )
    assert set(SITE_PROBES) | set(SERVER_PROBES) == set(ProbeKind)


def test_candidates_reject_unknown_site() -> None:
    with pytest.raises(KeyError):
        candidate_probes("nowhere", TOPO)


# ------------------------------------------------------------------ information gain


def test_outcome_likelihoods_use_the_probed_targets_state() -> None:
    net = _network(hw_check=_perfect_hw_check())
    rows = outcome_likelihoods(net, _uniform_belief(), ProbeRequest(ProbeKind.HW_CHECK, A), TOPO)
    hs = hypotheses(SITE, TOPO)
    assert rows.shape == (len(hs), len(OUTCOMES[ProbeKind.HW_CHECK]))
    faulty = [i for i, h in enumerate(hs) if rows[i, 1] == 1.0]
    assert [hs[i] for i in faulty] == [Hypothesis("hardware", A)]  # not hardware on B


@pytest.mark.parametrize(
    "request_",
    [
        ProbeRequest(ProbeKind.REMOTE_DNS, sorted(TOPO.sites)[1]),  # another site
        ProbeRequest(ProbeKind.LOCAL_DNS, sorted(TOPO.servers)[-1]),  # another site's server
        ProbeRequest(ProbeKind.LOCAL_DNS, SITE),  # server probe aimed at a site
    ],
)
def test_probe_about_another_target_is_rejected(request_: ProbeRequest) -> None:
    with pytest.raises(ValueError, match="says nothing about"):
        expected_information_gain(_network(), _uniform_belief(), request_, TOPO)


def test_uninformative_probe_gains_nothing() -> None:
    net = _network()
    for request in candidate_probes(SITE, TOPO):
        assert expected_information_gain(net, _uniform_belief(), request, TOPO) == pytest.approx(0)


def test_certain_belief_gains_nothing() -> None:
    net = _network(hw_check=_perfect_hw_check())
    belief = _belief({"crash on " + A: 1.0})
    assert expected_information_gain(net, belief, ProbeRequest(ProbeKind.HW_CHECK, A), TOPO) == 0


def test_perfect_probe_on_a_coin_flip_gains_one_bit() -> None:
    net = _network(hw_check=_perfect_hw_check())
    belief = _belief({f"hardware on {A}": 1.0, f"crash on {A}": 1.0})
    gain = expected_information_gain(net, belief, ProbeRequest(ProbeKind.HW_CHECK, A), TOPO)
    assert gain == pytest.approx(1.0)
    # The same probe on the other server cannot tell these two apart.
    other = expected_information_gain(net, belief, ProbeRequest(ProbeKind.HW_CHECK, B), TOPO)
    assert other == pytest.approx(0.0)


def test_gain_equals_mutual_information_on_random_networks() -> None:
    """IG = H(h) - E[H(h|x)] must equal I(h; X) = H(X) - E_h[H(X|h)]."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        tables = {str(k): rng.dirichlet(np.ones(len(o)), size=N) for k, o in OUTCOMES.items()}
        net = _network(**tables)
        belief = _belief(
            {str(h): w for h, w in zip(hypotheses(SITE, TOPO), rng.random(11), strict=True)}
        )
        for request in candidate_probes(SITE, TOPO):
            gain = expected_information_gain(net, belief, request, TOPO)
            rows = outcome_likelihoods(net, belief, request, TOPO)
            p_x = belief.probs @ rows
            h_x = -(p_x * np.log2(p_x)).sum()
            h_x_given_h = -(rows * np.log2(rows)).sum(axis=1) @ belief.probs
            assert gain == pytest.approx(h_x - h_x_given_h, abs=1e-9)
            assert 0.0 <= gain <= belief.entropy() + 1e-12


def test_gain_matches_the_posterior_update() -> None:
    """Averaging the entropy of real posteriors over outcomes gives the same gain."""
    net = _fitted()
    prior = posterior(net, SITE, TOPO, exclude_healthy=True)
    request = ProbeRequest(ProbeKind.LOCAL_DNS, A)
    p_x = prior.probs @ outcome_likelihoods(net, prior, request, TOPO)
    after = 0.0
    for outcome, p in zip(OUTCOMES[request.kind], p_x.tolist(), strict=True):
        result = ProbeResult(request, issued=0, completed=2, outcome=outcome)
        after += p * posterior(net, SITE, TOPO, probes=[result], exclude_healthy=True).entropy()
    gain = expected_information_gain(net, prior, request, TOPO)
    assert gain == pytest.approx(prior.entropy() - after)


# ------------------------------------------------------------------ scoring


def test_quicker_probe_wins_when_equally_informative() -> None:
    perfect = _perfect_hw_check()
    local = np.zeros((N, len(OUTCOMES[ProbeKind.LOCAL_DNS])))
    local[:, 0] = 1.0
    local[FAULT_STATES.index("hardware")] = [0.0, 0.0, 0.0, 1.0]  # timeout iff hardware
    net = _network(hw_check=perfect, local_dns=local)
    belief = _belief({f"hardware on {A}": 1.0, f"crash on {B}": 1.0})
    scores = score_probes(net, belief, TOPO)
    best = scores[0]
    assert best.request == ProbeRequest(ProbeKind.LOCAL_DNS, A)  # 2 s beats 15 s for the same bit
    assert best.gain_bits == pytest.approx(1.0)
    assert best.bits_per_second == pytest.approx(1.0 / PROBE_TICKS[ProbeKind.LOCAL_DNS])
    hw = next(s for s in scores if s.request == ProbeRequest(ProbeKind.HW_CHECK, A))
    assert hw.gain_bits == pytest.approx(1.0)
    assert scores.index(hw) > 0


def test_scores_are_sorted_and_cover_every_candidate_once() -> None:
    net = _fitted()
    belief = posterior(net, SITE, TOPO, exclude_healthy=True)
    scores = score_probes(net, belief, TOPO)
    assert len(scores) == len(candidate_probes(SITE, TOPO))
    assert {s.request for s in scores} == set(candidate_probes(SITE, TOPO))
    rates = [s.bits_per_second for s in scores]
    assert rates == sorted(rates, reverse=True)
    assert all(s.seconds == PROBE_TICKS[s.request.kind] for s in scores)


def test_ties_are_broken_deterministically() -> None:
    scores = score_probes(_network(), _uniform_belief(), TOPO)
    # Every gain is 0: quicker first, then candidate order.
    assert [s.request for s in scores] == sorted(
        candidate_probes(SITE, TOPO),
        key=lambda r: (PROBE_TICKS[r.kind], candidate_probes(SITE, TOPO).index(r)),
    )


def test_excluded_probes_are_not_scored() -> None:
    running = (ProbeRequest(ProbeKind.LOCAL_DNS, A), ProbeRequest(ProbeKind.REMOTE_DNS, SITE))
    scores = score_probes(_network(), _uniform_belief(), TOPO, exclude=running)
    assert len(scores) == len(candidate_probes(SITE, TOPO)) - 2
    assert not set(running) & {s.request for s in scores}


def test_probe_score_rate() -> None:
    assert ProbeScore(ProbeRequest(ProbeKind.HW_CHECK, A), 0.6, 15).bits_per_second == 0.04


# ------------------------------------------------------------------ next_probe


def test_confident_belief_stops_probing() -> None:
    choice = next_probe(_network(), _belief({f"crash on {A}": 0.95, "flood": 0.05}), TOPO)
    assert (choice.probe, choice.stop, choice.scores) == (None, StopReason.CONFIDENT, ())


def test_spent_budget_stops_probing() -> None:
    net = _network(hw_check=_perfect_hw_check())
    choice = next_probe(net, _uniform_belief(), TOPO, probes_used=6, max_probes=6)
    assert (choice.probe, choice.stop) == (None, StopReason.BUDGET)


def test_no_useful_probe_stops_probing() -> None:
    choice = next_probe(_network(), _uniform_belief(), TOPO)
    assert choice.probe is None
    assert choice.stop == StopReason.NO_GAIN
    assert len(choice.scores) == len(candidate_probes(SITE, TOPO))


def test_everything_running_stops_probing() -> None:
    choice = next_probe(
        _network(hw_check=_perfect_hw_check()),
        _uniform_belief(),
        TOPO,
        running=candidate_probes(SITE, TOPO),
    )
    assert (choice.probe, choice.stop, choice.scores) == (None, StopReason.NO_GAIN, ())


def test_best_probe_is_chosen_and_running_ones_skipped() -> None:
    net = _network(hw_check=_perfect_hw_check())
    belief = _belief({f"hardware on {A}": 1.0, f"hardware on {B}": 1.0})
    first = next_probe(net, belief, TOPO)
    assert first.stop is None
    assert first.probe == ProbeRequest(ProbeKind.HW_CHECK, A)
    second = next_probe(net, belief, TOPO, probes_used=1, running=[first.probe])
    assert second.probe == ProbeRequest(ProbeKind.HW_CHECK, B)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0.0},
        {"confidence": 1.5},
        {"max_probes": -1},
        {"probes_used": -1},
        {"min_gain_bits": -0.1},
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        next_probe(_network(), _uniform_belief(), TOPO, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ real runs


@cache
def _fitted() -> DiagnosisNetwork:
    """A network fitted on a few simulated training runs (built once)."""
    families = ("crash", "hardware", "flood", "benign_surge")
    scenarios = [s for s in split_scenarios(Split.TRAIN, TOPO, families) if s.seed < 2]
    return fit_network(build_dataset(scenarios, TOPO))


def _diagnose(family: str, seed: int, *, greedy: bool) -> tuple[Hypothesis, float, int]:
    """Probe a faulty site from 30 ticks after onset.

    Returns:
        The best hypothesis, its probability and the seconds spent probing.
    """
    net = _fitted()
    scenario = next(s for s in split_scenarios(Split.TUNE, TOPO, (family,)) if s.seed == seed)
    fault = scenario.faults[0]
    site = TOPO.servers[fault.target].site if fault.target in TOPO.servers else fault.target
    env = DnsEnv(TOPO, scenario.faults, scenario.horizon, scenario.seed)
    obs = env.reset()
    while obs.tick < fault.onset + 30:
        obs = env.step(())
    start, results, fixed = obs.tick, [], candidate_probes(site, TOPO)
    while True:
        belief = posterior(net, site, TOPO, probes=results, exclude_healthy=True)
        if greedy:
            request = next_probe(net, belief, TOPO, probes_used=len(results)).probe
        else:
            done = belief.best()[1] >= DEFAULT_CONFIDENCE or len(results) == len(fixed)
            request = None if done else fixed[len(results)]
        if request is None:
            break
        obs = env.step([request])
        while not obs.probe_results:
            obs = env.step(())
        results.extend(obs.probe_results)
    best, p = belief.best()
    return best, p, obs.tick - start


@pytest.mark.parametrize(("family", "seed"), [("crash", 100), ("crash", 101), ("hardware", 100)])
def test_greedy_probing_finds_the_faulty_server_faster_than_a_fixed_order(
    family: str, seed: int
) -> None:
    scenario = next(s for s in split_scenarios(Split.TUNE, TOPO, (family,)) if s.seed == seed)
    truth = Hypothesis(family, scenario.faults[0].target)
    greedy, p, greedy_secs = _diagnose(family, seed, greedy=True)
    assert greedy == truth
    assert p >= DEFAULT_CONFIDENCE
    fixed, _, fixed_secs = _diagnose(family, seed, greedy=False)
    assert greedy_secs < fixed_secs
    assert fixed.kind != HEALTHY
