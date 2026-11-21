"""Tests for exact diagnosis over site hypotheses."""

from __future__ import annotations

from functools import cache
from types import MappingProxyType

import numpy as np
import pytest

from sentinelops.diagnose import FAULT_STATES, LEVELS, SYMPTOMS, DiagnosisNetwork, fit_network
from sentinelops.diagnose.inference import (
    SERVER_KINDS,
    Belief,
    Hypothesis,
    hypotheses,
    posterior,
    target_state,
)
from sentinelops.eval.dataset import HEALTHY, build_dataset
from sentinelops.eval.splits import Split, split_scenarios
from sentinelops.sim.probes import OUTCOMES, ProbeKind, ProbeOutcome, ProbeRequest, ProbeResult
from sentinelops.sim.topology import ServerId, default_topology

TOPO = default_topology()
SITE = sorted(TOPO.sites)[0]
A, B = sorted(TOPO.sites[SITE].servers)
OTHER_SITE = sorted(TOPO.sites)[1]
N = len(FAULT_STATES)


def _uniform_network(prior: np.ndarray | None = None) -> DiagnosisNetwork:
    """A network whose every table is uniform (evidence changes nothing) unless edited."""
    return DiagnosisNetwork(
        prior=np.full(N, 1 / N) if prior is None else prior,
        bounds=np.zeros((len(SYMPTOMS), 2)),
        symptoms=np.full((len(SYMPTOMS), N, len(LEVELS)), 1 / len(LEVELS)),
        probes=MappingProxyType({k: np.full((N, len(o)), 1 / len(o)) for k, o in OUTCOMES.items()}),
    )


def _result(kind: ProbeKind, target: str, outcome: ProbeOutcome) -> ProbeResult:
    return ProbeResult(ProbeRequest(kind, target), issued=0, completed=1, outcome=outcome)


def _probs(belief: Belief) -> dict[str, float]:
    return {str(h): p for h, p in zip(belief.hypotheses, belief.probs.tolist(), strict=True)}


# --------------------------------------------------------------- hypotheses


def test_hypotheses_name_a_server_for_server_faults_only() -> None:
    hs = hypotheses(SITE, TOPO)
    assert len(hs) == 1 + 2 * len(SERVER_KINDS) + 4  # healthy + 3 kinds x 2 servers + 4
    assert hs[0] == Hypothesis(HEALTHY)
    assert [h.kind for h in hs] == sorted((h.kind for h in hs), key=FAULT_STATES.index)
    for h in hs:
        assert (h.server is not None) == (h.kind in SERVER_KINDS)
    assert [h.server for h in hs if h.kind == "crash"] == [A, B]


def test_hypotheses_of_an_unknown_site() -> None:
    with pytest.raises(KeyError):
        hypotheses("atlantis", TOPO)


def test_hypothesis_names() -> None:
    assert str(Hypothesis("crash", ServerId("lon-1"))) == "crash on lon-1"
    assert str(Hypothesis("flood")) == "flood"


@pytest.mark.parametrize(
    ("hypothesis", "target", "state"),
    [
        (Hypothesis("crash", A), A, "crash"),  # the crashed server
        (Hypothesis("crash", A), B, HEALTHY),  # its sibling is fine
        (Hypothesis("crash", A), SITE, "crash"),  # the site is affected
        (Hypothesis("site_outage"), B, "site_outage"),  # site faults reach every server
        (Hypothesis("flood"), A, "flood"),
        (Hypothesis(HEALTHY), SITE, HEALTHY),
    ],
)
def test_target_state(hypothesis: Hypothesis, target: str, state: str) -> None:
    assert target_state(hypothesis, target, TOPO) == state


# ----------------------------------------------------------- by-hand posteriors


def test_no_evidence_gives_the_prior_split_over_servers() -> None:
    belief = posterior(_uniform_network(), SITE, TOPO)
    p = _probs(belief)
    assert p["healthy"] == pytest.approx(1 / 8)
    assert p[f"crash on {A}"] == p[f"crash on {B}"] == pytest.approx(1 / 16)
    assert belief.probs.sum() == pytest.approx(1.0)


def test_excluding_healthy_renormalises_the_rest() -> None:
    belief = posterior(_uniform_network(), SITE, TOPO, exclude_healthy=True)
    p = _probs(belief)
    assert "healthy" not in p
    assert p["flood"] == pytest.approx(1 / 7)
    assert p[f"hardware on {A}"] == pytest.approx(1 / 14)


def test_one_probe_by_hand() -> None:
    net = _uniform_network()
    refused = OUTCOMES[ProbeKind.LOCAL_DNS].index(ProbeOutcome.REFUSED)
    table = np.full((N, 4), 0.3)
    table[:, refused] = 0.1  # rows sum to 1
    table[FAULT_STATES.index("crash")] = [0.1 / 3] * 4
    table[FAULT_STATES.index("crash"), refused] = 0.9
    net = DiagnosisNetwork(
        net.prior,
        net.bounds,
        net.symptoms,
        MappingProxyType({**net.probes, ProbeKind.LOCAL_DNS: table}),
    )
    belief = posterior(
        net, SITE, TOPO, probes=[_result(ProbeKind.LOCAL_DNS, A, ProbeOutcome.REFUSED)]
    )
    # crash on A: 1/16 * 0.9; every other hypothesis: prior * 0.1 (crash on B leaves A healthy).
    evidence = 1 / 16 * 0.9 + (1 - 1 / 16) * 0.1
    p = _probs(belief)
    assert p[f"crash on {A}"] == pytest.approx(1 / 16 * 0.9 / evidence)  # 0.375
    assert p[f"crash on {B}"] == pytest.approx(1 / 16 * 0.1 / evidence)
    assert p["healthy"] == pytest.approx(1 / 8 * 0.1 / evidence)
    best, p_best = belief.best()
    assert best == Hypothesis("crash", A)
    assert p_best == pytest.approx(0.375)


def test_symptoms_by_hand() -> None:
    net = _uniform_network()
    symptoms = np.array(net.symptoms)
    loss = SYMPTOMS.index("loss_mean")
    symptoms[loss, :, :] = [0.1, 0.8, 0.1]
    symptoms[loss, FAULT_STATES.index("path_loss")] = [0.1, 0.1, 0.8]
    net = DiagnosisNetwork(net.prior, net.bounds, symptoms, net.probes)
    levels = np.ones(len(SYMPTOMS), dtype=np.int64)
    levels[loss] = 2  # loss is high
    p = _probs(posterior(net, SITE, TOPO, levels=levels))
    evidence = 1 / 8 * 0.8 + 7 / 8 * 0.1
    assert p["path_loss"] == pytest.approx(1 / 8 * 0.8 / evidence)
    assert p["healthy"] == pytest.approx(1 / 8 * 0.1 / evidence)


def test_evidence_order_does_not_matter() -> None:
    net = _fitted()
    probes = [
        _result(ProbeKind.LOCAL_DNS, A, ProbeOutcome.TIMEOUT),
        _result(ProbeKind.HW_CHECK, A, ProbeOutcome.HARDWARE_FAULT),
        _result(ProbeKind.REMOTE_DNS, SITE, ProbeOutcome.SLOW),
    ]
    forward = posterior(net, SITE, TOPO, probes=probes)
    backward = posterior(net, SITE, TOPO, probes=probes[::-1])
    assert forward.probs == pytest.approx(backward.probs)


def test_much_evidence_does_not_underflow() -> None:
    net = _fitted()
    probes = [_result(ProbeKind.LOCAL_DNS, A, ProbeOutcome.REFUSED)] * 500
    belief = posterior(net, SITE, TOPO, probes=probes)
    assert np.all(np.isfinite(belief.probs))
    assert belief.best()[0] == Hypothesis("crash", A)
    assert belief.best()[1] == pytest.approx(1.0)


# ------------------------------------------------------------------ belief


def test_kinds_sum_servers_and_entropy_measures_uncertainty() -> None:
    belief = posterior(_uniform_network(), SITE, TOPO)
    kinds = belief.kinds()
    assert list(kinds) == list(FAULT_STATES)
    assert kinds["crash"] == pytest.approx(1 / 8)
    assert sum(kinds.values()) == pytest.approx(1.0)
    # healthy + 4 site/region kinds at 1/8 (3 bits each), 6 server hypotheses at 1/16 (4 bits).
    assert belief.entropy() == pytest.approx(5 * (1 / 8) * 3 + 6 * (1 / 16) * 4)
    certain = Belief(belief.site, belief.hypotheses[:2], np.array([1.0, 0.0]))
    assert certain.entropy() == 0.0


def test_probabilities_are_read_only() -> None:
    belief = posterior(_uniform_network(), SITE, TOPO)
    with pytest.raises(ValueError):
        belief.probs[0] = 1.0


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize(
    ("kind", "target"),
    [
        (ProbeKind.LOCAL_DNS, sorted(TOPO.sites[OTHER_SITE].servers)[0]),  # another site's server
        (ProbeKind.REMOTE_DNS, OTHER_SITE),  # another site
        (ProbeKind.HW_CHECK, SITE),  # a server probe aimed at a site
    ],
)
def test_probe_about_something_else_is_rejected(kind: ProbeKind, target: str) -> None:
    outcome = OUTCOMES[kind][0]
    with pytest.raises(ValueError, match="not evidence"):
        posterior(_uniform_network(), SITE, TOPO, probes=[_result(kind, target, outcome)])


@pytest.mark.parametrize("levels", [[1, 1, 1], [1, 1, 1, 1, 1, 3], [[1] * 6]])
def test_bad_levels_are_rejected(levels: list[int]) -> None:
    with pytest.raises(ValueError, match="levels"):
        posterior(_uniform_network(), SITE, TOPO, levels=levels)


# --------------------------------------------------------- on simulated runs


@cache
def _fitted() -> DiagnosisNetwork:
    """A network fitted on a few simulated training runs (built once)."""
    families = ("crash", "hardware", "flood", "benign_surge")
    scenarios = [s for s in split_scenarios(Split.TRAIN, TOPO, families) if s.seed < 2]
    return fit_network(build_dataset(scenarios, TOPO))


def test_probes_tell_crash_from_hardware_and_find_the_server() -> None:
    net = _fitted()
    crash = posterior(
        net,
        SITE,
        TOPO,
        probes=[
            _result(ProbeKind.LOCAL_DNS, A, ProbeOutcome.OK),
            _result(ProbeKind.LOCAL_DNS, B, ProbeOutcome.REFUSED),
        ],
        exclude_healthy=True,
    )
    assert crash.best()[0] == Hypothesis("crash", B)
    hardware = posterior(
        net,
        SITE,
        TOPO,
        probes=[
            _result(ProbeKind.LOCAL_DNS, A, ProbeOutcome.TIMEOUT),
            _result(ProbeKind.HW_CHECK, A, ProbeOutcome.HARDWARE_FAULT),
        ],
        exclude_healthy=True,
    )
    assert hardware.best()[0] == Hypothesis("hardware", A)
    assert hardware.best()[1] > 0.9


def test_traffic_sample_moves_belief_between_flood_and_surge() -> None:
    net = _fitted()

    def flood_vs_surge(outcome: ProbeOutcome) -> float:
        kinds = posterior(
            net, SITE, TOPO, probes=[_result(ProbeKind.TRAFFIC_SAMPLE, SITE, outcome)]
        ).kinds()
        return kinds["flood"] / kinds["benign_surge"]

    assert flood_vs_surge(ProbeOutcome.ATTACK_PATTERN) > 3
    assert flood_vs_surge(ProbeOutcome.BROAD_SURGE) < 1 / 3
    # The ambiguous resolver burst (D-024) leaves the two close: the agent should not be sure.
    assert 0.5 < flood_vs_surge(ProbeOutcome.RESOLVER_BURST) < 3
