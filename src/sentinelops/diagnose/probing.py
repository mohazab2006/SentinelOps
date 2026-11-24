r"""Choosing the next probe: the most information per second of waiting.

A diagnosis is a belief over hypotheses (:mod:`sentinelops.diagnose.inference`).
Probes sharpen it, but every probe takes time (``PROBE_TICKS``), and while the
agent waits the site keeps failing queries. So the agent should run the probe
that is expected to remove the most uncertainty *per second*.

For a probe :math:`X` and the current belief :math:`P(h)` the network predicts
each outcome's probability, :math:`P(x) = \sum_h P(h) P(x \mid h)`, and the
belief each outcome would leave behind. The expected information gain is the
drop in entropy, averaged over the outcomes::

    IG(X) = H(h) - sum_x P(x) H(h | X = x)        (bits; equals I(h; X))

Dividing by the probe's duration gives bits per second, and the best probe is
picked greedily. Greedy selection by information gain is the standard approach
to active diagnosis (Zheng, Rish and Beygelzimer, UAI 2005) and needs only
one cheap enumeration per candidate: a site has 11 hypotheses and 7 probes.

Probing stops when the agent is confident enough to act, when it has used its
probe budget, or when no probe is expected to tell it anything more. In the
last two cases without confidence the agent should hand the incident to an
engineer rather than act on a guess.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from sentinelops.diagnose.bn import FAULT_STATES, DiagnosisNetwork
from sentinelops.diagnose.inference import Belief, target_state
from sentinelops.sim.probes import PROBE_TICKS, ProbeKind, ProbeRequest
from sentinelops.sim.topology import SiteId, Topology

FloatArray = npt.NDArray[np.float64]

SITE_PROBES: tuple[ProbeKind, ...] = (
    ProbeKind.REMOTE_DNS,
    ProbeKind.NEIGHBOUR_DNS,
    ProbeKind.TRAFFIC_SAMPLE,
)
"""Probe kinds whose target is a site."""

SERVER_PROBES: tuple[ProbeKind, ...] = (ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK)
"""Probe kinds whose target is a server."""

DEFAULT_CONFIDENCE = 0.9
"""Probability of the best hypothesis at which the agent stops probing and acts."""

DEFAULT_MAX_PROBES = 6
"""Probes the agent may spend on one incident before it escalates."""

DEFAULT_MIN_GAIN_BITS = 0.01
"""Below this expected gain a probe is not worth its time."""

_TIE_DIGITS = 12


class StopReason(StrEnum):
    """Why probing stopped."""

    CONFIDENT = "confident"
    """The best hypothesis reached the confidence threshold: act on it."""

    BUDGET = "budget"
    """The probe budget is spent without enough confidence: escalate."""

    NO_GAIN = "no_gain"
    """No remaining probe is expected to help: escalate."""


@dataclass(frozen=True, slots=True)
class ProbeScore:
    """How useful one candidate probe is expected to be.

    Attributes:
        request: The probe.
        gain_bits: Expected entropy reduction of the belief, in bits.
        seconds: How long the probe takes (``PROBE_TICKS``; one tick is one second).
    """

    request: ProbeRequest
    gain_bits: float
    seconds: int

    @property
    def bits_per_second(self) -> float:
        """Expected gain per second of waiting: what the selection maximises."""
        return self.gain_bits / self.seconds


@dataclass(frozen=True, slots=True)
class ProbeChoice:
    """The outcome of one selection step.

    Exactly one of ``probe`` and ``stop`` is set.

    Attributes:
        probe: The probe to run next, or ``None`` when probing stops.
        stop: Why probing stopped, or ``None`` while it continues.
        scores: Every candidate's score, best first (empty when stopped for
            confidence or budget, which need no scoring).
    """

    probe: ProbeRequest | None
    stop: StopReason | None
    scores: tuple[ProbeScore, ...] = ()


def candidate_probes(site: str, topology: Topology) -> tuple[ProbeRequest, ...]:
    """Every probe that gives evidence about ``site``: site probes, then each server's.

    Raises:
        KeyError: If ``site`` is not a site of ``topology``.
    """
    servers = sorted(topology.sites[SiteId(site)].servers)
    site_probes = [ProbeRequest(kind, site) for kind in SITE_PROBES]
    server_probes = [ProbeRequest(kind, server) for server in servers for kind in SERVER_PROBES]
    return (*site_probes, *server_probes)


def _entropy(probs: FloatArray) -> FloatArray:
    """Entropy in bits of each row of ``probs`` (rows sum to 1)."""
    safe = np.where(probs > 0, probs, 1.0)
    result: FloatArray = -(probs * np.log2(safe)).sum(axis=-1)
    return result


def outcome_likelihoods(
    network: DiagnosisNetwork, belief: Belief, request: ProbeRequest, topology: Topology
) -> FloatArray:
    """``P(outcome | h)`` for every hypothesis of ``belief`` (rows) and outcome (columns).

    Columns follow ``OUTCOMES[request.kind]``.

    Raises:
        ValueError: If ``request`` is not a probe of ``belief.site`` or one of
            its servers (see :func:`candidate_probes`).
    """
    if request not in candidate_probes(belief.site, topology):
        raise ValueError(
            f"{request.kind} probe on {request.target!r} says nothing about {belief.site!r}"
        )
    table = network.probes[request.kind]
    states = [
        FAULT_STATES.index(target_state(h, request.target, topology)) for h in belief.hypotheses
    ]
    likelihoods: FloatArray = table[states]
    return likelihoods


def expected_information_gain(
    network: DiagnosisNetwork, belief: Belief, request: ProbeRequest, topology: Topology
) -> float:
    """Expected entropy reduction of ``belief`` from running ``request``, in bits.

    Never negative: on average evidence cannot make the belief less certain.

    Raises:
        ValueError: If ``request`` is not a probe of ``belief.site`` or its servers.
    """
    likelihoods = outcome_likelihoods(network, belief, request, topology)
    joint = belief.probs[:, None] * likelihoods  # P(h, x)
    p_outcome = joint.sum(axis=0)  # P(x)
    seen = p_outcome > 0
    after = (joint[:, seen] / p_outcome[seen]).T  # P(h | x), one row per outcome
    expected_after = float((p_outcome[seen] * _entropy(after)).sum())
    return max(0.0, belief.entropy() - expected_after)


def score_probes(
    network: DiagnosisNetwork,
    belief: Belief,
    topology: Topology,
    *,
    exclude: Iterable[ProbeRequest] = (),
) -> tuple[ProbeScore, ...]:
    """Score every candidate probe for ``belief.site``, best first.

    Order: highest bits per second first; ties go to the quicker probe, then to
    the order of :func:`candidate_probes`, so the choice is deterministic.

    Args:
        network: The fitted network.
        belief: The current belief about one site.
        topology: The network's topology.
        exclude: Probes not to consider, e.g. ones already running.
    """
    skip = set(exclude)
    candidates = [r for r in candidate_probes(belief.site, topology) if r not in skip]
    scores = [
        ProbeScore(r, expected_information_gain(network, belief, r, topology), PROBE_TICKS[r.kind])
        for r in candidates
    ]
    # Rounding makes gains that differ only by floating-point noise count as ties.
    order = sorted(
        range(len(scores)),
        key=lambda i: (-round(scores[i].bits_per_second, _TIE_DIGITS), scores[i].seconds, i),
    )
    return tuple(scores[i] for i in order)


def next_probe(
    network: DiagnosisNetwork,
    belief: Belief,
    topology: Topology,
    *,
    probes_used: int = 0,
    running: Iterable[ProbeRequest] = (),
    confidence: float = DEFAULT_CONFIDENCE,
    max_probes: int = DEFAULT_MAX_PROBES,
    min_gain_bits: float = DEFAULT_MIN_GAIN_BITS,
) -> ProbeChoice:
    """Decide whether to probe again and, if so, which probe to run.

    Checks, in order: confident enough (stop, act); budget spent (stop,
    escalate); otherwise the best probe by bits per second, unless even the best
    is expected to gain less than ``min_gain_bits`` (stop, escalate).

    Args:
        network: The fitted network.
        belief: The current belief about one site.
        topology: The network's topology.
        probes_used: Probes already spent on this incident (finished or running).
        running: Probes still running; they are not chosen again.
        confidence: Best-hypothesis probability at which to stop, in ``(0, 1]``.
        max_probes: The probe budget for one incident, ``>= 0``.
        min_gain_bits: Smallest expected gain worth waiting for, ``>= 0``.

    Raises:
        ValueError: If a threshold is out of range or ``probes_used`` is negative.
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1], got {confidence}")
    if max_probes < 0 or probes_used < 0:
        raise ValueError("max_probes and probes_used must be non-negative")
    if min_gain_bits < 0:
        raise ValueError(f"min_gain_bits must be non-negative, got {min_gain_bits}")
    if belief.best()[1] >= confidence:
        return ProbeChoice(None, StopReason.CONFIDENT)
    if probes_used >= max_probes:
        return ProbeChoice(None, StopReason.BUDGET)
    scores = score_probes(network, belief, topology, exclude=running)
    if not scores or scores[0].gain_bits < min_gain_bits:
        return ProbeChoice(None, StopReason.NO_GAIN, scores)
    return ProbeChoice(scores[0].request, None, scores)
