"""Exact diagnosis: the probability of every explanation of what is wrong with a site.

The network (:mod:`sentinelops.diagnose.bn`) knows how each fault *kind* shows
up. A diagnosis also has to say *where*: "crash on lon-1" and "crash on lon-2"
are different repairs. So inference reasons over **hypotheses** about one site:

* ``healthy`` - nothing is wrong;
* a server fault on one of the site's servers (crash, exhaustion, hardware);
* a site fault or a region fault reaching the site (path loss, site outage,
  flood, benign surge).

Each hypothesis fixes the state of every target at the site: under "crash on
lon-1" server lon-1 is crashed, its sibling lon-2 is healthy, and the site
itself is affected by the crash. Each piece of evidence is then scored with the
network's table for that target's state, and Bayes' rule gives::

    P(h | evidence) ∝ P(h) · Π P(symptom level | kind of h)
                           · Π P(probe outcome | state of the probed target under h)

A site has only a handful of hypotheses (11 for two servers), so the posterior
is computed exactly by enumerating them all; no approximation is needed.
Products are summed in log space so that many small probabilities do not
underflow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sentinelops.diagnose.bn import FAULT_STATES, LEVELS, SYMPTOMS, DiagnosisNetwork
from sentinelops.eval.dataset import HEALTHY
from sentinelops.sim.faults import FaultKind
from sentinelops.sim.probes import OUTCOMES, ProbeKind, ProbeResult
from sentinelops.sim.topology import ServerId, SiteId, Topology

FloatArray = npt.NDArray[np.float64]

SERVER_KINDS: tuple[str, ...] = (FaultKind.CRASH, FaultKind.EXHAUSTION, FaultKind.HARDWARE)
"""Fault kinds that hit one server, so a hypothesis must also name the server."""

_SERVER_PROBES = frozenset({ProbeKind.LOCAL_DNS, ProbeKind.HW_CHECK})


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One explanation of a site's state.

    Attributes:
        kind: A :data:`~sentinelops.diagnose.bn.FAULT_STATES` value.
        server: The faulty server for a server fault; ``None`` otherwise.
    """

    kind: str
    server: ServerId | None = None

    def __str__(self) -> str:
        """``"crash on lon-1"``, ``"flood"`` or ``"healthy"``."""
        return self.kind if self.server is None else f"{self.kind} on {self.server}"


def hypotheses(site: str, topology: Topology) -> tuple[Hypothesis, ...]:
    """Every hypothesis about ``site``, in :data:`FAULT_STATES` order, servers sorted.

    Raises:
        KeyError: If ``site`` is not a site of ``topology``.
    """
    servers = sorted(topology.sites[SiteId(site)].servers)
    result: list[Hypothesis] = []
    for kind in FAULT_STATES:
        if kind in SERVER_KINDS:
            result.extend(Hypothesis(kind, server) for server in servers)
        else:
            result.append(Hypothesis(kind))
    return tuple(result)


def target_state(hypothesis: Hypothesis, target: str, topology: Topology) -> str:
    """The state of ``target`` (a site or a server) under ``hypothesis``.

    A server fault leaves the other servers healthy but affects its site; every
    other kind affects the site and all its servers. This matches how the
    training rows were labelled (:func:`sentinelops.eval.dataset.affects`).
    """
    if hypothesis.server is not None and target in topology.servers:
        return hypothesis.kind if target == hypothesis.server else HEALTHY
    return hypothesis.kind


@dataclass(frozen=True, slots=True)
class Belief:
    """A posterior over the hypotheses of one site.

    Attributes:
        site: The site diagnosed.
        hypotheses: The hypotheses, in :func:`hypotheses` order.
        probs: Their probabilities, read-only, summing to 1.
    """

    site: SiteId
    hypotheses: tuple[Hypothesis, ...]
    probs: FloatArray

    def best(self) -> tuple[Hypothesis, float]:
        """The most probable hypothesis and its probability (first one on a tie)."""
        i = int(np.argmax(self.probs))
        return self.hypotheses[i], float(self.probs[i])

    def kinds(self) -> Mapping[str, float]:
        """Probability of each fault kind, summed over servers, in ``FAULT_STATES`` order."""
        totals = dict.fromkeys(FAULT_STATES, 0.0)
        for hypothesis, p in zip(self.hypotheses, self.probs.tolist(), strict=True):
            totals[hypothesis.kind] += p
        return totals

    def entropy(self) -> float:
        """Shannon entropy of the posterior in bits: 0 when certain."""
        p = self.probs[self.probs > 0]
        return float(-(p * np.log2(p)).sum())


def posterior(
    network: DiagnosisNetwork,
    site: str,
    topology: Topology,
    *,
    levels: npt.ArrayLike | None = None,
    probes: Iterable[ProbeResult] = (),
    exclude_healthy: bool = False,
) -> Belief:
    """Compute the exact posterior over every hypothesis about ``site``.

    Args:
        network: The fitted network.
        site: The site to diagnose.
        topology: The network's topology (which servers the site has).
        levels: Symptom levels of the site's latest window, shape
            ``(len(SYMPTOMS),)`` with values 0-2 (see
            :meth:`DiagnosisNetwork.levels`); ``None`` to ignore telemetry.
        probes: Finished probes on the site or its servers.
        exclude_healthy: Drop the ``healthy`` hypothesis, i.e. diagnose
            *given* that something is wrong (after an alert).

    Returns:
        The posterior. With no evidence it is the prior.

    Raises:
        KeyError: If ``site`` is not a site.
        ValueError: If ``levels`` has the wrong shape or values, or a probe's
            target is not this site (site probes) or one of its servers
            (server probes).
    """
    candidates = hypotheses(site, topology)
    if exclude_healthy:
        candidates = tuple(h for h in candidates if h.kind != HEALTHY)
    kinds = np.array([FAULT_STATES.index(h.kind) for h in candidates])
    servers = topology.sites[SiteId(site)].servers
    # A server fault's prior is shared evenly between the site's servers.
    shares = np.array([len(servers) if h.server is not None else 1 for h in candidates])
    log_p = np.log(network.prior[kinds]) - np.log(shares)

    if levels is not None:
        observed = np.asarray(levels)
        if observed.shape != (len(SYMPTOMS),) or not np.isin(observed, range(len(LEVELS))).all():
            raise ValueError(
                f"levels must be {len(SYMPTOMS)} values in 0-{len(LEVELS) - 1}, got {observed}"
            )
        for i, level in enumerate(observed.tolist()):
            log_p += np.log(network.symptoms[i, kinds, level])

    for result in probes:
        kind = result.request.kind
        target = result.request.target
        is_server = kind in _SERVER_PROBES
        if (is_server and target not in servers) or (not is_server and target != site):
            raise ValueError(f"{kind} probe on {target!r} is not evidence about site {site!r}")
        table = network.probes[kind]
        column = OUTCOMES[kind].index(result.outcome)
        states = [FAULT_STATES.index(target_state(h, target, topology)) for h in candidates]
        log_p += np.log(table[states, column])

    probs = np.exp(log_p - log_p.max())
    probs /= probs.sum()
    probs.flags.writeable = False
    return Belief(SiteId(site), candidates, probs)
