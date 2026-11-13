"""What an agent observes each tick: noisy, sometimes missing telemetry.

Agents never see the true state of the network. They see what an operator's
monitoring would show: per-site traffic and reply counts, latency, loss and
queue depth, plus per-server CPU, memory and served queries. Every reading
carries measurement noise and is occasionally missing (``None``), as real
monitoring data is.

Control-plane facts the operator sets themselves (whether a site is announced,
and at what weight) are always known exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sentinelops.sim.actions import ActionUpdate
from sentinelops.sim.topology import ServerId, SiteId

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ServerTelemetry:
    """Readings for one server.

    ``cpu`` and ``mem`` come from the host; ``served_qps`` comes from the site's
    load balancer, so it is still reported when the host itself is unreachable.

    Attributes:
        cpu: CPU utilisation in ``[0, 1]``, or ``None`` if missing.
        mem: Memory utilisation in ``[0, 1]``, or ``None`` if missing.
        served_qps: Queries answered by this server this tick, or ``None``.
    """

    cpu: float | None
    mem: float | None
    served_qps: float | None


@dataclass(frozen=True, slots=True)
class SiteTelemetry:
    """Readings for one site.

    Attributes:
        qps: Queries arriving at the site this tick.
        p95_latency_ms: 95th-percentile response time seen by clients, in ms.
        loss: Fraction of queries lost on the way to the site, as measured by
            external monitoring, in ``[0, 1]``.
        queue_depth: Queries waiting at the end of the tick.
        ok_replies: Queries answered this tick.
        failed_replies: Queries that failed this tick (rejected, overflowed,
            timed out or lost).
        announced: Whether the site is announcing the service address.
        weight: Current announcement weight in ``[0, 1]``.
    """

    qps: float | None
    p95_latency_ms: float | None
    loss: float | None
    queue_depth: float | None
    ok_replies: float | None
    failed_replies: float | None
    announced: bool
    weight: float


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything an agent sees at one tick.

    Attributes:
        tick: The tick this observation describes.
        sites: Telemetry for every site.
        servers: Telemetry for every server.
        action_updates: Status changes of the agent's actions during this tick.
    """

    tick: int
    sites: Mapping[SiteId, SiteTelemetry]
    servers: Mapping[ServerId, ServerTelemetry]
    action_updates: tuple[ActionUpdate, ...]


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """How noisy and incomplete the monitoring is.

    Attributes:
        relative_cv: Coefficient of variation of the multiplicative noise on
            counts, rates, latency and queue depth.
        fraction_sd: Standard deviation of the additive noise on fractions
            (CPU, memory, loss).
        missing_prob: Probability that any single reading is missing.
    """

    relative_cv: float = 0.03
    fraction_sd: float = 0.01
    missing_prob: float = 0.002

    def __post_init__(self) -> None:
        """Reject negative noise or a probability outside ``[0, 1]``."""
        if self.relative_cv < 0 or self.fraction_sd < 0:
            raise ValueError("noise levels must be non-negative")
        if not 0.0 <= self.missing_prob <= 1.0:
            raise ValueError("missing_prob must be in [0, 1]")


def add_noise(
    rng: np.random.Generator,
    amounts: FloatArray,
    fractions: FloatArray,
    config: NoiseConfig,
) -> tuple[FloatArray, FloatArray]:
    """Return noisy copies of true readings, with ``NaN`` for missing ones.

    ``amounts`` (non-negative quantities) get multiplicative Gaussian noise and
    stay non-negative; ``fractions`` get additive Gaussian noise and stay in
    ``[0, 1]``. A ``NaN`` in the input means "not measurable" and stays ``NaN``.

    The number of random draws depends only on the array sizes, never on the
    values, so the noise stream is identical across runs that see different
    network states. That keeps paired comparisons between agents fair.
    """
    amount_noise = rng.normal(0.0, config.relative_cv, size=amounts.shape)
    fraction_noise = rng.normal(0.0, config.fraction_sd, size=fractions.shape)
    amount_missing = rng.random(size=amounts.shape) < config.missing_prob
    fraction_missing = rng.random(size=fractions.shape) < config.missing_prob

    noisy_amounts = np.maximum(0.0, amounts * (1.0 + amount_noise))
    noisy_fractions = np.clip(fractions + fraction_noise, 0.0, 1.0)
    noisy_amounts[amount_missing] = np.nan
    noisy_fractions[fraction_missing] = np.nan
    return noisy_amounts, noisy_fractions


def reading(value: float) -> float | None:
    """Convert an array value to a telemetry reading: ``NaN`` becomes ``None``."""
    return None if np.isnan(value) else float(value)
