"""Recovery verification: prove a site is healthy again before closing an incident.

After the agent fixes something, one good tick proves nothing: telemetry is
noisy and some faults come and go. A site counts as recovered only when it
passes every check for ``required_ticks`` (30) ticks **in a row**.

Each tick, :func:`check_site` turns a site's telemetry into four checks:
fail ratio, p95 latency, loss and queue depth. A missing reading makes its
check *unknown* rather than failed, because monitoring drops readings now and
then even when everything is fine.

:class:`SiteVerifier` then keeps two streaks:

* a tick where every check passes extends the good streak and resets the bad one;
* a tick where any check fails resets the good streak and extends the bad one;
* a tick with only missing readings and no failure changes neither.

It reports PASSED after ``required_ticks`` good ticks in a row, and FAILED
after ``fail_after`` (10) bad ticks in a row (the fix did not work, or the
site regressed) or when ``deadline_ticks`` (300) pass without PASSED.

The default limits were measured on 5 healthy runs x 6 sites: healthy sites
never fail a query, p95 stays under 93 ms, loss under 4.6% and the queue stays
empty. A crashed server sends its site to 25% failed queries and 2.7 s p95.
The limits leave room for noise yet sit far below a broken site's values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sentinelops.sim.telemetry import SiteTelemetry
from sentinelops.sim.topology import SiteId, Topology


@dataclass(frozen=True, slots=True)
class HealthLimits:
    """The highest value each check accepts (a value equal to the limit passes).

    Attributes:
        max_fail_ratio: Highest share of failed queries, ``failed / (ok + failed)``.
        max_p95_ms: Highest 95th-percentile latency, in milliseconds.
        max_loss: Highest fraction of queries lost on the network path.
        max_queue_fraction: Highest queue depth as a fraction of the site's
            ``queue_limit``.
    """

    max_fail_ratio: float = 0.01
    max_p95_ms: float = 250.0
    max_loss: float = 0.05
    max_queue_fraction: float = 0.1

    def __post_init__(self) -> None:
        """Reject negative limits.

        Raises:
            ValueError: If any limit is negative.
        """
        for name in ("max_fail_ratio", "max_p95_ms", "max_loss", "max_queue_fraction"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one health check at one tick.

    Attributes:
        name: ``"fail_ratio"``, ``"p95_latency"``, ``"loss"`` or ``"queue"``.
        value: The measured value, or ``None`` if the reading was missing.
        limit: The highest value that passes.
        ok: ``True`` if ``value <= limit``, ``False`` if above, ``None`` if
            unknown (missing reading).
    """

    name: str
    value: float | None
    limit: float
    ok: bool | None


def _check(name: str, value: float | None, limit: float) -> CheckResult:
    return CheckResult(name, value, limit, None if value is None else value <= limit)


def _fail_ratio(telemetry: SiteTelemetry) -> float | None:
    ok, failed = telemetry.ok_replies, telemetry.failed_replies
    if ok is None or failed is None:
        return None
    total = ok + failed
    return failed / total if total > 0 else 0.0  # no traffic: nothing failed


def _queue_fraction(queue_depth: float | None, queue_limit: float) -> float | None:
    if queue_depth is None:
        return None
    if queue_limit <= 0:  # a site without a queue: any waiting query is too many
        return 0.0 if queue_depth <= 0 else float("inf")
    return queue_depth / queue_limit


def check_site(
    telemetry: SiteTelemetry, queue_limit: float, limits: HealthLimits | None = None
) -> tuple[CheckResult, ...]:
    """Run the four health checks on one tick of a site's telemetry.

    Args:
        telemetry: The site's readings for this tick.
        queue_limit: The site's ``queue_limit`` (queue depth is judged
            relative to it).
        limits: Limits to apply; the measured defaults if ``None``.

    Returns:
        Checks in the order fail_ratio, p95_latency, loss, queue.
    """
    lim = limits if limits is not None else HealthLimits()
    queue_fraction = _queue_fraction(telemetry.queue_depth, queue_limit)
    return (
        _check("fail_ratio", _fail_ratio(telemetry), lim.max_fail_ratio),
        _check("p95_latency", telemetry.p95_latency_ms, lim.max_p95_ms),
        _check("loss", telemetry.loss, lim.max_loss),
        _check("queue", queue_fraction, lim.max_queue_fraction),
    )


class VerifyStatus(StrEnum):
    """Where a site's recovery verification stands."""

    PENDING = "pending"
    """Still watching."""

    PASSED = "passed"
    """Healthy for ``required_ticks`` ticks in a row: the site has recovered."""

    FAILED = "failed"
    """Too many bad ticks in a row, or no recovery before the deadline."""


class SiteVerifier:
    """Watches one site after a fix until it passes or fails verification.

    Args:
        site: The site to verify.
        topology: Used for the site's ``queue_limit``.
        start_tick: Tick the verification starts (usually when the fix finished).
        required_ticks: Good ticks in a row needed to pass.
        fail_after: Bad ticks in a row that fail the verification.
        deadline_ticks: Ticks after ``start_tick`` by which the site must pass.
        limits: Health limits; the measured defaults if ``None``.

    Raises:
        ValueError: If the site is not in ``topology``, ``required_ticks`` or
            ``fail_after`` is below 1, or ``deadline_ticks`` is below
            ``required_ticks``.
    """

    def __init__(
        self,
        site: SiteId,
        topology: Topology,
        start_tick: int,
        *,
        required_ticks: int = 30,
        fail_after: int = 10,
        deadline_ticks: int = 300,
        limits: HealthLimits | None = None,
    ) -> None:
        """Check the settings and start in PENDING with empty streaks."""
        if site not in topology.sites:
            raise ValueError(f"unknown site {site!r}")
        if required_ticks < 1:
            raise ValueError(f"required_ticks must be >= 1, got {required_ticks}")
        if fail_after < 1:
            raise ValueError(f"fail_after must be >= 1, got {fail_after}")
        if deadline_ticks < required_ticks:
            raise ValueError(
                f"deadline_ticks ({deadline_ticks}) must be >= required_ticks ({required_ticks})"
            )
        self._site = site
        self._queue_limit = topology.sites[site].queue_limit
        self._start_tick = start_tick
        self._required = required_ticks
        self._fail_after = fail_after
        self._deadline = deadline_ticks
        self._limits = limits if limits is not None else HealthLimits()
        self._status = VerifyStatus.PENDING
        self._streak = 0
        self._bad_streak = 0
        self._last_checks: tuple[CheckResult, ...] = ()

    @property
    def status(self) -> VerifyStatus:
        """The current verification status (final once PASSED or FAILED)."""
        return self._status

    @property
    def streak(self) -> int:
        """Healthy ticks in a row so far."""
        return self._streak

    @property
    def last_checks(self) -> tuple[CheckResult, ...]:
        """The checks of the last tick fed to :meth:`update` (empty before the first)."""
        return self._last_checks

    def update(self, tick: int, telemetry: SiteTelemetry) -> VerifyStatus:
        """Feed one tick of the site's telemetry and return the new status.

        Args:
            tick: The tick the telemetry describes.
            telemetry: The site's readings for that tick.

        Returns:
            The status after this tick. Once PASSED or FAILED it never changes.
        """
        if self._status != VerifyStatus.PENDING:
            return self._status
        self._last_checks = check_site(telemetry, self._queue_limit, self._limits)
        results = [check.ok for check in self._last_checks]
        if False in results:
            self._streak = 0
            self._bad_streak += 1
        elif None not in results:
            self._streak += 1
            self._bad_streak = 0
        # only missing readings and no failure: both streaks stay as they are
        if self._streak >= self._required:
            self._status = VerifyStatus.PASSED
        elif self._bad_streak >= self._fail_after or tick - self._start_tick >= self._deadline:
            self._status = VerifyStatus.FAILED
        return self._status
