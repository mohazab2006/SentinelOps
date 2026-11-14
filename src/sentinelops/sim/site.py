"""What happens to queries at one anycast site during one tick.

Traffic is a fluid: amounts of legitimate and attack queries, kept separate so
that scoring can count failed *legitimate* queries while the site itself (like a
real DNS server) cannot tell the two apart.

Each tick, queries routed to a site go through these stages, and every query
that does not get an answer is counted under exactly one cause:

1. **lost** on the network path (path-loss faults or a site outage);
2. **rejected** by the site's rate limit, if one is set;
3. **overflow**: dropped because the bounded queue is full;
4. **served** first-in-first-out, up to the site's capacity this tick;
5. **timeout**: still queued after waiting ``timeout_ticks`` ticks.

Queries can wait across ticks, so admitted traffic is conserved over time:
everything admitted is eventually served, timed out or still queued.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

RATE_LIMIT_COLLATERAL = 0.05
"""Share of legitimate queries a rate limiter wrongly drops while it is shedding load.

A limiter keys on simple signatures (source, query-name pattern), so it drops
attack traffic first but misclassifies some legitimate queries. When there is
not enough attack traffic to meet the limit (a benign surge), it has to drop
legitimate queries, which is why rate limiting a surge hurts real users.
"""


@dataclass(frozen=True, slots=True)
class Traffic:
    """An amount of queries, split into legitimate and attack.

    Attributes:
        legit: Legitimate queries.
        attack: Attack queries.
    """

    legit: float = 0.0
    attack: float = 0.0

    @property
    def total(self) -> float:
        """All queries, legitimate and attack."""
        return self.legit + self.attack

    def __add__(self, other: Traffic) -> Traffic:
        """Add two amounts class by class."""
        return Traffic(self.legit + other.legit, self.attack + other.attack)

    def scaled(self, factor: float) -> Traffic:
        """Return both classes multiplied by ``factor``."""
        return Traffic(self.legit * factor, self.attack * factor)


@dataclass(slots=True)
class _Batch:
    arrival: int
    legit: float
    attack: float


class SiteQueue:
    """A bounded first-in-first-out queue of query batches.

    Queries that arrive in the same tick form one batch. Batches keep their
    arrival tick so the queue can time out queries that waited too long.
    """

    def __init__(self, limit: float, timeout_ticks: int) -> None:
        """Create an empty queue holding at most ``limit`` queries.

        Raises:
            ValueError: If ``limit`` is negative or ``timeout_ticks`` < 1.
        """
        if limit < 0:
            raise ValueError("queue limit must be non-negative")
        if timeout_ticks < 1:
            raise ValueError("timeout_ticks must be at least 1")
        self.limit = limit
        self.timeout_ticks = timeout_ticks
        self._batches: deque[_Batch] = deque()

    @property
    def queued(self) -> Traffic:
        """Queries currently waiting, by class."""
        return Traffic(sum(b.legit for b in self._batches), sum(b.attack for b in self._batches))

    @property
    def depth(self) -> float:
        """Number of queries currently waiting."""
        return self.queued.total

    def offer(self, tick: int, arriving: Traffic) -> Traffic:
        """Admit as much of ``arriving`` as fits and return what overflowed.

        When only part fits, legitimate and attack queries are admitted in the
        same proportion: the queue cannot tell them apart.
        """
        if arriving.total <= 0.0:
            return Traffic()
        space = max(0.0, self.limit - self.depth)
        admitted_share = min(1.0, space / arriving.total)
        admitted = arriving.scaled(admitted_share)
        if admitted.total > 0.0:
            self._batches.append(_Batch(tick, admitted.legit, admitted.attack))
        return Traffic(arriving.legit - admitted.legit, arriving.attack - admitted.attack)

    def serve(self, capacity: float) -> Traffic:
        """Answer up to ``capacity`` queries, oldest batch first, and return them."""
        served = Traffic()
        remaining = max(0.0, capacity)
        while self._batches and remaining > 0.0:
            batch = self._batches[0]
            size = batch.legit + batch.attack
            if size <= remaining:
                served += Traffic(batch.legit, batch.attack)
                remaining -= size
                self._batches.popleft()
            else:
                share = remaining / size
                served += Traffic(batch.legit * share, batch.attack * share)
                batch.legit -= batch.legit * share
                batch.attack -= batch.attack * share
                remaining = 0.0
        return served

    def expire(self, tick: int) -> Traffic:
        """Remove and return queries that have now waited ``timeout_ticks`` ticks.

        A batch that arrived at tick ``a`` has had one chance to be served in
        each of ticks ``a`` to ``tick``; after ``timeout_ticks`` chances it fails.
        """
        expired = Traffic()
        while self._batches and tick - self._batches[0].arrival + 1 >= self.timeout_ticks:
            batch = self._batches.popleft()
            expired += Traffic(batch.legit, batch.attack)
        return expired


@dataclass(frozen=True, slots=True)
class SiteTick:
    """Everything that happened to queries at one site in one tick.

    Attributes:
        arrived: Queries routed to the site (before path loss).
        lost: Lost on the network path.
        rejected: Dropped by the rate limit.
        overflow: Dropped because the queue was full.
        served: Answered.
        timeout: Failed after waiting too long (may include earlier arrivals).
        queue_depth: Queries still waiting at the end of the tick.
    """

    arrived: Traffic
    lost: Traffic
    rejected: Traffic
    overflow: Traffic
    served: Traffic
    timeout: Traffic
    queue_depth: float

    @property
    def failed(self) -> Traffic:
        """Queries that failed this tick, for any reason."""
        return self.lost + self.rejected + self.overflow + self.timeout


def rate_limit(arriving: Traffic, limit_qps: float | None) -> tuple[Traffic, Traffic]:
    """Apply a rate limit and return ``(passed, rejected)``.

    Nothing is dropped while traffic is within the limit. Above it, the limiter
    drops the excess from attack traffic first, and legitimate traffic only if
    attack traffic alone cannot cover the excess. On top of that it wrongly
    drops a share :data:`RATE_LIMIT_COLLATERAL` of legitimate queries.
    """
    if limit_qps is None or arriving.total <= limit_qps:
        return arriving, Traffic()
    excess = arriving.total - limit_qps
    attack_drop = min(arriving.attack, excess)
    shortfall = excess - attack_drop
    legit_drop = min(arriving.legit, shortfall + RATE_LIMIT_COLLATERAL * arriving.legit)
    rejected = Traffic(legit_drop, attack_drop)
    return Traffic(arriving.legit - legit_drop, arriving.attack - attack_drop), rejected


def process_tick(
    queue: SiteQueue,
    tick: int,
    arriving: Traffic,
    *,
    path_loss: float,
    limit_qps: float | None,
    capacity: float,
) -> SiteTick:
    """Run one tick of the site pipeline: loss, rate limit, queue, serve, time out.

    Args:
        queue: The site's queue; updated in place.
        tick: The current tick.
        arriving: Queries routed to the site this tick.
        path_loss: Fraction of arriving queries lost on the way, in ``[0, 1]``.
        limit_qps: Rate limit in queries per tick, or ``None`` for no limit.
        capacity: Queries the site's working servers can answer this tick.

    Returns:
        What happened to the queries, by cause.
    """
    if not 0.0 <= path_loss <= 1.0:
        raise ValueError(f"path_loss must be in [0, 1], got {path_loss}")
    lost = arriving.scaled(path_loss)
    reached = Traffic(arriving.legit - lost.legit, arriving.attack - lost.attack)
    passed, rejected = rate_limit(reached, limit_qps)
    overflow = queue.offer(tick, passed)
    served = queue.serve(capacity)
    timeout = queue.expire(tick)
    return SiteTick(
        arrived=arriving,
        lost=lost,
        rejected=rejected,
        overflow=overflow,
        served=served,
        timeout=timeout,
        queue_depth=queue.depth,
    )
