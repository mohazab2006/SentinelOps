"""Tests for the per-site query pipeline: rate limit, bounded queue, timeouts."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sentinelops.sim.site import (
    RATE_LIMIT_COLLATERAL,
    SiteQueue,
    Traffic,
    process_tick,
    rate_limit,
)


def test_traffic_arithmetic() -> None:
    t = Traffic(3.0, 1.0) + Traffic(1.0, 1.0)
    assert t == Traffic(4.0, 2.0)
    assert t.total == 6.0
    assert t.scaled(0.5) == Traffic(2.0, 1.0)


# ------------------------------------------------------------------------- queue


def test_queue_admits_until_full_and_overflows_proportionally() -> None:
    queue = SiteQueue(limit=100, timeout_ticks=3)
    assert queue.offer(0, Traffic(60.0, 20.0)) == Traffic()
    overflow = queue.offer(1, Traffic(30.0, 10.0))  # only 20 of 40 fit
    assert overflow == Traffic(15.0, 5.0)
    assert queue.depth == pytest.approx(100.0)
    assert queue.queued == Traffic(75.0, 25.0)


def test_queue_serves_oldest_first_and_splits_partial_batches() -> None:
    queue = SiteQueue(limit=1000, timeout_ticks=5)
    queue.offer(0, Traffic(10.0, 0.0))
    queue.offer(1, Traffic(0.0, 10.0))
    served = queue.serve(15.0)
    assert served == Traffic(10.0, 5.0)
    assert queue.queued == Traffic(0.0, 5.0)
    assert queue.serve(0.0) == Traffic()
    assert queue.serve(-5.0) == Traffic()


def test_queue_times_out_after_timeout_ticks_chances() -> None:
    queue = SiteQueue(limit=1000, timeout_ticks=3)
    queue.offer(10, Traffic(5.0, 0.0))
    assert queue.expire(10) == Traffic()  # first chance
    assert queue.expire(11) == Traffic()  # second chance
    assert queue.expire(12) == Traffic(5.0, 0.0)  # third chance used up
    assert queue.depth == 0.0


def test_empty_arrivals_change_nothing() -> None:
    queue = SiteQueue(limit=10, timeout_ticks=1)
    assert queue.offer(0, Traffic()) == Traffic()
    assert queue.depth == 0.0


@pytest.mark.parametrize(("limit", "timeout"), [(-1, 3), (10, 0)])
def test_queue_rejects_bad_settings(limit: float, timeout: int) -> None:
    with pytest.raises(ValueError):
        SiteQueue(limit, timeout)


# -------------------------------------------------------------------- rate limit


def test_rate_limit_passes_traffic_under_the_limit() -> None:
    assert rate_limit(Traffic(50.0, 10.0), 100.0) == (Traffic(50.0, 10.0), Traffic())
    assert rate_limit(Traffic(50.0, 10.0), None) == (Traffic(50.0, 10.0), Traffic())


def test_rate_limit_drops_attack_first_with_small_collateral() -> None:
    passed, rejected = rate_limit(Traffic(100.0, 900.0), limit_qps=200.0)
    assert rejected.attack == pytest.approx(800.0)
    assert rejected.legit == pytest.approx(RATE_LIMIT_COLLATERAL * 100.0)
    assert passed.legit == pytest.approx(95.0)


def test_rate_limit_on_a_benign_surge_drops_real_users() -> None:
    passed, rejected = rate_limit(Traffic(1000.0, 0.0), limit_qps=600.0)
    assert rejected.legit == pytest.approx(400.0 + RATE_LIMIT_COLLATERAL * 1000.0)
    assert passed.total == pytest.approx(550.0)


# ------------------------------------------------------------------ whole tick


def test_process_tick_counts_each_cause() -> None:
    queue = SiteQueue(limit=50, timeout_ticks=1)
    result = process_tick(
        queue, 0, Traffic(200.0, 0.0), path_loss=0.5, limit_qps=80.0, capacity=10.0
    )
    assert result.lost == Traffic(100.0, 0.0)
    assert result.rejected.legit == pytest.approx(20.0 + RATE_LIMIT_COLLATERAL * 100.0)
    assert result.overflow.legit == pytest.approx(25.0)
    assert result.served == Traffic(10.0, 0.0)
    assert result.timeout.legit == pytest.approx(40.0)  # timeout_ticks=1: unserved fail now
    assert result.queue_depth == 0.0


def test_process_tick_rejects_bad_loss() -> None:
    with pytest.raises(ValueError):
        process_tick(SiteQueue(1, 1), 0, Traffic(), path_loss=1.5, limit_qps=None, capacity=1)


ticks = st.lists(
    st.tuples(
        st.floats(0, 5000),
        st.floats(0, 5000),
        st.floats(0, 1),
        st.one_of(st.none(), st.floats(0, 6000)),
        st.floats(0, 4000),
    ),
    min_size=1,
    max_size=40,
)


@given(ticks=ticks, limit=st.floats(0, 8000), timeout=st.integers(1, 5))
def test_pipeline_conserves_queries(
    ticks: list[tuple[float, float, float, float | None, float]], limit: float, timeout: int
) -> None:
    queue = SiteQueue(limit, timeout)
    arrived = accounted = Traffic()
    for tick, (legit, attack, loss, limit_qps, capacity) in enumerate(ticks):
        r = process_tick(
            queue,
            tick,
            Traffic(legit, attack),
            path_loss=loss,
            limit_qps=limit_qps,
            capacity=capacity,
        )
        arrived += r.arrived
        accounted += r.served + r.failed
        assert min(r.served.legit, r.failed.legit, r.served.attack, r.failed.attack) >= -1e-9
        assert queue.depth <= limit + 1e-6
    accounted += queue.queued
    assert accounted.legit == pytest.approx(arrived.legit, rel=1e-9, abs=1e-6)
    assert accounted.attack == pytest.approx(arrived.attack, rel=1e-9, abs=1e-6)
