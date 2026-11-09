# SentinelOps

**AI-assisted diagnosis and self-healing for a simulated anycast DNS network.**

[![CI](https://github.com/mohazab2006/SentinelOps/actions/workflows/ci.yml/badge.svg)](https://github.com/mohazab2006/SentinelOps/actions/workflows/ci.yml)

SentinelOps is an agent that watches a simulated global DNS service, works out
what has gone wrong when something breaks, repairs it when operator policy
allows, verifies that the repair worked, and hands the incident to an engineer
with a clear explanation when it cannot. It is a COMP 3106 (Introduction to
Artificial Intelligence) project at Carleton University, Fall 2026.

> **Status:** in active development — see [ROADMAP.md](ROADMAP.md) for
> milestone progress.

## Motivation

Operating anycast DNS means many sites announce the same service address and
each client is routed to its nearest site. When a site degrades, engineers
compare CPU, packet-loss and latency graphs, test the affected servers, decide
whether to restart a server or withdraw the site, and then confirm recovery.
Even routine incidents demand this attention, and the wrong move — withdrawing
a site whose neighbours cannot absorb its traffic — can widen an outage.
SentinelOps automates these repeated checks and permitted repairs, and is
evaluated on whether it reduces failed legitimate queries and operator
interventions compared with fixed threshold rules.

## Approach

| Stage | Method |
|---|---|
| Detect | Isolation Forest fitted on windows of healthy telemetry |
| Diagnose | Bayesian network over faults, symptoms and probe results; the next probe is chosen by expected entropy reduction per second |
| Plan | Forecast of bounded queues over a fixed horizon for each candidate action and plausible fault, ranked by expected reduction in failed legitimate queries minus a disruption penalty |
| Authorise | Rule-based autonomy policies that allow, block or require approval for each action |
| Verify | Sustained recovery checks, gradual traffic restoration and rollback if errors recur |

The environment is a discrete-time simulator of six sites (Ottawa, Toronto,
London, Frankfurt, Paris, Singapore) with two servers and a bounded queue each.
Seeded scenarios inject service crashes, resource exhaustion, hardware faults,
path loss, site outages and malicious query floods, alongside benign traffic
surges of similar volume.

## Getting started

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the right Python
version automatically).

```bash
git clone https://github.com/mohazab2006/SentinelOps.git
cd SentinelOps
uv sync
uv run sentinelops --help
```

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```

Every push runs the same checks in GitHub Actions on Linux and Windows.

## Project layout

```
src/sentinelops/
  sim/        simulator: topology, routing, traffic, queues, faults, actions
  scenarios/  seeded fault scenario families
  detect/     anomaly detection
  diagnose/   Bayesian diagnosis and probe selection
  plan/       recovery forecasting and incident prioritisation
  policy/     autonomy policies and approvals
  verify/     recovery checks and gradual restoration
  agent/      the SentinelOps agent
  baselines/  threshold-rule baseline and ablations
  eval/       experiment harness and statistics
  api/        API for the operations console
tests/        unit and integration tests
```

## Team

- **Mohamed Azab** — simulator, diagnosis, recovery planning, agent, evaluation, console
- **Yassin Koubeisy** — faults and scenarios, anomaly detection, policy engine, verification, baseline

## References

1. F. T. Liu, K. M. Ting and Z.-H. Zhou, "Isolation Forest," *IEEE ICDM*, 2008.
2. I. Rish et al., "Adaptive Diagnosis in Distributed Systems," *IEEE Transactions on Neural Networks*, 16(5), 2005.
3. A. S. M. Rizvi, L. Bertholdo, J. Ceron and J. Heidemann, "Anycast Agility: Network Playbooks to Fight DDoS," *USENIX Security*, 2022.
