# Roadmap

SentinelOps is delivered in five milestones that follow the weekly schedule in
our project proposal (submitted November 6, 2026). Each task below is delivered
as a single reviewed commit whose message ends with `Refs: <task id>` and closes
the task's GitHub issue. Tasks are also tracked as
[issues](https://github.com/mohazab2006/SentinelOps/issues) grouped into
[milestones](https://github.com/mohazab2006/SentinelOps/milestones).
Task IDs are prefixed by owner: **M** = Mohamed Azab, **Y** = Yassin Koubeisy.

| Milestone | Window | Status |
|---|---|---|
| [M1 — Simulator and baseline](#m1--simulator-and-baseline) | Nov 9 – Nov 15 | 🟡 In progress (2/8) |
| [M2 — Learned models](#m2--learned-models) | Nov 16 – Nov 22 | ⚪ Planned |
| [M3 — Decision-making agent](#m3--decision-making-agent) | Nov 23 – Nov 29 | ⚪ Planned |
| [M4 — Evaluation and demo features](#m4--evaluation-and-demo-features) | Nov 30 – Dec 4 | ⚪ Planned |
| [M5 — Release](#m5--release) | Dec 5 – Dec 11 | ⚪ Planned |

Key dates: feature freeze **Dec 4** · live demonstration **Dec 7 – 11** ·
code and report submission **Dec 11**.

---

## M1 — Simulator and baseline

*Exit criterion:* a seeded crash scenario can be run end-to-end against the
rule-based baseline from the command line, reporting the share of failed
legitimate queries, with identical results for identical seeds.

- [x] ~~M1.1 Project scaffold, tooling and continuous integration ([#1](https://github.com/mohazab2006/SentinelOps/issues/1))~~ — Mohamed · Nov 9
- [x] ~~Y1.1 Architecture overview and interface contracts ([#2](https://github.com/mohazab2006/SentinelOps/issues/2))~~ — Yassin · Nov 10
- [ ] M1.2 Topology, routing preferences and traffic model ([#3](https://github.com/mohazab2006/SentinelOps/issues/3)) — Mohamed
- [ ] Y1.2 Fault model for servers, network paths and traffic ([#4](https://github.com/mohazab2006/SentinelOps/issues/4)) — Yassin
- [ ] M1.3 Simulation engine with bounded queues and failure accounting ([#5](https://github.com/mohazab2006/SentinelOps/issues/5)) — Mohamed
- [ ] Y1.3 Seeded scenario families ([#6](https://github.com/mohazab2006/SentinelOps/issues/6)) — Yassin
- [ ] M1.4 Run harness and legitimate-query failure metric ([#7](https://github.com/mohazab2006/SentinelOps/issues/7)) — Mohamed
- [ ] Y1.4 Threshold-rule baseline agent with runbooks ([#8](https://github.com/mohazab2006/SentinelOps/issues/8)) — Yassin

## M2 — Learned models

*Exit criterion:* detector precision/recall and diagnosis accuracy are reported
on the tuning split; the console globe reflects live site health.

- [ ] M2.1 Diagnostic probes and action durations ([#9](https://github.com/mohazab2006/SentinelOps/issues/9)) — Mohamed
- [ ] Y2.1 Telemetry window features ([#10](https://github.com/mohazab2006/SentinelOps/issues/10)) — Yassin
- [ ] M2.2 Labelled run generation and train/tune/test splits ([#11](https://github.com/mohazab2006/SentinelOps/issues/11)) — Mohamed
- [ ] Y2.2 Isolation Forest anomaly detector ([#12](https://github.com/mohazab2006/SentinelOps/issues/12)) — Yassin
- [ ] M2.3 Bayesian network structure and parameter fitting ([#13](https://github.com/mohazab2006/SentinelOps/issues/13)) — Mohamed
- [ ] Y2.3 Alert threshold tuning and detection metrics ([#14](https://github.com/mohazab2006/SentinelOps/issues/14)) — Yassin
- [ ] M2.4 Bayesian inference, console API and live globe ([#15](https://github.com/mohazab2006/SentinelOps/issues/15)) — Mohamed
- [ ] Y2.4 Policy rule engine ([#16](https://github.com/mohazab2006/SentinelOps/issues/16)) — Yassin

## M3 — Decision-making agent

*Exit criterion:* a service crash is repaired and verified fully autonomously; a
hardware fault is escalated, approved, repaired and returned to service
gradually after verification.

- [ ] M3.1 Probe selection by expected entropy reduction per second ([#17](https://github.com/mohazab2006/SentinelOps/issues/17)) — Mohamed
- [ ] Y3.1 Autonomy presets, approvals and maintenance holds ([#18](https://github.com/mohazab2006/SentinelOps/issues/18)) — Yassin
- [ ] M3.2 Recovery forecasting over candidate actions and fault hypotheses ([#19](https://github.com/mohazab2006/SentinelOps/issues/19)) — Mohamed
- [ ] Y3.2 Recovery verification checklist ([#20](https://github.com/mohazab2006/SentinelOps/issues/20)) — Yassin
- [ ] M3.3 Action ranking and incident prioritisation ([#21](https://github.com/mohazab2006/SentinelOps/issues/21)) — Mohamed
- [ ] Y3.3 Gradual traffic restoration with rollback ([#22](https://github.com/mohazab2006/SentinelOps/issues/22)) — Yassin
- [ ] M3.4 Agent orchestration and incident/approval console panels ([#23](https://github.com/mohazab2006/SentinelOps/issues/23)) — Mohamed
- [ ] Y3.4 Engineer handoff reports ([#24](https://github.com/mohazab2006/SentinelOps/issues/24)) — Yassin

## M4 — Evaluation and demo features

*Exit criterion:* a single command regenerates every table and figure in the
report from held-out seeds.

- [ ] M4.1 Parallel experiment runner and paired bootstrap confidence intervals ([#25](https://github.com/mohazab2006/SentinelOps/issues/25)) — Mohamed
- [ ] Y4.1 Stress scenarios: noise, missing readings and overlapping faults ([#26](https://github.com/mohazab2006/SentinelOps/issues/26)) — Yassin
- [ ] M4.2 Ablations: fixed probe order and site-local forecasting ([#27](https://github.com/mohazab2006/SentinelOps/issues/27)) — Mohamed
- [ ] Y4.2 Autonomy-preset comparison experiment ([#28](https://github.com/mohazab2006/SentinelOps/issues/28)) — Yassin
- [ ] M4.3 Report figures ([#29](https://github.com/mohazab2006/SentinelOps/issues/29)) — Mohamed
- [ ] Y4.3 Incident report data (before/after series) ([#30](https://github.com/mohazab2006/SentinelOps/issues/30)) — Yassin
- [ ] M4.4 Chaos injection, replay and evaluation views in the console ([#31](https://github.com/mohazab2006/SentinelOps/issues/31)) — Mohamed
- [ ] Y4.4 Report draft: background and related work ([#32](https://github.com/mohazab2006/SentinelOps/issues/32)) — Yassin

## M5 — Release

*Exit criterion:* a fresh clone runs the demo and experiments with one command
each; the report is submitted.

- [ ] M5.1 Documentation and quickstart ([#33](https://github.com/mohazab2006/SentinelOps/issues/33)) — Mohamed
- [ ] Y5.1 Fresh-clone reproducibility check ([#34](https://github.com/mohazab2006/SentinelOps/issues/34)) — Yassin
- [ ] M5.2 Report methods and results; demo preparation ([#35](https://github.com/mohazab2006/SentinelOps/issues/35)) — Mohamed
- [ ] Y5.2 Report introduction, limitations and contributions; demo preparation ([#36](https://github.com/mohazab2006/SentinelOps/issues/36)) — Yassin

---

## Scope changes

Changes to the plan above are recorded here with the reason.

| Date | Change | Reason |
|---|---|---|
| — | — | — |

## Backlog

Stretch goals, taken on only if the milestones above are on schedule.

- [ ] Spare-capacity action (temporarily add a server at a site)
- [ ] First-in-first-out versus priority incident scheduling experiment
- [ ] Belief timeline in the incident replay view
