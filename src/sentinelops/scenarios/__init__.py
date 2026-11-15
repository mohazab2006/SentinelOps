"""Seeded, reproducible fault scenarios grouped into families.

Disjoint seed ranges provide the training, tuning and held-out test sets.
"""

from sentinelops.scenarios.families import FAMILIES, Scenario, generate, generate_many

__all__ = ["FAMILIES", "Scenario", "generate", "generate_many"]
