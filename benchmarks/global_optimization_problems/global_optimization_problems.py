"""
Global optimization benchmark functions.

These used to be reimplemented in this file; they have since been upstreamed
into ``syne_tune.blackbox_repository.global_optimization_problems`` (same
functions, same numerics) so this module now just re-exports them. Doing so
also picks up the 24-function BBOB/COCO suite added to syne_tune's collection
(``syne_tune.blackbox_repository.bbob``, merged in automatically), without
duplicating that implementation here.
"""

from syne_tune.blackbox_repository.global_optimization_problems import (
    Ackley,
    Branin,
    Eggholder,
    Forrester,
    GoldsteinPrice,
    Hartman,
    Hartman3,
    Hartman6,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    Sphere,
    StyblinskiTang,
    SumPowers,
    SixHumpCamel,
    global_optimization_problem_collection,
)

__all__ = [
    "Ackley",
    "Branin",
    "Eggholder",
    "Forrester",
    "GoldsteinPrice",
    "Hartman",
    "Hartman3",
    "Hartman6",
    "Michalewicz",
    "Rastrigin",
    "Rosenbrock",
    "Sphere",
    "StyblinskiTang",
    "SumPowers",
    "SixHumpCamel",
    "global_optimization_problem_collection",
]
