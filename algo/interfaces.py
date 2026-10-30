# -*- coding: utf-8 -*-
# @Time     : 2026/06/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


__all__ = [
    'MainSolution',
    'SubProblemResult',
    'DualInfo',
    'CutRequest',
    'CutOutcome',
    'AggregationPolicy',
    'GeneratedCut',
    'InnerMinimizationResult',
    'ScenarioGroup',
    'IterationContext',
    'SolutionGenerator',
    'NoOpSolutionGenerator',
    'StaticSolutionGenerator',
]


@dataclass(frozen=True)
class ScenarioGroup:
    scenarios: Sequence[Any]

    def __post_init__(self):
        object.__setattr__(self, 'scenarios', tuple(self.scenarios))

    @property
    def key(self) -> Tuple[Any, ...]:
        return tuple(self.scenarios)

    def as_list(self) -> list:
        return list(self.scenarios)

    def probability_weight(self, probability_dict: dict) -> float:
        return sum(probability_dict[s] for s in self.scenarios)


@dataclass(frozen=True)
class MainSolution:
    values: Dict[str, Dict[Any, Any]]
    name: Optional[str] = None
    is_relaxed: bool = False

    def as_model_result(self) -> Dict[str, Dict[Any, Any]]:
        return self.values


@dataclass(frozen=True)
class DualInfo:
    dual_values: Dict[str, float]
    var_map: Dict[str, Tuple[str, Any]]
    local_copy_constraint_names: Sequence[str] = ()
    source: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, 'local_copy_constraint_names', tuple(self.local_copy_constraint_names))

    @property
    def constr_dual_info(self) -> Dict[str, float]:
        return self.dual_values

    @property
    def constr_var_map(self) -> Dict[str, Tuple[str, Any]]:
        return self.var_map


class CutOutcome(str, Enum):
    ADDED = 'added'
    REPEATED = 'repeated'
    NOT_APPLICABLE = 'not_applicable'
    NOT_VIOLATED = 'not_violated'


class AggregationPolicy(str, Enum):
    PER_GROUP = 'per_group'
    AGGREGATE = 'aggregate'


@dataclass(frozen=True)
class CutRequest:
    """Uniform input contract for every registered cut generator."""
    scenario_group: ScenarioGroup
    ite_name: str
    anchor_solution: MainSolution
    dual_info: Optional[DualInfo] = None
    relaxed_obj: Optional[float] = None
    exact_obj_bound: Optional[float] = None
    theta_values: Optional[Dict[Any, float]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedCut:
    """Normalized cut output containing both master expressions and the CGLP row form."""
    cut_type: str
    cut_name: str
    scenario_group: ScenarioGroup
    ite_name: str
    lhs: Any
    rhs: Any
    lower_bound_row: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InnerMinimizationResult:
    r"""
    Outcome of one Lagrangian inner minimization ``min_{x,y in X_sub} c^T x - lambda^T (y - y*)``.

    The two objective quantities point in OPPOSITE directions and are never interchangeable:

    * ``lagrangian_lower_bound`` -- the solver's dual bound, ``L_lb <= L(lambda)``. The only value a
      cut may be anchored on; a MIP gap or time limit then weakens the cut instead of invalidating it.
    * ``incumbent_sub_obj`` / ``incumbent_main_sol`` -- the returned feasible point, whose objective
      is an UPPER bound on ``L(lambda)``. Valid input for the multiplier heuristic's outer
      approximation, invalid as a cut anchor.
    """
    scenario_group: ScenarioGroup
    multiplier: Dict[str, float]
    lagrangian_lower_bound: float
    incumbent_sub_obj: float
    incumbent_main_sol: Dict[str, Dict[Any, Any]]
    solve_status: Optional[str] = None


@dataclass(frozen=True)
class IterationContext:
    ite_name: str
    iteration_number: Optional[int] = None
    current_solution: Optional[MainSolution] = None
    fractional_solution: Optional[MainSolution] = None
    main_stage_obj: Optional[float] = None
    candidate_cut_types: Sequence[str] = ()

    def __post_init__(self):
        object.__setattr__(self, 'candidate_cut_types', tuple(self.candidate_cut_types))


class SolutionGenerator:
    """Hook for producing extra first-stage points used as cut anchors."""

    def generate(self, context: IterationContext) -> List[MainSolution]:
        raise NotImplementedError


class NoOpSolutionGenerator(SolutionGenerator):
    """Default generator; preserves the original decomposition trajectory."""

    def generate(self, context: IterationContext) -> List[MainSolution]:
        return []


class StaticSolutionGenerator(SolutionGenerator):
    """Return a fixed collection of anchors, with runner-side repetition filtering."""

    def __init__(self, solutions: Sequence[MainSolution]):
        self._solutions = tuple(solutions)

    def generate(self, context: IterationContext) -> List[MainSolution]:
        return list(self._solutions)


@dataclass(frozen=True)
class SubProblemResult:
    scenario_group: ScenarioGroup
    main_solution: MainSolution
    relaxed_obj_value: float
    benders_dual: DualInfo
    #: Incumbent objective of the exact subproblem -- an UPPER bound on Q_G(x). Incumbent accounting
    #: only; never anchor a cut on it.
    exact_obj_value: Optional[float] = None
    #: Dual bound of the exact subproblem -- a LOWER bound on Q_G(x), valid under any MIP gap or
    #: time limit. This is the value every cut anchor must use.
    exact_obj_bound: Optional[float] = None
    fractional_solution: Optional[MainSolution] = None
    fractional_relaxed_obj_value: Optional[float] = None
    fractional_dual: Optional[DualInfo] = None
    incumbent_obj_with_main: Optional[float] = None
    incumbent_result: Optional[dict] = None

    @property
    def local_copy_constraint_names(self) -> Tuple[str, ...]:
        """Expose the canonical local-copy rows used for dual collection."""
        return tuple(self.benders_dual.local_copy_constraint_names)

    @property
    def local_copy_var_map(self) -> Dict[str, Tuple[str, Any]]:
        """Expose the local-copy constraint to master-state variable mapping."""
        return self.benders_dual.var_map
