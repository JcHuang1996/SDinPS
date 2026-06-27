# -*- coding: utf-8 -*-
# @Time     : 2026/06/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


__all__ = [
    'MainSolution',
    'SubProblemResult',
    'DualInfo',
    'CutCandidate',
    'GeneratedCut',
    'ScenarioGroup',
    'IterationContext',
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


@dataclass(frozen=True)
class CutCandidate:
    cut_type: str
    scenario_group: ScenarioGroup
    ite_name: str
    main_solution: MainSolution
    dual_info: Optional[DualInfo] = None
    sub_obj_value: Optional[float] = None
    exact_sub_obj_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
class IterationContext:
    ite_name: str
    iteration_number: Optional[int] = None
    current_solution: Optional[MainSolution] = None
    fractional_solution: Optional[MainSolution] = None
    main_stage_obj: Optional[float] = None
    candidate_cut_types: Sequence[str] = ()

    def __post_init__(self):
        object.__setattr__(self, 'candidate_cut_types', tuple(self.candidate_cut_types))


@dataclass(frozen=True)
class SubProblemResult:
    scenario_group: ScenarioGroup
    main_solution: MainSolution
    relaxed_obj_value: float
    benders_dual: DualInfo
    exact_obj_value: Optional[float] = None
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

    def as_legacy_tuple(self):
        return (
            self.relaxed_obj_value,
            self.benders_dual.dual_values,
            self.benders_dual.var_map,
            self.fractional_dual.dual_values if self.fractional_dual is not None else None,
            self.fractional_dual.var_map if self.fractional_dual is not None else None,
            self.exact_obj_value,
            self.fractional_relaxed_obj_value,
        )

    def __iter__(self):
        return iter(self.as_legacy_tuple())
