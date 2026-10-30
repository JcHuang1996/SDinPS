# -*- coding: utf-8 -*-
# @Time     : 2026/06/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

from dataclasses import dataclass, replace
from typing import Dict, Optional

from util.names import SolverRoleName


@dataclass(frozen=True)
class SolverConfig:
    solver_name: str = 'gurobi_direct'
    fallback_solver_name: Optional[str] = None
    mip_gap: Optional[float] = 0.002
    time_limit: Optional[float] = None
    tee: bool = True

    def to_solver_options(self, solver_name: Optional[str] = None) -> Dict[str, float]:
        solver_key = (solver_name or self.solver_name).lower()

        if 'gurobi' in solver_key:
            mip_gap_key = 'MIPGap'
            time_limit_key = 'TimeLimit'
        elif 'highs' in solver_key:
            mip_gap_key = 'mip_rel_gap'
            time_limit_key = 'time_limit'
        elif 'cbc' in solver_key:
            mip_gap_key = 'ratioGap'
            time_limit_key = 'seconds'
        elif 'glpk' in solver_key:
            mip_gap_key = 'mipgap'
            time_limit_key = 'tmlim'
        else:
            mip_gap_key = 'MIPGap'
            time_limit_key = 'TimeLimit'

        options = {}
        if self.mip_gap is not None:
            options[mip_gap_key] = self.mip_gap
        if self.time_limit is not None:
            options[time_limit_key] = self.time_limit

        return options


@dataclass(frozen=True)
class SolverConfigSet:
    """
    Role-scoped solver configuration for the whole decomposition run.

    Each model built inside the algorithm asks for the config of its own role instead of falling
    back to a single default plus inline ``set_parameters`` calls. The correctness rule behind the
    defaults is:

        cuts are anchored on DUAL bounds, upper bounds use incumbents.

    Because the cut anchors read ``ModelBase.get_obj_bound()``, a nonzero gap or a time limit on
    ``inner_min`` / ``subproblem_exact`` only weakens a cut; it can never invalidate it.
    """

    master: SolverConfig
    subproblem_lp: SolverConfig
    subproblem_exact: SolverConfig
    inner_min: SolverConfig
    lagrangian_master: SolverConfig
    cglp: SolverConfig

    @classmethod
    def build(
        cls,
        solver_name: str = 'gurobi_direct',
        fallback_solver_name: Optional[str] = None,
        tee: bool = False,
        master_mip_gap: Optional[float] = 0.002,
        master_time_limit: Optional[float] = 120,
        subproblem_exact_mip_gap: Optional[float] = 0.0,
        subproblem_exact_time_limit: Optional[float] = None,
        inner_min_mip_gap: Optional[float] = 0.0,
        inner_min_time_limit: Optional[float] = None,
        lagrangian_master_mip_gap: Optional[float] = 0.002,
        lagrangian_master_time_limit: Optional[float] = None,
        lp_time_limit: Optional[float] = None,
    ) -> 'SolverConfigSet':
        """Create one config per role from a small set of run-level knobs."""

        def _make(mip_gap, time_limit):
            return SolverConfig(
                solver_name=solver_name,
                fallback_solver_name=fallback_solver_name,
                mip_gap=mip_gap,
                time_limit=time_limit,
                tee=tee,
            )

        return cls(
            master=_make(master_mip_gap, master_time_limit),
            # LP roles: a MIP gap is meaningless and some solvers reject the option outright.
            subproblem_lp=_make(None, lp_time_limit),
            subproblem_exact=_make(subproblem_exact_mip_gap, subproblem_exact_time_limit),
            inner_min=_make(inner_min_mip_gap, inner_min_time_limit),
            lagrangian_master=_make(lagrangian_master_mip_gap, lagrangian_master_time_limit),
            cglp=_make(None, lp_time_limit),
        )

    @classmethod
    def from_single_config(cls, solver_config: Optional[SolverConfig] = None) -> 'SolverConfigSet':
        """Wrap one SolverConfig into a set, keeping the LP roles free of a MIP gap."""
        base = solver_config or SolverConfig()
        return cls(
            master=base,
            subproblem_lp=replace(base, mip_gap=None),
            subproblem_exact=replace(base, mip_gap=0.0),
            inner_min=replace(base, mip_gap=0.0),
            lagrangian_master=base,
            cglp=replace(base, mip_gap=None),
        )

    def get(self, role: str) -> SolverConfig:
        """Return the config registered for one SolverRoleName."""
        if role not in SolverRoleName.ALL_ROLES:
            raise ValueError(f'Unknown solver role: {role}')
        return getattr(self, role)

    def with_role(self, role: str, solver_config: SolverConfig) -> 'SolverConfigSet':
        """Return a copy of this set with one role replaced."""
        if role not in SolverRoleName.ALL_ROLES:
            raise ValueError(f'Unknown solver role: {role}')
        return replace(self, **{role: solver_config})

    def as_dict(self) -> Dict[str, Dict]:
        """Flatten every role's config for run-parameter reporting."""
        return {
            role: {
                'solver_name': self.get(role).solver_name,
                'fallback_solver_name': self.get(role).fallback_solver_name,
                'mip_gap': self.get(role).mip_gap,
                'time_limit': self.get(role).time_limit,
                'tee': self.get(role).tee,
            }
            for role in SolverRoleName.ALL_ROLES
        }
