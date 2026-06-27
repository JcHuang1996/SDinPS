# -*- coding: utf-8 -*-
# @Time     : 2026/06/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

from dataclasses import dataclass
from typing import Dict, Optional


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
