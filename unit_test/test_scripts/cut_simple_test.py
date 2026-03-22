# -*- coding: utf-8 -*-
# @Time     : 2026/03/18
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
Logic Check
-----------
This test now matches the reference note `bdd_toy_example_reference.md` exactly:
1. the master variable y is relaxed to 0 <= y <= 1,
2. the common first two cuts are generated from the LP subproblem,
3. the strengthened cut is generated with the implemented inner-minimization model,
4. the Lagrangian cut is generated with the implemented regularized-multiplier heuristic,
5. all cut coefficients and master solutions are checked against the reference sequence.

If the implemented Lagrangian heuristic does not reproduce the reference cut, this
script reports the mismatch instead of silently passing.
"""

import os
import sys
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from model.model_base import ModelBase
from model.model_lagrangian import (
    ModelInnerMinimizationProblem,
    ModelLagrangianMultiplierHeuristic,
)
from util.names import ObjName, VarName

logger = logging.getLogger(__name__)

CONSTR_NAME = 'copy_y_0'
CONSTR_VAR_MAP = {CONSTR_NAME: ('y', 0)}
SOLVER_NAME = 'gurobi_direct'
TOL = 1e-6


@dataclass(frozen=True)
class CutRecord:
    """Cut in the form theta >= intercept + slope * y."""
    name: str
    intercept: float
    slope: float


@dataclass(frozen=True)
class MasterPoint:
    y: float
    theta: float


REFERENCE_CLASSICAL_CUTS: List[CutRecord] = [
    CutRecord(name='classical_0', intercept=8.0, slope=-15.0),
    CutRecord(name='classical_1', intercept=-24.5, slope=35.0),
    CutRecord(name='classical_2', intercept=-0.5, slope=5.0),
    CutRecord(name='classical_3', intercept=13.0 / 3.0, slope=-10.0 / 3.0),
]

REFERENCE_CLASSICAL_MASTER_POINTS: List[MasterPoint] = [
    MasterPoint(y=0.0, theta=float('nan')),      # seeded initial point from the reference
    MasterPoint(y=1.0, theta=-7.0),
    MasterPoint(y=13.0 / 20.0, theta=-7.0 / 4.0),
    MasterPoint(y=17.0 / 40.0, theta=13.0 / 8.0),
    MasterPoint(y=29.0 / 50.0, theta=12.0 / 5.0),
]

REFERENCE_STRENGTHENED_CUT = CutRecord(
    name='strengthened_2',
    intercept=11.0 / 2.0,
    slope=5.0,
)
REFERENCE_STRENGTHENED_MASTER = MasterPoint(y=1.0 / 8.0, theta=49.0 / 8.0)

REFERENCE_LAGRANGIAN_CUT = CutRecord(
    name='lagrangian_2',
    intercept=8.0,
    slope=5.0 / 2.0,
)
REFERENCE_LAGRANGIAN_MASTER = MasterPoint(y=0.0, theta=8.0)


class SimpleInnerMinimization(ModelInnerMinimizationProblem):
    """Inner minimization for the textbook example."""

    def build_sub_vars_and_constraints(self):
        self.var['x'] = {0: self.add_var(domain=pyo.NonNegativeReals, name='x')}
        self.var['y'] = {0: self.add_var(domain=pyo.Binary, name='y')}

        x = self.var['x'][0]
        y = self.var['y'][0]

        self.add_constr(x + 15 * y >= 8, name='c1')
        self.add_constr(3 * x + 10 * y >= 13, name='c2')
        self.add_constr(x + 10 * y >= 7, name='c3')
        self.add_constr(2 * x - 10 * y >= -1, name='c4')
        self.add_constr(2 * x - 70 * y >= -49, name='c5')

    def get_sub_objective_expr(self):
        return self.var['x'][0]


# ============================================================
# Small helpers
# ============================================================

def _assert_close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    if math.isnan(expected):
        return
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f'{label}: expected {expected:.10f}, got {actual:.10f}')


def _format_cut(cut: CutRecord) -> str:
    sign = '+' if cut.slope >= 0 else '-'
    return f"theta >= {cut.intercept:.6f} {sign} {abs(cut.slope):.6f} * y"


def _make_cut_from_ctx_z_lambda(name: str, c_tx: float, z_y: float, lag_multiplier: float) -> CutRecord:
    return CutRecord(
        name=name,
        intercept=c_tx - lag_multiplier * z_y,
        slope=lag_multiplier,
    )


# ============================================================
# Core solves for the toy problem
# ============================================================

def solve_relaxed_master(cuts: List[CutRecord]) -> MasterPoint:
    """Solve min theta with 0 <= y <= 1 and the provided cuts."""
    master = ModelBase(model_name='toy_master')
    master.var['y'] = {0: master.add_var(domain=pyo.UnitInterval, name='y')}
    master.var['theta'] = {0: master.add_var(domain=pyo.Reals, lb=-1.0e6, name='theta')}

    for cut in cuts:
        master.add_constr(
            master.var['theta'][0] >= cut.intercept + cut.slope * master.var['y'][0],
            name=cut.name,
        )

    master.set_objective(master.var['theta'][0], sense=pyo.minimize)
    master.solve()
    result = master.get_result(['y', 'theta'])
    return MasterPoint(y=result['y'][0], theta=result['theta'][0])


def solve_lp_subproblem(fixed_y: float) -> Tuple[float, float]:
    """
    Solve the LP subproblem at y = fixed_y and return:
        (subproblem objective value, dual multiplier of z = fixed_y).
    """
    sub = ModelBase(model_name='toy_sub_lp')
    sub.var['x'] = {0: sub.add_var(domain=pyo.NonNegativeReals, name='x')}
    sub.var['y'] = {0: sub.add_var(domain=pyo.UnitInterval, name='y')}

    x = sub.var['x'][0]
    y = sub.var['y'][0]

    sub.add_constr(x + 15 * y >= 8, name='c1')
    sub.add_constr(3 * x + 10 * y >= 13, name='c2')
    sub.add_constr(x + 10 * y >= 7, name='c3')
    sub.add_constr(2 * x - 10 * y >= -1, name='c4')
    sub.add_constr(2 * x - 70 * y >= -49, name='c5')
    sub.add_constr(y == fixed_y, name=CONSTR_NAME)

    sub.set_objective(x, sense=pyo.minimize)

    sub.model_relax = sub.model.clone()
    pyo.TransformationFactory('core.relax_integer_vars').apply_to(sub.model_relax)
    sub.model_relax.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = SolverFactory(SOLVER_NAME)
    solver.solve(sub.model_relax, tee=False, load_solutions=True)

    lp_obj = pyo.value(sub.model_relax.find_component('_obj'))
    dual_val = sub.model_relax.dual[sub.get_constr_item_in_relax_by_name(CONSTR_NAME)]
    return lp_obj, dual_val


def generate_classical_cut(cut_name: str, fixed_y: float) -> Tuple[CutRecord, float]:
    """Generate the classical Benders cut from the LP subproblem at fixed_y."""
    lp_obj, dual_val = solve_lp_subproblem(fixed_y=fixed_y)
    cut = CutRecord(
        name=cut_name,
        intercept=lp_obj - dual_val * fixed_y,
        slope=dual_val,
    )
    return cut, lp_obj


def generate_strengthened_cut(cut_name: str, fixed_y: float, dual_val: float) -> CutRecord:
    """Generate the strengthened cut with the implemented inner minimization model."""
    inner_model = SimpleInnerMinimization(
        model_name='toy_strengthened_inner',
        main_result={'y': {0: fixed_y}},
        constr_dual_info={CONSTR_NAME: dual_val},
        constr_var_map=CONSTR_VAR_MAP,
    )
    inner_model.build_model()
    inner_model.solve()

    obtained_main_sol = inner_model.get_result(['y'])
    inner_model.cal_detailed_obj()
    c_tx = inner_model.obj_term_value[ObjName.SUB_OBJ_FUNCTION]
    z_y = obtained_main_sol['y'][0]
    return _make_cut_from_ctx_z_lambda(
        name=cut_name,
        c_tx=c_tx,
        z_y=z_y,
        lag_multiplier=dual_val,
    )


def generate_lagrangian_cut_with_heuristic(
    cut_name: str,
    fixed_y: float,
    dual_val: float,
    max_ite_num: int = 10,
) -> CutRecord:
    """Generate the Lagrangian cut with the implemented regularized heuristic."""
    inner_model = SimpleInnerMinimization(
        model_name='toy_lagrangian_inner',
        main_result={'y': {0: fixed_y}},
        constr_dual_info={CONSTR_NAME: dual_val},
        constr_var_map=CONSTR_VAR_MAP,
    )
    inner_model.build_model()
    inner_model.solve()

    obtained_main_sol = inner_model.get_result(['y'])
    inner_model.cal_detailed_obj()
    c_tx = inner_model.obj_term_value[ObjName.SUB_OBJ_FUNCTION]

    curr_lag_multiplier = {CONSTR_NAME: dual_val}

    best_L = c_tx + dual_val * (fixed_y - obtained_main_sol['y'][0])
    potential_c_tx = c_tx
    potential_z = {k: dict(v) for k, v in obtained_main_sol.items()}
    potential_lambda = curr_lag_multiplier.copy()

    lag_mult_model = ModelLagrangianMultiplierHeuristic(
        model_name='toy_lagrangian_multiplier',
        main_result={'y': {0: fixed_y}},
        constr_dual_info={CONSTR_NAME: dual_val},
        constr_var_map=CONSTR_VAR_MAP,
    )
    lag_mult_model.build_ini_model(
        ini_sub_obj_value=c_tx,
        ini_main_sol=obtained_main_sol,
        ini_multiplier=curr_lag_multiplier,
        ini_stabilization_param=0.01,
    )

    for ite_num in range(1, max_ite_num + 1):
        lag_mult_model.add_constr_for_regularized_model(
            sub_obj_value=c_tx,
            main_sol=obtained_main_sol,
        )
        lag_mult_model.set_objective_function(
            stabilization_param=0.01 / (1 + ite_num),
            prev_multiplier=curr_lag_multiplier,
        )
        lag_mult_model.solve()

        eta_ub = lag_mult_model.obtain_lift_value()

        curr_lag_multiplier = lag_mult_model.get_result(
            var_name_list=[VarName.LAG_MULTIPLIER]
        )[VarName.LAG_MULTIPLIER]

        inner_model.main_result = {'y': {0: fixed_y}}
        inner_model.set_objective_function(lag_multiplier=curr_lag_multiplier)
        inner_model.solve()

        obtained_main_sol = inner_model.get_result(['y'])
        inner_model.cal_detailed_obj()
        c_tx = inner_model.obj_term_value[ObjName.SUB_OBJ_FUNCTION]

        actual_L = c_tx + curr_lag_multiplier[CONSTR_NAME] * (fixed_y - obtained_main_sol['y'][0])

        if actual_L > best_L + TOL:
            best_L = actual_L
            potential_c_tx = c_tx
            potential_z = {k: dict(v) for k, v in obtained_main_sol.items()}
            potential_lambda = curr_lag_multiplier.copy()

        if eta_ub - best_L <= 0.001 * math.fabs(best_L):
            break

    return _make_cut_from_ctx_z_lambda(
        name=cut_name,
        c_tx=potential_c_tx,
        z_y=potential_z['y'][0],
        lag_multiplier=potential_lambda[CONSTR_NAME],
    )


# ============================================================
# Reference checks
# ============================================================

def check_classical_reference() -> None:
    print('\n' + '=' * 70)
    print('Classical Benders cuts on the toy example')
    print('=' * 70)

    generated_cuts: List[CutRecord] = []
    current_master_y = REFERENCE_CLASSICAL_MASTER_POINTS[0].y

    for idx, expected_cut in enumerate(REFERENCE_CLASSICAL_CUTS):
        cut, _ = generate_classical_cut(
            cut_name=f'classical_{idx}',
            fixed_y=current_master_y,
        )

        _assert_close(cut.intercept, expected_cut.intercept, f'classical cut {idx} intercept')
        _assert_close(cut.slope, expected_cut.slope, f'classical cut {idx} slope')
        print(f'Iteration {idx}: PASS -> {_format_cut(cut)}')
        generated_cuts.append(cut)

        master_point = solve_relaxed_master(generated_cuts)
        expected_master = REFERENCE_CLASSICAL_MASTER_POINTS[idx + 1]
        _assert_close(master_point.y, expected_master.y, f'classical master y after cut {idx}')
        _assert_close(master_point.theta, expected_master.theta, f'classical master theta after cut {idx}')
        print(
            f'  Master after cut {idx}: y={master_point.y:.6f}, '
            f'theta={master_point.theta:.6f}'
        )
        current_master_y = master_point.y


def check_strengthened_reference() -> None:
    print('\n' + '=' * 70)
    print('Strengthened Benders cut on the toy example')
    print('=' * 70)

    common_cuts = REFERENCE_CLASSICAL_CUTS[:2]
    common_master = solve_relaxed_master(common_cuts)
    _assert_close(common_master.y, 13.0 / 20.0, 'common master y before strengthened cut')
    _assert_close(common_master.theta, -7.0 / 4.0, 'common master theta before strengthened cut')

    _, dual_val = solve_lp_subproblem(fixed_y=common_master.y)
    strengthened_cut = generate_strengthened_cut(
        cut_name='strengthened_2',
        fixed_y=common_master.y,
        dual_val=dual_val,
    )

    _assert_close(strengthened_cut.intercept, REFERENCE_STRENGTHENED_CUT.intercept, 'strengthened cut intercept')
    _assert_close(strengthened_cut.slope, REFERENCE_STRENGTHENED_CUT.slope, 'strengthened cut slope')
    print(f'PASS -> {_format_cut(strengthened_cut)}')

    master_after_strengthened = solve_relaxed_master(common_cuts + [strengthened_cut])
    _assert_close(master_after_strengthened.y, REFERENCE_STRENGTHENED_MASTER.y, 'strengthened master y')
    _assert_close(master_after_strengthened.theta, REFERENCE_STRENGTHENED_MASTER.theta, 'strengthened master theta')
    print(
        f'  Master after strengthened cut: y={master_after_strengthened.y:.6f}, '
        f'theta={master_after_strengthened.theta:.6f}'
    )


def check_lagrangian_reference() -> List[str]:
    print('\n' + '=' * 70)
    print('Lagrangian cut on the toy example')
    print('=' * 70)

    common_cuts = REFERENCE_CLASSICAL_CUTS[:2]
    common_master = solve_relaxed_master(common_cuts)
    _, dual_val = solve_lp_subproblem(fixed_y=common_master.y)

    lagrangian_cut = generate_lagrangian_cut_with_heuristic(
        cut_name='lagrangian_2',
        fixed_y=common_master.y,
        dual_val=dual_val,
        max_ite_num=10,
    )

    issues: List[str] = []
    if not math.isclose(
        lagrangian_cut.intercept,
        REFERENCE_LAGRANGIAN_CUT.intercept,
        rel_tol=TOL,
        abs_tol=TOL,
    ) or not math.isclose(
        lagrangian_cut.slope,
        REFERENCE_LAGRANGIAN_CUT.slope,
        rel_tol=TOL,
        abs_tol=TOL,
    ):
        issues.append(
            'Implemented Lagrangian heuristic does not match the reference cut: '
            f'generated `{_format_cut(lagrangian_cut)}`, expected `{_format_cut(REFERENCE_LAGRANGIAN_CUT)}`.'
        )
        print('REPORT -> ' + issues[-1])
        return issues

    print(f'PASS -> {_format_cut(lagrangian_cut)}')
    master_after_lagrangian = solve_relaxed_master(common_cuts + [lagrangian_cut])
    if not math.isclose(master_after_lagrangian.y, REFERENCE_LAGRANGIAN_MASTER.y, rel_tol=TOL, abs_tol=TOL):
        issues.append(
            'Implemented Lagrangian cut matches the cut coefficients but not the reference master y: '
            f'generated y={master_after_lagrangian.y:.10f}, '
            f'expected y={REFERENCE_LAGRANGIAN_MASTER.y:.10f}.'
        )
    if not math.isclose(master_after_lagrangian.theta, REFERENCE_LAGRANGIAN_MASTER.theta, rel_tol=TOL, abs_tol=TOL):
        issues.append(
            'Implemented Lagrangian cut matches the cut coefficients but not the reference master theta: '
            f'generated theta={master_after_lagrangian.theta:.10f}, '
            f'expected theta={REFERENCE_LAGRANGIAN_MASTER.theta:.10f}.'
        )

    if issues:
        for issue in issues:
            print('REPORT -> ' + issue)
    else:
        print(
            f'  Master after Lagrangian cut: y={master_after_lagrangian.y:.6f}, '
            f'theta={master_after_lagrangian.theta:.6f}'
        )

    return issues


def run_test() -> bool:
    print('=' * 70)
    print('Toy-example cut verification against bdd_toy_example_reference.md')
    print(f'Solver: {SOLVER_NAME}')
    print('=' * 70)

    check_classical_reference()
    check_strengthened_reference()
    lagrangian_issues = check_lagrangian_reference()

    print('\n' + '=' * 70)
    if lagrangian_issues:
        print('Classical and strengthened checks passed.')
        print('Lagrangian check reported a mismatch with the current implementation.')
        print('=' * 70)
        return False

    print('All three cut checks passed.')
    print('=' * 70)
    return True


if __name__ == '__main__':
    success = run_test()
    raise SystemExit(0 if success else 1)

