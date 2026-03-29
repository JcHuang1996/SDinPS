# -*- coding: utf-8 -*-
# @Time     : 2026/03/22
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
Logic Check
-----------
This script checks the current CGLP implementation with the toy examples in
`CGLP_node_example.md` / `CGLP_test_example.md`.

It performs two kinds of checks:
1. exact set-recursion checks for the paper-style 3-variable example,
2. cut-generation checks for Examples A / B / C.

For the cut tests, the generated cut is first checked functionally:
- tight at the current point,
- valid on all exact points,
- valid on unseen points under the lower bound L = 0.

If the generated cut does not match the reference coefficients exactly, this
script checks the reference cut as well and reports the mismatch instead of
assuming the reference is correct.
"""

import itertools
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory, check_available_solvers

    from algo.algo_simple_tools import prefix_to_w_vector
    from model.model_base import ModelBase
    from model.model_cglp import ModelCGLP
except ModuleNotFoundError as exc:
    raise SystemExit(
        'This script requires a Python environment with `pyomo` installed. '
        'In this project, `conda run -n DailyOpt python unit_test/test_scripts/cglp_simple_test.py` works.'
    ) from exc


PREFERRED_SOLVER_ORDER = ('gurobi_direct', 'appsi_highs', 'highs', 'cbc')
TOL = 1e-6
LOWER_BOUND_L = 0.0


@dataclass(frozen=True)
class CutRecord:
    intercept: float
    coef: Tuple[float, ...]


@dataclass(frozen=True)
class CutExample:
    name: str
    dim: int
    exact_points: List[Tuple[int, ...]]
    q_value_table: Dict[Tuple[int, ...], float]
    target_point: Tuple[int, ...]
    expected_hat_v: Dict[int, set]
    expected_w: Dict[int, set]
    reference_cut: CutRecord


SET_EXAMPLE_V = [
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 1),
]

SET_EXAMPLE_EXPECTED_V = {
    1: {(0,), (1,)},
    2: {(0, 1), (1, 0)},
    3: {(0, 1, 0), (0, 1, 1), (1, 0, 1)},
}

SET_EXAMPLE_EXPECTED_HAT_V = {
    1: set(),
    2: {(0, 0), (1, 1)},
    3: {(1, 0, 0)},
}

SET_EXAMPLE_EXPECTED_W = {
    1: set(),
    2: {(0, 0, 0), (1, 1, 0)},
    3: {(1, 0, 0)},
}


CUT_EXAMPLES = [
    CutExample(
        name='Example A',
        dim=2,
        exact_points=[(1, 0), (1, 1)],
        q_value_table={
            (0, 0): 0.0,
            (0, 1): 0.0,
            (1, 0): 2.0,
            (1, 1): 4.0,
        },
        target_point=(1, 1),
        expected_hat_v={
            1: {(0,)},
            2: set(),
        },
        expected_w={
            1: {(0, 0)},
            2: set(),
        },
        reference_cut=CutRecord(intercept=-2.0, coef=(4.0, 2.0)),
    ),
    CutExample(
        name='Example B',
        dim=2,
        exact_points=[(0, 1), (1, 1)],
        q_value_table={
            (0, 0): 0.0,
            (1, 0): 0.0,
            (0, 1): 3.0,
            (1, 1): 5.0,
        },
        target_point=(1, 1),
        expected_hat_v={
            1: set(),
            2: {(0, 0), (1, 0)},
        },
        expected_w={
            1: set(),
            2: {(0, 0), (1, 0)},
        },
        reference_cut=CutRecord(intercept=-2.0, coef=(2.0, 5.0)),
    ),
    CutExample(
        name='Example C',
        dim=3,
        exact_points=[(0, 1, 0), (0, 1, 1), (1, 0, 1)],
        q_value_table={
            (0, 0, 0): 0.0,
            (0, 0, 1): 0.0,
            (0, 1, 0): 1.0,
            (0, 1, 1): 1.0,
            (1, 0, 0): 0.0,
            (1, 0, 1): 4.0,
            (1, 1, 0): 1.0,
            (1, 1, 1): 5.0,
        },
        target_point=(1, 0, 1),
        expected_hat_v={
            1: set(),
            2: {(0, 0), (1, 1)},
            3: {(1, 0, 0)},
        },
        expected_w={
            1: set(),
            2: {(0, 0, 0), (1, 1, 0)},
            3: {(1, 0, 0)},
        },
        reference_cut=CutRecord(intercept=-4.0, coef=(4.0, -4.0, 4.0)),
    ),
]


def _all_binary_points(dim: int) -> List[Tuple[int, ...]]:
    return list(itertools.product((0, 1), repeat=dim))


def _assert_set_equal(actual: Iterable[Tuple[int, ...]], expected: Iterable[Tuple[int, ...]], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        raise AssertionError(f'{label}: expected {sorted(expected_set)}, got {sorted(actual_set)}')


def _assert_close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f'{label}: expected {expected:.10f}, got {actual:.10f}')


def _evaluate_cut(cut: CutRecord, x_point: Sequence[int]) -> float:
    return float(cut.intercept + sum(cut.coef[idx] * x_point[idx] for idx in range(len(cut.coef))))


def _format_cut(cut: CutRecord) -> str:
    terms = [f'{cut.intercept:.6f}']
    for idx, coef in enumerate(cut.coef, start=1):
        sign = '+' if coef >= 0 else '-'
        terms.append(f'{sign} {abs(coef):.6f}*x_{idx}')
    return 'theta >= ' + ' '.join(terms)


def _extract_generated_cut(cglp_model: ModelCGLP) -> CutRecord:
    cut_row = cglp_model.build_lower_bound_row(cut_name='generated_cut')
    coef = tuple(cut_row['x_coef'][x_key] for x_key in cglp_model.x_var_keys)
    intercept = -cut_row['rhs']
    return CutRecord(intercept=intercept, coef=coef)


def _check_cut_functionality(example: CutExample, cut: CutRecord, label: str) -> List[str]:
    issues = []

    target_value = _evaluate_cut(cut=cut, x_point=example.target_point)
    try:
        _assert_close(
            actual=target_value,
            expected=example.q_value_table[example.target_point],
            label=f'{example.name} {label} tightness at target',
        )
    except AssertionError as exc:
        issues.append(str(exc))

    exact_point_set = set(example.exact_points)
    for x_point, q_value in sorted(example.q_value_table.items()):
        cut_value = _evaluate_cut(cut=cut, x_point=x_point)
        upper_value = q_value if x_point in exact_point_set else LOWER_BOUND_L
        if cut_value > upper_value + TOL:
            issues.append(
                f'{example.name} {label} invalid at {x_point}: cut value {cut_value:.10f}, upper bound {upper_value:.10f}'
            )

    return issues


def _create_dummy_main_model(dim: int) -> Tuple[ModelBase, List[Tuple[str, int]]]:
    main_model = ModelBase(model_name=f'cglp_master_{dim}')
    main_model.var['x'] = {
        idx: main_model.add_var(domain=pyo.Binary, name=f'x_{idx}')
        for idx in range(1, dim + 1)
    }
    main_model.var['theta'] = {
        0: main_model.add_var(domain=pyo.Reals, lb=LOWER_BOUND_L, name='theta_0')
    }
    x_var_keys = [('x', idx) for idx in range(1, dim + 1)]
    return main_model, x_var_keys


def _resolve_solver_name() -> str:
    requested_solver = os.environ.get('CGLP_SOLVER')
    if requested_solver is not None:
        return requested_solver

    available_solver_list = check_available_solvers(*PREFERRED_SOLVER_ORDER)
    if not available_solver_list:
        raise RuntimeError(
            'No supported LP solver is available. Set CGLP_SOLVER or install one of '
            f'{PREFERRED_SOLVER_ORDER}.'
        )
    return available_solver_list[0]


def _build_cglp_for_example(example: CutExample) -> ModelCGLP:
    main_model, x_var_keys = _create_dummy_main_model(dim=example.dim)

    cglp_model = ModelCGLP(
        main_model=main_model,
        x_var_keys=x_var_keys,
        z_var_keys=[],
        structural_constr_names=[],
        theta_keys=[0],
        lower_bound_rows=[{'name': 'lb0', 'x_coef': {}, 'rhs': 0.0}],
        model_name=f"cglp_{example.name.replace(' ', '_').lower()}",
        exact_point_data=[
            (x_point, example.q_value_table[x_point])
            for x_point in example.exact_points
        ],
    )
    cglp_model.solver = SolverFactory(_resolve_solver_name())
    cglp_model.build_model()

    return cglp_model


def _solve_example_cut(cglp_model: ModelCGLP) -> CutRecord:
    cglp_model.solve_and_build_group_cut()
    return _extract_generated_cut(cglp_model=cglp_model)


def check_set_processing_reference() -> None:
    print('\n' + '=' * 70)
    print('CGLP set-processing check from the paper-style toy example')
    print('=' * 70)

    set_example = CutExample(
        name='Set Example',
        dim=3,
        exact_points=list(SET_EXAMPLE_V),
        q_value_table={point: 0.0 for point in _all_binary_points(3)},
        target_point=(1, 0, 1),
        expected_hat_v=SET_EXAMPLE_EXPECTED_HAT_V,
        expected_w=SET_EXAMPLE_EXPECTED_W,
        reference_cut=CutRecord(intercept=0.0, coef=(0.0, 0.0, 0.0)),
    )

    cglp_model = _build_cglp_for_example(example=set_example)

    for prefix_len, expected_v_i in SET_EXAMPLE_EXPECTED_V.items():
        _assert_set_equal(
            actual=cglp_model.v_prefix_sets[prefix_len],
            expected=expected_v_i,
            label=f'V^{prefix_len}',
        )

    for prefix_len, expected_hat_v_i in SET_EXAMPLE_EXPECTED_HAT_V.items():
        _assert_set_equal(
            actual=cglp_model.hat_v_prefix_sets[prefix_len],
            expected=expected_hat_v_i,
            label=f'hat_V^{prefix_len}',
        )

    for prefix_len, expected_w_i in SET_EXAMPLE_EXPECTED_W.items():
        actual_w_i = {
            prefix_to_w_vector(prefix_tuple=prefix_tuple, total_dim=3)
            for prefix_tuple in cglp_model.active_w_row_name_dict[prefix_len].keys()
        }
        _assert_set_equal(actual=actual_w_i, expected=expected_w_i, label=f'W_{prefix_len}')

    v2_complement = {(0, 0), (1, 1)}
    lifted_term = {(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)}
    hat_v3 = {(1, 0, 0)}
    final_complement = {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)}

    _assert_set_equal(actual=v2_complement, expected={(0, 0), (1, 1)}, label='complement of V^2')
    _assert_set_equal(actual=lifted_term | hat_v3, expected=final_complement, label='recursion identity')

    print('PASS -> set recursion, projected sets, forbidden-prefix sets, and W-sets all match the note.')


def check_cut_examples() -> None:
    print('\n' + '=' * 70)
    print('CGLP cut-generation checks')
    print('=' * 70)

    overall_failures = []
    overall_notes = []

    for example in CUT_EXAMPLES:
        print(f'\n{example.name}')
        print('-' * len(example.name))

        cglp_model = _build_cglp_for_example(example=example)

        for prefix_len, expected_hat_v_i in example.expected_hat_v.items():
            _assert_set_equal(
                actual=cglp_model.hat_v_prefix_sets[prefix_len],
                expected=expected_hat_v_i,
                label=f'{example.name} hat_V^{prefix_len}',
            )

        for prefix_len, expected_w_i in example.expected_w.items():
            actual_w_i = {
                prefix_to_w_vector(prefix_tuple=prefix_tuple, total_dim=example.dim)
                for prefix_tuple in cglp_model.active_w_row_name_dict[prefix_len].keys()
            }
            _assert_set_equal(
                actual=actual_w_i,
                expected=expected_w_i,
                label=f'{example.name} W_{prefix_len}',
            )

        if cglp_model.current_tight_point != example.target_point:
            raise AssertionError(
                f'{example.name} target equality row mismatch: expected {example.target_point}, got {cglp_model.current_tight_point}'
            )

        generated_cut = _solve_example_cut(cglp_model=cglp_model)

        generated_issues = _check_cut_functionality(example=example, cut=generated_cut, label='generated cut')
        reference_issues = _check_cut_functionality(example=example, cut=example.reference_cut, label='reference cut')

        exact_match = (
            math.isclose(generated_cut.intercept, example.reference_cut.intercept, rel_tol=TOL, abs_tol=TOL)
            and all(
                math.isclose(generated_cut.coef[idx], example.reference_cut.coef[idx], rel_tol=TOL, abs_tol=TOL)
                for idx in range(example.dim)
            )
        )

        print(f'Generated cut: {_format_cut(generated_cut)}')
        print(f'Reference cut: {_format_cut(example.reference_cut)}')

        if generated_issues:
            overall_failures.extend(generated_issues)
            print('FAIL -> generated cut is not functionally valid.')
            continue

        if exact_match:
            print('PASS -> generated cut matches the reference coefficients exactly.')
            continue

        if reference_issues:
            overall_notes.extend(reference_issues)
            print('PASS -> generated cut is functionally valid, but the reference cut failed the same checks.')
            continue

        overall_notes.append(
            f'{example.name}: generated cut is valid but differs from the reference facet. '
            f'Generated `{_format_cut(generated_cut)}` vs reference `{_format_cut(example.reference_cut)}`.'
        )
        print('PASS -> generated cut is functionally valid, but it does not match the reference coefficients exactly.')

    if overall_notes:
        print('\nNotes')
        print('-' * 5)
        for note in overall_notes:
            print(note)

    if overall_failures:
        print('\nFailures')
        print('-' * 8)
        for failure in overall_failures:
            print(failure)
        raise AssertionError('One or more generated CGLP cuts failed the functional checks.')


def main() -> None:
    check_set_processing_reference()
    check_cut_examples()
    print('\nAll required CGLP checks completed.')


if __name__ == '__main__':
    main()
