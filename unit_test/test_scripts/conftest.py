# -*- coding: utf-8 -*-
# @Time     : 2026/09/17
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
pytest configuration for the SDinPS test scripts (docs/development_record.md, item B5).

Two kinds of test live side by side here:

* STRUCTURAL tests build models and inspect variables, constraints, bounds and data contracts.
  They need pyomo but no solver, so they run anywhere.
* NUMERICAL tests solve something. They are marked ``@pytest.mark.requires_solver`` and are skipped
  cleanly -- not failed -- when no solver is installed.

Run:

    pytest unit_test/test_scripts                       # structural only if no solver is present
    pytest unit_test/test_scripts --solver appsi_highs  # also run the numerical tests on HiGHS
    pytest unit_test/test_scripts -m requires_solver    # only the numerical tests

``--solver`` also selects which solver the numerical tests use; the default is read from the
SDINPS_TEST_SOLVER environment variable, falling back to gurobi_direct.
Tests marked requires_quadratic_solver probe the selected interface and skip explicitly when its
quadratic objective support is unavailable.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_SOLVER_NAME = os.environ.get('SDINPS_TEST_SOLVER', 'gurobi_direct')

#: Solvers tried in order when --solver is not given and the default is unavailable.
CANDIDATE_SOLVER_NAMES = (
    'gurobi_direct',
    'gurobi',
    'appsi_highs',
    'highs',
    'cbc',
    'glpk',
)


def pytest_addoption(parser):
    parser.addoption(
        '--solver',
        action='store',
        default=None,
        help=(
            'Pyomo solver used by tests marked requires_solver '
            f'(default: {DEFAULT_SOLVER_NAME}, or the first available of {CANDIDATE_SOLVER_NAMES}).'
        ),
    )
    parser.addoption(
        '--micro-bench-instance',
        action='store',
        default='m6_two_scenario_hedging',
        help='micro_bench instance used by the numerical tests.',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'requires_solver: test needs an installed MILP solver; skipped when none is available.',
    )
    config.addinivalue_line('markers', 'requires_quadratic_solver: test also needs a quadratic objective solver.')


def _solver_is_available(solver_name):
    try:
        from pyomo.opt import SolverFactory
    except ModuleNotFoundError:
        return False

    try:
        solver = SolverFactory(solver_name)
    except Exception:
        return False

    if solver is None:
        return False

    try:
        return bool(solver.available(exception_flag=False))
    except Exception:
        return False


def resolve_solver_name(config):
    """Return a usable solver name, or None when nothing is installed."""
    requested = config.getoption('--solver')
    if requested is not None:
        return requested if _solver_is_available(requested) else None

    candidates = (DEFAULT_SOLVER_NAME,) + CANDIDATE_SOLVER_NAMES
    for solver_name in candidates:
        if _solver_is_available(solver_name):
            return solver_name
    return None


@pytest.fixture(scope='session')
def solver_name(pytestconfig):
    """The solver the numerical tests should use; skips the test when none is available."""
    resolved = resolve_solver_name(pytestconfig)
    if resolved is None:
        pytest.skip('no MILP solver available in this environment')
    return resolved


@pytest.fixture(scope='session')
def micro_bench_instance(pytestconfig):
    return pytestconfig.getoption('--micro-bench-instance')


@pytest.fixture(scope='session')
def local_file_path():
    return os.path.join(PROJECT_ROOT, 'unit_test', 'test_local_csv_file')


@pytest.fixture(scope='session')
def quadratic_solver_name(solver_name):
    """Probe the selected interface, since an installed MILP solver need not support the QP."""
    import pyomo.environ as pyo
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(-1, 1))
    model.obj = pyo.Objective(expr=model.x ** 2)
    try:
        results = pyo.SolverFactory(solver_name).solve(model)
    except Exception as exc:
        pytest.skip(f'{solver_name} cannot solve the multiplier QP ({type(exc).__name__}).')
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        pytest.skip(f'{solver_name} did not solve the quadratic capability probe to optimality.')
    return solver_name


@pytest.fixture(autouse=True)
def require_quadratic_capability(request):
    if request.node.get_closest_marker('requires_quadratic_solver') is not None:
        request.getfixturevalue('quadratic_solver_name')


def pytest_collection_modifyitems(config, items):
    """Skip solver-dependent tests up front when no solver can be found."""
    if resolve_solver_name(config) is not None:
        return

    requested = config.getoption('--solver')
    reason = (
        f'solver {requested!r} is not available'
        if requested is not None
        else 'no MILP solver available in this environment'
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if 'requires_solver' in item.keywords:
            item.add_marker(skip_marker)
