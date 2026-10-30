# -*- coding: utf-8 -*-
# @Time     : 2026/09/17
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
pytest front end for the P0 acceptance criteria (docs/development_record.md).

The work is done by ``decomp_validity_harness``; this file only states the criteria as tests so that
``pytest unit_test/test_scripts`` reports them. Every test here needs a solver and is skipped
cleanly without one.

    pytest unit_test/test_scripts/test_decomp_validity.py --solver appsi_highs -v
"""

import pytest

from unit_test.test_scripts.decomp_validity_harness import run_validity_harness
from util.names import BoundModeName, CutTypeName


pytestmark = pytest.mark.requires_solver


def _assert_clean(report):
    assert not report['failures'], '\n'.join(str(item) for item in report['failures'])


@pytest.mark.parametrize('family', [
    pytest.param(family, marks=pytest.mark.requires_quadratic_solver)
    if family == CutTypeName.LAGRANGIAN else family
    for family in CutTypeName.ALL_TYPES
])
def test_cut_family_is_valid(family, solver_name, micro_bench_instance, local_file_path, tmp_path):
    """
    B3, one cut family at a time.

    Every cut the family produces must satisfy ``sum_s c_s Q_s(x) >= r(x)`` at EVERY first-stage
    design. A failure here means the family can cut off the true optimum.
    """
    report = run_validity_harness(
        instance_name=micro_bench_instance,
        local_file_path=local_file_path,
        solver_name=solver_name,
        families=[family],
        max_iterations=3,
        report_path=str(tmp_path / 'validity.json'),
    )
    _assert_clean(report)


@pytest.mark.requires_quadratic_solver
def test_all_families_together(solver_name, micro_bench_instance, local_file_path, tmp_path):
    """B2 + B3 + B4 with every family enabled at once."""
    report = run_validity_harness(
        instance_name=micro_bench_instance,
        local_file_path=local_file_path,
        solver_name=solver_name,
        families=list(CutTypeName.ALL_TYPES),
        max_iterations=4,
        report_path=str(tmp_path / 'validity.json'),
    )
    _assert_clean(report)


@pytest.mark.requires_quadratic_solver
def test_validity_survives_a_starved_inner_minimization(solver_name, micro_bench_instance, local_file_path, tmp_path):
    """
    P1 acceptance criterion.

    With the inner Lagrangian MILP deliberately starved (1 s, 50 % gap) the cuts must get WEAKER,
    never invalid: the anchor is a dual bound, so validity is tolerance-independent. This is the
    test that fails on the pre-P1 code.
    """
    report = run_validity_harness(
        instance_name=micro_bench_instance,
        local_file_path=local_file_path,
        solver_name=solver_name,
        families=[CutTypeName.STRENGTHEN_BENDERS, CutTypeName.LAGRANGIAN],
        max_iterations=3,
        inner_min_time_limit=1,
        inner_min_mip_gap=0.5,
        subproblem_exact_mip_gap=0.5,
        report_path=str(tmp_path / 'validity.json'),
    )
    _assert_clean(report)


@pytest.mark.parametrize('added_obj_term_weight', [0.0, 2.0])
def test_bound_validity_under_penalty_weight(
    added_obj_term_weight, solver_name, micro_bench_instance, local_file_path, tmp_path
):
    """
    P2 acceptance criterion.

    ``LB_k <= z* + eps`` must hold at w = 0 and at w = 2. In RESOLVE_AT_ZERO_WEIGHT mode the bound
    comes from a second master solve at w = 0; the pre-P2 code reports the penalized minimizer's
    value and fails this at w = 2.
    """
    report = run_validity_harness(
        instance_name=micro_bench_instance,
        local_file_path=local_file_path,
        solver_name=solver_name,
        families=[CutTypeName.BENDERS],
        max_iterations=3,
        added_obj_term_weight=added_obj_term_weight,
        bound_mode=BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
        report_path=str(tmp_path / 'validity.json'),
    )
    _assert_clean(report)


def test_non_singleton_group_with_unequal_probabilities(solver_name, local_file_path, tmp_path):
    """
    P3 acceptance criterion.

    m6 has two scenarios with probabilities 0.25 / 0.75. Put them in ONE group: the cut
    ``sum_{s in G} P_s theta_s >= ...`` must still bound the master correctly and the decomposition
    must reproduce z*. The pre-P3 code mixes an unweighted group objective with a weighted master
    term and fails here.
    """
    report = run_validity_harness(
        instance_name='m6_two_scenario_hedging',
        local_file_path=local_file_path,
        solver_name=solver_name,
        families=[CutTypeName.BENDERS, CutTypeName.INTEGER_OPT],
        max_iterations=20,
        scenario_groups='s_1+s_2',
        require_convergence=True,
        report_path=str(tmp_path / 'validity.json'),
    )
    _assert_clean(report)
