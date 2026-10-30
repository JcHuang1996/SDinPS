# -*- coding: utf-8 -*-
# @Time     : 2026/03/24
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from algo.algo_simple_tools import CutRepetitionTracker, normalize_affine_lower_bound_row
from algo.interfaces import (
    IterationContext,
    MainSolution,
    NoOpSolutionGenerator,
    StaticSolutionGenerator,
)
from algo.runner import PhasePolicy
from util.names import BranchName, CutTypeName, PhaseName, VarName
from util.tools import load_main_stage_solution_json


REFERENCE_SOLUTION_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    '..',
    'test_local_csv_file',
    'function_test_fixed_rated_p',
    'solutions',
    's_1_s_3_s_5.json',
))


def _assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_load_main_stage_solution_json() -> None:
    solution = load_main_stage_solution_json(REFERENCE_SOLUTION_PATH)
    _assert_true(set(solution.keys()) == {VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE}, "Unexpected top-level keys")
    _assert_true(solution[VarName.DG_INSTALL]["node_10"] == 1, "xg loader failed for node_10")
    _assert_true(solution[VarName.LINE_HARDEN][("node_12", "node_13")] == 1, "xl tuple-key loader failed")
    _assert_true(solution[VarName.DG_INSTALL_TYPE][("node_10", "T3")] == 1, "xg_type tuple-key loader failed")


def test_cut_repetition_tracker_cross_family_preview() -> None:
    tracker = CutRepetitionTracker()
    row = normalize_affine_lower_bound_row(
        cut_name="row_0",
        constant_term=5.0,
        var_coef_dict={VarName.DG_INSTALL: {"node_10": 2.0}},
    )

    tracker.record_by_row(
        cut_type="benders",
        ite_name=0,
        cut_name="bd_0",
        row=row,
    )

    default_peek = tracker.peek_by_row(
        cut_type="integer_opt",
        row=row,
        include_cut_type=True,
    )
    _assert_true(not default_peek["is_repeated"], "Default repetition mode should stay family-specific")

    tracker_cross_family = CutRepetitionTracker()
    tracker_cross_family.record_by_row(
        cut_type="benders",
        ite_name=0,
        cut_name="bd_0",
        row=row,
        include_cut_type=False,
    )

    cross_family_peek = tracker_cross_family.peek_by_row(
        cut_type="integer_opt",
        row=row,
        include_cut_type=False,
    )
    _assert_true(cross_family_peek["is_repeated"], "Cross-family peek should detect the existing row")

    repeated_record = tracker_cross_family.record_by_row(
        cut_type="integer_opt",
        ite_name=1,
        cut_name="int_1",
        row=row,
        include_cut_type=False,
    )
    _assert_true(repeated_record["is_repeated"], "Cross-family record should mark identical row as repeated")


def test_phase_and_branch_helpers() -> None:
    policy = PhasePolicy(1, 3, 5)
    _assert_true(policy.phase_name(0) == PhaseName.PHASE_1, "Iteration 0 should be in phase_1")
    _assert_true(policy.phase_name(2) == PhaseName.PHASE_2, "Iteration 2 should be in phase_2")
    _assert_true(policy.phase_name(5) == PhaseName.PHASE_3, "Iteration 5 should be in phase_3")

    _assert_true(
        policy.branch_name(5, frac_repeated=True, current_repeated=False)
        == BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR,
        "Unexpected branch for repeated frac and new curr main result",
    )
    _assert_true(
        policy.candidate_cut_types(BranchName.PHASE_3_REPEATED_FRAC_REPEATED_CURR)[-1]
        == CutTypeName.INTEGER_OPT,
        "Phase-3 repeated/repeated branch should include integer_opt candidate",
    )


def test_dynamic_branch_rules() -> None:
    _assert_true(
        PhasePolicy.should_add_integer_fallback(BranchName.PHASE_1_NEW_FRAC, any_repeated=True),
        "Phase-1 new-frac branch should fall back to integer-opt when a candidate repeats",
    )
    _assert_true(
        not PhasePolicy.should_add_integer_fallback(
            BranchName.PHASE_3_REPEATED_FRAC_REPEATED_CURR,
            any_repeated=True,
        ),
        "Phase-3 repeated/repeated branch should not use integer-opt fallback logic",
    )
    _assert_true(
        PhasePolicy.should_try_post_integer_cglp(BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR),
        "Phase-3 repeated-frac/new-curr branch should try the post-integer CGLP cut",
    )
    _assert_true(
        not PhasePolicy.should_try_post_integer_cglp(BranchName.PHASE_3_NEW_FRAC),
        "Only the repeated-frac/new-curr branch should use the post-integer CGLP path",
    )


def test_solution_generator_contracts() -> None:
    context = IterationContext(ite_name='0')
    solution = MainSolution(values={VarName.DG_INSTALL: {'node_1': 1}}, name='fixed')
    _assert_true(NoOpSolutionGenerator().generate(context) == [], 'No-op generator must stay empty')
    generated = StaticSolutionGenerator([solution]).generate(context)
    _assert_true(generated == [solution], 'Static generator did not return its fixed anchor')


def run_all_tests() -> None:
    test_load_main_stage_solution_json()
    test_cut_repetition_tracker_cross_family_preview()
    test_phase_and_branch_helpers()
    test_dynamic_branch_rules()
    test_solution_generator_contracts()
    print("dynamic_algorithm_helper_test.py: all checks passed")


if __name__ == "__main__":
    run_all_tests()
