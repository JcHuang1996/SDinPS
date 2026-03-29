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
from unit_test.test_scripts.dynamic_decomposition_test import (
    _get_branch_name,
    _get_candidate_cut_types,
    _get_phase_name,
    _should_add_integer_opt_fallback,
    _should_try_post_integer_cglp,
)
from util.names import VarName
from util.tools import load_main_stage_solution_json


REFERENCE_SOLUTION_PATH = (
    '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/'
    'function_test_fixed_rated_p/solutions/s_1_s_3_s_5.json'
)


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
    _assert_true(_get_phase_name(0, 1, 3, 5) == "phase_1", "Iteration 0 should be in phase_1")
    _assert_true(_get_phase_name(2, 1, 3, 5) == "phase_2", "Iteration 2 should be in phase_2")
    _assert_true(_get_phase_name(5, 1, 3, 5) == "phase_3", "Iteration 5 should be in phase_3")

    _assert_true(
        _get_branch_name("phase_3", frac_is_repeated=True, curr_is_repeated=False) == "phase_3_repeated_frac_new_curr",
        "Unexpected branch for repeated frac and new curr main result",
    )
    _assert_true(
        _get_candidate_cut_types("phase_3_repeated_frac_repeated_curr")[-1] == "integer_opt",
        "Phase-3 repeated/repeated branch should include integer_opt candidate",
    )


def test_dynamic_branch_rules() -> None:
    _assert_true(
        _should_add_integer_opt_fallback("phase_1_new_frac", any_candidate_repeated=True),
        "Phase-1 new-frac branch should fall back to integer-opt when a candidate repeats",
    )
    _assert_true(
        not _should_add_integer_opt_fallback("phase_3_repeated_frac_repeated_curr", any_candidate_repeated=True),
        "Phase-3 repeated/repeated branch should not use integer-opt fallback logic",
    )
    _assert_true(
        _should_try_post_integer_cglp("phase_3_repeated_frac_new_curr"),
        "Phase-3 repeated-frac/new-curr branch should try the post-integer CGLP cut",
    )
    _assert_true(
        not _should_try_post_integer_cglp("phase_3_new_frac"),
        "Only the repeated-frac/new-curr branch should use the post-integer CGLP path",
    )


def run_all_tests() -> None:
    test_load_main_stage_solution_json()
    test_cut_repetition_tracker_cross_family_preview()
    test_phase_and_branch_helpers()
    test_dynamic_branch_rules()
    print("dynamic_algorithm_helper_test.py: all checks passed")


if __name__ == "__main__":
    run_all_tests()
