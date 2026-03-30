# -*- coding: utf-8 -*-
# @Time     : 2026/03/29
# @Author   : Codex


import os
import sys
import pickle
from typing import Any, Dict, List, Optional

import pandas as pd


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


from util.names import VarName
from util.tools import load_main_stage_solution_json


DEFAULT_MAIN_VAR_NAMES = [
    VarName.DG_INSTALL,
    VarName.LINE_HARDEN,
    VarName.DG_INSTALL_TYPE,
]


def _load_main_result_history(report_pkl_path: str) -> List[Dict[str, Dict[Any, Any]]]:
    with open(report_pkl_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected pickle payload to be dict, got {type(payload)}")

    history = payload.get("curr_main_result_history")
    if not isinstance(history, list):
        raise ValueError("Pickle payload does not contain list field 'curr_main_result_history'")

    return history


def _get_var_names(history: List[Dict[str, Dict[Any, Any]]]) -> List[str]:
    if not history:
        return list(DEFAULT_MAIN_VAR_NAMES)

    first_result = history[0]
    ordered_names = [var_name for var_name in DEFAULT_MAIN_VAR_NAMES if var_name in first_result]
    ordered_names.extend([var_name for var_name in first_result.keys() if var_name not in ordered_names])
    return ordered_names


def _validate_result_shape(
    result: Dict[str, Dict[Any, Any]],
    expected_var_names: List[str],
    expected_keys_map: Dict[str, set],
    source_name: str,
) -> None:
    missing_var_names = [var_name for var_name in expected_var_names if var_name not in result]
    if missing_var_names:
        raise ValueError(f"{source_name} misses variable groups: {missing_var_names}")

    for var_name in expected_var_names:
        current_keys = set(result[var_name].keys())
        if current_keys != expected_keys_map[var_name]:
            raise ValueError(
                f"{source_name} has inconsistent keys for '{var_name}'. "
                f"Expected {len(expected_keys_map[var_name])} keys, got {len(current_keys)}."
            )


def _values_different(lhs: Any, rhs: Any, tol: float = 1e-9) -> bool:
    if pd.isna(lhs) and pd.isna(rhs):
        return False
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return abs(float(lhs) - float(rhs)) > tol
    return lhs != rhs


def _count_differences(
    lhs_result: Dict[str, Dict[Any, Any]],
    rhs_result: Dict[str, Dict[Any, Any]],
    var_names: List[str],
) -> Dict[str, int]:
    diff_counts: Dict[str, int] = {}
    for var_name in var_names:
        diff_counts[var_name] = sum(
            1
            for key in lhs_result[var_name]
            if _values_different(lhs_result[var_name][key], rhs_result[var_name][key])
        )
    return diff_counts


def analyze_main_decision_varibility(
    report_pkl_path: str,
    reference_solution_path: Optional[str] = None,
    output_csv_name: str = "main_decision_varibility.csv",
) -> str:
    report_pkl_path = os.path.abspath(report_pkl_path)
    history = _load_main_result_history(report_pkl_path)
    var_names = _get_var_names(history)

    if not history:
        raise ValueError("curr_main_result_history is empty; no iteration result is available for analysis")

    expected_keys_map = {
        var_name: set(history[0][var_name].keys())
        for var_name in var_names
    }
    total_var_num = sum(len(expected_keys_map[var_name]) for var_name in var_names)

    reference_result = None
    if reference_solution_path is not None:
        reference_solution_path = os.path.abspath(reference_solution_path)
        reference_result = load_main_stage_solution_json(reference_solution_path)
        _validate_result_shape(reference_result, var_names, expected_keys_map, "reference solution")

    rows = []
    previous_result = None
    for iteration, curr_result in enumerate(history):
        _validate_result_shape(curr_result, var_names, expected_keys_map, f"iteration {iteration}")

        row = {"iteration": iteration}

        if previous_result is None:
            prev_diff_counts = {var_name: 0 for var_name in var_names}
        else:
            prev_diff_counts = _count_differences(curr_result, previous_result, var_names)

        prev_total_diff = sum(prev_diff_counts.values())
        for var_name in var_names:
            var_total = len(expected_keys_map[var_name])
            row[f"{var_name}_diff_num_prev"] = prev_diff_counts[var_name]
            row[f"{var_name}_diff_ratio_prev"] = prev_diff_counts[var_name] / var_total if var_total else 0.0

        row["total_diff_num_prev"] = prev_total_diff
        row["total_diff_ratio_prev"] = prev_total_diff / total_var_num if total_var_num else 0.0

        if reference_result is not None:
            ref_diff_counts = _count_differences(curr_result, reference_result, var_names)
            ref_total_diff = sum(ref_diff_counts.values())
            for var_name in var_names:
                var_total = len(expected_keys_map[var_name])
                row[f"{var_name}_diff_ratio_ref"] = ref_diff_counts[var_name] / var_total if var_total else 0.0
            row["total_diff_ratio_ref"] = ref_total_diff / total_var_num if total_var_num else 0.0

        rows.append(row)
        previous_result = curr_result

    df = pd.DataFrame(rows)
    output_dir = os.path.dirname(report_pkl_path)
    output_csv_path = os.path.join(output_dir, output_csv_name)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


if __name__ == "__main__":
    REPORT_PKL_PATH = (
        '/Users/huangjiacheng/SDinPS/output/20260329151400_dynamic_algorithm_new_m/main_result_repetition_report.pkl'
    )
    REFERENCE_SOLUTION_PATH = (
        "/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p/solutions/s_1_S_2_s_3_s_4_s_5_s_6.json"
    )

    output_path = analyze_main_decision_varibility(
        report_pkl_path=REPORT_PKL_PATH,
        reference_solution_path=REFERENCE_SOLUTION_PATH,
    )
    print(f"Analysis CSV written to: {output_path}")
