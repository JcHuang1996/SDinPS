# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import ast
import pandas as pd
import os
import pickle
import json
from typing import Any, Dict, List, Optional, Tuple

from util.names import DECOMPOSITION_STATE_VAR_NAMES, VarName


def enrich_iter_general_dataframe(
    df: pd.DataFrame,
    drop_existing_best_rows: bool = True,
) -> pd.DataFrame:
    """
    Add ``gap`` = (best_incumbent_obj_value - valid_lower_bound) / abs(best_incumbent_obj_value).
    Missing certified bounds leave the gap blank; a zero incumbent and zero bound give zero gap.

    Append a summary row with ite_num == 'best', best_incumbent_obj_value = min(incumbent column),
    valid_lower_bound = max(certified bound column), plus the independent diagnostic maximum.

    If ``drop_existing_best_rows`` is True, rows whose ite_num string-equals 'best' are dropped first
    (so re-running on an already-enriched file does not duplicate the summary row).
    """
    if df is None:
        df = pd.DataFrame(columns=["ite_num", "best_incumbent_obj_value", "best_bound_objective_value"])

    out = df.copy()
    if "ite_num" not in out.columns:
        raise ValueError("DataFrame must contain column 'ite_num'")

    if "gap" in out.columns:
        out = out.drop(columns=["gap"])

    if not out.empty:
        ite_str = out["ite_num"].astype(str).str.strip()
        if drop_existing_best_rows:
            out = out.loc[ite_str != "best"].reset_index(drop=True)

    inc = (
        pd.to_numeric(out["best_incumbent_obj_value"], errors="coerce")
        if "best_incumbent_obj_value" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="Float64")
    )
    bnd = (
        pd.to_numeric(out["valid_lower_bound"], errors="coerce")
        if "valid_lower_bound" in out.columns
        else pd.Series(pd.NA, index=out.index, dtype="Float64")
    )

    bnd = bnd.replace([float('inf'), -float('inf')], pd.NA)
    valid = inc.notna() & bnd.notna() & (inc != 0)
    gap = pd.Series(pd.NA, index=out.index, dtype="Float64")
    gap.loc[valid] = (inc.loc[valid] - bnd.loc[valid]) / inc.loc[valid].abs()
    gap.loc[(inc == 0) & (bnd == 0)] = 0.0
    out["gap"] = gap

    # column order: gap immediately after best_bound_objective_value when present
    cols = [c for c in out.columns if c != "gap"]
    if "best_bound_objective_value" in cols:
        i = cols.index("best_bound_objective_value") + 1
        cols = cols[:i] + ["gap"] + cols[i:]
    else:
        cols = cols + ["gap"]
    out = out[cols]

    min_inc = inc.min(skipna=True)
    max_bnd = bnd.max(skipna=True)
    best_gap = float("nan")
    if pd.notna(min_inc) and pd.notna(max_bnd) and float(min_inc) != 0.0:
        best_gap = (float(min_inc) - float(max_bnd)) / abs(float(min_inc))
    elif pd.notna(min_inc) and pd.notna(max_bnd) and float(min_inc) == float(max_bnd) == 0.0:
        best_gap = 0.0

    # Use float NaN for non-labeled cells so concat dtypes stay stable (avoids all-NA FutureWarning).
    summary = {c: float("nan") for c in out.columns}
    summary["ite_num"] = "best"
    if "best_incumbent_obj_value" in out.columns:
        summary["best_incumbent_obj_value"] = float(min_inc) if pd.notna(min_inc) else float("nan")
    if "best_bound_objective_value" in out.columns:
        diagnostic = pd.to_numeric(out["best_bound_objective_value"], errors="coerce").max(skipna=True)
        summary["best_bound_objective_value"] = diagnostic
    if "valid_lower_bound" in out.columns:
        summary["valid_lower_bound"] = float(max_bnd) if pd.notna(max_bnd) else float("nan")
    if "gap" in out.columns:
        summary["gap"] = best_gap

    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    return out


def write_iter_general_enriched_csv(df: pd.DataFrame, path: str) -> str:
    """
    Write enriched iter-general dataframe to CSV. Forces ``ite_num`` to string so numeric iterations
    and the label 'best' do not mix types in a way that confuses readers.
    """
    out = df.copy()
    if "ite_num" in out.columns:
        out["ite_num"] = out["ite_num"].astype(str)
    out.to_csv(path, index=False)
    return path


def iter_general_csv(ite_obj_value_dict=None, output_dir=None):
    """
    Write or overwrite iter_general_info.csv from ite_obj_value_dict.

    Each row includes a gap from its certified lower bound when available, plus a trailing summary
    row with ``ite_num`` == 'best' (min incumbent, max certified bound over the run).
    ``ite_num`` is written as text so iteration indices and 'best' stay unambiguous in CSV.
    Handles empty dict (header-only plus best row).
    """
    general_records = []
    for ite_num in ite_obj_value_dict:
        incumbent = ite_obj_value_dict[ite_num].get('sub_obj(best_incumbent)', {}).get('sub_p_total', None)
        bound_data = ite_obj_value_dict[ite_num].get('main_obj(bound)', {})
        bound_total = bound_data.get('total', None)
        detail_dict = bound_data.get('detail', {})

        record = {
            'ite_num': ite_num,
            'best_incumbent_obj_value': incumbent,
            # Diagnostic design + sum P_s theta_s at the master incumbent; a certificate is
            # reported separately in valid_lower_bound (including under incomplete solves).
            'best_bound_objective_value': bound_total,
            'valid_lower_bound': bound_data.get('valid_lower_bound', None),
            'added_obj_term_weight': bound_data.get('added_obj_term_weight', None),
        }
        # merge cost components
        record.update(detail_dict)
        general_records.append(record)

    if not general_records:
        df_general = pd.DataFrame(columns=['ite_num', 'best_incumbent_obj_value', 'best_bound_objective_value'])
    else:
        df_general = pd.DataFrame(general_records)
    enriched = enrich_iter_general_dataframe(df_general, drop_existing_best_rows=False)
    csv_general_path = os.path.join(output_dir, 'iter_general_info.csv')
    write_iter_general_enriched_csv(enriched, csv_general_path)

def iter_sub_prob_info(ite_obj_value_dict=None, output_dir=None, scenario_list=None):
    """Write or overwrite iter_subproblem_info.csv from ite_obj_value_dict. Handles empty dict (header-only)."""
    sub_records = []
    scenario_set = set(scenario_list)
    for ite_num in ite_obj_value_dict:
        for group, sub_info in ite_obj_value_dict[ite_num].items():
            if not isinstance(group, tuple) or not set(group).issubset(scenario_set):
                continue
            s = group[0] if len(group) == 1 else json.dumps(list(group))
            sub_total = sub_info.get('total', None)
            sub_detail = sub_info.get('detail', {})
            record = {'ite_num': ite_num, 'scenario': s}
            record.update(sub_detail)
            record['sub_obj_total'] = sub_total
            sub_records.append(record)

    if not sub_records:
        df_sub = pd.DataFrame(columns=['ite_num', 'scenario', 'sub_obj_total'])
    else:
        df_sub = pd.DataFrame(sub_records)
    csv_sub_path = os.path.join(output_dir, 'iter_subproblem_info.csv')
    df_sub.to_csv(csv_sub_path, index=False)


def write_power_usage_capacity_ratio(
    best_sub_results: Optional[Dict[str, Any]],
    output_dir: str,
    detailed_csv_name: str = "power_usage_capacity_ratio_detailed.csv",
    averaged_csv_name: str = "power_usage_capacity_ratio_averaged.csv",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute and write power usage capacity ratio from best-iteration sub results.

    best_sub_results: dict with VarName.DG_ACTIVE_POWER -> {(j,t,s): value},
        VarName.DG_RATED_POWER -> {(j,s): value}. If None or empty, no files are written.
    output_dir: directory to write CSV files.
    detailed_csv_name / averaged_csv_name: output file names.

    Power usage capacity ratio = active_power / rated_power (per node, scenario, time).
    Averaged ratio = mean over timeslots for each (node, scenario).

    Returns (path_to_detailed, path_to_averaged), or (None, None) if no data.
    """
    if not best_sub_results:
        return None, None
    pg = best_sub_results.get(VarName.DG_ACTIVE_POWER, {})
    pgrt = best_sub_results.get(VarName.DG_RATED_POWER, {})
    if not pg or not pgrt:
        return None, None

    # Detailed: node, scenario, time, active_power, rated_power, ratio
    detailed_rows = []
    for (j, t, s), active_power in pg.items():
        rated_power = pgrt.get((j, s), None)
        if rated_power is None:
            continue
        ratio = (active_power / rated_power) if rated_power != 0 else float("nan")
        detailed_rows.append({
            "node": j,
            "scenario": s,
            "time": t,
            "active_power": active_power,
            "rated_power": rated_power,
            "ratio": ratio,
        })
    if not detailed_rows:
        return None, None
    df_detailed = pd.DataFrame(detailed_rows)
    os.makedirs(output_dir, exist_ok=True)
    path_detailed = os.path.join(output_dir, detailed_csv_name)
    df_detailed.to_csv(path_detailed, index=False)

    # Averaged: node, scenario, rated_power, averaged_ratio
    df = df_detailed.copy()
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    agg = df.groupby(["node", "scenario"], as_index=False).agg(
        rated_power=("rated_power", "first"),
        averaged_ratio=("ratio", "mean"),
    )
    path_averaged = os.path.join(output_dir, averaged_csv_name)
    agg.to_csv(path_averaged, index=False)

    return path_detailed, path_averaged


def write_decomp_run_parameters(
    output_dir: str,
    params: dict,
    file_name: str = "decomp_run_parameters.txt",
) -> str:
    """
    Write a text file in output_dir with one line per key-value pair in params.
    Returns the path of the written file.
    """
    path = os.path.join(output_dir, file_name)
    lines = [f"{k}: {v}" for k, v in params.items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_main_result_repetition_report(
    output_dir: str,
    curr_main_result_history: List[dict],
    frac_main_result_history: List[dict],
    curr_main_repetition_records: List[dict],
    frac_main_repetition_records: List[dict],
    file_name: str = "main_result_repetition_report.pkl",
) -> str:
    """
    Write repetition report for curr_main_result / frac_main_result to a pickle file.
    Returns the written file path.
    """
    curr_repeated_iterations = [r["iteration"] for r in curr_main_repetition_records if r["is_repeated"]]
    frac_repeated_iterations = [r["iteration"] for r in frac_main_repetition_records if r["is_repeated"]]
    curr_repetition_detected = len(curr_repeated_iterations) > 0
    frac_repetition_detected = len(frac_repeated_iterations) > 0

    payload = {
        "curr_main_result_history": curr_main_result_history,
        "frac_main_result_history": frac_main_result_history,
        "curr_main_result_repetition": {
            "repeat_count": len(curr_repeated_iterations),
            "repeated_iterations": curr_repeated_iterations,
            "is_repeated": [r["is_repeated"] for r in curr_main_repetition_records],
            "first_seen_iteration": [r["first_seen_iteration"] for r in curr_main_repetition_records],
            "repetition_detected": curr_repetition_detected,
        },
        "frac_main_result_repetition": {
            "repeat_count": len(frac_repeated_iterations),
            "repeated_iterations": frac_repeated_iterations,
            "is_repeated": [r["is_repeated"] for r in frac_main_repetition_records],
            "first_seen_iteration": [r["first_seen_iteration"] for r in frac_main_repetition_records],
            "repetition_detected": frac_repetition_detected,
        },
        "iteration_records": [
            {
                "iteration": ite_num,
                "curr_main_result": curr_main_result_history[ite_num],
                "frac_main_result": frac_main_result_history[ite_num],
                "curr_main_result_is_repeated": curr_main_repetition_records[ite_num]["is_repeated"],
                "curr_main_result_first_seen_iteration": curr_main_repetition_records[ite_num]["first_seen_iteration"],
                "frac_main_result_is_repeated": frac_main_repetition_records[ite_num]["is_repeated"],
                "frac_main_result_first_seen_iteration": frac_main_repetition_records[ite_num]["first_seen_iteration"],
            }
            for ite_num in range(len(curr_main_result_history))
        ],
    }

    path = os.path.join(output_dir, file_name)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def write_cut_repetition_report(
    output_dir: str,
    repetitive_cut_records: List[dict],
    file_name: str = "cut_repetition_report.json",
) -> str:
    """
    Write repeated-cut records into a JSON file.
    Returns the written file path.
    """
    payload = {
        "repeat_count": len(repetitive_cut_records),
        "repetitive_cut_records": repetitive_cut_records,
    }
    path = os.path.join(output_dir, file_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def report_repetitive_cuts(
    repetitive_cut_records: List[dict],
    enable_result_output: bool,
    output_dir: str,
    logger=None,
) -> Optional[str]:
    """
    Print/log repetitive cuts and optionally write JSON report.
    Returns report path when written, else None.
    """
    msg = f"repetitive cuts count: {len(repetitive_cut_records)}"
    print(msg)
    if logger is not None:
        logger.info(msg)

    for record in repetitive_cut_records:
        line = (
            f"repetitive cut detected | type={record['cut_type']} | "
            f"iteration={record['iteration']} | cut={record['cut_name']} | "
            f"first_seen_iteration={record['first_seen_iteration']} | "
            f"first_seen_cut={record['first_seen_cut_name']}"
        )
        print(line)
        if logger is not None:
            logger.info(line)

    if not enable_result_output:
        return None
    return write_cut_repetition_report(
        output_dir=output_dir,
        repetitive_cut_records=repetitive_cut_records,
    )


def load_main_stage_solution_json(path: str) -> Dict[str, Dict[Any, Any]]:
    """
    Load a main-stage solution JSON into the nested dict shape returned by model_main.get_result(...).
    Expected top-level keys: xg, xl, xg_type. Stringified tuple keys in xl / xg_type are restored.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    out = {var_name: {} for var_name in DECOMPOSITION_STATE_VAR_NAMES}

    for var_name in DECOMPOSITION_STATE_VAR_NAMES:
        for raw_key, raw_value in payload.get(var_name, {}).items():
            key = raw_key
            if var_name in [VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]:
                key = ast.literal_eval(raw_key)
            out[var_name][key] = raw_value

    return out
