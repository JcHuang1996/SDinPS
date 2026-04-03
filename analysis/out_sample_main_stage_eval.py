# -*- coding: utf-8 -*-
# @Time     : 2026/03/30
# @Author   : Codex

import argparse
import ast
import copy
import json
import os
import pickle
import sys
from typing import Any, Dict, Optional, Sequence

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from util.headers import (
    BranchHeader,
    DgRatedPowerHeader,
    NodeHeader,
    ParameterKey,
    ScenarioLineStateHeader,
    ScenarioProbHeader,
)
from util.names import InputMethodName, VarName


def _parse_json_like_key(key: Any) -> Any:
    if not isinstance(key, str) or not key.startswith("("):
        return key
    try:
        return ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return key


def normalize_main_solution(main_solution: Dict[str, Dict[Any, Any]]) -> Dict[str, Dict[Any, Any]]:
    payload = main_solution
    for wrapper_key in ["main_result", "curr_main_result", "solution"]:
        wrapped_payload = payload.get(wrapper_key)
        if isinstance(wrapped_payload, dict):
            payload = wrapped_payload
            break

    normalized = {}
    for var_name in [VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]:
        normalized[var_name] = {
            _parse_json_like_key(key): value
            for key, value in payload[var_name].items()
        }
    return normalized


def load_main_solution_from_json(solution_json_path: str) -> Dict[str, Dict[Any, Any]]:
    with open(solution_json_path, "r", encoding="utf-8") as file_obj:
        return normalize_main_solution(json.load(file_obj))


def select_iteration_from_iter_general(
    iter_general_csv_path: str,
    selection_rule: str,
    ite_num: Optional[int] = None,
) -> int:
    df = pd.read_csv(iter_general_csv_path, dtype={"ite_num": str}, keep_default_na=True)
    df = df[df["ite_num"].astype(str).str.strip() != "best"].copy()
    df["ite_num_num"] = pd.to_numeric(df["ite_num"], errors="coerce")
    df = df[df["ite_num_num"].notna()].copy()
    if df.empty:
        raise ValueError(f"No numeric iterations found in {iter_general_csv_path}")

    if selection_rule == "ite_num":
        return int(ite_num)

    if selection_rule == "latest":
        return int(df["ite_num_num"].max())

    if selection_rule != "best":
        raise ValueError(f"Unsupported selection_rule: {selection_rule}")

    df["best_incumbent_obj_value"] = pd.to_numeric(df["best_incumbent_obj_value"], errors="coerce")
    df = df[df["best_incumbent_obj_value"].notna()].copy()
    if df.empty:
        raise ValueError(f"No incumbent rows found in {iter_general_csv_path} for selection_rule='best'")
    best_row = df.loc[df["best_incumbent_obj_value"].idxmin()]
    return int(best_row["ite_num_num"])


def load_main_solution_from_result_dir(
    result_dir: str,
    selection_rule: str,
    ite_num: Optional[int] = None,
) -> tuple[Dict[str, Dict[Any, Any]], int]:
    iter_general_csv_path = os.path.join(result_dir, "iter_general_info.csv")
    main_result_report_path = os.path.join(result_dir, "main_result_repetition_report.pkl")

    iteration = select_iteration_from_iter_general(
        iter_general_csv_path=iter_general_csv_path,
        selection_rule=selection_rule,
        ite_num=ite_num,
    )

    with open(main_result_report_path, "rb") as file_obj:
        payload = pickle.load(file_obj)

    history = payload["curr_main_result_history"]
    return normalize_main_solution(history[iteration]), iteration


def load_main_solution(
    main_solution_json_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    selection_rule: str = "best",
    ite_num: Optional[int] = None,
) -> tuple[Dict[str, Dict[Any, Any]], str, Optional[int]]:
    if main_solution_json_path:
        return load_main_solution_from_json(main_solution_json_path), "json", None
    if result_dir:
        solution, iteration = load_main_solution_from_result_dir(
            result_dir=result_dir,
            selection_rule=selection_rule,
            ite_num=ite_num,
        )
        return solution, "result_dir", iteration
    raise ValueError("Either main_solution_json_path or result_dir must be provided")


def load_base_raw_data(data_dir: str) -> Dict[str, Any]:
    from dao.data_reader import DataReader

    data_dir = os.path.abspath(data_dir)
    local_file_path = os.path.dirname(data_dir)
    data_set_name = os.path.basename(data_dir.rstrip(os.sep))

    reader = DataReader(
        read_method=InputMethodName.LOCAL_CSV,
        local_file_path=local_file_path,
        data_set_name=data_set_name,
    )
    reader.read()
    return copy.deepcopy(reader.raw_data)


def filter_sample_dataframe(
    sample_df: pd.DataFrame,
    scenario_list: Optional[Sequence[str]] = None,
    time_list: Optional[Sequence[int]] = None,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
) -> pd.DataFrame:
    filtered_df = sample_df.copy()

    if scenario_list:
        filtered_df = filtered_df[
            filtered_df[ScenarioLineStateHeader.SCENARIO_ID].isin(scenario_list)
        ].copy()

    if time_list:
        filtered_df = filtered_df[
            filtered_df[ScenarioLineStateHeader.TIME_IDX].isin(time_list)
        ].copy()

    if time_start is not None:
        filtered_df = filtered_df[
            filtered_df[ScenarioLineStateHeader.TIME_IDX] >= time_start
        ].copy()

    if time_end is not None:
        filtered_df = filtered_df[
            filtered_df[ScenarioLineStateHeader.TIME_IDX] <= time_end
        ].copy()

    return filtered_df.sort_values(
        [
            ScenarioLineStateHeader.SCENARIO_ID,
            "sample_id",
            ScenarioLineStateHeader.TIME_IDX,
            ScenarioLineStateHeader.FROM_NODE,
            ScenarioLineStateHeader.TO_NODE,
        ]
    ).reset_index(drop=True)


def load_probability_map(probability_csv_path: str) -> Dict[str, float]:
    prob_df = pd.read_csv(probability_csv_path)
    return prob_df.set_index(ScenarioProbHeader.SCENARIO_ID)[ScenarioProbHeader.SCENARIO_PROB].to_dict()


def evaluate_design_cost(main_solution: Dict[str, Dict[Any, Any]], raw_data: Dict[str, Any]) -> float:
    parameter_dict = raw_data["parameters_dict"]
    node_df = raw_data["node_df"]
    branch_df = raw_data["branch_df"]
    dg_rated_power_df = raw_data["dg_rated_power_df"]

    if NodeHeader.FIXED_C in node_df.columns:
        dg_fixed_cost_map = node_df.set_index(NodeHeader.NODE_ID)[NodeHeader.FIXED_C].to_dict()
    else:
        dg_fixed_cost_map = {
            node_id: parameter_dict[ParameterKey.DG_CF]
            for node_id in node_df[NodeHeader.NODE_ID].tolist()
        }

    line_harden_cost_map = {
        (row[BranchHeader.FROM_NODE], row[BranchHeader.TO_NODE]):
            row[BranchHeader.POLE_NUMBER] * parameter_dict[ParameterKey.POLE_HARDEN_COST]
        for _, row in branch_df.iterrows()
    }

    dg_variant_cost_map = {
        row[DgRatedPowerHeader.DG_TYPE]:
            row[DgRatedPowerHeader.RATED_POWER] * row[DgRatedPowerHeader.UNIT_PRICE]
            + row[DgRatedPowerHeader.EXTRA_ADJUSTMENT]
        for _, row in dg_rated_power_df.iterrows()
    }

    return (
        sum(
            dg_fixed_cost_map[node_id] * value
            for node_id, value in main_solution[VarName.DG_INSTALL].items()
        )
        + sum(
            dg_variant_cost_map[dg_type] * value
            for (_, dg_type), value in main_solution[VarName.DG_INSTALL_TYPE].items()
        )
        + sum(
            line_harden_cost_map[line_key] * value
            for line_key, value in main_solution[VarName.LINE_HARDEN].items()
        )
    )


def evaluate_one_sample(
    main_solution: Dict[str, Dict[Any, Any]],
    sample_line_state_df: pd.DataFrame,
    base_raw_data: Dict[str, Any],
    time_list: Optional[Sequence[int]] = None,
    solver_param_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from dao.data_processor import DataProcessor
        from model.model_sub import ModelSub
        from util.names import InputDataName
    except ImportError as exc:
        raise RuntimeError(
            "Subproblem evaluation requires the project optimization stack (including pyomo)."
        ) from exc

    scenario_id = sample_line_state_df[ScenarioLineStateHeader.SCENARIO_ID].iloc[0]
    sample_time_list = list(time_list) if time_list is not None else sorted(
        sample_line_state_df[ScenarioLineStateHeader.TIME_IDX].unique().tolist()
    )

    sample_raw_data = copy.deepcopy(base_raw_data)
    sample_raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF] = sample_line_state_df.drop(
        columns=["sample_id"],
        errors="ignore",
    ).copy()

    data_processor = DataProcessor(raw_data=sample_raw_data)
    sub_model_data = data_processor.data_process(
        scenario_list_assigned=[scenario_id],
        time_list_assigned=sample_time_list,
    )
    data_processor.clear_existing_data()

    sub_model = ModelSub(
        model_name=f"sample_eval_{scenario_id}",
        model_data=sub_model_data,
        main_result=main_solution,
        sub_model_sce_list=[scenario_id],
    )
    sub_model.build_sub_model_redo()
    if solver_param_dict is None:
        sub_model.solve()
    else:
        sub_model.solve(param_dict=solver_param_dict)
    sub_model.cal_detailed_obj()

    return {
        "scenario_id": scenario_id,
        "sample_objective_value": sub_model.get_obj_value(),
        "solve_status": sub_model.solve_status,
    }


def evaluate_main_solution_out_of_sample(
    sample_csv_path: str,
    data_dir: str,
    main_solution_json_path: Optional[str] = None,
    result_dir: Optional[str] = None,
    selection_rule: str = "best",
    ite_num: Optional[int] = None,
    probability_csv_path: Optional[str] = None,
    scenario_list: Optional[Sequence[str]] = None,
    time_list: Optional[Sequence[int]] = None,
    time_start: Optional[int] = None,
    time_end: Optional[int] = None,
    output_dir: Optional[str] = None,
    solver_param_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample_df = pd.read_csv(sample_csv_path)
    filtered_sample_df = filter_sample_dataframe(
        sample_df=sample_df,
        scenario_list=scenario_list,
        time_list=time_list,
        time_start=time_start,
        time_end=time_end,
    )
    if filtered_sample_df.empty:
        raise ValueError("No sampled rows remain after filtering")

    base_raw_data = load_base_raw_data(data_dir=data_dir)
    main_solution, solution_source, selected_iteration = load_main_solution(
        main_solution_json_path=main_solution_json_path,
        result_dir=result_dir,
        selection_rule=selection_rule,
        ite_num=ite_num,
    )

    if probability_csv_path is None:
        probability_csv_path = os.path.join(data_dir, "s_probability.csv")
    probability_map = load_probability_map(probability_csv_path)

    sample_time_list = sorted(filtered_sample_df[ScenarioLineStateHeader.TIME_IDX].unique().tolist())
    detail_records = []
    grouped_sample_df = filtered_sample_df.groupby(
        [ScenarioLineStateHeader.SCENARIO_ID, "sample_id"],
        sort=True,
    )
    for (_, sample_id), one_sample_df in grouped_sample_df:
        eval_result = evaluate_one_sample(
            main_solution=main_solution,
            sample_line_state_df=one_sample_df,
            base_raw_data=base_raw_data,
            solver_param_dict=solver_param_dict,
        )
        detail_records.append(
            {
                "scenario_id": eval_result["scenario_id"],
                "sample_id": sample_id,
                "sample_objective_value": eval_result["sample_objective_value"],
                "solve_status": eval_result["solve_status"],
            }
        )

    detail_df = pd.DataFrame(detail_records)
    summary_df = detail_df.groupby("scenario_id", as_index=False).agg(
        sample_count=("sample_id", "count"),
        sample_objective_mean=("sample_objective_value", "mean"),
    )
    summary_df["probability"] = summary_df["scenario_id"].map(probability_map)
    if summary_df["probability"].isna().any():
        missing_scenarios = summary_df.loc[summary_df["probability"].isna(), "scenario_id"].tolist()
        raise ValueError(f"Missing probabilities for scenarios: {missing_scenarios}")
    summary_df["weighted_objective_contribution"] = (
        summary_df["sample_objective_mean"] * summary_df["probability"]
    )

    expected_subproblem_objective = summary_df["weighted_objective_contribution"].sum()
    design_cost = evaluate_design_cost(main_solution=main_solution, raw_data=base_raw_data)
    expected_total_objective = design_cost + expected_subproblem_objective

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(sample_csv_path))
    os.makedirs(output_dir, exist_ok=True)

    summary_output_row = pd.DataFrame(
        [
            {
                "scenario_id": "total_expected_sp_objective",
                "sample_count": detail_df["sample_id"].count(),
                "sample_objective_mean": float("nan"),
                "probability": summary_df["probability"].sum(),
                "weighted_objective_contribution": expected_subproblem_objective,
            }
        ]
    ).astype({**summary_df.dtypes.astype(str).to_dict(), "scenario_id": "object"})
    summary_output_df = pd.concat(
        [summary_df.astype({"scenario_id": "object"}), summary_output_row],
        ignore_index=True,
    )

    detail_output_path = os.path.join(output_dir, "sample_objective_detail.csv")
    summary_output_path = os.path.join(output_dir, "sample_objective_summary.csv")
    detail_df.to_csv(detail_output_path, index=False)
    summary_output_df.to_csv(summary_output_path, index=False)

    return {
        "detail_df": detail_df,
        "summary_df": summary_output_df,
        "detail_output_path": detail_output_path,
        "summary_output_path": summary_output_path,
        "solution_source": solution_source,
        "selected_iteration": selected_iteration,
        "sample_time_list": sample_time_list,
        "design_cost": design_cost,
        "expected_subproblem_objective": expected_subproblem_objective,
        "expected_total_objective": expected_total_objective,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one fixed main-stage solution on many sampled subproblems."
    )
    parser.add_argument("--sample-csv-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--probability-csv-path")
    parser.add_argument("--selection-rule", choices=["best", "latest", "ite_num"], default="best")
    parser.add_argument("--ite-num", type=int)
    parser.add_argument("--scenario-id", nargs="+")
    parser.add_argument("--time-idx", nargs="+", type=int)
    parser.add_argument("--time-start", type=int)
    parser.add_argument("--time-end", type=int)
    parser.add_argument("--output-dir")

    solution_group = parser.add_mutually_exclusive_group(required=True)
    solution_group.add_argument("--main-solution-json")
    solution_group.add_argument("--result-dir")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    result = evaluate_main_solution_out_of_sample(
        sample_csv_path=args.sample_csv_path,
        data_dir=args.data_dir,
        main_solution_json_path=args.main_solution_json,
        result_dir=args.result_dir,
        selection_rule=args.selection_rule,
        ite_num=args.ite_num,
        probability_csv_path=args.probability_csv_path,
        scenario_list=args.scenario_id,
        time_list=args.time_idx,
        time_start=args.time_start,
        time_end=args.time_end,
        output_dir=args.output_dir,
    )

    print(f"solution_source: {result['solution_source']}")
    if result["selected_iteration"] is not None:
        print(f"selected_iteration: {result['selected_iteration']}")
    print(f"sample_times: {result['sample_time_list']}")
    print(f"design_cost: {result['design_cost']}")
    print(f"expected_subproblem_objective: {result['expected_subproblem_objective']}")
    print(f"expected_total_objective: {result['expected_total_objective']}")
    print(f"detail_output_path: {result['detail_output_path']}")
    print(f"summary_output_path: {result['summary_output_path']}")

def run_local_example() -> None:
    result = evaluate_main_solution_out_of_sample(
        sample_csv_path='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/out_sample_data/out_sample_line_status.csv',
        data_dir='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p',
        main_solution_json_path=None,
        result_dir='/Users/huangjiacheng/SDinPS/output/20260331192031_new_m_agg_t_n_pret',
        selection_rule='best',
        # ite_num=1,
        probability_csv_path=None,
        # scenario_list=['s_1', 's_3', 's_5'],
        scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        time_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        time_start=None,
        time_end=None,
        output_dir='/Users/huangjiacheng/SDinPS/output/20260331192031_new_m_agg_t_n_pret',
    )
    # result = evaluate_main_solution_out_of_sample(
    #     sample_csv_path='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/out_sample_data/out_sample_line_status.csv',
    #     data_dir='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p',
    #     # main_solution_json_path='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p/solutions/s_1_s_2_s_3_s_4_s_5_s_6.json',
    #     main_solution_json_path='/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p/solutions/s_1_s_3_s_5.json',
    #     result_dir=None,
    #     selection_rule='best',  # ignored when main_solution_json_path is provided
    #     probability_csv_path=None,
    #     scenario_list=['s_1', 's_3', 's_5'],
    #     time_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     time_start=None,
    #     time_end=None,
    #     output_dir='/Users/huangjiacheng/SDinPS/output/20260329150235_dynamic_algorithm_new_m',
    # )

    print(f"solution_source: {result['solution_source']}")
    if result["selected_iteration"] is not None:
        print(f"selected_iteration: {result['selected_iteration']}")
    print(f"sample_times: {result['sample_time_list']}")
    print(f"design_cost: {result['design_cost']}")
    print(f"expected_subproblem_objective: {result['expected_subproblem_objective']}")
    print(f"expected_total_objective: {result['expected_total_objective']}")
    print(f"detail_output_path: {result['detail_output_path']}")
    print(f"summary_output_path: {result['summary_output_path']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_local_example()
