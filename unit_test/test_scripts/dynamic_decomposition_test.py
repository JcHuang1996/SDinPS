# -*- coding: utf-8 -*-
# @Time     : 2026/03/24
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys
import json

from algo.two_stage_decomp_redo import TwoStageDecompRedo
from algo.algo_simple_tools import analyze_scenario_solutions, detect_and_record_repetition

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from util.project_logger import init_logger
from util.names import VarName, InputMethodName, MainProblemHatDataMethodName, MainProblemModelTypeName
from dao.data_reader import DataReader

from util.converge_visual import plot_iter_obj_curves

from util.tools import (
    load_main_stage_solution_json,
    report_repetitive_cuts,
    write_decomp_run_parameters,
    write_main_result_repetition_report,
    write_power_usage_capacity_ratio,
)

from datetime import datetime
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def _check_phase_end_iterations(
    phase_1_end_iteration: Optional[int],
    phase_2_end_iteration: Optional[int],
    phase_3_end_iteration: Optional[int],
    max_iterations: int,
) -> None:
    if phase_1_end_iteration is None or phase_2_end_iteration is None or phase_3_end_iteration is None:
        raise ValueError("phase_1_end_iteration, phase_2_end_iteration, and phase_3_end_iteration must all be set")
    if not (0 <= phase_1_end_iteration <= phase_2_end_iteration <= phase_3_end_iteration == max_iterations - 1):
        raise ValueError(
            "Phase endpoints must satisfy "
            "0 <= phase_1_end_iteration <= phase_2_end_iteration <= phase_3_end_iteration == max_iterations - 1"
        )


def _get_phase_name(
    ite_num: int,
    phase_1_end_iteration: int,
    phase_2_end_iteration: int,
    phase_3_end_iteration: int,
) -> str:
    if ite_num <= phase_1_end_iteration:
        return "phase_1"
    if ite_num <= phase_2_end_iteration:
        return "phase_2"
    if ite_num <= phase_3_end_iteration:
        return "phase_3"
    raise ValueError(f"Iteration {ite_num} is outside configured phase ranges")


def _get_branch_name(phase_name: str, frac_is_repeated: bool, curr_is_repeated: bool) -> str:
    if phase_name == "phase_1":
        return "phase_1_repeated_frac" if frac_is_repeated else "phase_1_new_frac"
    if phase_name == "phase_2":
        return "phase_2_repeated_frac" if frac_is_repeated else "phase_2_new_frac"
    if phase_name == "phase_3":
        if not frac_is_repeated:
            return "phase_3_new_frac"
        if not curr_is_repeated:
            return "phase_3_repeated_frac_new_curr"
        return "phase_3_repeated_frac_repeated_curr"
    raise ValueError(f"Unsupported phase_name: {phase_name}")


def _get_added_obj_term_weight_for_phase(
    phase_name: str,
    default_weight: float,
    phase_1_weight: Optional[float],
    phase_2_weight: Optional[float],
    phase_3_weight: Optional[float],
) -> float:
    if phase_name == "phase_1" and phase_1_weight is not None:
        return float(phase_1_weight)
    if phase_name == "phase_2" and phase_2_weight is not None:
        return float(phase_2_weight)
    if phase_name == "phase_3" and phase_3_weight is not None:
        return float(phase_3_weight)
    return float(default_weight)


def _get_candidate_cut_types(branch_name: str) -> List[str]:
    branch_cut_map = {
        "phase_1_new_frac": ["benders"],
        "phase_1_repeated_frac": [],
        "phase_2_new_frac": ["benders", "strengthen_benders"],
        "phase_2_repeated_frac": [],
        "phase_3_new_frac": ["benders", "strengthen_benders", "cglp", "lagrangian"],
        "phase_3_repeated_frac_new_curr": [],
        "phase_3_repeated_frac_repeated_curr": [
            "benders",
            "strengthen_benders",
            "cglp",
            "lagrangian",
            "integer_opt",
        ],
    }
    return branch_cut_map[branch_name]


def _should_add_integer_opt_fallback(branch_name: str, any_candidate_repeated: bool) -> bool:
    if branch_name in ["phase_1_repeated_frac", "phase_2_repeated_frac", "phase_3_repeated_frac_new_curr"]:
        return True
    if branch_name in ["phase_1_new_frac", "phase_2_new_frac", "phase_3_new_frac"]:
        return any_candidate_repeated
    return False


def _should_try_post_integer_cglp(branch_name: str) -> bool:
    return branch_name == "phase_3_repeated_frac_new_curr"


def _make_cross_family_row_guard(two_stage_decomp_module: TwoStageDecompRedo, cut_type: str, preview: Dict[str, Any]):
    def _guard(cut_row: dict, cut_name: str) -> bool:
        peek_record = two_stage_decomp_module.cut_repetition_tracker.peek_by_row(
            cut_type=cut_type,
            row=cut_row,
            include_cut_type=False,
        )
        preview.clear()
        preview.update(peek_record)
        preview["cut_name"] = cut_name
        return not peek_record["is_repeated"]

    return _guard


def _add_candidate_cut(
    two_stage_decomp_module: TwoStageDecompRedo,
    cut_type: str,
    sub_sce_list: List,
    ite_name: str,
    curr_main_result: dict,
    frac_main_result: dict,
    sub_obj_value_for_benders: float,
    frac_sub_obj_value_for_benders: Optional[float],
    constr_dual_bi: Any,
    constr_var_map_bi: Any,
    constr_dual_frac: Any,
    constr_var_map_frac: Any,
    exact_sub_obj_value: Optional[float],
    max_lagrangian_ite: int = 10,
) -> Dict[str, Any]:
    preview = {"is_repeated": False}
    row_guard = _make_cross_family_row_guard(
        two_stage_decomp_module=two_stage_decomp_module,
        cut_type=cut_type,
        preview=preview,
    )

    if cut_type == "benders":
        if constr_dual_frac is None or constr_var_map_frac is None or frac_sub_obj_value_for_benders is None:
            raise ValueError("Fractional Benders cut requires fractional dual info, var map, and relaxed sub objective.")
        two_stage_decomp_module.add_benders_cut(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            sub_obj_value=frac_sub_obj_value_for_benders,
            constr_dual_info=constr_dual_frac,
            constr_var_map=constr_var_map_frac,
            forward_sol=frac_main_result,
            should_add_cut_row=row_guard,
            include_cut_type_in_repetition=False,
        )
    elif cut_type == "strengthen_benders" and constr_dual_frac is not None:
        two_stage_decomp_module.add_strengthen_benders_cut(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            given_main_result=frac_main_result,
            given_dual_info=constr_dual_frac,
            constr_var_map=constr_var_map_frac,
            should_add_cut_row=row_guard,
            include_cut_type_in_repetition=False,
        )
    elif cut_type == "cglp" and exact_sub_obj_value is not None:
        two_stage_decomp_module.add_cglp_cut(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            forward_sol=curr_main_result,
            exact_sub_obj_value=exact_sub_obj_value,
            should_add_cut_row=row_guard,
            include_cut_type_in_repetition=False,
        )
    elif cut_type == "lagrangian" and constr_dual_frac is not None:
        two_stage_decomp_module.add_lagrangian_cut(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            given_main_result=frac_main_result,
            given_dual_info=constr_dual_frac,
            constr_var_map=constr_var_map_frac,
            max_ite_num=max_lagrangian_ite,
            should_add_cut_row=row_guard,
            include_cut_type_in_repetition=False,
        )
    elif cut_type == "integer_opt" and exact_sub_obj_value is not None:
        two_stage_decomp_module.add_integer_opt_cut(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            exact_sub_obj_value=exact_sub_obj_value,
            forward_sol=curr_main_result,
            lower_bound=0.0,
            should_add_cut_row=row_guard,
            include_cut_type_in_repetition=False,
        )

    return preview


def _add_integer_opt_fallback(
    two_stage_decomp_module: TwoStageDecompRedo,
    sub_sce_list: List,
    ite_name: str,
    curr_main_result: dict,
    exact_sub_obj_value: float,
) -> None:
    two_stage_decomp_module.add_integer_opt_cut(
        sub_sce_list=sub_sce_list,
        ite_name=ite_name,
        exact_sub_obj_value=exact_sub_obj_value,
        forward_sol=curr_main_result,
        lower_bound=0.0,
        include_cut_type_in_repetition=False,
    )


def _ensure_exact_sub_obj_value(
    two_stage_decomp_module: TwoStageDecompRedo,
    sub_sce_list: List,
    curr_main_result: dict,
    exact_sub_obj_value: Optional[float],
) -> float:
    if exact_sub_obj_value is not None:
        return exact_sub_obj_value

    two_stage_decomp_module.solve_sub_model()
    exact_sub_obj_value = two_stage_decomp_module.current_sub_model.get_obj_value()
    two_stage_decomp_module._register_exact_point_for_group(
        sub_sce_list=sub_sce_list,
        forward_sol=curr_main_result,
        exact_sub_obj_value=exact_sub_obj_value,
    )
    return exact_sub_obj_value


def _compute_main_stage_obj_value(two_stage_decomp_module: TwoStageDecompRedo, main_result: dict) -> float:
    return two_stage_decomp_module.model_main.evaluate_design_objective_for_solution(main_result)


def _write_iteration_branch_cut_report(
    output_dir: str,
    iteration_branch_cut_records: List[dict],
    file_name: str = "iteration_branch_cut_report.json",
) -> str:
    path = os.path.join(output_dir, file_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"iteration_branch_cut_records": iteration_branch_cut_records}, f, indent=2)
    return path


def _summarize_incumbent_records(sce_prob: dict, incumbent_records: List[tuple]) -> Tuple[float, dict]:
    incumbent_obj_value = 0.0
    temp_iter_sub_results = {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}}

    for sub_sce_list, sub_obj_w_main, sub_res in incumbent_records:
        incumbent_obj_value += sub_obj_w_main * sum(sce_prob[s_idx] for s_idx in sub_sce_list)
        for key, val in sub_res[VarName.DG_ACTIVE_POWER].items():
            temp_iter_sub_results[VarName.DG_ACTIVE_POWER][key] = val
        for j, val in sub_res[VarName.DG_RATED_POWER].items():
            for s in sub_sce_list:
                temp_iter_sub_results[VarName.DG_RATED_POWER][(j, s)] = val

    return incumbent_obj_value, temp_iter_sub_results


def _evaluate_incumbent_for_main_result(
    two_stage_decomp_module: TwoStageDecompRedo,
    sce_group_list: List[List],
    ite_name: str,
    main_result: dict,
    main_stage_obj: float,
    sce_prob: dict,
) -> Tuple[float, dict]:
    incumbent_records = []
    for sub_sce_list in sce_group_list:
        two_stage_decomp_module._gather_one_group_cut_data(
            sub_sce_list=sub_sce_list,
            curr_main_result=main_result,
            frac_main_result=main_result,
            ite_name=ite_name,
            add_strengthen=False,
            add_lagrangian=False,
            add_cglp=False,
            need_frac_sub_model=False,
            need_exact_sub_obj=False,
            main_stage_obj=main_stage_obj,
            incumbent_records=incumbent_records,
        )

    return _summarize_incumbent_records(sce_prob=sce_prob, incumbent_records=incumbent_records)


def _solve_fixed_main_result_lower_bound(
    two_stage_decomp_module: TwoStageDecompRedo,
    ite_name: str,
    main_result: dict,
) -> float:
    try:
        two_stage_decomp_module.model_main.fix_variable_value(var_fix_info=main_result)
        two_stage_decomp_module.solve_main_stage_model()
        _, main_bound = two_stage_decomp_module.record_main_stage_model(ite_name=ite_name)
    finally:
        two_stage_decomp_module.model_main.recover_variable_from_fixed()

    return main_bound


def run_decomp_module_test(
    enable_log_output=False,
    enable_result_output=True,
    scenario_list=None,
    time_list=None,
    test_read_method=None,
    test_file_path=None,
    data_set_name='function_test',
    max_iterations=4,
    output_label=None,
    phase_1_end_iteration: Optional[int] = None,
    phase_2_end_iteration: Optional[int] = None,
    phase_3_end_iteration: Optional[int] = None,
    reference_solution_path: Optional[str] = None,
    record_incumbent_every_k: Optional[int] = None,
    use_aggregated_cuts: bool = False,
    main_problem_model_type: str = MainProblemModelTypeName.COLLAPSED_NO_TIME,
    main_problem_hat_data_method: str = MainProblemHatDataMethodName.WEIGHTED_AVERAGE,
    added_obj_term_weight: float = 0.0,
    phase_1_added_obj_term_weight: Optional[float] = None,
    phase_2_added_obj_term_weight: Optional[float] = None,
    phase_3_added_obj_term_weight: Optional[float] = None,
):
    if scenario_list is None:
        scenario_list = [
            's_1',
            's_2',
            's_3',
        ]
    if time_list is None:
        time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if test_read_method is None:
        test_read_method = InputMethodName.LOCAL_CSV
    if test_file_path is None:
        test_file_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file'
    if data_set_name is None:
        data_set_name = 'function_test'
    if enable_result_output is None:
        enable_result_output = False

    _check_phase_end_iterations(
        phase_1_end_iteration=phase_1_end_iteration,
        phase_2_end_iteration=phase_2_end_iteration,
        phase_3_end_iteration=phase_3_end_iteration,
        max_iterations=max_iterations,
    )

    if record_incumbent_every_k is not None and (not isinstance(record_incumbent_every_k, int) or record_incumbent_every_k < 1):
        raise ValueError("record_incumbent_every_k must be a positive integer or None")
    if use_aggregated_cuts:
        raise NotImplementedError("dynamic_decomposition_test.py requires use_aggregated_cuts=False.")

    init_logger(enable_file_output=enable_log_output)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', timestamp)
    if output_label is not None:
        output_dir = output_dir + '_' + output_label
    if enable_result_output:
        os.makedirs(output_dir, exist_ok=True)

    r = DataReader(
        read_method=test_read_method,
        local_file_path=test_file_path,
        data_set_name=data_set_name
    )

    r.read()
    print("CSV read success:")

    data_read_path = test_file_path + '/' + data_set_name + '/'

    two_stage_decomp_module = TwoStageDecompRedo(
        raw_data=r.raw_data,
        time_list=time_list,
        scenario_list=scenario_list,
    )
    two_stage_decomp_module.data_processor_module.main_problem_model_type = main_problem_model_type
    two_stage_decomp_module.data_processor_module.main_problem_hat_data_method = main_problem_hat_data_method

    two_stage_decomp_module.build_main_stage_model()
    two_stage_decomp_module.model_main.set_added_obj_term_weight(added_obj_term_weight)

    (expectation, keys_above, keys_below, sum_above, sum_below,
     true_keys_above, true_keys_below, true_sum_above, true_sum_below, best_hat) = analyze_scenario_solutions(
        dataset_path=data_read_path,
        scenario_list=scenario_list,
        hat=0.3334
    )

    if (
        true_keys_above is not None
        and true_keys_below is not None
        and true_sum_above is not None
        and true_sum_below is not None
    ):
        true_sum_above_value = sum(true_sum_above.values())
        true_sum_below_value = sum(true_sum_below.values())
        two_stage_decomp_module.model_main.set_pretrained_cut(
            above_var=true_keys_above,
            below_var=true_keys_below,
            above_sum=true_sum_above_value,
            below_sum=true_sum_below_value,
        )
    else:
        sum_above_value = sum(sum_above.values())
        sum_below_value = sum(sum_below.values())
        two_stage_decomp_module.model_main.set_pretrained_cut(
            above_var=keys_above,
            below_var=keys_below,
            above_sum=sum_above_value,
            below_sum=sum_below_value,
        )

    sce_group_list = [[s] for s in scenario_list]

    best_obj = float('inf')
    best_sub_results = None
    curr_main_result_history = []
    frac_main_result_history = []
    curr_main_repetition_records = []
    frac_main_repetition_records = []
    iteration_branch_cut_records = []
    curr_main_seen_signatures = {}
    frac_main_seen_signatures = {}

    if enable_result_output:
        two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

    for ite_num in range(max_iterations):
        logger.info(f"Current iteration: {ite_num}")

        ite_name = str(ite_num)
        ite_phase_name = _get_phase_name(
            ite_num=ite_num,
            phase_1_end_iteration=phase_1_end_iteration,
            phase_2_end_iteration=phase_2_end_iteration,
            phase_3_end_iteration=phase_3_end_iteration,
        )
        temp_iter_sub_results = {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}}

        curr_added_obj_term_weight = _get_added_obj_term_weight_for_phase(
            phase_name=ite_phase_name,
            default_weight=added_obj_term_weight,
            phase_1_weight=phase_1_added_obj_term_weight,
            phase_2_weight=phase_2_added_obj_term_weight,
            phase_3_weight=phase_3_added_obj_term_weight,
        )
        two_stage_decomp_module.model_main.set_added_obj_term_weight(curr_added_obj_term_weight)

        two_stage_decomp_module.solve_main_stage_model()
        main_stage_obj, main_bound = two_stage_decomp_module.record_main_stage_model(ite_name=ite_name)

        curr_main_result = two_stage_decomp_module.model_main.get_result(
            [VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]
        )

        curr_incumbent_obj_value = 0
        do_record_incumbent = record_incumbent_every_k is None or ite_num % record_incumbent_every_k == 0

        two_stage_decomp_module.solve_main_stage_relaxed_model()
        frac_main_result = two_stage_decomp_module.model_main.get_result_relaxed(
            [VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]
        )

        curr_is_repeated, curr_first_seen_ite = detect_and_record_repetition(
            value=curr_main_result,
            seen_signatures=curr_main_seen_signatures,
            iteration=ite_num,
            tol=1e-6,
        )
        curr_main_repetition_records.append({
            "iteration": ite_num,
            "is_repeated": curr_is_repeated,
            "first_seen_iteration": curr_first_seen_ite,
        })
        frac_is_repeated, frac_first_seen_ite = detect_and_record_repetition(
            value=frac_main_result,
            seen_signatures=frac_main_seen_signatures,
            iteration=ite_num,
            tol=1e-6,
        )
        frac_main_repetition_records.append({
            "iteration": ite_num,
            "is_repeated": frac_is_repeated,
            "first_seen_iteration": frac_first_seen_ite,
        })
        curr_main_result_history.append(copy.deepcopy(curr_main_result))
        frac_main_result_history.append(copy.deepcopy(frac_main_result))

        branch_name = _get_branch_name(
            phase_name=ite_phase_name,
            frac_is_repeated=frac_is_repeated,
            curr_is_repeated=curr_is_repeated,
        )
        candidate_cut_types = _get_candidate_cut_types(branch_name)
        iteration_cut_record = {
            "iteration": ite_name,
            "phase_name": ite_phase_name,
            "branch_name": branch_name,
            "candidate_cut_types": list(candidate_cut_types),
            "scenario_group_records": [],
        }
        add_strengthen = ite_phase_name in ["phase_2", "phase_3"]
        add_lagrangian = ite_phase_name == "phase_3"
        need_exact_sub_obj = do_record_incumbent or frac_is_repeated or ite_phase_name == "phase_3"
        sce_prob = two_stage_decomp_module.sce_prob_dict
        incumbent_records = [] if do_record_incumbent else None

        for sub_sce_list in sce_group_list:
            (
                sub_obj_value_for_benders,
                constr_dual_bi,
                constr_var_map_bi,
                constr_dual_frac,
                constr_var_map_frac,
                exact_sub_obj_value,
                frac_sub_obj_value_for_benders,
            ) = two_stage_decomp_module._gather_one_group_cut_data(
                sub_sce_list=sub_sce_list,
                curr_main_result=curr_main_result,
                frac_main_result=frac_main_result,
                ite_name=ite_name,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=False,
                need_frac_sub_model="benders" in candidate_cut_types,
                need_exact_sub_obj=need_exact_sub_obj,
                main_stage_obj=main_stage_obj if do_record_incumbent else None,
                incumbent_records=incumbent_records,
            )

            any_candidate_repeated = False
            group_record = {
                "scenario_group": list(sub_sce_list),
                "attempted_cut_types": list(candidate_cut_types),
                "candidate_cut_types": list(candidate_cut_types),
                "added_cut_types": [],
                "repeated_candidate_cut_types": [],
                "integer_opt_fallback_added": False,
                "post_integer_cglp_added": False,
            }
            for cut_type in candidate_cut_types:
                preview = _add_candidate_cut(
                    two_stage_decomp_module=two_stage_decomp_module,
                    cut_type=cut_type,
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    curr_main_result=curr_main_result,
                    frac_main_result=frac_main_result,
                    sub_obj_value_for_benders=sub_obj_value_for_benders,
                    frac_sub_obj_value_for_benders=frac_sub_obj_value_for_benders,
                    constr_dual_bi=constr_dual_bi,
                    constr_var_map_bi=constr_var_map_bi,
                    constr_dual_frac=constr_dual_frac,
                    constr_var_map_frac=constr_var_map_frac,
                    exact_sub_obj_value=exact_sub_obj_value,
                    max_lagrangian_ite=10,
                )
                if preview.get("is_repeated", False):
                    any_candidate_repeated = True
                    group_record["repeated_candidate_cut_types"].append(cut_type)
                else:
                    group_record["added_cut_types"].append(cut_type)

            if _should_add_integer_opt_fallback(branch_name, any_candidate_repeated):
                exact_sub_obj_value = _ensure_exact_sub_obj_value(
                    two_stage_decomp_module=two_stage_decomp_module,
                    sub_sce_list=sub_sce_list,
                    curr_main_result=curr_main_result,
                    exact_sub_obj_value=exact_sub_obj_value,
                )
                _add_integer_opt_fallback(
                    two_stage_decomp_module=two_stage_decomp_module,
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    curr_main_result=curr_main_result,
                    exact_sub_obj_value=exact_sub_obj_value,
                )
                group_record["integer_opt_fallback_added"] = True

            if _should_try_post_integer_cglp(branch_name):
                preview = _add_candidate_cut(
                    two_stage_decomp_module=two_stage_decomp_module,
                    cut_type="cglp",
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    curr_main_result=curr_main_result,
                    frac_main_result=frac_main_result,
                    sub_obj_value_for_benders=sub_obj_value_for_benders,
                    frac_sub_obj_value_for_benders=frac_sub_obj_value_for_benders,
                    constr_dual_bi=constr_dual_bi,
                    constr_var_map_bi=constr_var_map_bi,
                    constr_dual_frac=constr_dual_frac,
                    constr_var_map_frac=constr_var_map_frac,
                    exact_sub_obj_value=exact_sub_obj_value,
                    max_lagrangian_ite=10,
                )
                if preview.get("is_repeated", False):
                    group_record["repeated_candidate_cut_types"].append("cglp_post_integer")
                else:
                    group_record["added_cut_types"].append("cglp_post_integer")
                    group_record["post_integer_cglp_added"] = True

            iteration_cut_record["scenario_group_records"].append(group_record)

        if incumbent_records is not None:
            curr_incumbent_obj_value, temp_iter_sub_results = _summarize_incumbent_records(
                sce_prob=sce_prob,
                incumbent_records=incumbent_records,
            )

        if do_record_incumbent:
            two_stage_decomp_module.ite_obj_value_dict[ite_name]['sub_obj(best_incumbent)'] = {
                'sub_p_total': curr_incumbent_obj_value
            }
            new_best = curr_incumbent_obj_value < best_obj
            best_obj = min(best_obj, curr_incumbent_obj_value)
            if new_best:
                best_sub_results = copy.deepcopy(temp_iter_sub_results)

        if enable_result_output:
            two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

        if do_record_incumbent and best_obj - main_bound <= 0.01 * best_obj:
            iteration_branch_cut_records.append(iteration_cut_record)
            logger.info(f'Ite {ite_name} has reached convergence by the gap of 1%')
            break

        iteration_branch_cut_records.append(iteration_cut_record)

    if reference_solution_path is not None:
        two_stage_decomp_module.model_main.set_added_obj_term_weight(added_obj_term_weight)
        reference_main_result = load_main_stage_solution_json(reference_solution_path)
        reference_main_stage_obj = _compute_main_stage_obj_value(
            two_stage_decomp_module=two_stage_decomp_module,
            main_result=reference_main_result,
        )
        ref_0_name = "ref_0"
        ref_0_incumbent_records = []
        two_stage_decomp_module.run_iteration_cuts_for_given_main_result(
            sce_group_list=sce_group_list,
            ite_name=ref_0_name,
            given_main_result=reference_main_result,
            add_benders=True,
            add_strengthen=True,
            add_lagrangian=True,
            add_cglp=True,
            main_stage_obj=reference_main_stage_obj,
            incumbent_records=ref_0_incumbent_records,
            max_lagrangian_ite=10,
        )
        ref_0_incumbent_obj_value, temp_iter_sub_results = _summarize_incumbent_records(
            sce_prob=sce_prob,
            incumbent_records=ref_0_incumbent_records,
        )
        two_stage_decomp_module.ite_obj_value_dict[ref_0_name]['sub_obj(best_incumbent)'] = {
            'sub_p_total': ref_0_incumbent_obj_value
        }
        _solve_fixed_main_result_lower_bound(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_0_name,
            main_result=reference_main_result,
        )
        new_best = ref_0_incumbent_obj_value < best_obj
        best_obj = min(best_obj, ref_0_incumbent_obj_value)
        if new_best:
            best_sub_results = copy.deepcopy(temp_iter_sub_results)
        iteration_branch_cut_records.append({
            "iteration": ref_0_name,
            "phase_name": "reference_solution_phase",
            "branch_name": "reference_generate_all_cuts",
            "candidate_cut_types": ["benders", "strengthen_benders", "lagrangian", "cglp"],
            "scenario_group_records": [
                {
                    "scenario_group": list(sub_sce_list),
                    "attempted_cut_types": ["benders", "strengthen_benders", "lagrangian", "cglp"],
                    "candidate_cut_types": ["benders", "strengthen_benders", "lagrangian", "cglp"],
                    "added_cut_types": [],
                    "repeated_candidate_cut_types": [],
                    "integer_opt_fallback_added": False,
                    "post_integer_cglp_added": False,
                }
                for sub_sce_list in sce_group_list
            ],
        })
        if enable_result_output:
            two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

        ref_1_name = "ref_1"
        two_stage_decomp_module.solve_main_stage_model()
        ref_1_main_stage_obj, ref_1_main_bound = two_stage_decomp_module.record_main_stage_model(ite_name=ref_1_name)
        ref_1_main_result = two_stage_decomp_module.model_main.get_result(
            [VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]
        )
        ref_1_incumbent_obj_value, temp_iter_sub_results = _evaluate_incumbent_for_main_result(
            two_stage_decomp_module=two_stage_decomp_module,
            sce_group_list=sce_group_list,
            ite_name=ref_1_name,
            main_result=ref_1_main_result,
            main_stage_obj=ref_1_main_stage_obj,
            sce_prob=sce_prob,
        )
        two_stage_decomp_module.ite_obj_value_dict[ref_1_name]['sub_obj(best_incumbent)'] = {
            'sub_p_total': ref_1_incumbent_obj_value
        }
        new_best = ref_1_incumbent_obj_value < best_obj
        best_obj = min(best_obj, ref_1_incumbent_obj_value)
        if new_best:
            best_sub_results = copy.deepcopy(temp_iter_sub_results)
        iteration_branch_cut_records.append({
            "iteration": ref_1_name,
            "phase_name": "reference_solution_phase",
            "branch_name": "reference_solved_main_iteration",
            "candidate_cut_types": [],
            "scenario_group_records": [],
        })
        if enable_result_output:
            two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

        ref_2_name = "ref_2"
        ref_2_main_stage_obj = reference_main_stage_obj
        ref_2_incumbent_obj_value, temp_iter_sub_results = _evaluate_incumbent_for_main_result(
            two_stage_decomp_module=two_stage_decomp_module,
            sce_group_list=sce_group_list,
            ite_name=ref_2_name,
            main_result=reference_main_result,
            main_stage_obj=ref_2_main_stage_obj,
            sce_prob=sce_prob,
        )
        two_stage_decomp_module.ite_obj_value_dict[ref_2_name]['sub_obj(best_incumbent)'] = {
            'sub_p_total': ref_2_incumbent_obj_value
        }
        _solve_fixed_main_result_lower_bound(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_2_name,
            main_result=reference_main_result,
        )
        new_best = ref_2_incumbent_obj_value < best_obj
        best_obj = min(best_obj, ref_2_incumbent_obj_value)
        if new_best:
            best_sub_results = copy.deepcopy(temp_iter_sub_results)
        iteration_branch_cut_records.append({
            "iteration": ref_2_name,
            "phase_name": "reference_solution_phase",
            "branch_name": "reference_fixed_main_tightness_check",
            "candidate_cut_types": [],
            "scenario_group_records": [],
        })
        if enable_result_output:
            two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

    os.makedirs(output_dir, exist_ok=True)
    two_stage_decomp_module.model_main.write_file(file_name=output_dir + '/main_model.lp')

    write_decomp_run_parameters(
        output_dir=output_dir,
        params={
            "scenario_list": scenario_list,
            "data_set_name": data_set_name,
            "max_iterations": max_iterations,
            "phase_1_end_iteration": phase_1_end_iteration,
            "phase_2_end_iteration": phase_2_end_iteration,
            "phase_3_end_iteration": phase_3_end_iteration,
            "reference_solution_path": reference_solution_path,
            "record_incumbent_every_k": record_incumbent_every_k,
            "use_aggregated_cuts": use_aggregated_cuts,
            "main_problem_model_type": main_problem_model_type,
            "main_problem_hat_data_method": main_problem_hat_data_method,
            "added_obj_term_weight": added_obj_term_weight,
            "phase_1_added_obj_term_weight": phase_1_added_obj_term_weight,
            "phase_2_added_obj_term_weight": phase_2_added_obj_term_weight,
            "phase_3_added_obj_term_weight": phase_3_added_obj_term_weight,
        },
    )

    if enable_result_output and best_sub_results is not None:
        write_power_usage_capacity_ratio(
            best_sub_results=best_sub_results,
            output_dir=output_dir,
        )

    repetitive_cut_records = two_stage_decomp_module.get_repetitive_cut_records()
    report_repetitive_cuts(
        repetitive_cut_records=repetitive_cut_records,
        enable_result_output=enable_result_output,
        output_dir=output_dir,
        logger=logger,
    )
    if enable_result_output:
        write_main_result_repetition_report(
            output_dir=output_dir,
            curr_main_result_history=curr_main_result_history,
            frac_main_result_history=frac_main_result_history,
            curr_main_repetition_records=curr_main_repetition_records,
            frac_main_repetition_records=frac_main_repetition_records,
        )
        _write_iteration_branch_cut_report(
            output_dir=output_dir,
            iteration_branch_cut_records=iteration_branch_cut_records,
        )

    plot_iter_obj_curves(
        ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
        output_dir=output_dir,
        real_objective_value=3822465
    )
    return {
        "repetitive_cut_count": len(repetitive_cut_records),
        "repetitive_cut_records": repetitive_cut_records,
    }


if __name__ == "__main__":
    _max_iter = 55
    _phase_1_end = 0
    _phase_2_end = _max_iter - 4
    _phase_3_end = _max_iter - 1
    run_decomp_module_test(
        enable_log_output=False,
        enable_result_output=True,
        scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        # scenario_list=['s_3'],
        # scenario_list=['s_1', 's_2', 's_3'],
        time_list=None,
        test_read_method=None,
        test_file_path=None,
        data_set_name='function_test_fixed_rated_p',
        # data_set_name='IEEE123bus',
        max_iterations=_max_iter,
        output_label='new_m_agg_t',
        phase_1_end_iteration=_phase_1_end,
        phase_2_end_iteration=_phase_2_end,
        phase_3_end_iteration=_phase_3_end,
        # reference_solution_path=(
        #     '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/'
        #     'function_test_fixed_rated_p/solutions/s_1_s_3_s_5.json'
        # ),
        record_incumbent_every_k=2,
        use_aggregated_cuts=False,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS,
        added_obj_term_weight=1.0,
        phase_1_added_obj_term_weight=None,
        phase_2_added_obj_term_weight=None,
        phase_3_added_obj_term_weight=0,
    )
