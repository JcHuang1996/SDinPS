# -*- coding: utf-8 -*-
# @Time     : 2026/03/24
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

from algo.two_stage_decomp_redo import TwoStageDecompRedo
from algo.algo_simple_tools import detect_and_record_repetition
from algo.runner_helpers import (
    IncumbentTracker,
    add_candidate_cut as _add_candidate_cut,
    add_integer_opt_fallback as _add_integer_opt_fallback,
    check_phase_end_iterations as _check_phase_end_iterations,
    compute_main_stage_obj_value as _compute_main_stage_obj_value,
    ensure_exact_sub_obj_value as _ensure_exact_sub_obj_value,
    evaluate_incumbent_for_main_result as _evaluate_incumbent_for_main_result,
    get_added_obj_term_weight_for_phase as _get_added_obj_term_weight_for_phase,
    get_branch_name as _get_branch_name,
    get_candidate_cut_types as _get_candidate_cut_types,
    get_phase_name as _get_phase_name,
    setup_pretrained_cut,
    should_add_integer_opt_fallback as _should_add_integer_opt_fallback,
    should_try_post_integer_cglp as _should_try_post_integer_cglp,
    solve_fixed_main_result_lower_bound as _solve_fixed_main_result_lower_bound,
    write_iteration_branch_cut_report as _write_iteration_branch_cut_report,
)

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
from typing import List, Optional


logger = logging.getLogger(__name__)


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

    setup_pretrained_cut(
        two_stage_decomp_module=two_stage_decomp_module,
        dataset_path=data_read_path,
        scenario_list=scenario_list,
        hat=0.3334,
        use_fallback_partition=True,
        logger=logger,
    )

    sce_group_list = [[s] for s in scenario_list]

    incumbent_tracker = IncumbentTracker()
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
            curr_incumbent_obj_value, _, _ = incumbent_tracker.update_from_records(
                two_stage_decomp_module=two_stage_decomp_module,
                ite_name=ite_name,
                sce_prob=sce_prob,
                incumbent_records=incumbent_records,
            )

        if enable_result_output:
            two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)

        if do_record_incumbent and incumbent_tracker.has_converged(main_bound=main_bound, rel_gap=0.01):
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
        ref_0_incumbent_obj_value, _, _ = incumbent_tracker.update_from_records(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_0_name,
            sce_prob=sce_prob,
            incumbent_records=ref_0_incumbent_records,
        )
        _solve_fixed_main_result_lower_bound(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_0_name,
            main_result=reference_main_result,
        )
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
        incumbent_tracker.update_from_value(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_1_name,
            incumbent_obj_value=ref_1_incumbent_obj_value,
            sub_results=temp_iter_sub_results,
        )
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
        _solve_fixed_main_result_lower_bound(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_2_name,
            main_result=reference_main_result,
        )
        incumbent_tracker.update_from_value(
            two_stage_decomp_module=two_stage_decomp_module,
            ite_name=ref_2_name,
            incumbent_obj_value=ref_2_incumbent_obj_value,
            sub_results=temp_iter_sub_results,
        )
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

    if enable_result_output and incumbent_tracker.best_sub_results is not None:
        write_power_usage_capacity_ratio(
            best_sub_results=incumbent_tracker.best_sub_results,
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
