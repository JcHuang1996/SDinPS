# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

from algo.two_stage_decomp_redo import TwoStageDecompRedo

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from util.project_logger import init_logger
from util.names import VarName, InputMethodName
from dao.data_reader import DataReader

from util.converge_visual import plot_iter_obj_curves

from util.tools import iter_general_csv, iter_sub_prob_info, write_decomp_run_parameters, write_power_usage_capacity_ratio

from datetime import datetime
import copy
import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def _check_iter_range(name: str, val: Tuple[int, int], max_iterations: int) -> None:
    """Raise ValueError if (a, b) does not satisfy 0 <= a <= b <= max_iterations."""
    if len(val) != 2:
        raise ValueError(f"{name} must be a 2-tuple (first_iter, last_iter), got {val}")
    a, b = val
    if not (0 <= a <= b <= max_iterations):
        raise ValueError(
            f"{name} (first_iter, last_iter) must satisfy 0 <= first <= last <= max_iterations "
            f"(max_iterations={max_iterations}), got ({a}, {b})"
        )


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
    benders_cut_iter_range: Optional[Tuple[int, int]] = None,
    strengthen_benders_cut_iter_range: Optional[Tuple[int, int]] = None,
    lagrangian_cut_iter_range: Optional[Tuple[int, int]] = None,
    record_incumbent_every_k: Optional[int] = None,
):
    """Run the decomposition module test.

    Args:
        enable_log_output: Whether to write running logs to log output folder (default: False)
        enable_result_output: Whether to write result output (default: True)
        scenario_list: List of scenarios to test (default: ['s_1', 's_2', 's_3'])
        time_list: List of time periods (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        test_read_method: Input method for reading data (default: InputMethodName.LOCAL_CSV)
        test_file_path: Path to test data files (default: '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file')
        data_set_name: Name of the data set (default: 'function test')
        max_iterations: Maximum number of iterations to run (default: 10)
        benders_cut_iter_range: (first_iter, last_iter) to add Benders cuts; None to skip.
        strengthen_benders_cut_iter_range: (first_iter, last_iter) to add strengthened Benders cuts; None to skip.
        lagrangian_cut_iter_range: (first_iter, last_iter) to add Lagrangian cuts; None to skip.
        record_incumbent_every_k: If a positive integer k, record incumbent and run convergence check only every k
            iterations (0, k, 2k, ...). If None, record every iteration. When not recording, solve_sub_model() is
            skipped and Benders cut uses relaxed LP objective (get_relaxed_obj_value()); strengthen/Lagrangian
            cuts are unchanged (they use curr_sub_frac_model only, not current_sub_model).
    """
    # Set default values
    if scenario_list is None:
        scenario_list = [
            's_1',
            's_2',
            's_3',
            # 's_4',
            # 's_5',
            # 's_6',
            # 's_7',
            # 's_8',
            # 's_9',
            # 's_10'
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

    for name, val in [
        ("benders_cut_iter_range", benders_cut_iter_range),
        ("strengthen_benders_cut_iter_range", strengthen_benders_cut_iter_range),
        ("lagrangian_cut_iter_range", lagrangian_cut_iter_range),
    ]:
        if val is not None:
            _check_iter_range(name, val, max_iterations)

    if record_incumbent_every_k is not None and (not isinstance(record_incumbent_every_k, int) or record_incumbent_every_k < 1):
        raise ValueError("record_incumbent_every_k must be a positive integer or None")

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

    two_stage_decomp_module = TwoStageDecompRedo(
        raw_data=r.raw_data,
        time_list=time_list,
        scenario_list=scenario_list,
    )

    # build main model
    two_stage_decomp_module.build_main_stage_model()

    # decide how to group and iterate the scenarios
    sce_group_list = [
        [s] for s in scenario_list
    ]

    best_obj = float('inf')
    best_sub_results = None  # best-iteration sub results: {VarName.DG_ACTIVE_POWER: {(j,t,s): v}, VarName.DG_RATED_POWER: {(j,s): v}}

    for ite_num in range(max_iterations):

        ite_name = str(ite_num)
        temp_iter_sub_results = {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}}

        # solve main model
        two_stage_decomp_module.solve_main_stage_model()

        # record the main model result
        main_stage_obj, main_bound = two_stage_decomp_module.record_main_stage_model(ite_name=ite_name)

        # record the result of the main model required by the sub problems
        curr_main_result = two_stage_decomp_module.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])

        # the best incumbent objective value will be given by the weighted sum of sub-problem objective values
        # starting from 0 (only when recording)
        curr_incumbent_obj_value = 0
        do_record_incumbent = record_incumbent_every_k is None or ite_num % record_incumbent_every_k == 0

        # the following part aims on collecting relaxed solution from main problem for better cut
        two_stage_decomp_module.solve_main_stage_relaxed_model()
        frac_main_result = two_stage_decomp_module.model_main.get_result_relaxed([VarName.DG_INSTALL, VarName.LINE_HARDEN])

        # iterating by the above division
        for sub_sce_list in sce_group_list:

            # solve sub model based on binary main solution for classical benders cut and recording real sub problem obj
            two_stage_decomp_module.build_sub_model_redo(
                sub_model_sce_list=sub_sce_list,
                given_main_result=curr_main_result,
            )

            local_copy_var_constr_dict = two_stage_decomp_module.current_sub_model.local_copy_constr_info.copy()
            local_copy_var_constr_name = sorted(local_copy_var_constr_dict.keys())

            two_stage_decomp_module.solve_relaxed_sub_model()
            # Benders cut uses relaxed LP objective; collect_dual_opt_sol reads from model_relax (see model_sub.py)
            sub_obj_value_for_benders = two_stage_decomp_module.current_sub_model.get_relaxed_obj_value()

            if do_record_incumbent:
                two_stage_decomp_module.solve_sub_model()
                sub_obj_w_main, _ = two_stage_decomp_module.record_sub_model(
                    main_stage_obj_value=main_stage_obj,
                    ite_name=ite_name,
                )
                curr_incumbent_obj_value += sub_obj_w_main * sum(
                    two_stage_decomp_module.sce_prob_dict[s_idx]
                    for s_idx in sub_sce_list
                )
                # collect pg and pgrt from current sub for possible best-iteration save
                sub_res = two_stage_decomp_module.current_sub_model.get_result(
                    [VarName.DG_ACTIVE_POWER, VarName.DG_RATED_POWER]
                )
                for key, val in sub_res[VarName.DG_ACTIVE_POWER].items():
                    temp_iter_sub_results[VarName.DG_ACTIVE_POWER][key] = val
                for j, val in sub_res[VarName.DG_RATED_POWER].items():
                    for s in sub_sce_list:
                        temp_iter_sub_results[VarName.DG_RATED_POWER][(j, s)] = val

            constr_dual_via_bi_main, constr_var_map_via_bi_main = two_stage_decomp_module.current_sub_model.collect_dual_opt_sol(
                constr_to_collect_list=local_copy_var_constr_name
            )

            if benders_cut_iter_range is not None:
                first_bd, last_bd = benders_cut_iter_range
                if first_bd <= ite_num <= last_bd:
                    two_stage_decomp_module.add_benders_cut(
                        sub_sce_list=sub_sce_list,
                        ite_name=ite_name,
                        sub_obj_value=sub_obj_value_for_benders,
                        constr_dual_info=constr_dual_via_bi_main,
                        constr_var_map=constr_var_map_via_bi_main,
                        forward_sol=curr_main_result,
                    )

            add_strengthen = (
                strengthen_benders_cut_iter_range is not None
                and strengthen_benders_cut_iter_range[0] <= ite_num <= strengthen_benders_cut_iter_range[1]
            )
            add_lagrangian = (
                lagrangian_cut_iter_range is not None
                and lagrangian_cut_iter_range[0] <= ite_num <= lagrangian_cut_iter_range[1]
            )
            # Strengthen Benders and Lagrangian use curr_sub_frac_model only (frac_main_result, duals from
            # solve_sub_frac_model); they do not use current_sub_model or solve_sub_model result.
            if add_strengthen or add_lagrangian:
                two_stage_decomp_module.build_sub_frac_model(
                    sub_model_sce_list=sub_sce_list,
                    given_main_frac_result=frac_main_result
                )
                two_stage_decomp_module.solve_sub_frac_model()
                constr_dual_via_frac_main, constr_var_map_via_frac_main = two_stage_decomp_module.curr_sub_frac_model.collect_dual_opt_sol(
                    constr_to_collect_list=local_copy_var_constr_name
                )

                if add_strengthen:
                    two_stage_decomp_module.add_strengthen_benders_cut(
                        sub_sce_list=sub_sce_list,
                        ite_name=ite_name,
                        given_main_result=frac_main_result,
                        given_dual_info=constr_dual_via_frac_main,
                        constr_var_map=constr_var_map_via_frac_main,
                    )

                if add_lagrangian:
                    two_stage_decomp_module.add_lagrangian_cut(
                        sub_sce_list=sub_sce_list,
                        ite_name=ite_name,
                        given_main_result=frac_main_result,
                        given_dual_info=constr_dual_via_frac_main,
                        constr_var_map=constr_var_map_via_frac_main,
                        max_ite_num=10,
                    )

        if do_record_incumbent:
            two_stage_decomp_module.ite_obj_value_dict[ite_name]['sub_obj(best_incumbent)'] = {
                'sub_p_total': curr_incumbent_obj_value
            }
            new_best = curr_incumbent_obj_value < best_obj
            best_obj = min(best_obj, curr_incumbent_obj_value)
            if new_best:
                best_sub_results = copy.deepcopy(temp_iter_sub_results)
            if best_obj - main_bound <= 0.01 * best_obj:
                logger.info(f'Ite {ite_name} has reached convergence by the gap of 1%')
                break

    two_stage_decomp_module.model_main.write_file(file_name=output_dir+'/main_model.lp')

    iter_general_csv(
        ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
        output_dir=output_dir
    )
    iter_sub_prob_info(
        ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
        output_dir=output_dir,
        scenario_list=scenario_list
    )

    write_decomp_run_parameters(
        output_dir=output_dir,
        scenario_list=scenario_list,
        data_set_name=data_set_name,
        max_iterations=max_iterations,
        benders_cut_iter_range=benders_cut_iter_range,
        strengthen_benders_cut_iter_range=strengthen_benders_cut_iter_range,
        lagrangian_cut_iter_range=lagrangian_cut_iter_range,
    )

    if enable_result_output and best_sub_results is not None:
        write_power_usage_capacity_ratio(
            best_sub_results=best_sub_results,
            output_dir=output_dir,
        )

    # after finishing the algorithm
    plot_iter_obj_curves(
        ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
        output_dir=output_dir,
        real_objective_value=1729496
    )



if __name__ == "__main__":
    _max_iter = 20
    run_decomp_module_test(
        enable_log_output=False,
        enable_result_output=True,
        # scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        # scenario_list=['s_1', 's_3', 's_5'],
        scenario_list=['s_1', 's_2', 's_3'],
        time_list=None,
        test_read_method=None,
        test_file_path=None,
        data_set_name='function_test_symm_broken',
        max_iterations=_max_iter,
        output_label='test',
        benders_cut_iter_range=(0, _max_iter - 1),
        strengthen_benders_cut_iter_range=(0, _max_iter - 1),
        lagrangian_cut_iter_range=None,
        record_incumbent_every_k=2,
    )
