# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

from algo.two_stage_decomp_redo import TwoStageDecompRedo
from model import ModelInnerMinimizationProblem
from algo.algo_simple_tools import compare_rhs

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from util.project_logger import init_logger
from util.names import VarName, InputMethodName
from dao.data_reader import DataReader

from util.converge_visual import plot_iter_obj_curves

from util.tools import iter_general_csv, iter_sub_prob_info

from datetime import datetime
import logging


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

    for ite_num in range(max_iterations):

        ite_name = str(ite_num)

        # solve main model
        two_stage_decomp_module.solve_main_stage_model()

        # record the main model result
        main_stage_obj, total_obj = two_stage_decomp_module.record_main_stage_model(ite_name=ite_name)

        # record the result of the main model required by the sub problems
        curr_main_result = two_stage_decomp_module.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])

        # the best incumbent objective value will be given by the weighted sum of sub-problem objective values
        # starting from 0
        best_incumbent_obj_value = 0

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
            two_stage_decomp_module.solve_sub_model()

            sub_obj_w_main, sub_obj_value = two_stage_decomp_module.record_sub_model(
                main_stage_obj_value=main_stage_obj,
                ite_name=ite_name,
            )

            best_incumbent_obj_value += sub_obj_w_main * sum(
                two_stage_decomp_module.sce_prob_dict[s_idx]
                for s_idx in sub_sce_list
            )

            constr_dual_via_bi_main, constr_var_map_via_bi_main = two_stage_decomp_module.current_sub_model.collect_dual_opt_sol(
                constr_to_collect_list=local_copy_var_constr_name
            )
            sub_model_LP_relax_value = two_stage_decomp_module.current_sub_model.get_relaxed_obj_value()

            # lhs, rhs = two_stage_decomp_module.gen_sce_bds_opt_cut(
            #     lp_opt_value=sub_model_LP_relax_value,
            #     constr_dual_info=constr_dual_via_bi_main,
            #     constr_var_map=constr_var_map_via_bi_main
            # )

            bd_lhs, bd_rhs = two_stage_decomp_module.gen_sce_bds_opt_cut(
                lp_opt_value=sub_obj_value,
                constr_dual_info=constr_dual_via_bi_main,
                constr_var_map=constr_var_map_via_bi_main,
                forward_sol=curr_main_result
            )

            cut_name = f'bd_s_{str(sub_sce_list)}_{ite_name}'

            two_stage_decomp_module.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=bd_lhs, cut_rhs=bd_rhs)

            # solve sub model based on fractional main solution for strengthen benders cut and lagrangian cut
            two_stage_decomp_module.build_sub_frac_model(
                sub_model_sce_list=sub_sce_list,
                given_main_frac_result=frac_main_result
            )
            two_stage_decomp_module.solve_sub_frac_model()
            constr_dual_via_frac_main, constr_var_map_via_frac_main = two_stage_decomp_module.curr_sub_frac_model.collect_dual_opt_sol(
                constr_to_collect_list=local_copy_var_constr_name
            )

            sbd_lhs, sbd_rhs, obtained_sub_obj, obtained_main_sol = two_stage_decomp_module.gen_sce_strengthen_bds_cut(
                sub_model_sce_list=sub_sce_list,
                given_main_result=frac_main_result,
                given_dual_info=constr_dual_via_frac_main,
                constr_var_map=constr_var_map_via_frac_main,
            )
            cut_name = f'str_bd_s_{str(sub_sce_list)}_{ite_name}'
            is_same, max_abs_diff, location = compare_rhs(bd_rhs, sbd_rhs, tol=0.1, show_in_log=True, show_in_console=False)
            if not is_same:
                two_stage_decomp_module.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=sbd_lhs, cut_rhs=sbd_rhs)

            if ite_num >= max_iterations-25:
                lg_lhs, lg_rhs, sub_obj, main_sol = two_stage_decomp_module.gen_sub_sce_lag_cut_heuristic(
                    sub_model_sce_list=sub_sce_list,
                    given_main_result=frac_main_result,
                    given_dual_info=constr_dual_via_frac_main,
                    constr_var_map=constr_var_map_via_frac_main,
                    max_ite_num=10
                )
                cut_name = f'str_lag_s_{str(sub_sce_list)}_{ite_name}'
                is_same, max_abs_diff, location = compare_rhs(bd_rhs, lg_rhs, tol=0.1, show_in_log=True, show_in_console=False)
                if not is_same:
                    two_stage_decomp_module.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=lg_lhs, cut_rhs=lg_rhs)

        two_stage_decomp_module.ite_obj_value_dict[ite_name]['sub_obj(best_incumbent)'] = {
            'sub_p_total': best_incumbent_obj_value
        }

    # # # generate lagrangian cuts of each scenario
    # for sub_sce_list in sce_group_list:
    #     curr_main_result = two_stage_decomp_module.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])
    #     two_stage_decomp_module.gen_sub_sce_lagrangian_cut(
    #         sub_model_sce_list=sub_sce_list,
    #         ini_main_result=curr_main_result,
    #         max_ite_num=10
    #     )
    #
    # # solve main model
    # two_stage_decomp_module.solve_main_stage_model()
    #
    # # record the main model result
    # ite_name = 'lag'
    # main_stage_obj, total_obj = two_stage_decomp_module.record_main_stage_model(ite_name='lag')
    #
    # # record the result of the main model required by the sub problems
    # curr_main_result = two_stage_decomp_module.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])
    #
    # # iterating by the above division
    # for sub_sce_list in sce_group_list:
    #     two_stage_decomp_module.build_sub_model_redo(
    #         sub_model_sce_list=sub_sce_list,
    #         given_main_result=curr_main_result,
    #     )
    #
    #     two_stage_decomp_module.solve_relaxed_sub_model()
    #     two_stage_decomp_module.solve_sub_model()
    #
    #     sub_obj_w_main, sub_obj_value = two_stage_decomp_module.record_sub_model(
    #         main_stage_obj_value=main_stage_obj,
    #         ite_name=ite_name,
    #     )
    #
    #     best_incumbent_obj_value = 0
    #
    #     best_incumbent_obj_value += sub_obj_w_main * sum(
    #         two_stage_decomp_module.sce_prob_dict[s_idx]
    #         for s_idx in sub_sce_list
    #     )

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

    # after finishing the algorithm
    plot_iter_obj_curves(
        ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
        output_dir=output_dir,
        real_objective_value=2999173
    )



if __name__ == "__main__":
    run_decomp_module_test(
        enable_log_output=False,
        enable_result_output=True,
        # scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        scenario_list=['s_1', 's_3', 's_5'],
        time_list=None,
        test_read_method=None,
        test_file_path=None,
        data_set_name='function_test_symm_broken',
        max_iterations=250,
        output_label='longr_ns_correct_sbd_lgc_135s_right'
    )
