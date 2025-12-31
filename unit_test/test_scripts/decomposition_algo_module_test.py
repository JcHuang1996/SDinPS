# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com
from turtledemo.clock import current_day

from flask import current_app

from util.headers import *
from util.names import *
from util.tools import *
from util.project_logger import init_logger
from dao.data_reader import DataReader
from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub, ModelCombined
from algo.TwoStageDecomposition import TwoStageDecomposition
from algo.algo_simple_tools import *
from util.virsualization import plot_iter_obj_curves

import numpy as np
import pandas as pd
from datetime import datetime
from gurobipy import GRB
import logging
import os

# control whether to write running logs to log output folder
ENABLE_LOG_OUTPUT = False  # set to 'False' to disable log file creation
# ENABLE_RESULT_OUTPUT = False
ENABLE_RESULT_OUTPUT = True

init_logger(enable_file_output=ENABLE_LOG_OUTPUT)

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
time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

logger = logging.getLogger(__name__)

test_read_method = InputMethodName.LOCAL_CSV
test_file_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file'
data_set_name = 'function test'

timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', timestamp)
if ENABLE_RESULT_OUTPUT:
    os.makedirs(output_dir, exist_ok=True)

r = DataReader(
    read_method=test_read_method,
    local_file_path=test_file_path,
    data_set_name=data_set_name
)

r.read()
print("CSV read success:")

SDDiP_module = TwoStageDecomposition(raw_data=r.raw_data, scenario_list=scenario_list, time_list=time_list)

# =========================================
# estimate the obj value lb for every scenario
# =========================================

# by solving scenario full formulation
obj_lb_dict = {}
single_s_optimal_main_result_dict = {}
for s in scenario_list:
    obj_lb_dict[s], single_s_optimal_main_result_dict[s] = SDDiP_module.sub_model_lb_estimator(strategy='aggressive', sub_model_sce_list=[s])

# # by solving combined full formulation for aggressive bound
# obj_lb_real, optimal_main_result = SDDiP_module.sub_model_lb_estimator(strategy='aggressive', sub_model_sce_list=scenario_list)
#
# for s in obj_lb_dict.keys():
#     obj_lb_dict[s] = obj_lb_real * 0.965

# # =========================================
# # estimate the big-M
# # =========================================
# R_POWER_M, V_FLOW_M = SDDiP_module.big_M_estimator(sub_model_sce_list=scenario_list)
# note: in opt, R_POWER_M = 1.300015592, V_FLOW_M = 1.266

# build the main model
SDDiP_module.build_main_stage_model()

# # =========================================
# # user-iteration
# # cut generation are based on given main results
# # =========================================
#
# for s in sorted(single_s_optimal_main_result_dict.keys()):
#     s_main_result = single_s_optimal_main_result_dict[s]
#     ite_name = str('init' + s)
#     SDDiP_module.execute_single_iteration(
#         iteration_name=ite_name,
#         est_sub_lb_dict=obj_lb_dict,
#         given_main_result=s_main_result,
#         if_benders_cut=1,
#         if_l_shaped_cut=0,
#         l_shaped_cut_enforce=0,
#         record_incumbent=False
#     )

# # load warm start result
# warm_start_points = load_warm_start('/Users/huangjiacheng/OR591/unit test/test_local_csv_file/function test/warm_start.csv')
# for w_key in sorted(warm_start_points.keys()):
#     w_main_result = warm_start_points[w_key]
#     ite_name = str('init_' + w_key)
#     SDDiP_module.execute_single_iteration(
#         iteration_name=ite_name,
#         est_sub_lb_dict=obj_lb_dict,
#         given_main_result=w_main_result,
#         if_benders_cut=1,
#         if_l_shaped_cut=1,
#         l_shaped_cut_enforce=0,
#         record_incumbent=False
#     )
#
# node_to_est_list = [
#     'node_1',
#     'node_2',
#     'node_7',
#     'node_10',
#     'node_11',
#     'node_12',
#     'node_13',
#     'node_14',
#     'node_16',
#     'node_17',
#     'node_18',
#     'node_21',
#     'node_23',
#     'node_24',
#     'node_26',
#     'node_27',
#     'node_28',
#     'node_29',
#     'node_30',
#     'node_31',
#     'node_32',
#     'node_33'
# ]
#
# n_obj_list_for_debug = []
# for s in scenario_list:
#     for n_idx in node_to_est_list:
#         n_obj_value = SDDiP_module.user_node_decision_estimator(sub_model_sce_list=[s], node_to_est_idx=n_idx)
#         n_obj_list_for_debug.append(n_obj_value)
#         SDDiP_module.model_main.add_user_cut_node_estimation(
#             sub_problem_sce_list=[s],
#             sub_model_obj_value=n_obj_value,
#             sub_model_obj_lb=obj_lb_dict[s],
#             estimated_node_key=(VarName.DG_INSTALL, n_idx),
#             track_idx=n_idx
#         )

# =========================================
# free-iteration
# cut generation are only based on the solved main model
# =========================================

for ite_num in range(10):

    ite_name = str(ite_num)

    # if ite_num % 5 == 0:
    #     record_this_ite = True
    # else:
    #     record_this_ite = False

    if ite_num % 2 == 0:
        add_integer_l_shaped = 1
    else:
        add_integer_l_shaped = 0

    # if ite_num <= 25:
    #     SDDiP_module.execute_single_iteration(
    #         iteration_name=ite_name,
    #         est_sub_lb_dict=obj_lb_dict,
    #         if_benders_cut=1,
    #         if_l_shaped_cut=1,
    #         record_incumbent=1,
    #         l_shaped_cut_enforce=4
    #     )
    #
    # else:
    #     SDDiP_module.execute_single_iteration(
    #         iteration_name=ite_name,
    #         est_sub_lb_dict=obj_lb_dict,
    #         if_benders_cut=1,
    #         if_l_shaped_cut=int(add_integer_l_shaped),
    #         record_incumbent=1,
    #         l_shaped_cut_enforce=0
    #     )

    SDDiP_module.execute_single_iteration(
        iteration_name=ite_name,
        est_sub_lb_dict=obj_lb_dict,
        if_benders_cut=1,
        if_l_shaped_cut=0,
        record_incumbent=1,
        l_shaped_cut_enforce=0
    )


if ENABLE_RESULT_OUTPUT:
    iter_general_csv(
        ite_obj_value_dict=SDDiP_module.ite_obj_value_dict,
        output_dir=output_dir
    )
    iter_sub_prob_info(
        ite_obj_value_dict=SDDiP_module.ite_obj_value_dict,
        output_dir=output_dir,
        scenario_list=scenario_list
    )

    # after finishing the algorithm
    plot_iter_obj_curves(
        ite_obj_value_dict=SDDiP_module.ite_obj_value_dict,
        output_dir=output_dir,
        # real_objective_value=3441951
        real_objective_value=1707000
        # real_objective_value=2910637
    )

print('')


