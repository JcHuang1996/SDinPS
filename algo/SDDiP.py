# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com
from typing import Any

from util.headers import *
from util.names import *
from util.project_logger import init_logger
from dao.data_reader import DataReader
from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub, ModelCombined
from algo.algo_simple_tools import *

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import pyomo.environ as pyo
import logging
import os


logger = logging.getLogger(__name__)


class SDDiP_planning():

    def __init__(self, raw_data, time_list, scenario_list):

        self.data_processor_module = DataProcessor(raw_data=raw_data)
        self.time_list = time_list
        self.scenario_list = scenario_list

        self.main_model_data, self.sub_model_data_dict = {}, {}

        self.sce_prob_dict = {}

        self.model_main = None

        # the dict for collecting information of benders optimality cuts
        self.bds_cut_info_dict = {}

        # the dict for collecting info of integer L-shaped cuts
        self.L_cut_info_dict = {}

        # the dict for collecting objective values
        self.ite_obj_value_dict = {}

        # the dict for collecting the sub model's best possible objective function
        # note: though the estimation is computed by full formulation,
        # the full formulation's objective terms should only contain the terms of sub model.
        self.sub_obj_lb_dict = {}

    @staticmethod
    def _get_model_obj_value(pyomo_model):
        obj = next(pyomo_model.component_data_objects(pyo.Objective, active=True))
        return pyo.value(obj.expr)

    def build_main_stage_model(self):

        # process the data for main model
        self.main_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=self.scenario_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        self.sce_prob_dict = self.main_model_data[DataName.DICT_SC_PROB].copy()

        # build the main model with processed main model data
        self.model_main = ModelMain(model_name='Main', model_data=self.main_model_data)
        self.model_main.build_main_model()

    def solve_and_record_main_stage_model(self, ite_name=None):
        self.model_main.solve()
        self.model_main.cal_detailed_obj()

        # show and record the main model objective value (the best bound objective value)
        best_bound_objective_value = self._get_model_obj_value(self.model_main.model)
        best_main_stage_objective_value = (
                self.model_main.obj_term_value[ObjName.DG_FIXED_COST]
                # + self.model_main.obj_term_value[ObjName.DG_VARIANT_COST]
                + self.model_main.obj_term_value[ObjName.LINE_HARDEN_COST]
        )

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        # record the objective value and values of every terms
        self.ite_obj_value_dict[ite_name]['main_obj(bound)'] = {
            'total': best_bound_objective_value,
            'detail': self.model_main.obj_term_value.copy()
        }

        return best_main_stage_objective_value

    def build_sub_model(self, sub_model_sce_list=None, given_main_result=None):
        """
        The function iterates one round on the given scenarios based on the given main result.
        :param
        sub_model_sce_list: the list of scenarios considered in this sub model.
        given_main_result: the main stage result for current iteration
        :return:
        """

        # the sub model's scenario cannot be empty
        if sub_model_sce_list is None:
            raise ValueError('sub_model_sce_list cannot be None')

        # process the sub model data
        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        # ============================
        # build and solve sub problem models
        # ============================
        # build the sub model for the given scenario(s) in this iteration
        sce_sub_model = ModelSub(
            model_name=f'{tuple(sub_model_sce_list)}_model',
            model_data=sub_model_data,
            main_result=given_main_result
        )
        sce_sub_model.build_sub_model()
        return sce_sub_model

    def solve_and_record_sub_model(self, sub_model_sce_list=None, sub_model=None, main_stage_obj_value=None, ite_name=None):
        # solve the sub problem model
        sub_model.solve()
        sub_model.cal_detailed_obj()

        # collect the scenario's objective value.
        # Note: the 1st stage objective value is included.
        sce_obj_value_w_main = self._get_model_obj_value(sub_model.model) + main_stage_obj_value

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        self.ite_obj_value_dict[ite_name][tuple(sub_model_sce_list)] = {
            'total': sce_obj_value_w_main,
            'detail': sub_model.obj_term_value.copy()
        }

        return sce_obj_value_w_main

    def generate_benders_opt_cut(self, sub_model, sub_model_sce_list=None, ite_name=None):

        # solve sub model's LP relax for Benders optimality cut
        sub_model.solve_relaxed()

        # compute the information for generating Benders optimality cut
        constant_term, var_coeff_dict = sub_model.benders_opt_cut_info_generator()

        if ite_name not in self.bds_cut_info_dict:
            self.bds_cut_info_dict[ite_name] = {}

        # record the benders cut info
        self.bds_cut_info_dict[ite_name][tuple(sub_model_sce_list)] = [constant_term, var_coeff_dict]

    def collect_L_cut_info(self, sub_model_sce_list=None, ite_name=None, obj_lb=None, obj_value=None, zero_var_idx=None, one_var_idx=None):

        if ite_name not in self.L_cut_info_dict:
            self.L_cut_info_dict[ite_name] = {}

        self.L_cut_info_dict[ite_name][tuple(sub_model_sce_list)] = {
            'sub_obj_value': obj_value,
            'sub_obj_lb': obj_lb,
            'zero_var_idx': zero_var_idx.copy(),
            'one_var_idx': one_var_idx.copy()
        }

    def add_benders_cut(self, sub_model_sce_list=None, ite_name=None):
        sub_model_key = tuple(sub_model_sce_list)
        self.model_main.add_constr_benders_opt_cut(
            sub_problem_sce_list=sub_model_sce_list,
            constant_term=self.bds_cut_info_dict[ite_name][sub_model_key][0],
            var_coef_dict=self.bds_cut_info_dict[ite_name][sub_model_key][1],
            track_idx=f'i_{ite_name}_'
        )

    def add_integer_L_shaped_cut(self, sub_model_sce_list=None, ite_name=None, enforce_constant=None):
        sub_model_key = tuple(sub_model_sce_list)
        self.model_main.add_constr_integer_L_shaped_cut(
            sub_problem_sce_list=sub_model_sce_list,
            sub_model_obj_value=self.L_cut_info_dict[ite_name][sub_model_key]['sub_obj_value'],
            sub_model_obj_lb=self.L_cut_info_dict[ite_name][sub_model_key]['sub_obj_lb'],
            zero_var_idx=self.L_cut_info_dict[ite_name][sub_model_key]['zero_var_idx'],
            one_var_idx=self.L_cut_info_dict[ite_name][sub_model_key]['one_var_idx'],
            track_idx=f'i_{ite_name}_',
            enforce_constant=enforce_constant
        )

    def big_M_estimator(self, sub_model_sce_list=None):
        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        model_est = ModelCombined(model_name='m_est', model_data=sub_model_data)

        model_est.build_model_all_obj_terms()
        model_est.solve()
        model_result = model_est.get_result(
            var_name_list=[
                VarName.DG_RATED_POWER,
                VarName.DG_INSTALL,
                VarName.LINE_CONNECTED,
                VarName.BUS_VOLTAGE,
                VarName.LINE_ACTIVE_FLOW,
                VarName.LINE_REACTIVE_FLOW
            ]
        )

        V_FLOW_M_EST = 0

        for (i, j) in sub_model_data[DataName.LIST_LINE]:
            for t in sub_model_data[DataName.LIST_TIME]:
                for s in sub_model_data[DataName.LIST_SCENARIO]:

                    if model_result[VarName.LINE_CONNECTED][i, j, t, s] > 0.5:
                        continue

                    v_drop_on_ij = (
                            sub_model_data[DataName.DICT_LINE_RESISTANCE][i, j] * model_result[VarName.LINE_ACTIVE_FLOW][i, j, t, s]
                            + sub_model_data[DataName.DICT_LINE_REACTANCE][i, j] * model_result[VarName.LINE_REACTIVE_FLOW][i, j, t, s]
                    )

                    body_terms = (
                            model_result[VarName.BUS_VOLTAGE][i, t, s]
                            - model_result[VarName.BUS_VOLTAGE][j, t, s]
                            - v_drop_on_ij / sub_model_data[DataName.NUM_VOLTAGE_SLACK]
                    )

                    V_FLOW_M_EST = max(V_FLOW_M_EST, abs(body_terms))

        R_POWER_M_EST = 0
        for j in sub_model_data[DataName.LIST_NODE]:
            if model_result[VarName.DG_INSTALL][j] > 0.5:
                R_POWER_M_EST = max(R_POWER_M_EST, model_result[VarName.DG_RATED_POWER][j])

        return R_POWER_M_EST, V_FLOW_M_EST


    def sub_model_lb_estimator(self, strategy=None, sub_model_sce_list: list = None) -> tuple[Any, Any]:

        # process corresponding data set
        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        # build the corresponding sub model for estimating
        model_est = ModelCombined(model_name='m_est', model_data=sub_model_data)

        if strategy == 'safe':
            # obtaining safe but loose bound by only consider sub-model objective.
            # drawback: possible to add too many DG or harden too many lines
            model_est.build_model_given_obj_terms([
                ObjName.DG_VARIANT_COST,
                ObjName.DG_GENERATING_COST,
                ObjName.LOAD_SHED_COST
            ])
            model_est.solve()

            est_obj_value = self._get_model_obj_value(model_est.model)

        elif strategy == 'aggressive':
            # obtaining tight but risky bound by consider all objective terms.
            # drawback: too aggressive
            model_est.build_model_given_obj_terms([
                ObjName.DG_FIXED_COST,
                ObjName.LINE_HARDEN_COST,
                ObjName.DG_VARIANT_COST,
                ObjName.DG_GENERATING_COST,
                ObjName.LOAD_SHED_COST
            ])
            model_est.solve()

            model_est.cal_detailed_obj()
            est_obj_value = sum(
                model_est.obj_term_value[obj_name]
                for obj_name in [ObjName.DG_VARIANT_COST, ObjName.DG_GENERATING_COST, ObjName.LOAD_SHED_COST]
            )

        else:
            raise ValueError(f'Unknown strategy {strategy}')

        # record and return main result for future warm starting
        est_main_result = model_est.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])

        return est_obj_value, est_main_result

    def execute_single_iteration(
            self,
            iteration_name=None,
            est_sub_lb_dict=None,
            given_main_result=None,
            if_benders_cut=None,
            if_l_shaped_cut=None,
            l_shaped_cut_enforce=None,
            record_incumbent=None,
    ):

        ite_name = iteration_name

        # =======================================================
        # solve the main model and set main result at the beginning of the iteration
        # =======================================================

        if given_main_result is not None:
            self.model_main.fix_variable_value(var_fix_info=given_main_result)

        best_main_stage_obj_value = self.solve_and_record_main_stage_model(ite_name=ite_name)

        # record the result of the main model required by the sub problems
        curr_main_result = self.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])

        # if given_main_result is None:
        #     # record the result of the main model required by the sub problems
        #     curr_main_result = self.model_main.get_result([VarName.DG_INSTALL, VarName.LINE_HARDEN])
        #
        # else:
        #     curr_main_result = given_main_result

        # prepare the data for L-shaped cuts
        curr_main_result_zero_idx, curr_main_result_one_idx = bi_var_counter(curr_main_result)

        # the best incumbent objective value will be given by the weighted sum of sub-problem objective values
        # starting from 0
        best_incumbent_obj_value = 0

        # ============================
        # using the current main result, iterating scenarios
        # ============================

        # decide how to group and iterate the scenarios
        sce_group_list = [
            [s] for s in self.scenario_list
        ]

        # iterating by the above division
        for sub_sce_list in sce_group_list:
            # ============================
            # build and solve the corresponding sub problem model
            # ============================

            # build the sub model
            curr_sub_model = self.build_sub_model(
                sub_model_sce_list=sub_sce_list,
                given_main_result=curr_main_result
            )

            if record_incumbent:
                # solve the sub model and update the objective record, including the detailed record in the algo module
                sub_obj_value_w_main = self.solve_and_record_sub_model(
                    sub_model_sce_list=sub_sce_list,
                    sub_model=curr_sub_model,
                    main_stage_obj_value=best_main_stage_obj_value,
                    ite_name=ite_name
                )
                best_incumbent_obj_value += sub_obj_value_w_main * sum(
                    self.sce_prob_dict[s_idx]
                    for s_idx in sub_sce_list
                )

            if if_benders_cut == 1:
                # ===========================
                # generating Benders optimality cut
                # ===========================
                self.generate_benders_opt_cut(
                    sub_model=curr_sub_model,
                    sub_model_sce_list=sub_sce_list,
                    ite_name=ite_name
                )

            if if_l_shaped_cut == 1 and record_incumbent:
                # ==========================
                # collecting L-shaped cut info
                # ==========================
                self.collect_L_cut_info(
                    sub_model_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    obj_lb=est_sub_lb_dict[sub_sce_list[0]],
                    obj_value=self._get_model_obj_value(curr_sub_model.model),
                    zero_var_idx=curr_main_result_zero_idx,
                    one_var_idx=curr_main_result_one_idx
                )

        # ===============================
        # Operations after the solving process
        # ===============================
        if record_incumbent:
            # summarize the current iteration record
            self.ite_obj_value_dict[ite_name]['sub_obj(best_incumbent)'] = {
                'sub_p_total': best_incumbent_obj_value
            }

        # update the main model:

        # adding benders cuts from all scenarios
        if if_benders_cut == 1:
            for sub_sce_list in sce_group_list:
                self.add_benders_cut(
                    sub_model_sce_list=sub_sce_list,
                    ite_name=ite_name
                )

        # adding integer L-shaped cut
        if if_l_shaped_cut == 1 and record_incumbent:
            for sub_sce_list in sce_group_list:
                self.add_integer_L_shaped_cut(
                    sub_model_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    enforce_constant=l_shaped_cut_enforce
                )

        # release the fix of the main model (if there is)
        if given_main_result is not None:
            self.model_main.recover_variable_from_fixed()

    def user_node_decision_estimator(self, sub_model_sce_list: list = None, node_to_est_idx: str = None):

        # process corresponding data set
        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        # only fix the bi-var to be one
        bi_to_fix_info = {
            VarName.DG_INSTALL: {node_to_est_idx: 1}
        }

        # build the corresponding sub model for estimating
        model_est = ModelCombined(model_name='m_est', model_data=sub_model_data)

        model_est.build_model_given_obj_terms([
            ObjName.DG_FIXED_COST,
            ObjName.LINE_HARDEN_COST,
            ObjName.DG_VARIANT_COST,
            ObjName.DG_GENERATING_COST,
            ObjName.LOAD_SHED_COST
        ])

        model_est.fix_variable_value(var_fix_info=bi_to_fix_info)

        model_est.solve()

        model_est.cal_detailed_obj()
        est_obj_value = sum(
            model_est.obj_term_value[obj_name]
            for obj_name in [ObjName.DG_VARIANT_COST, ObjName.DG_GENERATING_COST, ObjName.LOAD_SHED_COST]
        )

        return est_obj_value
