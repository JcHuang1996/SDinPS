# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from typing import Any
from util.names import DataName, ObjName, VarName

from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub

import pyomo.environ as pyo
import logging
import os


logger = logging.getLogger(__name__)


class TwoStageDecompRedo:

    def __init__(self, raw_data, time_list, scenario_list):

        self.data_processor_module = DataProcessor(raw_data=raw_data)
        self.time_list = time_list
        self.scenario_list = scenario_list

        self.main_model_data, self.sub_model_data_dict = {}, {}

        self.sce_prob_dict = {}

        self.model_main, self.current_sub_model = None, None

        # the dict for collecting objective values
        self.ite_obj_value_dict = {}

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

    def solve_main_stage_model(self):
        # solve the main model, and compute the obj terms
        # return the optimal value of the solved main model
        self.model_main.solve()
        self.model_main.cal_detailed_obj()

    def record_main_stage_model(self, ite_name=None):
        # show and record the main model objective value (the best bound objective value)
        total_objective_value = self.model_main.get_obj_value()
        main_stage_objective_value = (
                self.model_main.obj_term_value[ObjName.DG_FIXED_COST]
                + self.model_main.obj_term_value[ObjName.LINE_HARDEN_COST]
        )

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        # record the objective value and values of every terms
        self.ite_obj_value_dict[ite_name]['main_obj(bound)'] = {
            'total': total_objective_value,
            'detail': self.model_main.obj_term_value.copy()
        }

        return main_stage_objective_value, total_objective_value

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
        self.current_sub_model = ModelSub(
            model_name=f'{tuple(sub_model_sce_list)}_model',
            model_data=sub_model_data,
            main_result=given_main_result,
            sub_model_sce_list=sub_model_sce_list
        )
        self.current_sub_model.build_sub_model_redo()

    def solve_sub_model(self):
        self.current_sub_model.solve()
        self.current_sub_model.cal_detailed_obj()

    def solve_relaxed_sub_model(self):
        self.current_sub_model.solve_relaxed()

    def record_sub_model(self, main_stage_obj_value=None, ite_name=None):
        # collect the scenario's objective value.
        # Note: the 1st stage objective value is included.
        sce_obj_value = self.current_sub_model.get_obj_value()
        sce_obj_value_w_main = sce_obj_value + main_stage_obj_value

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        self.ite_obj_value_dict[ite_name][tuple(self.current_sub_model.sub_model_sce_list)] = {
            'total': sce_obj_value_w_main,
            'detail': self.current_sub_model.obj_term_value.copy()
        }

        return sce_obj_value_w_main, sce_obj_value

    def gen_sce_bds_opt_cut(self, lp_opt_value=None, constr_dual_info=None, constr_var_map=None):
        """
        Idea: generate benders opt cuts components in the format of:
         \theta \geq Q^{LP}_{obtained_main_solution} + pi^{T} * (main var - obtained main solution)
         Reference: Zou, J., Ahmed, S., & Sun, X. (2019). Stochastic dual dynamic integer programming.
        :param lp_opt_value: the relaxed LP objective value of the corresponding sub model.
        :param constr_dual_info: the dict of constr-dual, for generating benders cuts.
                                key: constraint name; value: dual optimal solution value
        :param constr_var_map: the dict showing constr-var relationship, for generating benders cuts.
        :return: lhs, rhs
        lhs: the left hand side of the cut in above format
        rhs: the right hand side of the cut in above format
        """
        constr_list = sorted(constr_dual_info.keys())

        sub_sce_list = self.current_sub_model.sub_model_sce_list

        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_sce_list
        )

        rhs = lp_opt_value + pyo.quicksum(
            constr_dual_info[constr_name] * self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
            for constr_name in constr_list
        )

        return lhs, rhs




