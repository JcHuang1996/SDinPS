# -*- coding: utf-8 -*-
# @Time     : 2025/09/16
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined

from util.headers import *
from util.names import *

import pyomo.environ as pyo
import math
import os
import datetime
import logging

logger = logging.getLogger(__name__)


class ModelMain(ModelCombined):

    def build_main_model(self):

        # using existing defining functions from combined model
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_line()
        self.add_constr_DG_ub()
        # self.add_constr_DG_rated_power_ub()

        # adding specific components of main model
        self.add_vars_cut()

        # set objective of main model
        self.set_main_objective()

    def add_vars_cut(self):
        self.var[VarName.SUB_OBJ_EST] = {
            s: self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.SUB_OBJ_EST}_{s}'
            )
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def set_main_objective(self):

        self.obj_term[ObjName.DG_FIXED_COST] = pyo.quicksum(
            self.data[DataName.DICT_DG_COST_FIX][j] * self.var[VarName.DG_INSTALL][j]
            for j in self.data[DataName.LIST_NODE]
        )

        # self.obj_term[ObjName.DG_VARIANT_COST] = pyo.quicksum(
        #     self.data[DataName.DICT_DG_COST_VAR][j] * self.var[VarName.DG_RATED_POWER][j]
        #     for j in self.data[DataName.LIST_NODE]
        # )

        self.obj_term[ObjName.LINE_HARDEN_COST] = pyo.quicksum(
            self.data[DataName.DICT_LINE_COST_HARDEN][i, j] * self.var[VarName.LINE_HARDEN][i, j]
            for (i, j) in self.data[DataName.LIST_LINE]
        )

        self.obj_term[ObjName.SUB_OBJ_TERM] = pyo.quicksum(
            self.data[DataName.DICT_SC_PROB][s] * self.var[VarName.SUB_OBJ_EST][s]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.set_objective(
            self.obj_term[ObjName.DG_FIXED_COST]
            # + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
            + self.obj_term[ObjName.SUB_OBJ_TERM],
            sense=pyo.minimize
        )

    def add_constr_benders_opt_cut(self, sub_problem_sce_list=None, constant_term=None, var_coef_dict=None, track_idx=None):

        self.add_constr(
            pyo.quicksum(
                self.var[VarName.SUB_OBJ_EST][sce_idx]
                for sce_idx in sub_problem_sce_list
            ) >= 1.001 * (constant_term - pyo.quicksum(
                self.var[var_class_name][var_key] * var_coef_dict[var_class_name][var_key]
                for var_class_name in sorted(var_coef_dict.keys())
                for var_key in sorted(var_coef_dict[var_class_name].keys())
            )),
            name=f'B_OPT_C_{track_idx}_{sub_problem_sce_list}'
        )

    def add_constr_integer_L_shaped_cut(self,
                                 sub_problem_sce_list,
                                 sub_model_obj_value,
                                 sub_model_obj_lb,
                                 zero_var_idx,
                                 one_var_idx,
                                 track_idx,
                                 enforce_constant
                                 ):

        term_regarding_value_zero_index = pyo.quicksum(
            1 - (self.var[var_class_name][var_key])
            for var_class_name in sorted(one_var_idx.keys())
            for var_key in sorted(one_var_idx[var_class_name])
        )

        term_regarding_value_one_index = pyo.quicksum(
            self.var[var_class_name][var_key]
            for var_class_name in sorted(zero_var_idx.keys())
            for var_key in sorted(zero_var_idx[var_class_name])
        )

        self.add_constr(
            pyo.quicksum(
                self.var[VarName.SUB_OBJ_EST][sce_idx] for sce_idx in sub_problem_sce_list
            ) >= sub_model_obj_value
            - (sub_model_obj_value - sub_model_obj_lb) * (term_regarding_value_zero_index + term_regarding_value_one_index - enforce_constant),
            name=f'L_OPT_C_{track_idx}_{sub_problem_sce_list}'
        )

    def add_user_cut_node_estimation(self, sub_problem_sce_list, sub_model_obj_value, sub_model_obj_lb, estimated_node_key, track_idx):

        self.add_constr(
            pyo.quicksum(
                self.var[VarName.SUB_OBJ_EST][sce_idx] for sce_idx in sub_problem_sce_list
            ) >= sub_model_obj_value
            - (sub_model_obj_value - sub_model_obj_lb) * self.var[estimated_node_key[0]][estimated_node_key[1]],
            name=f'U_C_Bi_{track_idx}_{sub_problem_sce_list}'
        )
