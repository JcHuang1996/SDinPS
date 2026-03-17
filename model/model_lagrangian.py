# -*- coding: utf-8 -*-
# @Time     : 2026/02/07
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined
from model.model_base import ModelBase

from util.headers import *
from util.names import *

import pyomo.environ as pyo
import math
import os
import datetime
import logging

logger = logging.getLogger(__name__)


class ModelInnerMinimizationProblem(ModelCombined):
    """
    Reference:
    Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., & Rei, W. (2020).
    The Benders Dual Decomposition Method. Operations Research, 68(3), 878–895.
    https://doi-org.prox.lib.ncsu.edu/10.1287/opre.2019.1892

    The model solve MILP (9) in the reference paper for obtaining the first and second stage decisions, in order to
    generate the Lagrangian cut.
    """

    def __init__(self, model_name='default_m', model_data=None, main_result=None, constr_dual_info=None, constr_var_map=None):
        super().__init__(model_name=model_name, model_data=model_data)

        self.main_result = main_result

        self.constr_dual_info = constr_dual_info
        self.constr_var_map = constr_var_map

    def build_model(self):
        # add vars, the same as add_vars in model_combined
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_generator_c()
        self.add_vars_basic_line()
        self.add_vars_line_connectivity()
        self.add_vars_sys_operating()
        self.add_vars_sys_topology()

        # add constraints, the same as add_constraints in model_combine, except that DG_ub is removed
        self.add_constr_DG_rated_power_ub()
        self.add_constr_DG_operating()
        self.add_constr_line_connectivity()
        self.add_constr_system_operating()
        self.add_constr_system_topology_constraints()

        self.set_objective_function()

    def set_objective_function(self, lag_multiplier=None):

        self.obj_term[ObjName.SUB_OBJ_FUNCTION] = pyo.quicksum(
            self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.DG_ACTIVE_POWER][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        ) + pyo.quicksum(
            self.data[DataName.NUM_COST_SHED] * self.var[VarName.LOAD_SHED_RATIO][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        if lag_multiplier is None:
            considered_multiplier = self.constr_dual_info
        else:
            considered_multiplier = lag_multiplier

        constr_name_list = sorted(considered_multiplier.keys())
        self.obj_term[ObjName.COMPLEMENTARY_SLACK] = pyo.quicksum(
            considered_multiplier[constr_name]
            * (self.var[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]]
               - self.main_result[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]])
            for constr_name in constr_name_list
        )

        self.set_objective(
            self.obj_term[ObjName.SUB_OBJ_FUNCTION] - self.obj_term[ObjName.COMPLEMENTARY_SLACK],
            sense=pyo.minimize
        )


class ModelLagrangianMultiplierHeuristic(ModelBase):
    """
    Reference:
    Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., & Rei, W. (2020).
    The Benders Dual Decomposition Method. Operations Research, 68(3), 878–895.
    https://doi-org.prox.lib.ncsu.edu/10.1287/opre.2019.1892

    The model solve MILP (15) in the reference paper for obtaining the Lagrangian multiplier, in order to
    generate the Lagrangian cut.
    """

    def __init__(self, model_name='default_m', main_result=None, constr_dual_info=None, constr_var_map=None):
        super().__init__(model_name=model_name)

        self.main_result = main_result          # the prefixed y's value in model (15) in the reference

        self.constr_dual_info = constr_dual_info
        self.constr_var_map = constr_var_map

        self.constr_name_list = sorted(self.constr_dual_info.keys())

        self.ite_executed_num = 0

    def build_ini_model(self, ini_sub_obj_value, ini_main_sol, ini_multiplier, ini_stabilization_param):
        """
        ini_sub_obj_value: the c^{T}*\bar{x}^{1}
        ini_main_sol: the \bar{z}^{1}
        Build the initial version of the Lagrangian multiplier obtaining model, with following steps:
        1. add variable \eta
        2. add variables \lambda
        3. add one initial constraint in the format of \eta <= (a second stage objective value corresponding to x) + ((y - z) * lambda
        4. set initial objective function
        """

        ini_multiplier = self.constr_dual_info.copy()

        self.var[VarName.LIFT_VALUE] = self.add_var(domain=pyo.Reals, name=f'{VarName.LIFT_VALUE}')

        constr_name_list = sorted(self.constr_dual_info.keys())
        self.var[VarName.LAG_MULTIPLIER] = {
            constr_name: self.add_var(
                domain=pyo.Reals, name=f'{VarName.LAG_MULTIPLIER}_{constr_name}'
            )
            for constr_name in constr_name_list
        }

        # self.add_constr_for_regularized_model(sub_obj_value=ini_sub_obj_value, main_sol=ini_main_sol)

        self.set_objective_function(stabilization_param=ini_stabilization_param, prev_multiplier=ini_multiplier)

        self.ite_executed_num += 1

    def add_constr_for_regularized_model(self, sub_obj_value=None, main_sol=None):

        self.add_constr(
            self.var[VarName.LIFT_VALUE] <= sub_obj_value + pyo.quicksum(
                self.var[VarName.LAG_MULTIPLIER][constr_name]
                * (self.main_result[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]]
                   - main_sol[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]])
                for constr_name in self.constr_name_list
            ),
            name=f'inner_approx_{self.ite_executed_num+1}'
        )

    def set_objective_function(self, stabilization_param=None, prev_multiplier=None):

        obj_expr = self.var[VarName.LIFT_VALUE] - (stabilization_param / 2) * pyo.quicksum(
            (prev_multiplier[constr_name] - self.var[VarName.LAG_MULTIPLIER][constr_name]) ** 2
            for constr_name in self.constr_name_list
        )

        self.set_objective(obj_expr, sense=pyo.maximize)

    def obtain_lift_value(self):
        return pyo.value(self.var[VarName.LIFT_VALUE])


# class ModelLagrangianCutDeterministic(ModelCombined):
#     """
#     Reference:
#     Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., & Rei, W. (2020).
#     The Benders Dual Decomposition Method. Operations Research, 68(3), 878–895.
#     https://doi-org.prox.lib.ncsu.edu/10.1287/opre.2019.1892
#
#     The model solve MINLP (8) in the reference paper for generating exact Lagrangian cut.
#     """
#     def __init__(self, model_name='default_m', model_data=None, main_result=None, constr_dual_info=None, constr_var_map=None):
#         super().__init__(model_name=model_name, model_data=model_data)
#
#         self.main_result = main_result
#
#         self.constr_dual_info = constr_dual_info
#         self.constr_var_map = constr_var_map
#
#     def build_model(self):
#         # add vars, the same as add_vars in model_combined
#         self.add_vars_basic_generator_bi()
#         self.add_vars_basic_generator_c()
#         self.add_vars_basic_line()
#         self.add_vars_line_connectivity()
#         self.add_vars_sys_operating()
#         self.add_vars_sys_topology()
#
#         # add constraints, the same as add_constraints in model_combine, except that DG_ub is removed
#         self.add_constr_DG_rated_power_ub()
#         self.add_constr_DG_operating()
#         self.add_constr_line_connectivity()
#         self.add_constr_system_operating()
#         self.add_constr_system_topology_constraints()
#
#         # add Lagrangian cut generation components: \eta, the constraint for describing eta, objective function
#         self.add_lagrangian_generating_components()
#
#     def add_lagrangian_generating_components(self):
#
#         self.var[VarName.LAG_ESTIMATOR] = self.add_var(domain=pyo.Reals, name=f'{VarName.LAG_MULTIPLIER}')
#
#         self.var[VarName.AR_VAR_OBJ] = self.add_var(domain=pyo.Reals, name=f'{VarName.AR_VAR_OBJ}')
#
#         constr_name_list = sorted(self.constr_dual_info.keys())
#         self.var[VarName.LAG_MULTIPLIER] = {
#             constr_name: self.add_var(
#                 domain=pyo.Reals, name=f'{VarName.LAG_MULTIPLIER}_{constr_name}'
#             )
#             for constr_name in constr_name_list
#         }
#
#         self.add_constr(
#             self.var[VarName.AR_VAR_OBJ] == pyo.quicksum(
#                 self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.DG_ACTIVE_POWER][j, t, s]
#                 for j in self.data[DataName.LIST_NODE]
#                 for t in self.data[DataName.LIST_TIME]
#                 for s in self.data[DataName.LIST_SCENARIO]
#             ) + pyo.quicksum(
#                 self.data[DataName.NUM_COST_SHED] * self.var[VarName.LOAD_SHED_RATIO][j, t, s]
#                 for j in self.data[DataName.LIST_NODE]
#                 for t in self.data[DataName.LIST_TIME]
#                 for s in self.data[DataName.LIST_SCENARIO]
#             ), name='ctx_record'
#         )
#
#         self.add_constr(
#             self.var[VarName.LAG_ESTIMATOR] <= self.var[VarName.AR_VAR_OBJ] - pyo.quicksum(
#                 self.var[VarName.LAG_MULTIPLIER][constr_name]
#                 * (self.var[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]]
#                    - self.main_result[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]])
#                 for constr_name in constr_name_list
#             )
#         )
#
#         self.set_objective(self.var[VarName.LAG_ESTIMATOR], sense=pyo.maximize)

