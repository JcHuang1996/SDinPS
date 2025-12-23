# -*- coding: utf-8 -*-
# @Time     : 2025/10/20
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined

from util.headers import *
from util.names import *

from gurobipy import GRB
import gurobipy as gp

import numpy as np
import math
import os
import datetime
import logging

logger = logging.getLogger(__name__)


class ModelSub(ModelCombined):

    def __init__(self, model_name='default_m', model_data=None, main_result=None):
        super().__init__(model_name=model_name, model_data=model_data)

        self.main_result = main_result

        self.cross_constr_names = []

        self.model.setParam(GRB.Param.OutputFlag, 0)

    def build_sub_model(self):

        # using existing defining functions from combined model
        self.add_vars_basic_generator_c()
        self.add_vars_line_connectivity()
        self.add_vars_sys_operating()
        self.add_vars_sys_topology()
        self.add_constr_system_operating()
        self.add_constr_system_topology_constraints()

        # adding specific components of sub problem model
        self.add_sub_main_common_vars()

        self.add_constr_DG_rated_power_ub()
        self.add_constr_DG_operating()
        self.add_constr_line_connectivity()

        # set submodel objective
        self.set_sub_objective()

    def add_sub_main_common_vars(self):
        """
        The vars added in this section are the variables of main model.
        They should be added as variables, rather than values, so that corresponding columns / coefficients
        could be got by gurobi methods.
        """

        # self.var[VarName.DG_RATED_POWER] = {
        #     j: self.model.addVar(
        #         vtype=GRB.CONTINUOUS,
        #         lb=self.main_result[VarName.DG_RATED_POWER][j],
        #         ub=self.main_result[VarName.DG_RATED_POWER][j],
        #         name=f'{VarName.DG_RATED_POWER}_({j})'
        #     )
        #     for j in self.data[DataName.LIST_NODE]
        # }

        # X^{G}_{j}: Binary
        self.var[VarName.DG_INSTALL] = {
            j: self.model.addVar(
                vtype=GRB.BINARY,
                lb=self.main_result[VarName.DG_INSTALL][j],
                ub=self.main_result[VarName.DG_INSTALL][j],
                name=f'{VarName.DG_INSTALL}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        # x^L_{ij}: Binary
        self.var[VarName.LINE_HARDEN] = {
            (i, j): self.model.addVar(
                vtype=GRB.BINARY,
                lb=self.main_result[VarName.LINE_HARDEN][i, j],
                ub=self.main_result[VarName.LINE_HARDEN][i, j],
                name=f'{VarName.LINE_HARDEN}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

    def set_sub_objective(self):

        self.obj_term[ObjName.DG_VARIANT_COST] = gp.quicksum(
            self.data[DataName.DICT_DG_COST_VAR][j] * self.var[VarName.DG_RATED_POWER][j]
            for j in self.data[DataName.LIST_NODE]
        )

        self.obj_term[ObjName.DG_GENERATING_COST] = gp.quicksum(
            self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.DG_ACTIVE_POWER][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.obj_term[ObjName.LOAD_SHED_COST] = gp.quicksum(
            self.data[DataName.NUM_COST_SHED] * self.var[VarName.LOAD_SHED_RATIO][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.model.setObjective(
            self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.DG_GENERATING_COST]
            + self.obj_term[ObjName.LOAD_SHED_COST],
            GRB.MINIMIZE
        )

    def benders_opt_cut_info_generator(self):
        """
        After solved the RELAXED sub model, derive the coefficients and constants for generating corresponding benders cut.
        :return:
        constant_term: constant term of the benders opt cut (dot product of the dua solution and the RHS);
        var_coeff_dict: coefficients for generating benders opt cut, in the form of:
            {var_class_name: {var_key: \pi^{T} * the corresponding column of the variable in sub model}}
        """

        # compute the constant term
        duals_all = self.model_relax.getAttr(GRB.Attr.Pi)
        rhs_value_all = self.model_relax.getAttr(GRB.Attr.RHS)
        constant_term = np.dot(duals_all, rhs_value_all)

        # compute the var_coeff_dict
        var_coeff_dict = {
            var_class_name: {var_key: None for var_key in sorted(self.main_result[var_class_name].keys())}
            for var_class_name in sorted(self.main_result.keys())
        }

        for var_class_name in var_coeff_dict.keys():
            for var_key in var_coeff_dict[var_class_name].keys():
                var_exact_name = self.var[var_class_name][var_key].VarName
                col_info = self.model_relax.getCol(self.model_relax.getVarByName(var_exact_name))
                var_coef_in_cut = 0
                for i in range(col_info.size()):
                    pi_of_constr = col_info.getConstr(i).getAttr(GRB.Attr.Pi)
                    var_coed_in_constr = col_info.getCoeff(i)
                    var_coef_in_cut += pi_of_constr * var_coed_in_constr
                var_coeff_dict[var_class_name][var_key] = var_coef_in_cut

        return constant_term, var_coeff_dict
