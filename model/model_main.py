# -*- coding: utf-8 -*-
# @Time     : 2025/09/16
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined
from model.formulation_blocks import (
    add_design_objective_terms,
    add_dg_rated_power_link_constraints,
    add_dg_rated_power_vars,
    add_dg_operating_constraints,
    add_line_connectivity_constraints,
    add_line_connectivity_vars,
    add_operating_constraints,
    add_operating_objective_terms,
    add_operating_vars,
    add_topology_constraints,
    add_topology_vars,
    make_main_formulation_adapter,
)

from util.headers import *
from util.names import *

import pyomo.environ as pyo
import logging

logger = logging.getLogger(__name__)


class ModelMain(ModelCombined):

    def __init__(self, model_name='default_m', model_data=None, solver_config=None):
        super().__init__(model_name=model_name, model_data=model_data, solver_config=solver_config)

        self.added_obj_term_weight = 0.0
        self.main_problem_model_type = (
            model_data.get(DataName.MAIN_PROBLEM_MODEL_TYPE, MainProblemModelTypeName.COLLAPSED_NO_TIME)
            if model_data is not None else MainProblemModelTypeName.COLLAPSED_NO_TIME
        )
        self.decomposition_state_var_name_list = list(DECOMPOSITION_STATE_VAR_NAMES)
        self.decomposition_structural_constr_name_list = []

    def build_main_model(self):

        # using existing defining functions from combined model
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_line()
        self.add_constr_DG_ub()
        self.decomposition_structural_constr_name_list = list(self._constr_name_map.keys())

        self._validate_main_problem_model_type()

        # adding specific components of main model
        if self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
            self.add_vars_collapsed_sp_no_time()
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
            self.add_vars_collapsed_sp_by_time()
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
            self.add_vars_collapsed_sp_by_scenario()

        self.add_vars_cut()
        if self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
            self.add_constr_collapsed_sp_no_time()
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
            self.add_constr_collapsed_sp_by_time()
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
            self.add_constr_collapsed_sp_by_scenario()

        # set objective of main model
        self.set_main_objective()

    def _validate_main_problem_model_type(self):
        supported_types = {
            MainProblemModelTypeName.BASE,
            MainProblemModelTypeName.COLLAPSED_NO_TIME,
            MainProblemModelTypeName.COLLAPSED_BY_TIME,
            MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS,
        }
        if self.main_problem_model_type not in supported_types:
            raise ValueError(f'Unsupported main_problem_model_type: {self.main_problem_model_type}')

    def get_decomposition_state_var_name_list(self):
        """Return the canonical master-state blocks used by subproblems and CGLP."""
        return list(self.decomposition_state_var_name_list)

    def get_decomposition_structural_constr_name_list(self):
        """Return master structural rows used when building CGLP models."""
        return list(self.decomposition_structural_constr_name_list)

    def get_decomposition_var_key_lists_for_cglp(self):
        """Split canonical master-state variables into binary and continuous CGLP key lists."""
        x_var_key_list, z_var_key_list = [], []

        for var_name in self.get_decomposition_state_var_name_list():
            var_key_list = sorted(self.var[var_name].keys())
            if not var_key_list:
                continue

            first_var = self.var[var_name][var_key_list[0]]
            target_list = x_var_key_list if first_var.is_binary() else z_var_key_list
            target_list.extend((var_name, var_key) for var_key in var_key_list)

        return x_var_key_list, z_var_key_list

    def add_vars_cut(self):
        self.var[VarName.SUB_OBJ_EST] = {
            s: self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.SUB_OBJ_EST}_{s}'
            )
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def add_vars_collapsed_sp_no_time(self):
        self._add_collapsed_sp_vars(MainProblemModelTypeName.COLLAPSED_NO_TIME)

    def add_vars_collapsed_sp_by_time(self):
        self._add_collapsed_sp_vars(MainProblemModelTypeName.COLLAPSED_BY_TIME)

    def add_vars_collapsed_sp_by_scenario(self):
        self._add_collapsed_sp_vars(MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS)

    def _add_collapsed_sp_vars(self, main_problem_model_type):
        adapter = make_main_formulation_adapter(self.data, main_problem_model_type)
        add_dg_rated_power_vars(self)
        add_line_connectivity_vars(self, adapter)
        add_operating_vars(self, adapter)
        add_topology_vars(self, adapter)

    def add_constr_collapsed_sp_no_time(self):
        self._add_collapsed_sp_constraints(MainProblemModelTypeName.COLLAPSED_NO_TIME)

    def add_constr_collapsed_sp_by_time(self):
        self._add_collapsed_sp_constraints(MainProblemModelTypeName.COLLAPSED_BY_TIME)

    def add_constr_collapsed_sp_by_scenario(self):
        self._add_collapsed_sp_constraints(MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS)

    def _add_collapsed_sp_constraints(self, main_problem_model_type):
        adapter = make_main_formulation_adapter(self.data, main_problem_model_type)
        add_dg_rated_power_link_constraints(self, ConstrName.MAIN_DG_OPERATION)
        add_line_connectivity_constraints(self, adapter)
        add_topology_constraints(self, adapter)
        add_dg_operating_constraints(self, adapter)
        add_operating_constraints(self, adapter)

    def set_added_obj_term_weight(self, weight):
        self.added_obj_term_weight = float(weight)
        if self.obj_term:
            self.set_main_objective()

    def get_design_objective_value(self):
        return pyo.value(
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
        )

    def get_iteration_bound_value(self):
        return pyo.value(
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
            + self.obj_term[ObjName.SUB_OBJ_TERM]
        )

    def evaluate_design_objective_for_solution(self, main_result):
        return (
            sum(
                self.data[DataName.DICT_DG_COST_FIX][j] * main_result[VarName.DG_INSTALL][j]
                for j in self.data[DataName.LIST_NODE]
            )
            + sum(
                (
                    self.data[DataName.DICT_DG_RATED_POWER][typ] * self.data[DataName.DICT_DG_UNIT_PRICE][typ]
                    + self.data[DataName.DICT_DG_EXTRA_ADJUSTMENT][typ]
                ) * main_result[VarName.DG_INSTALL_TYPE][j, typ]
                for j in self.data[DataName.LIST_NODE]
                for typ in self.data[DataName.LIST_DG_TYPE]
            )
            + sum(
                self.data[DataName.DICT_LINE_COST_HARDEN][i, j] * main_result[VarName.LINE_HARDEN][i, j]
                for (i, j) in self.data[DataName.LIST_LINE]
            )
        )

    def set_main_objective(self):
        add_design_objective_terms(self)

        self.obj_term[ObjName.SUB_OBJ_TERM] = pyo.quicksum(
            self.data[DataName.DICT_SC_PROB][s] * self.var[VarName.SUB_OBJ_EST][s]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        if self.main_problem_model_type == MainProblemModelTypeName.BASE:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = 0.0
            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = 0.0
        else:
            add_operating_objective_terms(
                self,
                adapter=make_main_formulation_adapter(self.data, self.main_problem_model_type),
                dg_obj_name=ObjName.MAIN_APPROX_DG_GENERATING_COST,
                load_shed_obj_name=ObjName.MAIN_APPROX_LOAD_SHED_COST,
            )

        self.set_objective(
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
            + self.obj_term[ObjName.SUB_OBJ_TERM]
            + self.added_obj_term_weight * (
                self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST]
                + self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST]
            ),
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

    def add_custom_cut(self, cut_name, cut_lhs, cut_rhs):
        """
        Add a custom cut to the main model. The cut must be in the format of Expr1 \geq Expr2,
        where Expr1, Expr2 are two mathematical expressions.
        :param cut_name:
        :param cut_lhs:
        :param cut_rhs:
        :return:
        """
        self.add_constr(cut_lhs >= cut_rhs, name=cut_name)

    def set_pretrained_cut(self, above_var, below_var, above_sum, below_sum):

        above_var_name_list = sorted(above_var.keys())
        below_var_name_list = sorted(below_var.keys())

        self.add_constr(
            pyo.quicksum(
                self.var[var_name][var_key] for var_name in above_var_name_list for var_key in above_var[var_name]
            ) >= above_sum,
            name=f'PrTC_above'
        )
        self.add_constr(
            pyo.quicksum(
                self.var[var_name][var_key] for var_name in below_var_name_list for var_key in below_var[var_name]
            ) <= below_sum,
            name='PrTC_below'
        )
