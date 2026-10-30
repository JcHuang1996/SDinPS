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
import math

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
        self._main_formulation_adapter = None

    def get_main_formulation_adapter(self):
        if self._main_formulation_adapter is None:
            self._main_formulation_adapter = make_main_formulation_adapter(
                self.data,
                self.main_problem_model_type,
            )
        return self._main_formulation_adapter

    def build_main_model(self):

        # using existing defining functions from combined model
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_line()
        self.add_constr_DG_ub()
        self.decomposition_structural_constr_name_list = list(self._constr_name_map.keys())

        self._validate_main_problem_model_type()

        if self.main_problem_model_type != MainProblemModelTypeName.BASE:
            self._add_collapsed_sp_vars()
        self.add_vars_cut()
        if self.main_problem_model_type != MainProblemModelTypeName.BASE:
            self._add_collapsed_sp_constraints()

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

    def _add_collapsed_sp_vars(self):
        adapter = self.get_main_formulation_adapter()
        add_dg_rated_power_vars(self)
        add_line_connectivity_vars(self, adapter)
        add_operating_vars(self, adapter)
        add_topology_vars(self, adapter)

    def _add_collapsed_sp_constraints(self):
        adapter = self.get_main_formulation_adapter()
        add_dg_rated_power_link_constraints(self, ConstrName.MAIN_DG_OPERATION)
        add_line_connectivity_constraints(self, adapter)
        add_topology_constraints(self, adapter)
        add_dg_operating_constraints(self, adapter)
        add_operating_constraints(self, adapter)

    def set_added_obj_term_weight(self, weight):
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError('added_obj_term_weight must be finite.')
        if weight == self.added_obj_term_weight:
            return
        self.added_obj_term_weight = weight
        if self.obj_term:
            self.set_main_objective()

    def get_design_objective_value(self):
        return pyo.value(
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
        )

    def get_theta_weight(self, scenario):
        """
        Master weight of theta_s, i.e. the P_s in the objective term sum_s P_s * theta_s.

        Every cut left-hand side must use this weight so that a group cut
        ``sum_{s in G} P_s * theta_s >= (underestimator of sum_{s in G} P_s * Q_s(x))``
        tightens exactly the term it is meant to tighten, for any grouping (risk R3.4).
        """
        return self.data[DataName.DICT_SC_PROB_UNCOND][scenario]

    def get_weighted_theta_expr(self, scenario_list):
        """Return the canonical cut left-hand side sum_{s in scenario_list} P_s * theta_s."""
        return pyo.quicksum(
            self.get_theta_weight(s) * self.var[VarName.SUB_OBJ_EST][s]
            for s in scenario_list
        )

    def is_bound_objective_active(self) -> bool:
        """
        True when the master objective IS the bound objective, i.e. added_obj_term_weight == 0.

        With a positive weight the master minimizes ``design + sum P_s theta_s + w * (approximate
        operating terms)``. Its optimum is then a minimizer of a DIFFERENT function, so neither the
        value of ``design + sum P_s theta_s`` at that point nor the solver's dual bound is a lower
        bound on the two-stage problem (risk R3.3).
        """
        return self.added_obj_term_weight == 0.0

    def get_iteration_bound_value(self):
        """
        Value of the bound objective at the current solution, only when that objective was solved.
        """
        if not self.is_bound_objective_active():
            raise ValueError(
                'get_iteration_bound_value() requires added_obj_term_weight == 0. '
                'Use get_reported_base_objective_value() for a diagnostic value from a guidance solve.'
            )
        self.get_valid_lower_bound()
        return self.get_reported_base_objective_value()

    def get_reported_base_objective_value(self):
        """Diagnostic value of design + weighted theta at the current master point."""
        return pyo.value(
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
            + self.obj_term[ObjName.SUB_OBJ_TERM]
        )

    def get_valid_lower_bound(self):
        """
        Return a guaranteed lower bound on the two-stage objective from the last master solve.

        Raises when the penalized (guidance) objective was active, because no valid bound can be
        recovered from that solve.
        """
        if not self.is_bound_objective_active():
            raise ValueError(
                'get_valid_lower_bound() requires added_obj_term_weight == 0; the last master solve '
                f'used weight {self.added_obj_term_weight}, whose optimum bounds a different '
                'objective. Re-solve the master at weight 0, or run in heuristic bound mode.'
            )
        if self.has_pretrained_restrictions():
            raise ValueError('Pretrained restrictions do not certify a global lower bound.')
        return self.get_obj_bound()

    def has_pretrained_restrictions(self):
        """Whether the master currently searches a predicted subset of the original designs."""
        return any(
            name in self._constr_name_map and self.get_constr_item_by_name(name).active
            for name in (ConstrName.PRETRAINED_ABOVE, ConstrName.PRETRAINED_BELOW)
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

        # Convention: theta_s is the CONDITIONAL recourse cost of scenario s and the master carries
        # the probability weight. Cut left-hand sides must match it (see get_weighted_theta_expr).
        self.obj_term[ObjName.SUB_OBJ_TERM] = self.get_weighted_theta_expr(
            self.data[DataName.LIST_SCENARIO]
        )

        if self.main_problem_model_type == MainProblemModelTypeName.BASE:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = 0.0
            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = 0.0
        else:
            add_operating_objective_terms(
                self,
                adapter=self.get_main_formulation_adapter(),
                dg_obj_name=ObjName.MAIN_APPROX_DG_GENERATING_COST,
                load_shed_obj_name=ObjName.MAIN_APPROX_LOAD_SHED_COST,
            )

        self.bound_objective = (
            self.obj_term[ObjName.DG_FIXED_COST]
            + self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.LINE_HARDEN_COST]
            + self.obj_term[ObjName.SUB_OBJ_TERM]
        )
        self.guidance_objective = (
            self.bound_objective
            + self.added_obj_term_weight * (
                self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST]
                + self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST]
            )
        )
        self.set_objective(
            self.guidance_objective,
            sense=pyo.minimize
        )

    def add_custom_cut(self, cut_name, cut_lhs, cut_rhs):
        r"""
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
            name=ConstrName.PRETRAINED_ABOVE
        )
        self.add_constr(
            pyo.quicksum(
                self.var[var_name][var_key] for var_name in below_var_name_list for var_key in below_var[var_name]
            ) <= below_sum,
            name=ConstrName.PRETRAINED_BELOW
        )
