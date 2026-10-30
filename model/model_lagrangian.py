# -*- coding: utf-8 -*-
# @Time     : 2026/02/07
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined
from model.model_base import ModelBase
from model.formulation_blocks import (
    add_operating_objective_terms,
    make_sub_formulation_adapter,
)

from util.headers import *
from util.names import *

import pyomo.environ as pyo
import logging

logger = logging.getLogger(__name__)


class ModelInnerMinimizationProblem(ModelBase):
    """
    General inner-minimization model for Lagrangian cut generation.

    Reference:
    Rahmaniani, R., Ahmed, S., Crainic, T. G., Gendreau, M., & Rei, W. (2020).
    The Benders Dual Decomposition Method. Operations Research, 68(3), 878–895.
    https://doi-org.prox.lib.ncsu.edu/10.1287/opre.2019.1892

    Solves MILP (9): min_{x,y} c^T x - lambda^T (y - y*)
    to obtain first/second-stage decisions for Lagrangian cut generation.

    Subclass and override build_sub_vars_and_constraints() and get_sub_objective_expr(),
    or pass build_fn / sub_obj_fn callables to the constructor.
    """

    def __init__(self, model_name='default_m', model_data=None, main_result=None,
                 constr_dual_info=None, constr_var_map=None,
                 build_fn=None, sub_obj_fn=None, solver_config=None):
        super().__init__(model_name=model_name, model_data=model_data, solver_config=solver_config)

        self.main_result = main_result

        self.constr_dual_info = constr_dual_info
        self.constr_var_map = constr_var_map

        self._build_fn = build_fn
        self._sub_obj_fn = sub_obj_fn

    def build_model(self):
        self.build_sub_vars_and_constraints()
        self.set_objective_function()

    def build_sub_vars_and_constraints(self):
        """Build all subproblem variables and constraints (including local copies of master vars).
        Override in subclass or provide build_fn to constructor."""
        if self._build_fn is not None:
            self._build_fn(self)
        else:
            raise NotImplementedError(
                "Subclass must implement build_sub_vars_and_constraints() "
                "or pass build_fn to constructor."
            )

    def get_sub_objective_expr(self):
        """Return the Pyomo expression for the sub-objective c^T x (without Lagrangian penalty).
        Override in subclass or provide sub_obj_fn to constructor."""
        if self._sub_obj_fn is not None:
            return self._sub_obj_fn(self)
        raise NotImplementedError(
            "Subclass must implement get_sub_objective_expr() "
            "or pass sub_obj_fn to constructor."
        )

    def set_objective_function(self, lag_multiplier=None):

        self.obj_term[ObjName.SUB_OBJ_FUNCTION] = self.get_sub_objective_expr()

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


class PSInnerMinimizationProblem(ModelInnerMinimizationProblem, ModelCombined):
    """Power-system-specific inner minimization for Lagrangian cut generation.
    Backward-compatible replacement for the original ModelInnerMinimizationProblem."""

    def build_sub_vars_and_constraints(self):
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_generator_c()
        self.add_vars_basic_line()
        self.add_vars_line_connectivity()
        self.add_vars_sys_operating()
        self.add_vars_sys_topology()

        self.add_constr_DG_rated_power_ub()
        self.add_constr_DG_operating()
        self.add_constr_line_connectivity()
        self.add_constr_system_operating()
        self.add_constr_system_topology_constraints()

    def get_sub_objective_expr(self):
        """
        c^T x for the inner minimization, in the SAME units as the subproblem it belongs to:
        weighted by the unconditional scenario probabilities and built by the shared formulation
    block, so the recourse objective is written in exactly one place (workstream 6).
        """
        if not hasattr(self, '_sub_formulation_adapter'):
            self._sub_formulation_adapter = make_sub_formulation_adapter(self.data)
        add_operating_objective_terms(
            self,
            adapter=self._sub_formulation_adapter,
            dg_obj_name=ObjName.DG_GENERATING_COST,
            load_shed_obj_name=ObjName.LOAD_SHED_COST,
        )
        return (
            self.obj_term[ObjName.DG_GENERATING_COST]
            + self.obj_term[ObjName.LOAD_SHED_COST]
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

    def __init__(self, model_name='default_m', main_result=None, constr_dual_info=None, constr_var_map=None,
                 solver_config=None):
        super().__init__(model_name=model_name, solver_config=solver_config)

        self.main_result = main_result          # the prefixed y's value in model (15) in the reference

        self.constr_dual_info = constr_dual_info
        self.constr_var_map = constr_var_map

        self.constr_name_list = sorted(self.constr_dual_info.keys())

        self.ite_executed_num = 0

    def build_ini_model(self, ini_multiplier, ini_stabilization_param):
        """Build multiplier variables and their initial regularization; the caller adds oracle rows."""
        self.var[VarName.LIFT_VALUE] = self.add_var(domain=pyo.Reals, name=f'{VarName.LIFT_VALUE}')

        constr_name_list = sorted(self.constr_dual_info.keys())
        self.var[VarName.LAG_MULTIPLIER] = {
            constr_name: self.add_var(
                domain=pyo.Reals, name=f'{VarName.LAG_MULTIPLIER}_{constr_name}'
            )
            for constr_name in constr_name_list
        }

        self.set_objective_function(stabilization_param=ini_stabilization_param, prev_multiplier=ini_multiplier)

        self.ite_executed_num += 1

    def add_constr_for_regularized_model(self, sub_obj_value=None, main_sol=None):

        approximation_constr_name = f'inner_approx_{self.ite_executed_num+1}'
        self.add_constr(
            self.var[VarName.LIFT_VALUE] <= sub_obj_value + pyo.quicksum(
                self.var[VarName.LAG_MULTIPLIER][constr_name]
                * (self.main_result[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]]
                   - main_sol[self.constr_var_map[constr_name][0]][self.constr_var_map[constr_name][1]])
                for constr_name in self.constr_name_list
            ),
            name=approximation_constr_name
        )
        self.ite_executed_num += 1

    def set_objective_function(self, stabilization_param=None, prev_multiplier=None):

        obj_expr = self.var[VarName.LIFT_VALUE] - (stabilization_param / 2) * pyo.quicksum(
            (prev_multiplier[constr_name] - self.var[VarName.LAG_MULTIPLIER][constr_name]) ** 2
            for constr_name in self.constr_name_list
        )

        self.set_objective(obj_expr, sense=pyo.maximize)

    def obtain_lift_value(self):
        return pyo.value(self.var[VarName.LIFT_VALUE])
