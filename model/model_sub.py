# -*- coding: utf-8 -*-
# @Time     : 2025/10/20
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined

from util.headers import *
from util.names import *

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn

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
        Vars in this section are the (fixed) variables from the main model.
        They are added as Vars (not Params), fixed to the main solution,
        so their coefficients can be tracked when generating Benders cuts.
        """

        # X^{G}_{j}: Binary (fixed)
        self.var[VarName.DG_INSTALL] = {
            j: self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.DG_INSTALL}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }
        for j in self.data[DataName.LIST_NODE]:
            val = self.main_result[VarName.DG_INSTALL][j]
            self.var[VarName.DG_INSTALL][j].setlb(val)
            self.var[VarName.DG_INSTALL][j].setub(val)
            self.var[VarName.DG_INSTALL][j].set_value(val)

        # x^L_{ij}: Binary (fixed)
        self.var[VarName.LINE_HARDEN] = {
            (i, j): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.LINE_HARDEN}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }
        for (i, j) in self.data[DataName.LIST_LINE]:
            val = self.main_result[VarName.LINE_HARDEN][i, j]
            self.var[VarName.LINE_HARDEN][i, j].setlb(val)
            self.var[VarName.LINE_HARDEN][i, j].setub(val)
            self.var[VarName.LINE_HARDEN][i, j].set_value(val)

    def set_sub_objective(self):

        self.obj_term[ObjName.DG_VARIANT_COST] = pyo.quicksum(
            self.data[DataName.DICT_DG_COST_VAR][j] * self.var[VarName.DG_RATED_POWER][j]
            for j in self.data[DataName.LIST_NODE]
        )

        self.obj_term[ObjName.DG_GENERATING_COST] = pyo.quicksum(
            self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.DG_ACTIVE_POWER][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.obj_term[ObjName.LOAD_SHED_COST] = pyo.quicksum(
            self.data[DataName.NUM_COST_SHED] * self.var[VarName.LOAD_SHED_RATIO][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.set_objective(
            self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.DG_GENERATING_COST]
            + self.obj_term[ObjName.LOAD_SHED_COST],
            sense=pyo.minimize
        )

    def solve_relaxed(self):
        logger.info(f'Optimizing relaxed model of {self.model_name}')

        self.model_relax = self.model.clone()
        pyo.TransformationFactory('core.relax_integer_vars').apply_to(self.model_relax)

        self.model_relax.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        self._last_results_relax = self.solver.solve(self.model_relax, tee=True, load_solutions=True)

    def benders_opt_cut_info_generator(self):
        """
        After solving the RELAXED sub model, derive the coefficients and constants for generating
        the corresponding Benders optimality cut.
        """

        constant_term = 0.0

        var_coeff_dict = {
            var_class_name: {var_key: 0.0 for var_key in sorted(self.main_result[var_class_name].keys())}
            for var_class_name in sorted(self.main_result.keys())
            if var_class_name in self.var
        }

        relax_id_to_key = {}
        for var_class_name in sorted(var_coeff_dict.keys()):
            for var_key in sorted(var_coeff_dict[var_class_name].keys()):
                v = self.var[var_class_name][var_key]
                comp_name = self._var_comp_name_by_id[id(v)]
                v_relax = getattr(self.model_relax, comp_name)
                relax_id_to_key[id(v_relax)] = (var_class_name, var_key)

        for c in self.model_relax._constr_list.values():

            pi = self.model_relax.dual.get(c, 0.0)

            if c.upper is not None and c.lower is None:
                residual = c.body - c.upper
            elif c.lower is not None and c.upper is None:
                residual = c.body - c.lower
            else:
                rhs = c.upper if c.upper is not None else c.lower
                residual = c.body - rhs

            repn = generate_standard_repn(residual, compute_values=False)

            c0 = repn.constant if repn.constant is not None else 0.0
            constant_term += float(pi) * (-pyo.value(c0))

            if repn.linear_vars is None:
                continue

            for v_i, a_i in zip(repn.linear_vars, repn.linear_coefs):
                key = relax_id_to_key.get(id(v_i), None)
                if key is None:
                    continue
                var_class_name, var_key = key
                var_coeff_dict[var_class_name][var_key] += float(pi) * float(a_i)

        return constant_term, var_coeff_dict
