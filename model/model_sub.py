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

    def __init__(self, model_name='default_m', model_data=None, main_result=None, sub_model_sce_list=None,
                 solver_config=None):
        super().__init__(model_name=model_name, model_data=model_data, solver_config=solver_config)

        self.main_result = main_result

        # the dict for collecting necessary information of the local variable copy constraints.
        # key: constraint name
        # value: (var class, var idx), for finding corresponding var by self.var[var class][var name]
        self.local_copy_constr_info = {}

        # the dict for recording constraints' dual value
        # key: constraint name
        # value: dual value
        self.constr_dual_value = {}

        self.sub_model_sce_list = sub_model_sce_list

    def set_sub_objective(self):

        # self.obj_term[ObjName.DG_VARIANT_COST] = pyo.quicksum(
        #     self.data[DataName.DICT_DG_COST_VAR][j] * self.var[VarName.DG_RATED_POWER][j]
        #     for j in self.data[DataName.LIST_NODE]
        # )

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
            # self.obj_term[ObjName.DG_VARIANT_COST]
            + self.obj_term[ObjName.DG_GENERATING_COST]
            + self.obj_term[ObjName.LOAD_SHED_COST],
            sense=pyo.minimize
        )

    def solve_relaxed(self):
        logger.info(f'Optimizing relaxed model of {self.model_name}')

        self.model_relax = self.model.clone()
        pyo.TransformationFactory('core.relax_integer_vars').apply_to(self.model_relax)

        self.model_relax.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        self._last_results_relax = self._solve_model(
            self.model_relax,
            use_config_defaults=True,
            load_solutions=True
        )

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

    def get_decomposition_state_var_name_list(self):
        """Return the canonical master-state blocks copied into every subproblem."""
        return list(DECOMPOSITION_STATE_VAR_NAMES)

    def _validate_local_copy_main_result(self):
        """Check that the provided master solution exactly matches canonical state keys."""
        if self.main_result is None:
            raise ValueError('main_result is required for the canonical local-copy subproblem.')

        missing_var_names = [
            var_name
            for var_name in self.get_decomposition_state_var_name_list()
            if var_name not in self.main_result
        ]
        if missing_var_names:
            raise ValueError(f'main_result is missing decomposition state variables: {missing_var_names}')

        for var_name in self.get_decomposition_state_var_name_list():
            if var_name not in self.var:
                raise ValueError(f'Canonical subproblem does not contain variable block: {var_name}')
            expected_keys = set(self.var[var_name].keys())
            provided_keys = set(self.main_result[var_name].keys())
            missing_keys = sorted(expected_keys - provided_keys)
            extra_keys = sorted(provided_keys - expected_keys)
            if missing_keys or extra_keys:
                raise ValueError(
                    f'main_result keys mismatch for {var_name}: '
                    f'missing={missing_keys}, extra={extra_keys}'
                )

    def build_sub_model_canonical(self):
        """Build the subproblem with local-copy constraints as the only supported decomposition path."""

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

        # modularized design: add constraints to fix the main stage variables.
        self.add_constr_state_var_local_copy()

        # set submodel objective
        self.set_sub_objective()

    def add_constr_state_var_local_copy(self):
        """Link each canonical state variable to the supplied master solution and record its row map."""
        self._validate_local_copy_main_result()
        var_name_list = self.get_decomposition_state_var_name_list()
        for var_name in var_name_list:
            var_key_list = sorted(self.main_result[var_name].keys())
            for var_key in var_key_list:
                constr_name = f'{ConstrName.VAR_LOCAL_COPY}_{var_name}_{var_key}'
                self.add_constr(
                    self.var[var_name][var_key] == self.main_result[var_name][var_key],
                    name=constr_name,
                )
                self.local_copy_constr_info[constr_name] = (var_name, var_key)

    def collect_dual_opt_sol(self, constr_to_collect_list):
        """
        collect the dual optimal solution value corresponding to given constraints
        :param constr_to_collect_list: list of constraint names. constraint names should be defined when adding constraints
        :return:
        constr_dual_value: dict. key: constraint name; value: dual optimal solution value
        constr_var_map: dict. key: constraint name; value: (var class, var key)
        """
        constr_var_map = {}
        for constr_name in constr_to_collect_list:
            constr_item = self.get_constr_item_in_relax_by_name(constr_name)
            pi = self.model_relax.dual[constr_item]
            self.constr_dual_value[constr_name] = pi
            constr_var_map[constr_name] = self.local_copy_constr_info[constr_name]

        return self.constr_dual_value.copy(), constr_var_map
