# -*- coding: utf-8 -*-
# @Time     : 2025/10/20
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_combined import ModelCombined
from model.formulation_blocks import (
    add_operating_objective_terms,
    make_sub_formulation_adapter,
)

from util.headers import *
from util.names import *

import pyomo.environ as pyo

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

    def get_formulation_adapter(self):
        if self._formulation_adapter is None:
            self._formulation_adapter = make_sub_formulation_adapter(self.data)
        return self._formulation_adapter

    def set_sub_objective(self):
        """
        Recourse objective of one scenario group, weighted by the UNCONDITIONAL probabilities.

        Convention (docs/development_record.md, P3): the subproblem for group G minimizes
        ``sum_{s in G} P_s * Q_s(x)`` so that the matching cut is written on the weighted left-hand
        side ``sum_{s in G} P_s * theta_s`` and the master objective stays ``sum_s P_s * theta_s``.
        The expression is built by the shared formulation block, so the recourse objective now
        exists in exactly one place (workstream 6).
        """
        add_operating_objective_terms(
            self,
            adapter=self.get_formulation_adapter(),
            dg_obj_name=ObjName.DG_GENERATING_COST,
            load_shed_obj_name=ObjName.LOAD_SHED_COST,
        )

        self.set_objective(
            self.obj_term[ObjName.DG_GENERATING_COST]
            + self.obj_term[ObjName.LOAD_SHED_COST],
            sense=pyo.minimize
        )

    def get_scenario_probability_weights(self):
        """Return {s: P_s} (unconditional) for the scenarios carried by this subproblem."""
        return {
            s: self.data[DataName.DICT_SC_PROB_UNCOND][s]
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def _prepare_relaxed_model(self, relaxed_model):
        """Attach the dual Suffix so local-copy multipliers can be read after the LP solve."""
        relaxed_model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

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
        constr_dual_value, constr_var_map = {}, {}
        for constr_name in constr_to_collect_list:
            var_info = self.local_copy_constr_info.get(constr_name)
            if var_info is None:
                raise KeyError(
                    f'Constraint {constr_name} is not a registered local-copy row in {self.model_name}.'
                )
            constr_item = self.get_constr_item_in_relax_by_name(constr_name)
            if constr_item not in self.model_relax.dual:
                raise KeyError(
                    f'No dual value available for local-copy constraint {constr_name} in '
                    f'{self.model_name}; the relaxation must be solved with the dual Suffix attached.'
                )
            pi = self.model_relax.dual[constr_item]
            constr_dual_value[constr_name] = pi
            constr_var_map[constr_name] = var_info

        # Return only the rows that were asked for; self.constr_dual_value stays as the running
        # record for reporting.
        self.constr_dual_value.update(constr_dual_value)
        return constr_dual_value, constr_var_map
