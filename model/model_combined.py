# -*- coding: utf-8 -*-
# @Time     : 2025/09/26
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_base import ModelBase
from model.formulation_blocks import (
    add_design_objective_terms,
    add_dg_design_binary_vars,
    add_dg_install_constraints,
    add_dg_operating_constraints,
    add_dg_rated_power_link_constraints,
    add_dg_rated_power_vars,
    add_line_connectivity_constraints,
    add_line_connectivity_vars,
    add_line_design_vars,
    add_operating_constraints,
    add_operating_objective_terms,
    add_operating_vars,
    add_topology_constraints,
    add_topology_vars,
    make_full_formulation_adapter,
)
from util.headers import *
from util.names import *

import pyomo.environ as pyo
import logging

logger = logging.getLogger(__name__)


class ModelCombined(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._formulation_adapter = None

    def get_formulation_adapter(self):
        if self._formulation_adapter is None:
            self._formulation_adapter = make_full_formulation_adapter(self.data)
        return self._formulation_adapter

    def build_model_all_obj_terms(self):
        self.add_vars()
        self.add_constraints()
        self.derive_obj_terms()
        self.add_all_objective_terms()

    def build_model_given_obj_terms(self, input_term_list):
        self.add_vars()
        self.add_constraints()
        self.derive_obj_terms()
        self.add_objective_by_terms(term_list=input_term_list)

    def add_vars(self):
        self.add_vars_basic_generator_bi()
        self.add_vars_basic_generator_c()
        self.add_vars_basic_line()
        self.add_vars_line_connectivity()
        self.add_vars_sys_operating()
        self.add_vars_sys_topology()

    def add_constraints(self):
        self.add_constr_DG_ub()
        self.add_constr_DG_rated_power_ub()
        self.add_constr_DG_operating()
        self.add_constr_line_connectivity()
        self.add_constr_system_operating()
        self.add_constr_system_topology_constraints()

    def add_vars_basic_generator_bi(self):
        add_dg_design_binary_vars(self)

    def add_vars_basic_generator_c(self):
        add_dg_rated_power_vars(self)

    def add_vars_basic_line(self):
        add_line_design_vars(self)

    def add_vars_sys_topology(self):
        add_topology_vars(self, self.get_formulation_adapter())

    def add_vars_line_connectivity(self):
        add_line_connectivity_vars(self, self.get_formulation_adapter())

    def add_vars_sys_operating(self):
        add_operating_vars(self, self.get_formulation_adapter())

    def add_constr_DG_ub(self):
        add_dg_install_constraints(self)

    def add_constr_DG_rated_power_ub(self):
        add_dg_rated_power_link_constraints(self, ConstrName.DG_OPERATION)

    def add_constr_line_connectivity(self):
        add_line_connectivity_constraints(self, self.get_formulation_adapter())

    def add_constr_system_topology_constraints(self):
        add_topology_constraints(self, self.get_formulation_adapter())

    def add_constr_DG_operating(self):
        add_dg_operating_constraints(self, self.get_formulation_adapter())

    def add_constr_system_operating(self):
        add_operating_constraints(self, self.get_formulation_adapter())

    def derive_obj_terms(self):
        add_design_objective_terms(self)
        add_operating_objective_terms(
            self,
            adapter=self.get_formulation_adapter(),
            dg_obj_name=ObjName.DG_GENERATING_COST,
            load_shed_obj_name=ObjName.LOAD_SHED_COST,
        )

    def add_all_objective_terms(self):

        self.set_objective(
            pyo.quicksum(
                self.obj_term[obj_term_name] for obj_term_name in sorted(self.obj_term.keys())
            ),
            sense=pyo.minimize
        )

    def add_objective_by_terms(self, term_list=None):

        self.set_objective(
            pyo.quicksum(
                self.obj_term[obj_term_name] for obj_term_name in term_list
            ),
            sense=pyo.minimize
        )
