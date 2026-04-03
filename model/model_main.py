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

    def __init__(self, model_name='default_m', model_data=None):
        super().__init__(model_name=model_name, model_data=model_data)

        self.added_obj_term_weight = 0.0
        self.main_problem_model_type = (
            model_data.get(DataName.MAIN_PROBLEM_MODEL_TYPE, MainProblemModelTypeName.COLLAPSED_NO_TIME)
            if model_data is not None else MainProblemModelTypeName.COLLAPSED_NO_TIME
        )
        self.decomposition_state_var_name_list = [
            VarName.DG_INSTALL,
            VarName.LINE_HARDEN,
            VarName.DG_INSTALL_TYPE,
        ]
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
        return list(self.decomposition_state_var_name_list)

    def get_decomposition_structural_constr_name_list(self):
        return list(self.decomposition_structural_constr_name_list)

    def get_decomposition_var_key_lists_for_cglp(self):
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
        self.add_vars_basic_generator_c()

        self.var[VarName.MAIN_LINE_CONNECTED] = {
            (i, j): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_LINE_CONNECTED}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

        self.var[VarName.MAIN_LOAD_SHED_RATIO] = {
            j: self.add_var(
                domain=pyo.Reals, lb=0, ub=1,
                name=f'{VarName.MAIN_LOAD_SHED_RATIO}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        self.var[VarName.MAIN_BUS_VOLTAGE] = {
            j: self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_BUS_VOLTAGE}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        self.var[VarName.MAIN_DG_ACTIVE_POWER] = {
            j: self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_ACTIVE_POWER}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        self.var[VarName.MAIN_DG_REACTIVE_POWER] = {
            j: self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_REACTIVE_POWER}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        self.var[VarName.MAIN_LINE_ACTIVE_FLOW] = {
            (i, j): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_ACTIVE_FLOW}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

        self.var[VarName.MAIN_LINE_REACTIVE_FLOW] = {
            (i, j): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_REACTIVE_FLOW}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW] = {
            (i, j): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_VIRTUAL_LINE_FLOW}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

        self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR] = {
            j: self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_VIRTUAL_SOURCE_INDICATOR}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

        self.var[VarName.MAIN_VIRTUAL_INJECT_POWER] = {
            j: self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_VIRTUAL_INJECT_POWER}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

    def add_vars_collapsed_sp_by_time(self):
        self.add_vars_basic_generator_c()

        self.var[VarName.MAIN_LINE_CONNECTED] = {
            (i, j, t): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_LINE_CONNECTED}_({i},{j},{t})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_LOAD_SHED_RATIO] = {
            (j, t): self.add_var(
                domain=pyo.Reals, lb=0, ub=1,
                name=f'{VarName.MAIN_LOAD_SHED_RATIO}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_BUS_VOLTAGE] = {
            (j, t): self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_BUS_VOLTAGE}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_DG_ACTIVE_POWER] = {
            (j, t): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_ACTIVE_POWER}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_DG_REACTIVE_POWER] = {
            (j, t): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_REACTIVE_POWER}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_LINE_ACTIVE_FLOW] = {
            (i, j, t): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_ACTIVE_FLOW}_({i},{j},{t})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_LINE_REACTIVE_FLOW] = {
            (i, j, t): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_REACTIVE_FLOW}_({i},{j},{t})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW] = {
            (i, j, t): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_VIRTUAL_LINE_FLOW}_({i},{j},{t})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR] = {
            (j, t): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_VIRTUAL_SOURCE_INDICATOR}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

        self.var[VarName.MAIN_VIRTUAL_INJECT_POWER] = {
            (j, t): self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_VIRTUAL_INJECT_POWER}_({j},{t})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
        }

    def add_vars_collapsed_sp_by_scenario(self):
        self.add_vars_basic_generator_c()

        self.var[VarName.MAIN_LINE_CONNECTED] = {
            (i, j, s): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_LINE_CONNECTED}_({i},{j},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_LOAD_SHED_RATIO] = {
            (j, s): self.add_var(
                domain=pyo.Reals, lb=0, ub=1,
                name=f'{VarName.MAIN_LOAD_SHED_RATIO}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_BUS_VOLTAGE] = {
            (j, s): self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_BUS_VOLTAGE}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_DG_ACTIVE_POWER] = {
            (j, s): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_ACTIVE_POWER}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_DG_REACTIVE_POWER] = {
            (j, s): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_DG_REACTIVE_POWER}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_LINE_ACTIVE_FLOW] = {
            (i, j, s): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_ACTIVE_FLOW}_({i},{j},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_LINE_REACTIVE_FLOW] = {
            (i, j, s): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_LINE_REACTIVE_FLOW}_({i},{j},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW] = {
            (i, j, s): self.add_var(
                domain=pyo.Reals,
                name=f'{VarName.MAIN_VIRTUAL_LINE_FLOW}_({i},{j},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR] = {
            (j, s): self.add_var(
                domain=pyo.Binary,
                name=f'{VarName.MAIN_VIRTUAL_SOURCE_INDICATOR}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        self.var[VarName.MAIN_VIRTUAL_INJECT_POWER] = {
            (j, s): self.add_var(
                domain=pyo.Reals, lb=0,
                name=f'{VarName.MAIN_VIRTUAL_INJECT_POWER}_({j},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def add_constr_collapsed_sp_no_time(self):
        # Generator rated power linking
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                self.var[VarName.DG_RATED_POWER][j]
                == pyo.quicksum(
                    self.data[DataName.DICT_DG_RATED_POWER][typ] * self.var[VarName.DG_INSTALL_TYPE][j, typ]
                    for typ in self.data[DataName.LIST_DG_TYPE]
                ),
                name=f'{ConstrName.MAIN_DG_OPERATION}_{j}'
            )

        # Line connectivity
        for (i, j) in self.data[DataName.LIST_LINE]:
            self.add_constr(
                self.var[VarName.MAIN_LINE_CONNECTED][i, j]
                <= self.data[DataName.DICT_HAT_LINE_HEALTHY_NH][i, j] * (1 - self.var[VarName.LINE_HARDEN][i, j])
                + self.data[DataName.DICT_HAT_LINE_HEALTHY_H][i, j] * self.var[VarName.LINE_HARDEN][i, j],
                name=f'{ConstrName.MAIN_LINE_CONNECTED}_{i}_{j}'
            )

        # System topology constraints
        virtual_injection_m = len(self.data[DataName.LIST_NODE])
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                pyo.quicksum(
                    self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][j, k] for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j] for i in self.data[DataName.DICT_NODE_PARENTS][j]
                )
                == self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j] - 1,
                name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C1}_{j}'
            )

            self.add_constr(
                self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j]
                <= virtual_injection_m * self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j],
                name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C2}_{j}'
            )

        for (i, j) in self.data[DataName.LIST_LINE]:
            self.add_constr(
                -virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j]
                <= self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j],
                name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_L_{i}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j]
                <= virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j],
                name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_R_{i}_{j}'
            )

        self.add_constr(
            pyo.quicksum(
                self.var[VarName.MAIN_LINE_CONNECTED][i, j]
                for (i, j) in self.data[DataName.LIST_LINE]
            ) == len(self.data[DataName.LIST_NODE]) - pyo.quicksum(
                self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j] for j in self.data[DataName.LIST_NODE]
            ),
            name=ConstrName.MAIN_RADIALITY
        )

        # DG operating constraints
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                self.var[VarName.MAIN_DG_ACTIVE_POWER][j] <= self.var[VarName.DG_RATED_POWER][j],
                name=f'{ConstrName.MAIN_DG_ACTIVE_POWER_UB}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_DG_REACTIVE_POWER][j]
                <= self.data[DataName.DICT_DG_ALPHA_UB][j] * self.var[VarName.MAIN_DG_ACTIVE_POWER][j],
                name=f'{ConstrName.MAIN_DG_REACTIVE_POWER_UB}_{j}'
            )

        # System operating constraints
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                self.data[DataName.NUM_VOLTAGE_LB] <= self.var[VarName.MAIN_BUS_VOLTAGE][j],
                name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C1}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_BUS_VOLTAGE][j] <= self.data[DataName.NUM_VOLTAGE_UB],
                name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C2}_{j}'
            )

        self.add_constr(
            self.var[VarName.MAIN_BUS_VOLTAGE][self.data[DataName.SLACK_NODE_IDX]] == self.data[DataName.NUM_VOLTAGE_SLACK],
            name=ConstrName.MAIN_VOLTAGE_SLACK_BUS
        )

        voltage_flow_m = self.data[DataName.NUM_TOTAL_POWER]
        for (i, j) in self.data[DataName.LIST_LINE]:
            v_drop_on_ij = (
                self.data[DataName.DICT_LINE_RESISTANCE][i, j] * self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j]
                + self.data[DataName.DICT_LINE_REACTANCE][i, j] * self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j]
            )

            self.add_constr(
                -voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j])
                <= self.var[VarName.MAIN_BUS_VOLTAGE][i] - self.var[VarName.MAIN_BUS_VOLTAGE][j]
                - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK],
                name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_L_{i}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_BUS_VOLTAGE][i] - self.var[VarName.MAIN_BUS_VOLTAGE][j]
                - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK]
                <= voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j]),
                name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_R_{i}_{j}'
            )

        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                pyo.quicksum(
                    self.var[VarName.MAIN_LINE_ACTIVE_FLOW][j, k]
                    for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j]
                    for i in self.data[DataName.DICT_NODE_PARENTS][j]
                )
                == self.var[VarName.MAIN_DG_ACTIVE_POWER][j]
                - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j]) * self.data[DataName.DICT_HAT_DEMAND_ACTIVE][j],
                name=f'{ConstrName.MAIN_FLOW_BALANCE_C1}_{j}'
            )

            self.add_constr(
                pyo.quicksum(
                    self.var[VarName.MAIN_LINE_REACTIVE_FLOW][j, k]
                    for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j]
                    for i in self.data[DataName.DICT_NODE_PARENTS][j]
                )
                == self.var[VarName.MAIN_DG_REACTIVE_POWER][j]
                - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j]) * self.data[DataName.DICT_HAT_DEMAND_REACTIVE][j],
                name=f'{ConstrName.MAIN_FLOW_BALANCE_C2}_{j}'
            )

        for (i, j) in self.data[DataName.LIST_LINE]:
            line_connected = self.var[VarName.MAIN_LINE_CONNECTED][i, j]
            self.add_constr(
                -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j],
                name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_L_{i}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j] <= voltage_flow_m * line_connected,
                name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_R_{i}_{j}'
            )
            self.add_constr(
                -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j],
                name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_L_{i}_{j}'
            )
            self.add_constr(
                self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j] <= voltage_flow_m * line_connected,
                name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_R_{i}_{j}'
            )

        sqrt3 = math.sqrt(3.0)
        for (i, j) in self.data[DataName.LIST_LINE]:
            thermal_ub = self.data[DataName.DICT_LINE_THERMAL_UB][i, j]
            p_flow = self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j]
            q_flow = self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j]

            self.add_constr(
                -2 * thermal_ub <= sqrt3 * p_flow + q_flow,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_L_{i}_{j}'
            )
            self.add_constr(
                sqrt3 * p_flow + q_flow <= 2 * thermal_ub,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_R_{i}_{j}'
            )
            self.add_constr(
                -thermal_ub <= p_flow,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_L_{i}_{j}'
            )
            self.add_constr(
                p_flow <= thermal_ub,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_R_{i}_{j}'
            )
            self.add_constr(
                -2 * thermal_ub <= sqrt3 * p_flow - q_flow,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_L_{i}_{j}'
            )
            self.add_constr(
                sqrt3 * p_flow - q_flow <= 2 * thermal_ub,
                name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_R_{i}_{j}'
            )

    def add_constr_collapsed_sp_by_time(self):
        # Generator rated power linking
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                self.var[VarName.DG_RATED_POWER][j]
                == pyo.quicksum(
                    self.data[DataName.DICT_DG_RATED_POWER][typ] * self.var[VarName.DG_INSTALL_TYPE][j, typ]
                    for typ in self.data[DataName.LIST_DG_TYPE]
                ),
                name=f'{ConstrName.MAIN_DG_OPERATION}_{j}'
            )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    self.var[VarName.MAIN_LINE_CONNECTED][i, j, t]
                    <= self.data[DataName.DICT_HAT_LINE_HEALTHY_NH_BY_TIME][i, j, t] * (1 - self.var[VarName.LINE_HARDEN][i, j])
                    + self.data[DataName.DICT_HAT_LINE_HEALTHY_H_BY_TIME][i, j, t] * self.var[VarName.LINE_HARDEN][i, j],
                    name=f'{ConstrName.MAIN_LINE_CONNECTED}_{i}_{j}_{t}'
                )

        virtual_injection_m = len(self.data[DataName.LIST_NODE])
        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][j, k, t] for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, t] for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j, t] - 1,
                    name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C1}_{j}_{t}'
                )

                self.add_constr(
                    self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j, t]
                    <= virtual_injection_m * self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j, t],
                    name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C2}_{j}_{t}'
                )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    -virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j, t]
                    <= self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, t],
                    name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, t]
                    <= virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j, t],
                    name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_R_{i}_{j}_{t}'
                )

        for t in self.data[DataName.LIST_TIME]:
            self.add_constr(
                pyo.quicksum(
                    self.var[VarName.MAIN_LINE_CONNECTED][i, j, t]
                    for (i, j) in self.data[DataName.LIST_LINE]
                ) == len(self.data[DataName.LIST_NODE]) - pyo.quicksum(
                    self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j, t] for j in self.data[DataName.LIST_NODE]
                ),
                name=f'{ConstrName.MAIN_RADIALITY}_{t}'
            )

        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    self.var[VarName.MAIN_DG_ACTIVE_POWER][j, t] <= self.var[VarName.DG_RATED_POWER][j],
                    name=f'{ConstrName.MAIN_DG_ACTIVE_POWER_UB}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_DG_REACTIVE_POWER][j, t]
                    <= self.data[DataName.DICT_DG_ALPHA_UB][j] * self.var[VarName.MAIN_DG_ACTIVE_POWER][j, t],
                    name=f'{ConstrName.MAIN_DG_REACTIVE_POWER_UB}_{j}_{t}'
                )

        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    self.data[DataName.NUM_VOLTAGE_LB] <= self.var[VarName.MAIN_BUS_VOLTAGE][j, t],
                    name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C1}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_BUS_VOLTAGE][j, t] <= self.data[DataName.NUM_VOLTAGE_UB],
                    name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C2}_{j}_{t}'
                )

        for t in self.data[DataName.LIST_TIME]:
            self.add_constr(
                self.var[VarName.MAIN_BUS_VOLTAGE][self.data[DataName.SLACK_NODE_IDX], t] == self.data[DataName.NUM_VOLTAGE_SLACK],
                name=f'{ConstrName.MAIN_VOLTAGE_SLACK_BUS}_{t}'
            )

        voltage_flow_m = self.data[DataName.NUM_TOTAL_POWER]
        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                v_drop_on_ij = (
                    self.data[DataName.DICT_LINE_RESISTANCE][i, j] * self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, t]
                    + self.data[DataName.DICT_LINE_REACTANCE][i, j] * self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, t]
                )

                self.add_constr(
                    -voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j, t])
                    <= self.var[VarName.MAIN_BUS_VOLTAGE][i, t] - self.var[VarName.MAIN_BUS_VOLTAGE][j, t]
                    - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK],
                    name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_BUS_VOLTAGE][i, t] - self.var[VarName.MAIN_BUS_VOLTAGE][j, t]
                    - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK]
                    <= voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j, t]),
                    name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_R_{i}_{j}_{t}'
                )

        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_LINE_ACTIVE_FLOW][j, k, t]
                        for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, t]
                        for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_DG_ACTIVE_POWER][j, t]
                    - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j, t]) * self.data[DataName.DICT_HAT_DEMAND_ACTIVE_BY_TIME][j, t],
                    name=f'{ConstrName.MAIN_FLOW_BALANCE_C1}_{j}_{t}'
                )

                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_LINE_REACTIVE_FLOW][j, k, t]
                        for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, t]
                        for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_DG_REACTIVE_POWER][j, t]
                    - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j, t]) * self.data[DataName.DICT_HAT_DEMAND_REACTIVE_BY_TIME][j, t],
                    name=f'{ConstrName.MAIN_FLOW_BALANCE_C2}_{j}_{t}'
                )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                line_connected = self.var[VarName.MAIN_LINE_CONNECTED][i, j, t]
                self.add_constr(
                    -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, t],
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, t] <= voltage_flow_m * line_connected,
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_R_{i}_{j}_{t}'
                )
                self.add_constr(
                    -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, t],
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, t] <= voltage_flow_m * line_connected,
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_R_{i}_{j}_{t}'
                )

        sqrt3 = math.sqrt(3.0)
        for (i, j) in self.data[DataName.LIST_LINE]:
            thermal_ub = self.data[DataName.DICT_LINE_THERMAL_UB][i, j]
            for t in self.data[DataName.LIST_TIME]:
                p_flow = self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, t]
                q_flow = self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, t]

                self.add_constr(
                    -2 * thermal_ub <= sqrt3 * p_flow + q_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    sqrt3 * p_flow + q_flow <= 2 * thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_R_{i}_{j}_{t}'
                )
                self.add_constr(
                    -thermal_ub <= p_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    p_flow <= thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_R_{i}_{j}_{t}'
                )
                self.add_constr(
                    -2 * thermal_ub <= sqrt3 * p_flow - q_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_L_{i}_{j}_{t}'
                )
                self.add_constr(
                    sqrt3 * p_flow - q_flow <= 2 * thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_R_{i}_{j}_{t}'
                )

    def add_constr_collapsed_sp_by_scenario(self):
        # Generator rated power linking
        for j in self.data[DataName.LIST_NODE]:
            self.add_constr(
                self.var[VarName.DG_RATED_POWER][j]
                == pyo.quicksum(
                    self.data[DataName.DICT_DG_RATED_POWER][typ] * self.var[VarName.DG_INSTALL_TYPE][j, typ]
                    for typ in self.data[DataName.LIST_DG_TYPE]
                ),
                name=f'{ConstrName.MAIN_DG_OPERATION}_{j}'
            )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    self.var[VarName.MAIN_LINE_CONNECTED][i, j, s]
                    <= self.data[DataName.DICT_HAT_LINE_HEALTHY_NH_BY_SCENARIO][i, j, s] * (1 - self.var[VarName.LINE_HARDEN][i, j])
                    + self.data[DataName.DICT_HAT_LINE_HEALTHY_H_BY_SCENARIO][i, j, s] * self.var[VarName.LINE_HARDEN][i, j],
                    name=f'{ConstrName.MAIN_LINE_CONNECTED}_{i}_{j}_{s}'
                )

        virtual_injection_m = len(self.data[DataName.LIST_NODE])
        for j in self.data[DataName.LIST_NODE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][j, k, s] for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, s] for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j, s] - 1,
                    name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C1}_{j}_{s}'
                )

                self.add_constr(
                    self.var[VarName.MAIN_VIRTUAL_INJECT_POWER][j, s]
                    <= virtual_injection_m * self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j, s],
                    name=f'{ConstrName.MAIN_VIRTUAL_FLOW_BAL_C2}_{j}_{s}'
                )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    -virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j, s]
                    <= self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, s],
                    name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_VIRTUAL_LINE_FLOW][i, j, s]
                    <= virtual_injection_m * self.var[VarName.MAIN_LINE_CONNECTED][i, j, s],
                    name=f'{ConstrName.MAIN_NO_VIRTUAL_ON_OPEN}_R_{i}_{j}_{s}'
                )

        for s in self.data[DataName.LIST_SCENARIO]:
            self.add_constr(
                pyo.quicksum(
                    self.var[VarName.MAIN_LINE_CONNECTED][i, j, s]
                    for (i, j) in self.data[DataName.LIST_LINE]
                ) == len(self.data[DataName.LIST_NODE]) - pyo.quicksum(
                    self.var[VarName.MAIN_VIRTUAL_SOURCE_INDICATOR][j, s] for j in self.data[DataName.LIST_NODE]
                ),
                name=f'{ConstrName.MAIN_RADIALITY}_{s}'
            )

        for j in self.data[DataName.LIST_NODE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    self.var[VarName.MAIN_DG_ACTIVE_POWER][j, s] <= self.var[VarName.DG_RATED_POWER][j],
                    name=f'{ConstrName.MAIN_DG_ACTIVE_POWER_UB}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_DG_REACTIVE_POWER][j, s]
                    <= self.data[DataName.DICT_DG_ALPHA_UB][j] * self.var[VarName.MAIN_DG_ACTIVE_POWER][j, s],
                    name=f'{ConstrName.MAIN_DG_REACTIVE_POWER_UB}_{j}_{s}'
                )

        for j in self.data[DataName.LIST_NODE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    self.data[DataName.NUM_VOLTAGE_LB] <= self.var[VarName.MAIN_BUS_VOLTAGE][j, s],
                    name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C1}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_BUS_VOLTAGE][j, s] <= self.data[DataName.NUM_VOLTAGE_UB],
                    name=f'{ConstrName.MAIN_VOLTAGE_RANGE_C2}_{j}_{s}'
                )

        for s in self.data[DataName.LIST_SCENARIO]:
            self.add_constr(
                self.var[VarName.MAIN_BUS_VOLTAGE][self.data[DataName.SLACK_NODE_IDX], s] == self.data[DataName.NUM_VOLTAGE_SLACK],
                name=f'{ConstrName.MAIN_VOLTAGE_SLACK_BUS}_{s}'
            )

        voltage_flow_m = self.data[DataName.NUM_TOTAL_POWER]
        for (i, j) in self.data[DataName.LIST_LINE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                v_drop_on_ij = (
                    self.data[DataName.DICT_LINE_RESISTANCE][i, j] * self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, s]
                    + self.data[DataName.DICT_LINE_REACTANCE][i, j] * self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, s]
                )

                self.add_constr(
                    -voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j, s])
                    <= self.var[VarName.MAIN_BUS_VOLTAGE][i, s] - self.var[VarName.MAIN_BUS_VOLTAGE][j, s]
                    - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK],
                    name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_BUS_VOLTAGE][i, s] - self.var[VarName.MAIN_BUS_VOLTAGE][j, s]
                    - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK]
                    <= voltage_flow_m * (1 - self.var[VarName.MAIN_LINE_CONNECTED][i, j, s]),
                    name=f'{ConstrName.MAIN_VOLTAGE_FLOW_REL}_R_{i}_{j}_{s}'
                )

        for j in self.data[DataName.LIST_NODE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_LINE_ACTIVE_FLOW][j, k, s]
                        for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, s]
                        for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_DG_ACTIVE_POWER][j, s]
                    - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j, s]) * self.data[DataName.DICT_HAT_DEMAND_ACTIVE_BY_SCENARIO][j, s],
                    name=f'{ConstrName.MAIN_FLOW_BALANCE_C1}_{j}_{s}'
                )

                self.add_constr(
                    pyo.quicksum(
                        self.var[VarName.MAIN_LINE_REACTIVE_FLOW][j, k, s]
                        for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                    )
                    - pyo.quicksum(
                        self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, s]
                        for i in self.data[DataName.DICT_NODE_PARENTS][j]
                    )
                    == self.var[VarName.MAIN_DG_REACTIVE_POWER][j, s]
                    - (1 - self.var[VarName.MAIN_LOAD_SHED_RATIO][j, s]) * self.data[DataName.DICT_HAT_DEMAND_REACTIVE_BY_SCENARIO][j, s],
                    name=f'{ConstrName.MAIN_FLOW_BALANCE_C2}_{j}_{s}'
                )

        for (i, j) in self.data[DataName.LIST_LINE]:
            for s in self.data[DataName.LIST_SCENARIO]:
                line_connected = self.var[VarName.MAIN_LINE_CONNECTED][i, j, s]
                self.add_constr(
                    -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, s],
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, s] <= voltage_flow_m * line_connected,
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C1}_R_{i}_{j}_{s}'
                )
                self.add_constr(
                    -voltage_flow_m * line_connected <= self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, s],
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, s] <= voltage_flow_m * line_connected,
                    name=f'{ConstrName.MAIN_NO_FLOW_ON_OPEN_C2}_R_{i}_{j}_{s}'
                )

        sqrt3 = math.sqrt(3.0)
        for (i, j) in self.data[DataName.LIST_LINE]:
            thermal_ub = self.data[DataName.DICT_LINE_THERMAL_UB][i, j]
            for s in self.data[DataName.LIST_SCENARIO]:
                p_flow = self.var[VarName.MAIN_LINE_ACTIVE_FLOW][i, j, s]
                q_flow = self.var[VarName.MAIN_LINE_REACTIVE_FLOW][i, j, s]

                self.add_constr(
                    -2 * thermal_ub <= sqrt3 * p_flow + q_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    sqrt3 * p_flow + q_flow <= 2 * thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C1}_R_{i}_{j}_{s}'
                )
                self.add_constr(
                    -thermal_ub <= p_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    p_flow <= thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C2}_R_{i}_{j}_{s}'
                )
                self.add_constr(
                    -2 * thermal_ub <= sqrt3 * p_flow - q_flow,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_L_{i}_{j}_{s}'
                )
                self.add_constr(
                    sqrt3 * p_flow - q_flow <= 2 * thermal_ub,
                    name=f'{ConstrName.MAIN_LINE_THERMAL_C3}_R_{i}_{j}_{s}'
                )

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

        self.obj_term[ObjName.DG_FIXED_COST] = pyo.quicksum(
            self.data[DataName.DICT_DG_COST_FIX][j] * self.var[VarName.DG_INSTALL][j]
            for j in self.data[DataName.LIST_NODE]
        )

        # variant cost at node j: rated_power * unit_price + extra_adjustment (for chosen DG type)
        self.obj_term[ObjName.DG_VARIANT_COST] = pyo.quicksum(
            (self.data[DataName.DICT_DG_RATED_POWER][typ] * self.data[DataName.DICT_DG_UNIT_PRICE][typ]
             + self.data[DataName.DICT_DG_EXTRA_ADJUSTMENT][typ])
            * self.var[VarName.DG_INSTALL_TYPE][j, typ]
            for j in self.data[DataName.LIST_NODE]
            for typ in self.data[DataName.LIST_DG_TYPE]
        )

        self.obj_term[ObjName.LINE_HARDEN_COST] = pyo.quicksum(
            self.data[DataName.DICT_LINE_COST_HARDEN][i, j] * self.var[VarName.LINE_HARDEN][i, j]
            for (i, j) in self.data[DataName.LIST_LINE]
        )

        self.obj_term[ObjName.SUB_OBJ_TERM] = pyo.quicksum(
            self.data[DataName.DICT_SC_PROB][s] * self.var[VarName.SUB_OBJ_EST][s]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        if self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = pyo.quicksum(
                self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.MAIN_DG_ACTIVE_POWER][j]
                for j in self.data[DataName.LIST_NODE]
            )

            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = pyo.quicksum(
                self.data[DataName.NUM_COST_SHED] * self.var[VarName.MAIN_LOAD_SHED_RATIO][j]
                for j in self.data[DataName.LIST_NODE]
            )
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = pyo.quicksum(
                self.data[DataName.DICT_DG_COST_UNIT][j] * self.var[VarName.MAIN_DG_ACTIVE_POWER][j, t]
                for j in self.data[DataName.LIST_NODE]
                for t in self.data[DataName.LIST_TIME]
            )

            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = pyo.quicksum(
                self.data[DataName.NUM_COST_SHED] * self.var[VarName.MAIN_LOAD_SHED_RATIO][j, t]
                for j in self.data[DataName.LIST_NODE]
                for t in self.data[DataName.LIST_TIME]
            )
        elif self.main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = pyo.quicksum(
                self.data[DataName.DICT_SC_PROB][s]
                * self.data[DataName.DICT_DG_COST_UNIT][j]
                * self.var[VarName.MAIN_DG_ACTIVE_POWER][j, s]
                for j in self.data[DataName.LIST_NODE]
                for s in self.data[DataName.LIST_SCENARIO]
            )

            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = pyo.quicksum(
                self.data[DataName.DICT_SC_PROB][s]
                * self.data[DataName.NUM_COST_SHED]
                * self.var[VarName.MAIN_LOAD_SHED_RATIO][j, s]
                for j in self.data[DataName.LIST_NODE]
                for s in self.data[DataName.LIST_SCENARIO]
            )
        else:
            self.obj_term[ObjName.MAIN_APPROX_DG_GENERATING_COST] = 0.0
            self.obj_term[ObjName.MAIN_APPROX_LOAD_SHED_COST] = 0.0

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
