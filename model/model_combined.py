# -*- coding: utf-8 -*-
# @Time     : 2025/09/26
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from model.model_base import ModelBase
from util.headers import *
from util.names import *
from gurobipy import GRB

import gurobipy as gp
import math
import logging

logger = logging.getLogger(__name__)


class ModelCombined(ModelBase):

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

        self.var[VarName.DG_INSTALL] = {
            j: self.model.addVar(
                vtype=GRB.BINARY,
                name=f'{VarName.DG_INSTALL}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

    def add_vars_basic_generator_c(self):

        self.var[VarName.DG_RATED_POWER] = {
            j: self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0,
                name=f'{VarName.DG_RATED_POWER}_({j})'
            )
            for j in self.data[DataName.LIST_NODE]
        }

    def add_vars_basic_line(self):
        # x^L_{ij}: Binary
        self.var[VarName.LINE_HARDEN] = {
            (i, j): self.model.addVar(
                vtype=GRB.BINARY,
                name=f'{VarName.LINE_HARDEN}_({i},{j})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
        }

    def add_vars_sys_topology(self):
        # u^L_{ijts}: Continuous, free
        self.var[VarName.VIRTUAL_LINE_FLOW] = {
            (i, j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                name=f'{VarName.VIRTUAL_LINE_FLOW}_({i},{j},{t},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # u^S_{jts}: Binary
        self.var[VarName.VIRTUAL_SOURCE_INDICATOR] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.BINARY,
                name=f'{VarName.VIRTUAL_SOURCE_INDICATOR}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # u^I_{jts}: Continuous, non-negative
        self.var[VarName.VIRTUAL_INJECT_POWER] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0,
                name=f'{VarName.VIRTUAL_INJECT_POWER}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def add_vars_line_connectivity(self):
        # y^C_{ijts}: Binary
        self.var[VarName.LINE_CONNECTED] = {
            (i, j, t, s): self.model.addVar(
                vtype=GRB.BINARY,
                name=f'{VarName.LINE_CONNECTED}_({i},{j},{t},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def add_vars_sys_operating(self):
        # v_{jts}: Continuous, non-negative
        self.var[VarName.BUS_VOLTAGE] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0,
                name=f'{VarName.BUS_VOLTAGE}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # p^G_{jts}: Continuous, free
        self.var[VarName.DG_ACTIVE_POWER] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                name=f'{VarName.DG_ACTIVE_POWER}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # q^G_{jts}: Continuous, free
        self.var[VarName.DG_REACTIVE_POWER] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                name=f'{VarName.DG_REACTIVE_POWER}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # p_{ijts}: Continuous, free
        self.var[VarName.LINE_ACTIVE_FLOW] = {
            (i, j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                name=f'{VarName.LINE_ACTIVE_FLOW}_({i},{j},{t},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # q_{ijts}: Continuous, free
        self.var[VarName.LINE_REACTIVE_FLOW] = {
            (i, j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                name=f'{VarName.LINE_REACTIVE_FLOW}_({i},{j},{t},{s})'
            )
            for (i, j) in self.data[DataName.LIST_LINE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

        # y^L_{jts}: Continuous, non-negative (load shed ratio)
        self.var[VarName.LOAD_SHED_RATIO] = {
            (j, t, s): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1,
                name=f'{VarName.LOAD_SHED_RATIO}_({j},{t},{s})'
            )
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        }

    def add_constr_DG_ub(self):
        self.model.addConstr(
            gp.quicksum(
                self.var[VarName.DG_INSTALL][j] for j in self.data[DataName.LIST_NODE]
            ) <= self.data[DataName.NUM_DG_UB],
            name=ConstrName.DG_UPPERBOUND
        )

    def add_constr_DG_rated_power_ub(self):
        for j in self.data[DataName.LIST_NODE]:
            self.model.addConstr(
                self.var[VarName.DG_RATED_POWER][j] <= self.data[DataName.NUM_RATED_POWER_UB] * self.var[VarName.DG_INSTALL][j],
                name=f'{ConstrName.DG_OPERATION}_{j}'
            )

    def add_constr_line_connectivity(self):
        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    self.model.addConstr(
                        self.var[VarName.LINE_CONNECTED][i, j, t, s]
                        <= self.data[DataName.DICT_LINE_HEALTHY_NH][i, j, t, s] * (1 - self.var[VarName.LINE_HARDEN][i, j])
                        + self.data[DataName.DICT_LINE_HEALTHY_H][i, j, t, s] * self.var[VarName.LINE_HARDEN][i, j],
                        name=f'{ConstrName.LINE_CONNECTED}_{i}_{j}_{t}_{s}'
                    )

    def add_constr_system_topology_constraints(self):
        # Based on Single-commodity Flow (SCF)

        # Virtual power flow balance
        V_INJ_M = len(self.data[DataName.LIST_NODE])
        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:

                    # c1: sum_k u^L_{jkts} - sum_i u^L_{ijts} = u^I_{jts} - 1
                    self.model.addConstr(
                        gp.quicksum(
                            self.var[VarName.VIRTUAL_LINE_FLOW][j, k, t, s] for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                        )
                        - gp.quicksum(
                            self.var[VarName.VIRTUAL_LINE_FLOW][i, j, t, s] for i in self.data[DataName.DICT_NODE_PARENTS][j]
                        ) == self.var[VarName.VIRTUAL_INJECT_POWER][j, t, s] - 1,
                        name=f'{ConstrName.VIRTUAL_FLOW_BAL_C1}_{j}_{t}_{s}'
                    )

                    # c2: u^I_{jts} <= V_INJ_M u^{S}_{jts}
                    self.model.addConstr(
                        self.var[VarName.VIRTUAL_INJECT_POWER][j, t, s]
                        <= V_INJ_M * self.var[VarName.VIRTUAL_SOURCE_INDICATOR][j, t, s],
                        name=f'{ConstrName.VIRTUAL_FLOW_BAL_C2}_{j}_{t}_{s}'
                    )

        # No virtual flow on open line
        # -M y^C_{ijts} <= u^L_{ijts} <= M y^C_{ijts}
        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    self.model.addConstr(
                        -V_INJ_M * self.var[VarName.LINE_CONNECTED][i, j, t, s]
                        <= self.var[VarName.VIRTUAL_LINE_FLOW][i, j, t, s],
                        name=f'{ConstrName.NO_VIRTUAL_ON_OPEN}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        self.var[VarName.VIRTUAL_LINE_FLOW][i, j, t, s]
                        <= V_INJ_M * self.var[VarName.LINE_CONNECTED][i, j, t, s],
                        name=f'{ConstrName.NO_VIRTUAL_ON_OPEN}_R_{i}_{j}_{t}_{s}'
                    )

        # sum_{(i,j)} y^C_{ijts} = |J| - sum_j u^S_{jts}
        for t in self.data[DataName.LIST_TIME]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.model.addConstr(
                    gp.quicksum(self.var[VarName.LINE_CONNECTED][i, j, t, s]
                                        for (i, j) in self.data[DataName.LIST_LINE])
                    == len(self.data[DataName.LIST_NODE]) - gp.quicksum(
                        self.var[VarName.VIRTUAL_SOURCE_INDICATOR][j, t, s] for j in self.data[DataName.LIST_NODE]
                    ),
                    name=f'{ConstrName.RADIALITY}_{t}_{s}'
                )

    def add_constr_DG_operating(self):
        # p^G_{jts} <= p^{Grt}_j
        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    self.model.addConstr(
                        self.var[VarName.DG_ACTIVE_POWER][j, t, s] <= self.var[VarName.DG_RATED_POWER][j],
                        name=f'{ConstrName.DG_ACTIVE_POWER_UB}_{j}_{t}_{s}'
                    )
        # q^G_{jts} <= tan(acos(alpha^G_j)) * p^G_{jts}
        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    self.model.addConstr(
                        self.var[VarName.DG_REACTIVE_POWER][j, t, s]
                        <= self.data[DataName.DICT_DG_ALPHA_UB][j] * self.var[VarName.DG_ACTIVE_POWER][j, t, s],
                        name=f'{ConstrName.DG_REACTIVE_POWER_UB}_{j}_{t}_{s}'
                    )

    def add_constr_system_operating(self):
        # \underline{V} <= v_{jts} <= \overline{V}  and  v_{slack,ts} = V0

        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    # Left (lower) bound
                    self.model.addConstr(
                        self.data[DataName.NUM_VOLTAGE_LB] <= self.var[VarName.BUS_VOLTAGE][j, t, s],
                        name=f'{ConstrName.VOLTAGE_RANGE_C1}_{j}_{t}_{s}'
                    )
                    # Right (upper) bound
                    self.model.addConstr(
                        self.var[VarName.BUS_VOLTAGE][j, t, s] <= self.data[DataName.NUM_VOLTAGE_UB],
                        name=f'{ConstrName.VOLTAGE_RANGE_C2}_{j}_{t}_{s}'
                    )

        # Slack bus equality
        slk_node_idx = self.data[DataName.SLACK_NODE_IDX]
        for t in self.data[DataName.LIST_TIME]:
            for s in self.data[DataName.LIST_SCENARIO]:
                self.model.addConstr(
                    self.var[VarName.BUS_VOLTAGE][slk_node_idx, t, s] == self.data[DataName.NUM_VOLTAGE_SLACK],
                    name=f'{ConstrName.VOLTAGE_SLACK_BUS}_{t}_{s}'
                )

        # Voltage - flow relationships
        # -M(1 - y^C) <= v_i - v_j - (R_ij p_ij + X_ij q_ij)/V0 <= M(1 - y^C)
        V_FLOW_M = self.data[DataName.NUM_TOTAL_POWER]
        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:

                    v_drop_on_ij = (
                            self.data[DataName.DICT_LINE_RESISTANCE][i, j] * self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s]
                            + self.data[DataName.DICT_LINE_REACTANCE][i, j] * self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s]
                    )

                    # Left (lower) side
                    self.model.addConstr(
                        - V_FLOW_M * (1 - self.var[VarName.LINE_CONNECTED][i, j, t, s])
                        <= self.var[VarName.BUS_VOLTAGE][i, t, s] - self.var[VarName.BUS_VOLTAGE][j, t, s]
                        - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK],
                        name=f'{ConstrName.VOLTAGE_FLOW_REL}_L_{i}_{j}_{t}_{s}'
                    )
                    # Right (upper) side
                    self.model.addConstr(
                        self.var[VarName.BUS_VOLTAGE][i, t, s] - self.var[VarName.BUS_VOLTAGE][j, t, s]
                        - v_drop_on_ij / self.data[DataName.NUM_VOLTAGE_SLACK]
                        <= V_FLOW_M * (1 - self.var[VarName.LINE_CONNECTED][i, j, t, s]),
                        name=f'{ConstrName.VOLTAGE_FLOW_REL}_R_{i}_{j}_{t}_{s}'
                    )

        # Active & reactive nodal balances
        for j in self.data[DataName.LIST_NODE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:

                    self.model.addConstr(
                        gp.quicksum(
                            self.var[VarName.LINE_ACTIVE_FLOW][j, k, t, s]
                            for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                        )
                        - gp.quicksum(
                            self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s]
                            for i in self.data[DataName.DICT_NODE_PARENTS][j]
                        )
                        == self.var[VarName.DG_ACTIVE_POWER][j, t, s]
                        - (1 - self.var[VarName.LOAD_SHED_RATIO][j, t, s])
                        * self.data[DataName.DICT_DEMAND_ACTIVE][j, t, s],
                        name=f'{ConstrName.FLOW_BALANCE_C1}_{j}_{t}_{s}'
                    )

                    self.model.addConstr(
                        gp.quicksum(
                            self.var[VarName.LINE_REACTIVE_FLOW][j, k, t, s]
                            for k in self.data[DataName.DICT_NODE_CHILDREN][j]
                        )
                        - gp.quicksum(
                            self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s]
                            for i in self.data[DataName.DICT_NODE_PARENTS][j]
                        )
                        == self.var[VarName.DG_REACTIVE_POWER][j, t, s]
                        - (1 - self.var[VarName.LOAD_SHED_RATIO][j, t, s])
                        * self.data[DataName.DICT_DEMAND_REACTIVE][j, t, s],
                        name=f'{ConstrName.FLOW_BALANCE_C2}_{j}_{t}_{s}'
                    )

        # no_flow_on_open_line
        # -M y^C <= p_ij <= M y^C;  -M y^C <= q_ij <= M y^C
        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:
                    yC = self.var[VarName.LINE_CONNECTED][i, j, t, s]
                    # Active flow
                    self.model.addConstr(
                        -V_FLOW_M * yC <= self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s],
                        name=f'{ConstrName.NO_FLOW_ON_OPEN_C1}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] <= V_FLOW_M * yC,
                        name=f'{ConstrName.NO_FLOW_ON_OPEN_C1}_R_{i}_{j}_{t}_{s}'
                    )
                    # Reactive flow
                    self.model.addConstr(
                        -V_FLOW_M * yC <= self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s],
                        name=f'{ConstrName.NO_FLOW_ON_OPEN_C2}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s] <= V_FLOW_M * yC,
                        name=f'{ConstrName.NO_FLOW_ON_OPEN_C2}_R_{i}_{j}_{t}_{s}'
                    )

        # line_thermal_limits
        # Linearized thermal limit (hexagon) with proper coefficients
        sqrt3 = math.sqrt(3.0)

        for (i, j) in self.data[DataName.LIST_LINE]:
            for t in self.data[DataName.LIST_TIME]:
                for s in self.data[DataName.LIST_SCENARIO]:

                    # c1: -2 S̄_ij <= sqrt(3)*p + q <= 2 S̄_ij
                    self.model.addConstr(
                        -2 * self.data[DataName.DICT_LINE_THERMAL_UB][i, j]
                        <= sqrt3 * self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] + self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s],
                        name=f'{ConstrName.LINE_THERMAL_C1}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        sqrt3 * self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] + self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s]
                        <= 2 * self.data[DataName.DICT_LINE_THERMAL_UB][i, j],
                        name=f'{ConstrName.LINE_THERMAL_C1}_R_{i}_{j}_{t}_{s}'
                    )

                    # c2: -S̄_ij <= p <= S̄_ij
                    self.model.addConstr(
                        - self.data[DataName.DICT_LINE_THERMAL_UB][i, j] <= self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s],
                        name=f'{ConstrName.LINE_THERMAL_C2}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] <= self.data[DataName.DICT_LINE_THERMAL_UB][i, j],
                        name=f'{ConstrName.LINE_THERMAL_C2}_R_{i}_{j}_{t}_{s}'
                    )

                    # c3: -2 S̄_ij <= sqrt(3)*p - q <= 2 S̄_ij
                    self.model.addConstr(
                        -2 * self.data[DataName.DICT_LINE_THERMAL_UB][i, j]
                        <= sqrt3 * self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] - self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s],
                        name=f'{ConstrName.LINE_THERMAL_C3}_L_{i}_{j}_{t}_{s}'
                    )
                    self.model.addConstr(
                        sqrt3 * self.var[VarName.LINE_ACTIVE_FLOW][i, j, t, s] - self.var[VarName.LINE_REACTIVE_FLOW][i, j, t, s]
                        <= 2 * self.data[DataName.DICT_LINE_THERMAL_UB][i, j],
                        name=f'{ConstrName.LINE_THERMAL_C3}_R_{i}_{j}_{t}_{s}'
                    )

    def derive_obj_terms(self):

        self.obj_term[ObjName.DG_FIXED_COST] = gp.quicksum(
            self.data[DataName.DICT_DG_COST_FIX][j] * self.var[VarName.DG_INSTALL][j]
            for j in self.data[DataName.LIST_NODE]
        )

        self.obj_term[ObjName.DG_VARIANT_COST] = gp.quicksum(
            self.data[DataName.DICT_DG_COST_VAR][j] * self.var[VarName.DG_RATED_POWER][j]
            for j in self.data[DataName.LIST_NODE]
        )

        self.obj_term[ObjName.DG_GENERATING_COST] = gp.quicksum(
            self.data[DataName.DICT_SC_PROB][s]
            * self.data[DataName.DICT_DG_COST_UNIT][j]
            * self.var[VarName.DG_ACTIVE_POWER][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

        self.obj_term[ObjName.LINE_HARDEN_COST] = gp.quicksum(
            self.data[DataName.DICT_LINE_COST_HARDEN][i, j] * self.var[VarName.LINE_HARDEN][i, j]
            for (i, j) in self.data[DataName.LIST_LINE]
        )

        self.obj_term[ObjName.LOAD_SHED_COST] = gp.quicksum(
            self.data[DataName.DICT_SC_PROB][s]
            * self.data[DataName.NUM_COST_SHED]
            * self.var[VarName.LOAD_SHED_RATIO][j, t, s]
            for j in self.data[DataName.LIST_NODE]
            for t in self.data[DataName.LIST_TIME]
            for s in self.data[DataName.LIST_SCENARIO]
        )

    def add_all_objective_terms(self):

        self.model.setObjective(
            gp.quicksum(
                self.obj_term[obj_term_name] for obj_term_name in sorted(self.obj_term.keys())
            ),
            GRB.MINIMIZE
        )

    def add_objective_by_terms(self, term_list=None):

        self.model.setObjective(
            gp.quicksum(
                self.obj_term[obj_term_name] for obj_term_name in term_list
            ),
            GRB.MINIMIZE
        )
