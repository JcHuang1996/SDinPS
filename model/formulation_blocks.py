# -*- coding: utf-8 -*-
# @Time     : 2026/06/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import pyomo.environ as pyo

from util.names import ConstrName, DataName, MainProblemModelTypeName, ObjName, VarName


@dataclass(frozen=True)
class FormulationIndexAdapter:
    axes: Tuple[Tuple, ...]
    line_connected_var: str
    load_shed_var: str
    bus_voltage_var: str
    dg_active_power_var: str
    dg_reactive_power_var: str
    line_active_flow_var: str
    line_reactive_flow_var: str
    virtual_line_flow_var: str
    virtual_source_indicator_var: str
    virtual_inject_power_var: str
    line_healthy_nh_data: str
    line_healthy_h_data: str
    demand_active_data: str
    demand_reactive_data: str
    line_connected_constr: str
    virtual_flow_c1_constr: str
    virtual_flow_c2_constr: str
    no_virtual_on_open_constr: str
    radiality_constr: str
    dg_active_power_ub_constr: str
    dg_reactive_power_ub_constr: str
    voltage_range_c1_constr: str
    voltage_range_c2_constr: str
    voltage_slack_bus_constr: str
    voltage_flow_rel_constr: str
    flow_balance_c1_constr: str
    flow_balance_c2_constr: str
    no_flow_on_open_c1_constr: str
    no_flow_on_open_c2_constr: str
    line_thermal_c1_constr: str
    line_thermal_c2_constr: str
    line_thermal_c3_constr: str
    uses_scenario_probability: bool = False
    scenario_axis_pos: Optional[int] = None
    # Which probability table the recourse objective is weighted by. The master's collapsed 'hat'
    # data is built from the conditional DICT_SC_PROB; a decomposition subproblem must instead use
    # the unconditional DICT_SC_PROB_UNCOND so that its objective is sum_{s in G} P_s * Q_s and the
    # matching cut can be written on sum_{s in G} P_s * theta_s.
    scenario_probability_data: str = DataName.DICT_SC_PROB

    def node_key(self, node, axis):
        return node if not axis else (node, *axis)

    def line_key(self, line, axis):
        return line if not axis else (*line, *axis)

    def data_key(self, base_key, axis):
        return base_key if not axis else (*_as_tuple(base_key), *axis)

    def probability_weight(self, data, axis):
        if not self.uses_scenario_probability:
            return 1.0
        return data[self.scenario_probability_data][axis[self.scenario_axis_pos]]


def make_full_formulation_adapter(data, scenario_probability_data=DataName.DICT_SC_PROB):
    axes = tuple(
        (t, s)
        for t in data[DataName.LIST_TIME]
        for s in data[DataName.LIST_SCENARIO]
    )
    return FormulationIndexAdapter(
        axes=axes,
        line_connected_var=VarName.LINE_CONNECTED,
        load_shed_var=VarName.LOAD_SHED_RATIO,
        bus_voltage_var=VarName.BUS_VOLTAGE,
        dg_active_power_var=VarName.DG_ACTIVE_POWER,
        dg_reactive_power_var=VarName.DG_REACTIVE_POWER,
        line_active_flow_var=VarName.LINE_ACTIVE_FLOW,
        line_reactive_flow_var=VarName.LINE_REACTIVE_FLOW,
        virtual_line_flow_var=VarName.VIRTUAL_LINE_FLOW,
        virtual_source_indicator_var=VarName.VIRTUAL_SOURCE_INDICATOR,
        virtual_inject_power_var=VarName.VIRTUAL_INJECT_POWER,
        line_healthy_nh_data=DataName.DICT_LINE_HEALTHY_NH,
        line_healthy_h_data=DataName.DICT_LINE_HEALTHY_H,
        demand_active_data=DataName.DICT_DEMAND_ACTIVE,
        demand_reactive_data=DataName.DICT_DEMAND_REACTIVE,
        line_connected_constr=ConstrName.LINE_CONNECTED,
        virtual_flow_c1_constr=ConstrName.VIRTUAL_FLOW_BAL_C1,
        virtual_flow_c2_constr=ConstrName.VIRTUAL_FLOW_BAL_C2,
        no_virtual_on_open_constr=ConstrName.NO_VIRTUAL_ON_OPEN,
        radiality_constr=ConstrName.RADIALITY,
        dg_active_power_ub_constr=ConstrName.DG_ACTIVE_POWER_UB,
        dg_reactive_power_ub_constr=ConstrName.DG_REACTIVE_POWER_UB,
        voltage_range_c1_constr=ConstrName.VOLTAGE_RANGE_C1,
        voltage_range_c2_constr=ConstrName.VOLTAGE_RANGE_C2,
        voltage_slack_bus_constr=ConstrName.VOLTAGE_SLACK_BUS,
        voltage_flow_rel_constr=ConstrName.VOLTAGE_FLOW_REL,
        flow_balance_c1_constr=ConstrName.FLOW_BALANCE_C1,
        flow_balance_c2_constr=ConstrName.FLOW_BALANCE_C2,
        no_flow_on_open_c1_constr=ConstrName.NO_FLOW_ON_OPEN_C1,
        no_flow_on_open_c2_constr=ConstrName.NO_FLOW_ON_OPEN_C2,
        line_thermal_c1_constr=ConstrName.LINE_THERMAL_C1,
        line_thermal_c2_constr=ConstrName.LINE_THERMAL_C2,
        line_thermal_c3_constr=ConstrName.LINE_THERMAL_C3,
        uses_scenario_probability=True,
        scenario_axis_pos=1,
        scenario_probability_data=scenario_probability_data,
    )


def make_sub_formulation_adapter(data):
    """
    Adapter for a decomposition subproblem: the full (t, s) formulation weighted by the
    UNCONDITIONAL scenario probabilities.

    Convention (docs/development_record.md, P3): the master objective keeps sum_s P_s * theta_s, the
    subproblem for group G minimizes sum_{s in G} P_s * Q_s(x), and the corresponding cut is written
    on the weighted left-hand side sum_{s in G} P_s * theta_s. For singleton groups this is the old
    cut multiplied by P_s, i.e. numerically identical; for non-singleton groups with unequal P_s it
    is the correction for risk R3.4.
    """
    return make_full_formulation_adapter(
        data,
        scenario_probability_data=DataName.DICT_SC_PROB_UNCOND,
    )


def make_main_formulation_adapter(data, main_problem_model_type):
    if main_problem_model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
        axes = ((),)
        line_healthy_nh_data = DataName.DICT_HAT_LINE_HEALTHY_NH
        line_healthy_h_data = DataName.DICT_HAT_LINE_HEALTHY_H
        demand_active_data = DataName.DICT_HAT_DEMAND_ACTIVE
        demand_reactive_data = DataName.DICT_HAT_DEMAND_REACTIVE
        uses_scenario_probability = False
        scenario_axis_pos = None
    elif main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
        axes = tuple((t,) for t in data[DataName.LIST_TIME])
        line_healthy_nh_data = DataName.DICT_HAT_LINE_HEALTHY_NH_BY_TIME
        line_healthy_h_data = DataName.DICT_HAT_LINE_HEALTHY_H_BY_TIME
        demand_active_data = DataName.DICT_HAT_DEMAND_ACTIVE_BY_TIME
        demand_reactive_data = DataName.DICT_HAT_DEMAND_REACTIVE_BY_TIME
        uses_scenario_probability = False
        scenario_axis_pos = None
    elif main_problem_model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
        axes = tuple((s,) for s in data[DataName.LIST_SCENARIO])
        line_healthy_nh_data = DataName.DICT_HAT_LINE_HEALTHY_NH_BY_SCENARIO
        line_healthy_h_data = DataName.DICT_HAT_LINE_HEALTHY_H_BY_SCENARIO
        demand_active_data = DataName.DICT_HAT_DEMAND_ACTIVE_BY_SCENARIO
        demand_reactive_data = DataName.DICT_HAT_DEMAND_REACTIVE_BY_SCENARIO
        uses_scenario_probability = True
        scenario_axis_pos = 0
    else:
        raise ValueError(f'Unsupported main_problem_model_type: {main_problem_model_type}')

    return FormulationIndexAdapter(
        axes=axes,
        line_connected_var=VarName.MAIN_LINE_CONNECTED,
        load_shed_var=VarName.MAIN_LOAD_SHED_RATIO,
        bus_voltage_var=VarName.MAIN_BUS_VOLTAGE,
        dg_active_power_var=VarName.MAIN_DG_ACTIVE_POWER,
        dg_reactive_power_var=VarName.MAIN_DG_REACTIVE_POWER,
        line_active_flow_var=VarName.MAIN_LINE_ACTIVE_FLOW,
        line_reactive_flow_var=VarName.MAIN_LINE_REACTIVE_FLOW,
        virtual_line_flow_var=VarName.MAIN_VIRTUAL_LINE_FLOW,
        virtual_source_indicator_var=VarName.MAIN_VIRTUAL_SOURCE_INDICATOR,
        virtual_inject_power_var=VarName.MAIN_VIRTUAL_INJECT_POWER,
        line_healthy_nh_data=line_healthy_nh_data,
        line_healthy_h_data=line_healthy_h_data,
        demand_active_data=demand_active_data,
        demand_reactive_data=demand_reactive_data,
        line_connected_constr=ConstrName.MAIN_LINE_CONNECTED,
        virtual_flow_c1_constr=ConstrName.MAIN_VIRTUAL_FLOW_BAL_C1,
        virtual_flow_c2_constr=ConstrName.MAIN_VIRTUAL_FLOW_BAL_C2,
        no_virtual_on_open_constr=ConstrName.MAIN_NO_VIRTUAL_ON_OPEN,
        radiality_constr=ConstrName.MAIN_RADIALITY,
        dg_active_power_ub_constr=ConstrName.MAIN_DG_ACTIVE_POWER_UB,
        dg_reactive_power_ub_constr=ConstrName.MAIN_DG_REACTIVE_POWER_UB,
        voltage_range_c1_constr=ConstrName.MAIN_VOLTAGE_RANGE_C1,
        voltage_range_c2_constr=ConstrName.MAIN_VOLTAGE_RANGE_C2,
        voltage_slack_bus_constr=ConstrName.MAIN_VOLTAGE_SLACK_BUS,
        voltage_flow_rel_constr=ConstrName.MAIN_VOLTAGE_FLOW_REL,
        flow_balance_c1_constr=ConstrName.MAIN_FLOW_BALANCE_C1,
        flow_balance_c2_constr=ConstrName.MAIN_FLOW_BALANCE_C2,
        no_flow_on_open_c1_constr=ConstrName.MAIN_NO_FLOW_ON_OPEN_C1,
        no_flow_on_open_c2_constr=ConstrName.MAIN_NO_FLOW_ON_OPEN_C2,
        line_thermal_c1_constr=ConstrName.MAIN_LINE_THERMAL_C1,
        line_thermal_c2_constr=ConstrName.MAIN_LINE_THERMAL_C2,
        line_thermal_c3_constr=ConstrName.MAIN_LINE_THERMAL_C3,
        uses_scenario_probability=uses_scenario_probability,
        scenario_axis_pos=scenario_axis_pos,
    )


def add_dg_design_binary_vars(model):
    model.var[VarName.DG_INSTALL] = {
        j: model.add_var(
            domain=pyo.Binary,
            name=_var_label(VarName.DG_INSTALL, j),
        )
        for j in model.data[DataName.LIST_NODE]
    }

    model.var[VarName.DG_INSTALL_TYPE] = {
        (j, typ): model.add_var(
            domain=pyo.Binary,
            name=_var_label(VarName.DG_INSTALL_TYPE, (j, typ)),
        )
        for j in model.data[DataName.LIST_NODE]
        for typ in model.data[DataName.LIST_DG_TYPE]
    }


def add_dg_rated_power_vars(model):
    model.var[VarName.DG_RATED_POWER] = {
        j: model.add_var(
            domain=pyo.Reals,
            lb=0,
            name=_var_label(VarName.DG_RATED_POWER, j),
        )
        for j in model.data[DataName.LIST_NODE]
    }


def add_line_design_vars(model):
    model.var[VarName.LINE_HARDEN] = {
        (i, j): model.add_var(
            domain=pyo.Binary,
            name=_var_label(VarName.LINE_HARDEN, (i, j)),
        )
        for (i, j) in model.data[DataName.LIST_LINE]
    }


def add_line_connectivity_vars(model, adapter):
    model.var[adapter.line_connected_var] = {
        adapter.line_key((i, j), axis): model.add_var(
            domain=pyo.Binary,
            name=_var_label(adapter.line_connected_var, adapter.line_key((i, j), axis)),
        )
        for (i, j) in model.data[DataName.LIST_LINE]
        for axis in adapter.axes
    }


def add_topology_vars(model, adapter):
    model.var[adapter.virtual_line_flow_var] = {
        adapter.line_key((i, j), axis): model.add_var(
            domain=pyo.Reals,
            name=_var_label(adapter.virtual_line_flow_var, adapter.line_key((i, j), axis)),
        )
        for (i, j) in model.data[DataName.LIST_LINE]
        for axis in adapter.axes
    }

    model.var[adapter.virtual_source_indicator_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Binary,
            name=_var_label(adapter.virtual_source_indicator_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }

    model.var[adapter.virtual_inject_power_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Reals,
            lb=0,
            name=_var_label(adapter.virtual_inject_power_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }


def add_operating_vars(model, adapter):
    model.var[adapter.bus_voltage_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Reals,
            lb=0,
            name=_var_label(adapter.bus_voltage_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }

    # lb = 0: a DG cannot absorb active power. Leaving p^G free made negative injection a
    # cost-reducing direction (unit cost C^Gu > 0) bounded only by flow balance and the thermal
    # limits -- risk R3.6 in docs/development_record.md.
    model.var[adapter.dg_active_power_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Reals,
            lb=0,
            name=_var_label(adapter.dg_active_power_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }

    model.var[adapter.dg_reactive_power_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Reals,
            name=_var_label(adapter.dg_reactive_power_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }

    model.var[adapter.line_active_flow_var] = {
        adapter.line_key((i, j), axis): model.add_var(
            domain=pyo.Reals,
            name=_var_label(adapter.line_active_flow_var, adapter.line_key((i, j), axis)),
        )
        for (i, j) in model.data[DataName.LIST_LINE]
        for axis in adapter.axes
    }

    model.var[adapter.line_reactive_flow_var] = {
        adapter.line_key((i, j), axis): model.add_var(
            domain=pyo.Reals,
            name=_var_label(adapter.line_reactive_flow_var, adapter.line_key((i, j), axis)),
        )
        for (i, j) in model.data[DataName.LIST_LINE]
        for axis in adapter.axes
    }

    model.var[adapter.load_shed_var] = {
        adapter.node_key(j, axis): model.add_var(
            domain=pyo.Reals,
            lb=0,
            ub=1,
            name=_var_label(adapter.load_shed_var, adapter.node_key(j, axis)),
        )
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    }


def add_dg_install_constraints(model):
    model.add_constr(
        pyo.quicksum(
            model.var[VarName.DG_INSTALL][j] for j in model.data[DataName.LIST_NODE]
        ) <= model.data[DataName.NUM_DG_UB],
        name=ConstrName.DG_UPPERBOUND,
    )

    for j in model.data[DataName.LIST_NODE]:
        model.add_constr(
            pyo.quicksum(
                model.var[VarName.DG_INSTALL_TYPE][j, typ]
                for typ in model.data[DataName.LIST_DG_TYPE]
            ) == model.var[VarName.DG_INSTALL][j],
            name=_constr_name(ConstrName.DG_INSTALL_TYPE_ONE, j),
        )


def add_dg_rated_power_link_constraints(model, constr_name):
    for j in model.data[DataName.LIST_NODE]:
        model.add_constr(
            model.var[VarName.DG_RATED_POWER][j]
            == pyo.quicksum(
                model.data[DataName.DICT_DG_RATED_POWER][typ]
                * model.var[VarName.DG_INSTALL_TYPE][j, typ]
                for typ in model.data[DataName.LIST_DG_TYPE]
            ),
            name=_constr_name(constr_name, j),
        )


def add_line_connectivity_constraints(model, adapter):
    for (i, j) in model.data[DataName.LIST_LINE]:
        for axis in adapter.axes:
            line_key = adapter.line_key((i, j), axis)
            model.add_constr(
                model.var[adapter.line_connected_var][line_key]
                <= model.data[adapter.line_healthy_nh_data][line_key] * (1 - model.var[VarName.LINE_HARDEN][i, j])
                + model.data[adapter.line_healthy_h_data][line_key] * model.var[VarName.LINE_HARDEN][i, j],
                name=_constr_name(adapter.line_connected_constr, i, j, *axis),
            )


def add_topology_constraints(model, adapter):
    virtual_injection_m = len(model.data[DataName.LIST_NODE])

    for j in model.data[DataName.LIST_NODE]:
        for axis in adapter.axes:
            model.add_constr(
                pyo.quicksum(
                    model.var[adapter.virtual_line_flow_var][adapter.line_key((j, k), axis)]
                    for k in model.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    model.var[adapter.virtual_line_flow_var][adapter.line_key((i, j), axis)]
                    for i in model.data[DataName.DICT_NODE_PARENTS][j]
                )
                == model.var[adapter.virtual_inject_power_var][adapter.node_key(j, axis)] - 1,
                name=_constr_name(adapter.virtual_flow_c1_constr, j, *axis),
            )

            model.add_constr(
                model.var[adapter.virtual_inject_power_var][adapter.node_key(j, axis)]
                <= virtual_injection_m * model.var[adapter.virtual_source_indicator_var][adapter.node_key(j, axis)],
                name=_constr_name(adapter.virtual_flow_c2_constr, j, *axis),
            )

    for (i, j) in model.data[DataName.LIST_LINE]:
        for axis in adapter.axes:
            line_key = adapter.line_key((i, j), axis)
            model.add_constr(
                -virtual_injection_m * model.var[adapter.line_connected_var][line_key]
                <= model.var[adapter.virtual_line_flow_var][line_key],
                name=_constr_name(adapter.no_virtual_on_open_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                model.var[adapter.virtual_line_flow_var][line_key]
                <= virtual_injection_m * model.var[adapter.line_connected_var][line_key],
                name=_constr_name(adapter.no_virtual_on_open_constr, 'R', i, j, *axis),
            )

    for axis in adapter.axes:
        model.add_constr(
            pyo.quicksum(
                model.var[adapter.line_connected_var][adapter.line_key((i, j), axis)]
                for (i, j) in model.data[DataName.LIST_LINE]
            ) == len(model.data[DataName.LIST_NODE]) - pyo.quicksum(
                model.var[adapter.virtual_source_indicator_var][adapter.node_key(j, axis)]
                for j in model.data[DataName.LIST_NODE]
            ),
            name=_constr_name(adapter.radiality_constr, *axis),
        )


def add_dg_operating_constraints(model, adapter):
    for j in model.data[DataName.LIST_NODE]:
        for axis in adapter.axes:
            node_key = adapter.node_key(j, axis)
            model.add_constr(
                model.var[adapter.dg_active_power_var][node_key] <= model.var[VarName.DG_RATED_POWER][j],
                name=_constr_name(adapter.dg_active_power_ub_constr, j, *axis),
            )
            model.add_constr(
                model.var[adapter.dg_reactive_power_var][node_key]
                <= model.data[DataName.DICT_DG_ALPHA_UB][j] * model.var[adapter.dg_active_power_var][node_key],
                name=_constr_name(adapter.dg_reactive_power_ub_constr, j, *axis),
            )


def add_operating_constraints(model, adapter):
    for j in model.data[DataName.LIST_NODE]:
        for axis in adapter.axes:
            node_key = adapter.node_key(j, axis)
            model.add_constr(
                model.data[DataName.NUM_VOLTAGE_LB] <= model.var[adapter.bus_voltage_var][node_key],
                name=_constr_name(adapter.voltage_range_c1_constr, j, *axis),
            )
            model.add_constr(
                model.var[adapter.bus_voltage_var][node_key] <= model.data[DataName.NUM_VOLTAGE_UB],
                name=_constr_name(adapter.voltage_range_c2_constr, j, *axis),
            )

    slack_node_idx = model.data[DataName.SLACK_NODE_IDX]
    for axis in adapter.axes:
        model.add_constr(
            model.var[adapter.bus_voltage_var][adapter.node_key(slack_node_idx, axis)]
            == model.data[DataName.NUM_VOLTAGE_SLACK],
            name=_constr_name(adapter.voltage_slack_bus_constr, *axis),
        )

    # An open line has zero flow, so only the voltage range must be relaxed.
    voltage_relaxation = model.data[DataName.NUM_VOLTAGE_UB] - model.data[DataName.NUM_VOLTAGE_LB]
    for (i, j) in model.data[DataName.LIST_LINE]:
        for axis in adapter.axes:
            line_key = adapter.line_key((i, j), axis)
            v_drop_on_ij = (
                model.data[DataName.DICT_LINE_RESISTANCE][i, j] * model.var[adapter.line_active_flow_var][line_key]
                + model.data[DataName.DICT_LINE_REACTANCE][i, j] * model.var[adapter.line_reactive_flow_var][line_key]
            )
            voltage_drop_expr = (
                model.var[adapter.bus_voltage_var][adapter.node_key(i, axis)]
                - model.var[adapter.bus_voltage_var][adapter.node_key(j, axis)]
                - v_drop_on_ij / model.data[DataName.NUM_VOLTAGE_SLACK]
            )

            model.add_constr(
                -voltage_relaxation * (1 - model.var[adapter.line_connected_var][line_key])
                <= voltage_drop_expr,
                name=_constr_name(adapter.voltage_flow_rel_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                voltage_drop_expr
                <= voltage_relaxation * (1 - model.var[adapter.line_connected_var][line_key]),
                name=_constr_name(adapter.voltage_flow_rel_constr, 'R', i, j, *axis),
            )

    for j in model.data[DataName.LIST_NODE]:
        for axis in adapter.axes:
            node_key = adapter.node_key(j, axis)
            model.add_constr(
                pyo.quicksum(
                    model.var[adapter.line_active_flow_var][adapter.line_key((j, k), axis)]
                    for k in model.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    model.var[adapter.line_active_flow_var][adapter.line_key((i, j), axis)]
                    for i in model.data[DataName.DICT_NODE_PARENTS][j]
                )
                == model.var[adapter.dg_active_power_var][node_key]
                - (1 - model.var[adapter.load_shed_var][node_key])
                * model.data[adapter.demand_active_data][node_key],
                name=_constr_name(adapter.flow_balance_c1_constr, j, *axis),
            )

            model.add_constr(
                pyo.quicksum(
                    model.var[adapter.line_reactive_flow_var][adapter.line_key((j, k), axis)]
                    for k in model.data[DataName.DICT_NODE_CHILDREN][j]
                )
                - pyo.quicksum(
                    model.var[adapter.line_reactive_flow_var][adapter.line_key((i, j), axis)]
                    for i in model.data[DataName.DICT_NODE_PARENTS][j]
                )
                == model.var[adapter.dg_reactive_power_var][node_key]
                - (1 - model.var[adapter.load_shed_var][node_key])
                * model.data[adapter.demand_reactive_data][node_key],
                name=_constr_name(adapter.flow_balance_c2_constr, j, *axis),
            )

    for (i, j) in model.data[DataName.LIST_LINE]:
        for axis in adapter.axes:
            line_key = adapter.line_key((i, j), axis)
            line_connected = model.var[adapter.line_connected_var][line_key]
            # These component bounds follow from the existing thermal polygon.
            capacity = model.data[DataName.DICT_LINE_THERMAL_UB][i, j]
            model.add_constr(
                -capacity * line_connected <= model.var[adapter.line_active_flow_var][line_key],
                name=_constr_name(adapter.no_flow_on_open_c1_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                model.var[adapter.line_active_flow_var][line_key] <= capacity * line_connected,
                name=_constr_name(adapter.no_flow_on_open_c1_constr, 'R', i, j, *axis),
            )
            model.add_constr(
                -2 * capacity * line_connected <= model.var[adapter.line_reactive_flow_var][line_key],
                name=_constr_name(adapter.no_flow_on_open_c2_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                model.var[adapter.line_reactive_flow_var][line_key] <= 2 * capacity * line_connected,
                name=_constr_name(adapter.no_flow_on_open_c2_constr, 'R', i, j, *axis),
            )

    sqrt3 = math.sqrt(3.0)
    for (i, j) in model.data[DataName.LIST_LINE]:
        thermal_ub = model.data[DataName.DICT_LINE_THERMAL_UB][i, j]
        for axis in adapter.axes:
            line_key = adapter.line_key((i, j), axis)
            p_flow = model.var[adapter.line_active_flow_var][line_key]
            q_flow = model.var[adapter.line_reactive_flow_var][line_key]

            model.add_constr(
                -2 * thermal_ub <= sqrt3 * p_flow + q_flow,
                name=_constr_name(adapter.line_thermal_c1_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                sqrt3 * p_flow + q_flow <= 2 * thermal_ub,
                name=_constr_name(adapter.line_thermal_c1_constr, 'R', i, j, *axis),
            )
            model.add_constr(
                -thermal_ub <= p_flow,
                name=_constr_name(adapter.line_thermal_c2_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                p_flow <= thermal_ub,
                name=_constr_name(adapter.line_thermal_c2_constr, 'R', i, j, *axis),
            )
            model.add_constr(
                -2 * thermal_ub <= sqrt3 * p_flow - q_flow,
                name=_constr_name(adapter.line_thermal_c3_constr, 'L', i, j, *axis),
            )
            model.add_constr(
                sqrt3 * p_flow - q_flow <= 2 * thermal_ub,
                name=_constr_name(adapter.line_thermal_c3_constr, 'R', i, j, *axis),
            )


def add_design_objective_terms(model):
    model.obj_term[ObjName.DG_FIXED_COST] = pyo.quicksum(
        model.data[DataName.DICT_DG_COST_FIX][j] * model.var[VarName.DG_INSTALL][j]
        for j in model.data[DataName.LIST_NODE]
    )

    model.obj_term[ObjName.DG_VARIANT_COST] = pyo.quicksum(
        (
            model.data[DataName.DICT_DG_RATED_POWER][typ] * model.data[DataName.DICT_DG_UNIT_PRICE][typ]
            + model.data[DataName.DICT_DG_EXTRA_ADJUSTMENT][typ]
        )
        * model.var[VarName.DG_INSTALL_TYPE][j, typ]
        for j in model.data[DataName.LIST_NODE]
        for typ in model.data[DataName.LIST_DG_TYPE]
    )

    model.obj_term[ObjName.LINE_HARDEN_COST] = pyo.quicksum(
        model.data[DataName.DICT_LINE_COST_HARDEN][i, j] * model.var[VarName.LINE_HARDEN][i, j]
        for (i, j) in model.data[DataName.LIST_LINE]
    )


def add_operating_objective_terms(model, adapter, dg_obj_name, load_shed_obj_name):
    model.obj_term[dg_obj_name] = pyo.quicksum(
        adapter.probability_weight(model.data, axis)
        * model.data[DataName.DICT_DG_COST_UNIT][j]
        * model.var[adapter.dg_active_power_var][adapter.node_key(j, axis)]
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    )

    model.obj_term[load_shed_obj_name] = pyo.quicksum(
        adapter.probability_weight(model.data, axis)
        * model.data[DataName.NUM_COST_SHED]
        * model.var[adapter.load_shed_var][adapter.node_key(j, axis)]
        for j in model.data[DataName.LIST_NODE]
        for axis in adapter.axes
    )


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def _var_label(var_name, key):
    return f'{var_name}_({",".join(str(part) for part in _as_tuple(key))})'


def _constr_name(prefix, *parts):
    if not parts:
        return prefix
    return f'{prefix}_{"_".join(str(part) for part in parts)}'
