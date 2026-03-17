# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


# ===========================================
# All headers of input tables are stored here
# Decoupling the headers used in the program from headers of the original input
# Preventing errors caused by typo
# Helping checking headers' meanings
# ===========================================

class NodeHeader:
    NODE_ID = 'node_id'
    P_LOAD = 'Pload_MW'
    Q_LOAD = 'Qload_MW'
    FIXED_C = 'fixed_cost'


class BranchHeader:
    FROM_NODE = 'from_bus'
    TO_NODE = 'to_bus'
    R = 'R'
    X = 'X'
    INIT_STATE = 'init_state'
    POLE_NUMBER = 'pole_number'


class ScenarioProbHeader:
    SCENARIO_ID = 'scenario_id'
    SCENARIO_PROB = 'probability'


class ScenarioLineStateHeader:
    SCENARIO_ID = 'scenario_id'
    TIME_IDX = 'time_idx'
    FROM_NODE = 'from_bus'
    TO_NODE = 'to_bus'
    STATE_NO_HARDEN = 'state_no_harden'
    STATE_HARDEN = 'state_harden'


class ParameterHeader:
    PARAM = 'parameter'
    VALUE = 'value'


class ParameterKey:
    DG_NUM_LIM = 'dg_num_lim'
    DG_CV = 'dg_kv'             # cost per MW, when DG_RATE_POWER is variable
    DG_CF = 'dg_kf'             # fixed cost, when DG_RATE_POWER is a fixed value

    POLE_HARDEN_COST = 'pole_harden_cost'

    C_DG_OPERATING = 'c_dg_operating'
    C_LOAD_SHED = 'c_load_shed'

    V_0 = 'v_0'
    V_MIN = 'v_min'
    V_MAX = 'v_max'
    S_MAX = 's_max'
    DG_PF_LB = 'dg_pf_lb'

    SLACK_NODE_IDX = 'slack_node'   # str, the index of the slack node of the system


class ScenarioNodePowerHeader:
    SCENARIO_ID = 'scenario_id'
    NODE = 'node'
    P_LOAD_MW = 'P_LOAD_MW'
    Q_LOAD_MW = 'Q_LOAD_MW'


class NodeCoordinateHeader:
    NODE_ID = 'node_id'
    X = 'x'
    Y = 'y'


class DgRatedPowerHeader:
    DG_TYPE = 'DG_type'
    RATED_POWER = 'rated_power'
    UNIT_PRICE = 'unit_price'
    EXTRA_ADJUSTMENT = 'extra_adjustment'
