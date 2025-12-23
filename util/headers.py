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


class ScenarioFaultLineHeader:
    SCENARIO_ID = 'scenario_id'
    FROM_NODE = 'from_bus'
    TO_NODE = 'to_bus'

class ScenarioLineStateHeader:
    SCENARIO_ID = 'scenario_id'
    TIME_IDX = 'time_idx'
    NATURE_HOUR = 'nature_hour'
    FROM_NODE = 'from_bus'
    TO_NODE = 'to_bus'
    STATE_NO_HARDEN = 'state_no_harden'
    STATE_HARDEN = 'state_harden'
    BROKEN_POLE_NUM_NO_HARDEN = 'broken_pole_num_no_harden'
    BROKEN_POLE_NUM_HARDEN = 'broken_pole_num_harden'


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

class ModelForHeader:
    DG_ONLY = 'DG'
    DG_BES = 'DG_BES'
    DG_POLE = 'DG_POLE'
    DG_BES_POLE = 'DG_BES_POLE'

class PowerCurveHeader:
    TIME = 'Time'
    JAN = 'Jan'
    APR = 'Apr'
    JULY = 'July'
    OCT = 'Oct'

class ScenarioNodePowerHeader:
    SCENARIO_ID = 'scenario_id'
    NODE = 'node'
    P_LOAD_MW = 'P_LOAD_MW'
    Q_LOAD_MW = 'Q_LOAD_MW'


class PresentValueCoefHeader:
    PVC_DG = 'pvc_dg'
    PVC_BES = 'pvc_bes'
    PVC_POLE = 'pvc_pole'

class PHPenaltyHeader:
    DG_LOC = 'dg_loc'
    BES_LOC = 'bes_loc'
    HARDEN_LINE = 'harden_line'
    DG_PRT = 'dg_prt'
    BES_PRT = 'bes_prt'
    BES_ERT = 'bes_ert'