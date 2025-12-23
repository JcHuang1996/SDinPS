# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

# ============================================================
# Save all names for important vars used in the code here
# To change any name in the code, change the name here
# To check the detail of any vars, check it here
# Preventing errors caused by typo
# Helping checking the variables/datas' meanings
# ============================================================

# ============================================================
# Notations
# AFN: short for 'Algo File Notations', the notation of corresponding content in the algo file.
# ============================================================

class ModelStatus:
    SOLVED_OPT = 'opt'                                  # model is solved and an optimal solution (including solution within required gap) is found
    SOLVED_TIMEOUT_WS = 'timeout_with_s'                # time out, and a feasible solution is found
    SOLVED_TIMEOUT_WOS = 'timeout_without_s'            # time out, and no feasible solution is found
    UNSOLVED = 'unsolved'                               # model is not solved yet
    INFEASIBLE = 'infeasible'                           # model is infeasible. IIS can be solved.
    INF_OR_UNBD = 'inf_or_unbd'                         # model is infeasible or unbounded. No iis guaranteed
    UNKNOWN = 'unknown'                                 # prefix of an unknown model status


class InputMethodName:
    LOCAL_CSV = 'local_csv'                             # read input data from local csv file


class InputDataName:
    PARAMETERS_DICT = 'parameters_dict'                 # dict, all parameters of the model
    NODE_DF = 'node_df'                                 # dataframe, node data
    BRANCH_DF = 'branch_df'                             # dataframe, branch data
    POWER_CURVE_DF = 'power_curve_df'                   # dataframe, power curve
    SCENARIO_NODE_LOAD_DF = 'scenario_node_load_df'     # dataframe, power of each node with normal distribution fluctuation
    SCENARIO_PROB_DF = 'scenario_prob_dict'             # dataframe, prob of all scenario

    SCENARIO_FAULT_LINE_DF = 'scenario_fault_line_df'   # detaframe, fault line scenario

    SCENARIO_LINE_STATE_WO_HARDEN_DF = 'scenario_line_state_wo_harden_df'   # dataframe, line state at each time idx in each scenario with/without line harden
    SCENARIO_TIME_IDX_HOUR_DF = 'scenario_time_idx_hour_df'     # dataframe, (time idx - nature hour) correspondence in each scenario


class DataName:
    # === Set ===
    LIST_SCENARIO = 'list_scenario'                     # list, [sc_1, sc_2, ...], the set of all scenarios. AFN: S
    LIST_NODE = 'list_node'                             # list, [node_1, node_2, ...], the set of all nodes. AFN: J
    LIST_LINE = 'list_line'                             # list, [(i, j), ...], the set of all lines. AFN: L
    LIST_TIME = 'list_time'                             # list, [t1, t2, ...], the set of all time slots. AFN: T

    # === Parameters / Data ===
    NUM_DG_UB = 'dg_ub'                                 # integer, upperbound on number of generators. AFN: \overline{N^{G}} key: dg_num_lim
    DICT_SC_PROB = 'dict_scenario_prob'                 # dict, {sc_1: float, ...}, probability of each scenario. AFN: P_s
    DICT_DG_COST_FIX = 'dict_dg_cost_fix'               # dict, {j: float, ...}, fixed cost of installing DG at node j. AFN: C^{Gf}_j
    DICT_DG_COST_VAR = 'dict_dg_cost_var'               # dict, {j: float, ...}, variable cost of installing DG at node j. AFN: C^{Gv}_j
    DICT_DG_COST_UNIT = 'dict_dg_cost_unit'             # dict, {j: float, ...}, unit cost of generating power at node j. AFN: C^{Gu}_j
    DICT_DG_ALPHA_UB = 'dict_dg_alpha_ub'               # dict, {j: float, ...}, power factor upperbound of DG at node j. AFN: \Alpha^{G}_j
    DICT_LINE_COST_HARDEN = 'dict_line_cost_harden'     # dict, {(i,j): float, ...}, fixed cost of hardening line (i,j). AFN: C^{H}_{ij}
    DICT_LINE_RESISTANCE = 'dict_line_resistance'       # dict, {(i,j): float, ...}, resistance of line (i,j). AFN: R_{ij}
    DICT_LINE_REACTANCE = 'dict_line_reactance'         # dict, {(i,j): float, ...}, reactance of line (i,j). AFN: X_{ij}
    DICT_LINE_HEALTHY_NH = 'dict_line_healthy_n_harden'    # dict, {(i,j,t,s): status_binary, ...}, healthy status of line (i,j) at time t in scenario s if it was not hardened. AFN: H^{nh}_{ijts} column: state_no_harden
    DICT_LINE_HEALTHY_H = 'dict_line_healthy_harden'    # dict, {(i,j,t,s): status_binary, ...}, healthy status of line (i,j) at time t in scenario s if it was hardened. AFN: H^{h}_{ijts} column: state_harden
    NUM_VOLTAGE_LB = 'voltage_lb'                       # float, lower bound of bus voltage. AFN: \underline{V} key: v_min
    NUM_VOLTAGE_UB = 'voltage_ub'                       # float, upper bound of bus voltage. AFN: \overline{V} key: v_max
    NUM_VOLTAGE_SLACK = 'voltage_slack'                 # float, slack bus voltage. AFN: V_0 key: v_0
    DICT_LINE_THERMAL_UB = 'dict_line_thermal_ub'       # dict, {(i,j): float, ...}, thermal limit of line (i,j). AFN: \overline{S_{ij}}
    DICT_NODE_CHILDREN = 'dict_node_children'           # dict, {j: [child_nodes1, ...], ...}, child neighbors of node j. AFN: \Theta_j
    DICT_NODE_PARENTS = 'dict_node_parents'             # dict, {j: [parent_nodes1, ...], ...}, parent neighbors of node j. AFN: \Pi_j
    #
    # # --- Quote note from algo file ---
    # # Note: just using one 'all neighbors of node j' is sufficient, as it is formulated as an undirected graph.
    # # We divide into 'child' and 'parent' only for coding convenience.
    # # Given nodes i and j, implementing (i, j) = (j, i) is more complex,
    # # while only defining (i, j) and ensuring i ∈ Θ_j or Π_j is easier.
    #
    DICT_DEMAND_ACTIVE = 'dict_demand_active'           # dict, {(j,t,s): float, ...}, active power demand at node j at time t in scenario s. AFN: D^{p}_{jts}
    DICT_DEMAND_REACTIVE = 'dict_demand_reactive'       # dict, {(j,t,s): float, ...}, reactive power demand at node j at time t in scenario s. AFN: D^{q}_{jts}
    NUM_COST_SHED = 'cost_shed'                         # float, cost of load shedding. AFN: C^{L} key: c_load_shed
    #
    # # Important values and coefficient
    NUM_TOTAL_POWER = 'total_power'                     # num, the total power demand in the system
    SLACK_NODE_IDX = 'slack_node_idx'                   # num/str, the slack node's index in list of nodes
    NUM_RATED_POWER_UB = 'rated_power_ub'               # num, the potential upperbound for generator's rated power.


class VarName:
    # === Main Variables ===
    SUB_OBJ_EST = 'eta'               # Continuous, η_s value of the subproblem objective in scenario s. AFN: \eta_{s}
    DG_INSTALL = 'xg'                   # Binary, xg_j indicating whether DG is installed at node j. AFN: x^{G}_{j}
    LINE_HARDEN = 'xl'                  # Binary, xl_ij indicating whether line (i,j) is hardened. AFN: x^{L}_{ij}
    DG_RATED_POWER = 'pgrt'             # Continuous, pgrt_j the rated power capacity of DG at node j. AFN: p^{Grt}_{j}

    # === Sub Variables ===
    LINE_CONNECTED = 'yc'               # Binary, yc_ijts indicating whether line (i,j) is connected at time t in scenario s. AFN: y^{C}_{ijts}
    LOAD_SHED_RATIO = 'yl'              # Continuous, yl_jts percentage of load shed at node j at time t in scenario s. AFN: y^{L}_{jts}
    BUS_VOLTAGE = 'v'                   # Continuous, v_jts bus voltage magnitude at node j at time t in scenario s. AFN: v_{jts}
    DG_ACTIVE_POWER = 'pg'              # Continuous, pg_jts active power generation of DG at node j at time t in scenario s. AFN: p^{G}_{jts}
    DG_REACTIVE_POWER = 'qg'            # Continuous, qg_jts reactive power generation of DG at node j at time t in scenario s. AFN: q^{G}_{jts}
    LINE_ACTIVE_FLOW = 'p'              # Continuous, p_ijts active power flow on line (i,j) at time t in scenario s. AFN: p_{ijts}
    LINE_REACTIVE_FLOW = 'q'            # Continuous, q_ijts reactive power flow on line (i,j) at time t in scenario s. AFN: q_{ijts}

    # --- Virtual / Topology-related variables ---
    VIRTUAL_LINE_FLOW = 'ul'            # Continuous, ul_ijts virtual flow on line (i,j) at time t in scenario s. AFN: u^{L}_{ijts}
    VIRTUAL_SOURCE_INDICATOR = 'us'     # Binary, us_jts indicating whether node j is virtual source at time t in scenario s. AFN: u^{S}_{jts}
    VIRTUAL_INJECT_POWER = 'ui'         # Continuous, ui_jts virtual power injected from virtual source at node j at time t in scenario s. AFN: u^{I}_{jts}


class ConstrName:
    # === Main Problem Model Constraints ===
    DG_UPPERBOUND = 'dg_ub'                             # AFN: Generator number upperbound
    DG_OPERATION = 'dg_op'                              # AFN: Generator operation
    OPTIMAL_CUT = 'opt_cut'                             # AFN: Optimal Cut (generated by solving sub-problems)

    # === Sub-problem Model Constraints ===
    # Line connectivity
    LINE_CONNECTED = 'line_conn'                        # AFN: Line connected condition

    # System topology constraints
    VIRTUAL_FLOW_BAL_C1 = 'virt_flow_c1'                # AFN: Virtual power flow balance (eq. 1)
    VIRTUAL_FLOW_BAL_C2 = 'virt_flow_c2'                # AFN: Virtual power flow balance (eq. 2)

    VIRTUAL_INJECTION_UB = 'virt_inj_ub'                # AFN: Virtual power injection upperbound
    NO_VIRTUAL_ON_OPEN = 'virt_no_open'                 # AFN: No virtual flow on open line
    RADIALITY = 'radiality'                             # AFN: Radiality

    # DG operating constraints
    DG_ACTIVE_POWER_UB = 'dg_p'                         # AFN: Amount of generated active power
    DG_REACTIVE_POWER_UB = 'dg_q'                       # AFN: Amount of generated reactive power

    # System operating constraints
    VOLTAGE_RANGE_C1 = 'volt_rng_c1'                    # AFN: Voltage range (lower bound)
    VOLTAGE_RANGE_C2 = 'volt_rng_c2'                    # AFN: Voltage range (upper bound)
    VOLTAGE_SLACK_BUS = 'slack_bus'                     # AFN: Slack bus voltage equality

    VOLTAGE_FLOW_REL = 'volt_flow'                      # AFN: Voltage - flow relationships

    FLOW_BALANCE_C1 = 'flow_bal_p'                      # AFN: Flow balance (active power)
    FLOW_BALANCE_C2 = 'flow_bal_q'                      # AFN: Flow balance (reactive power)

    NO_FLOW_ON_OPEN_C1 = 'no_flow_open_p'               # AFN: No flow on open line (active)
    NO_FLOW_ON_OPEN_C2 = 'no_flow_open_q'               # AFN: No flow on open line (reactive)

    LINE_THERMAL_C1 = 'thermal_c1'                      # AFN: Linearized thermal limitation (part 1)
    LINE_THERMAL_C2 = 'thermal_c2'                      # AFN: Linearized thermal limitation (part 2)
    LINE_THERMAL_C3 = 'thermal_c3'                      # AFN: Linearized thermal limitation (part 3)


class ObjName:
    DG_FIXED_COST = 'dg_fixed_cost'
    DG_VARIANT_COST = 'dg_variant_cost'
    DG_GENERATING_COST = 'dg_generating_cost'
    LINE_HARDEN_COST = 'line_hard_cost'
    LOAD_SHED_COST = 'load_shed_cost'

    SUB_OBJ_TERM = 'sub_obj_term'