# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com
from numpy.ma.core import arccos

from util.headers import *
from util.names import *
from util.project_logger import init_logger

import math

import logging


logger = logging.getLogger(__name__)


class DataProcessor:

    def __init__(self, raw_data):

        # the original input data to be processed
        self.raw_data = raw_data

        # the processed data, ready to be used in later model
        self.data = {}

    def clear_existing_data(self):

        # clear the processed data to generate a new data set
        self.data = {}

    def scenario_time_specific_data_process(self, scenario_list_assigned=None, time_list_assigned=None):

        logger.info(f'Processing data for specific scenarios and times')
        logger.info(f'Scenarios: {scenario_list_assigned}')
        logger.info(f'Times: {time_list_assigned}')

        self.scenario_data(scenario_list_assigned=scenario_list_assigned, time_list_assigned=time_list_assigned)

    def data_process(self, scenario_list_assigned=None, time_list_assigned=None):

        logger.info(f'Data processing starts')

        self.node_data()
        self.line_data()
        self.node_data()
        self.scenario_data()
        self.parameter_data()

        if scenario_list_assigned is not None or time_list_assigned is not None:

            logger.info(f'Processing data for specific scenarios and times')
            logger.info(f'Scenarios: {scenario_list_assigned}')
            logger.info(f'Times: {time_list_assigned}')
            self.scenario_data(scenario_list_assigned=scenario_list_assigned, time_list_assigned=time_list_assigned)

        else:
            logger.info('Processing data for all scenarios and times')
            self.scenario_data()

        self.logic_process_data()

        logger.info(f'Data processing completed.')

        return self.data


    def node_data(self):
        """
        This function processes general node data which is independent of the scenarios.
        The function gives its output by directly modify the self.data, rather than returning.
        :return: None
        """
        # ========================================
        # process node data from node.csv:
        # LIST_NODE: 'node_id' in node.csv
        # ========================================
        df_node = self.raw_data[InputDataName.NODE_DF].copy()

        # sort the nodes by the order of their idx, avoid potential random in modeling process
        self.data[DataName.LIST_NODE] = sorted(list(df_node[NodeHeader.NODE_ID]), key=lambda x: int(x.split('_')[1]))

        # ========================================
        # process node data from node.csv and parameter dict (dict by the parameter.csv):
        # DICT_DG_COST_FIX: key: node, value: 'dg_kf' in parameter dict
        # DICT_DG_COST_VAR: key: node, value: 'dg_kv' in parameter dict
        # DICT_DG_COST_UNIT: key: node, value: 'dict_dg_cost_unit' in parameter dict
        # DICT_DG_ALPHA_UB: key: node, value: tan(arccos('dict_dg_alpha_ub')) in parameter dict
        # ========================================
        dict_parameter = self.raw_data[InputDataName.PARAMETERS_DICT].copy()

        self.data[DataName.DICT_DG_COST_FIX] = {
            j: dict_parameter[ParameterKey.DG_CF] for j in self.data[DataName.LIST_NODE]
        }

        self.data[DataName.DICT_DG_COST_VAR] = {
            j: dict_parameter[ParameterKey.DG_CV] for j in self.data[DataName.LIST_NODE]
        }

        self.data[DataName.DICT_DG_COST_UNIT] = {
            j: dict_parameter[ParameterKey.C_DG_OPERATING] for j in self.data[DataName.LIST_NODE]
        }

        alpha_value = math.tan(math.acos(dict_parameter[ParameterKey.DG_PF_LB]))
        self.data[DataName.DICT_DG_ALPHA_UB] = {
            j: alpha_value for j in self.data[DataName.LIST_NODE]
        }


    def line_data(self):
        """
        This function processes general node data which is independent of the scenarios.
        The function writes the output directly into the self.data, rather than returning.
        Inputs come from self.raw_data[InputDataName.BRANCH_DF] and self.raw_data[InputDataName.PARAMETERS_DICT].
        """

        # ========================================
        # process node data from branch.csv:
        # LIST_LINE: all ('from_bus', 'to_bus')'s in branch.csv
        # DICT_LINE_RESISTANCE: key: line, value: 'R'
        # DICT_LINE_REACTANCE: key: line, value: 'X'
        # DICT_NODE_CHILDREN: key: 'from_bus', value: list of all corresponding 'to_bus'
        # DICT_NODE_PARENTS: key: 'to_bus', value: list of all corresponding 'from_bus'
        # ========================================
        df_branch = self.raw_data[InputDataName.BRANCH_DF].copy()

        self.data[DataName.LIST_LINE] = []
        self.data[DataName.DICT_LINE_RESISTANCE] = {}
        self.data[DataName.DICT_LINE_REACTANCE] = {}
        self.data[DataName.DICT_NODE_CHILDREN] = {j: [] for j in self.data[DataName.LIST_NODE]}
        self.data[DataName.DICT_NODE_PARENTS] = {j: [] for j in self.data[DataName.LIST_NODE]}

        for _, row in df_branch.iterrows():
            i = row[BranchHeader.FROM_NODE]
            j = row[BranchHeader.TO_NODE]

            self.data[DataName.LIST_LINE].append((i, j))
            self.data[DataName.DICT_LINE_RESISTANCE][(i, j)] = row[BranchHeader.R]
            self.data[DataName.DICT_LINE_REACTANCE][(i, j)] = row[BranchHeader.X]
            self.data[DataName.DICT_NODE_CHILDREN][i].append(j)
            self.data[DataName.DICT_NODE_PARENTS][j].append(i)

        # ========================================
        # process node data from branch.csv & parameter dict (dict by the parameter.csv):
        # DICT_LINE_COST_HARDEN: key: line, value: 'pole_number' in branch.csv * 'pole_harden_cost' in parameter dict
        # DICT_LINE_THERMAL_UB: key: line, value: 's_max' in parameter dict
        # ========================================
        dict_parameter = self.raw_data[InputDataName.PARAMETERS_DICT].copy()

        self.data[DataName.DICT_LINE_COST_HARDEN], self.data[DataName.DICT_LINE_THERMAL_UB] = {}, {}
        self.data[DataName.DICT_LINE_COST_HARDEN], self.data[DataName.DICT_LINE_THERMAL_UB] = {}, {}

        for _, row in df_branch.iterrows():
            i = row[BranchHeader.FROM_NODE]
            j = row[BranchHeader.TO_NODE]

            self.data[DataName.DICT_LINE_COST_HARDEN][(i, j)] = (
                row[BranchHeader.POLE_NUMBER] * dict_parameter[ParameterKey.POLE_HARDEN_COST]
            )
            self.data[DataName.DICT_LINE_THERMAL_UB][(i, j)] = dict_parameter[ParameterKey.S_MAX]

        # ========================================
        # sorting for deterministic order
        # ========================================
        self.data[DataName.LIST_LINE] = sorted(
            self.data[DataName.LIST_LINE],
            key=lambda tup: (int(str(tup[0]).split('_')[1]), int(str(tup[1]).split('_')[1]))
        )

        for node in self.data[DataName.DICT_NODE_CHILDREN]:
            self.data[DataName.DICT_NODE_CHILDREN][node] = sorted(
                self.data[DataName.DICT_NODE_CHILDREN][node],
                key=lambda x: int(str(x).split('_')[1])
            )
        for node in self.data[DataName.DICT_NODE_PARENTS]:
            self.data[DataName.DICT_NODE_PARENTS][node] = sorted(
                self.data[DataName.DICT_NODE_PARENTS][node],
                key=lambda x: int(str(x).split('_')[1])
            )


    def parameter_data(self):
        """
        This function processes specific parameters, which are often independent of other .csv files.
        The function writes the output directly into the self.data, rather than returning.
        Inputs come from self.raw_data[InputDataName.PARAMETERS_DICT].
        """
        # ========================================
        # process parameters from parameter dict (dict by the parameter.csv):
        # NUM_DG_UB: the maximum number of DGs in the system, key: dg_num_lim
        # NUM_VOLTAGE_LB: lower bound of bus voltage. key: v_min
        # NUM_VOLTAGE_UB: upper bound of bus voltage. key: v_max
        # NUM_VOLTAGE_SLACK: slack bus voltage. key: v_0
        # SLACK_NODE_IDX: the index of the slack node. key: slack_node
        # NUM_COST_SHED: the cost of load shed. key: c_load_shed
        # ========================================
        dict_parameter = self.raw_data[InputDataName.PARAMETERS_DICT].copy()

        self.data[DataName.NUM_DG_UB] = dict_parameter[ParameterKey.DG_NUM_LIM]

        self.data[DataName.NUM_VOLTAGE_LB] = dict_parameter[ParameterKey.V_MIN]
        self.data[DataName.NUM_VOLTAGE_UB] = dict_parameter[ParameterKey.V_MAX]
        self.data[DataName.NUM_VOLTAGE_SLACK] = dict_parameter[ParameterKey.V_0]

        self.data[DataName.SLACK_NODE_IDX] = dict_parameter[ParameterKey.SLACK_NODE_IDX]

        self.data[DataName.NUM_COST_SHED] = dict_parameter[ParameterKey.C_LOAD_SHED]


    def scenario_data(self, scenario_list_assigned=None, time_list_assigned=None):
        """
        This function processes general scenario data.
        The function writes the output directly into the self.data, rather than returning.
        Inputs come from:
        (Optional) scenario_list_assigned (the list of scenarios should be considered)
        (Optional) time_list_assigned (the list of scenarios should be considered)
        self.raw_data[InputDataName.SCENARIO_PROB_DF]
        self.raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF]
        self.raw_data[InputDataName.SCENARIO_NODE_LOAD_DF]
        self.raw_data[InputDataName.PARAMETERS_DICT]
        """
        # ========================================
        # process data from s_probability.csv:
        # DICT_SC_PROB: the probability of the scenarios. key: 'scenario_id', value: 'probability'
        # Note: the s_probability will be filtered by the given list of scenarios that we should consider.
        # If the list is not given, then the default scenarios to be considered is all scenarios
        # ========================================

        if scenario_list_assigned is None:
            self.data[DataName.LIST_SCENARIO] = sorted(
                list(self.raw_data[InputDataName.SCENARIO_PROB_DF][ScenarioProbHeader.SCENARIO_ID])
            )
        else:
            self.data[DataName.LIST_SCENARIO] = scenario_list_assigned

        df_s_prob = self.raw_data[InputDataName.SCENARIO_PROB_DF][
            self.raw_data[InputDataName.SCENARIO_PROB_DF][ScenarioProbHeader.SCENARIO_ID].isin(self.data[DataName.LIST_SCENARIO])
        ].copy()

        # check: sum of probability of all scenarios != 1
        if 0.99 <= df_s_prob[ScenarioProbHeader.SCENARIO_PROB].sum() <= 1.01:
            pass
        else:   # if sum of probability != 1, unify it
            df_s_prob[ScenarioProbHeader.SCENARIO_PROB] = \
                    df_s_prob[ScenarioProbHeader.SCENARIO_PROB] / df_s_prob[ScenarioProbHeader.SCENARIO_PROB].sum()

            # logger.warning('Warning: Incorrect scenario probability, auto unified')

        self.data[DataName.DICT_SC_PROB] = \
            df_s_prob.set_index([ScenarioProbHeader.SCENARIO_ID])[ScenarioProbHeader.SCENARIO_PROB].to_dict()

        # ========================================
        # process data from s_line_state_w_o_harden.csv:
        # LIST_TIME: the list of all time slot indexes
        # DICT_LINE_HEALTHY_NH: healthy status of lines in scenarios if no harden. key: (from_bus, to_bus, time_idx, scenario_id), value: state_no_harden
        # Note: the data will be filtered by the given list of scenarios and list of time idx that we should consider.
        # If the lists are not given, then the default scenarios to be considered as above,
        # and the default time idx are all time idx given in the .csv.
        # ========================================
        if time_list_assigned is None:
            self.data[DataName.LIST_TIME] = sorted(
                self.raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF][ScenarioLineStateHeader.TIME_IDX].unique().tolist()
            )
        else:
            self.data[DataName.LIST_TIME] = time_list_assigned

        df_s_line_state = \
            self.raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF][
                self.raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF][ScenarioLineStateHeader.TIME_IDX].isin(self.data[DataName.LIST_TIME])
            ].copy()

        self.data[DataName.DICT_LINE_HEALTHY_H] = df_s_line_state.set_index(
            [ScenarioLineStateHeader.FROM_NODE, ScenarioLineStateHeader.TO_NODE,
             ScenarioLineStateHeader.TIME_IDX, ScenarioLineStateHeader.SCENARIO_ID]
        )[ScenarioLineStateHeader.STATE_HARDEN].to_dict()

        self.data[DataName.DICT_LINE_HEALTHY_NH] = df_s_line_state.set_index(
            [ScenarioLineStateHeader.FROM_NODE, ScenarioLineStateHeader.TO_NODE,
             ScenarioLineStateHeader.TIME_IDX, ScenarioLineStateHeader.SCENARIO_ID]
        )[ScenarioLineStateHeader.STATE_NO_HARDEN].to_dict()

        # ========================================
        # process data from node.csv & s_load_P3715kW_system.csv:
        # DICT_DEMAND_ACTIVE: active power demand at nodes at times in scenarios,
        # involving the benchmark load form node.csv, and the adjustment value from s_load_P3715kW_system.csv.
        # DICT_DEMAND_REACTIVE: reactive power demand at nodes at times in scenarios,
        # involving the benchmark load form node.csv, and the adjustment value from s_load_P3715kW_system.csv.
        # Note: the data will be filtered by the given list of scenarios that we should consider.
        # ========================================

        self.data[DataName.DICT_DEMAND_ACTIVE], self.data[DataName.DICT_DEMAND_REACTIVE] = {}, {}

        # df_node = self.raw_data[InputDataName.NODE_DF].copy()
        df_s_load = self.raw_data[InputDataName.SCENARIO_NODE_LOAD_DF][
            self.raw_data[InputDataName.SCENARIO_NODE_LOAD_DF][ScenarioNodePowerHeader.SCENARIO_ID].isin(self.data[DataName.LIST_SCENARIO])
        ].copy()

        # dict_node_benchmark_p_load = df_node.set_index([NodeHeader.NODE_ID])[NodeHeader.P_LOAD].to_dict()
        # dict_node_benchmark_q_load = df_node.set_index([NodeHeader.NODE_ID])[NodeHeader.Q_LOAD].to_dict()

        dict_node_s_p_load = df_s_load.set_index(
            [ScenarioNodePowerHeader.NODE, ScenarioNodePowerHeader.SCENARIO_ID]
        )[ScenarioNodePowerHeader.P_LOAD_MW].to_dict()
        dict_node_s_q_load = df_s_load.set_index(
            [ScenarioNodePowerHeader.NODE, ScenarioNodePowerHeader.SCENARIO_ID]
        )[ScenarioNodePowerHeader.Q_LOAD_MW].to_dict()

        for j in self.data[DataName.LIST_NODE]:
            # p_load_benchmark = dict_node_benchmark_p_load[j]
            # q_load_benchmark = dict_node_benchmark_q_load[j]
            for s in self.data[DataName.LIST_SCENARIO]:
                adj_p_load = dict_node_s_p_load[j, s]
                adj_q_load = dict_node_s_q_load[j, s]
                for t in self.data[DataName.LIST_TIME]:
                    # self.data[DataName.DICT_DEMAND_ACTIVE][j, t, s] = p_load_benchmark - adj_p_load
                    # self.data[DataName.DICT_DEMAND_REACTIVE][j, t, s] = q_load_benchmark - adj_q_load
                    self.data[DataName.DICT_DEMAND_ACTIVE][j, t, s] = adj_p_load
                    self.data[DataName.DICT_DEMAND_REACTIVE][j, t, s] = adj_q_load


    def logic_process_data(self):
        """
        This function processes parameters based on data in self.data that has been read in.
        For example, the big-M's for constraints.
        The function writes the output directly into the self.data, rather than returning.
        Inputs come from
        """

        # ====================
        # compute NUM_TOTAL_POWER, the big-M for Voltage - flow relationships constraints
        # value: the sum of all p-load of the system
        # ====================

        df_node = self.raw_data[InputDataName.NODE_DF].copy()
        # self.data[DataName.NUM_TOTAL_POWER] = df_node[NodeHeader.P_LOAD].sum()
        self.data[DataName.NUM_TOTAL_POWER] = 1.5

        # ====================
        # compute NUM_RATED_POWER_UB, the big-M for DG operation constraints
        # value: TBD
        # ====================
        self.data[DataName.NUM_RATED_POWER_UB] = 1.75


if __name__ == "__main__":
    init_logger()

    # Simple smoke test for this module
    print("Running DataReader smoke test...")

    test_read_method = InputMethodName.LOCAL_CSV
    test_file_path = '/Users/huangjiacheng/OR591/unit test/test_local_csv_file'
    data_set_name = 'function test'

    from dao.data_reader import DataReader

    r = DataReader(
        read_method=test_read_method,
        local_file_path=test_file_path,
        data_set_name=data_set_name
    )

    r.read()
    print("CSV read success:")

    DataProcessorModule = DataProcessor(r.raw_data)
    processed_data = DataProcessorModule.data_process()

    print('Data processing completed.')