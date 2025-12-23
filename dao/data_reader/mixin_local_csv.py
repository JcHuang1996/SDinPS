# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from __future__ import annotations
from typing import Optional, Any
import logging
import os
import ast
import pandas as pd

from util.names import InputDataName
from util.headers import *

logger = logging.getLogger(__name__)


class LocalCSVMixin:

    local_file_path: Optional[str]
    data_set_name: Optional[str]
    raw_data: Any

    def read_local_csv(self):
        if not self.local_file_path:
            raise ValueError("local_file_path is required for 'local_csv'")

        if not self.data_set_name:
            raise ValueError("data_set_name is required for 'local_csv'")

        read_path = self.local_file_path + '/' + self.data_set_name + '/'
        logger.info(f"Reading local csv file from {read_path}")
        logger.info(f"Data set name: {self.data_set_name}")

        def auto_cast(val):
            """
            Convert string representations in parameter.csv to proper Python types.
            Examples:
                '3.5'  -> 3.5
                '10'   -> 10
                '[1, 2, 3]' -> [1, 2, 3]
                "{'a': 1}"  -> {'a': 1}
                'text' -> 'text' (unchanged)
            """
            if isinstance(val, str):
                val = val.strip()
                # try literal_eval (handles lists, dicts, tuples, numbers, booleans)
                try:
                    return ast.literal_eval(val)
                except Exception:
                    # fall back to numeric conversion if possible
                    try:
                        return float(val) if '.' in val else int(val)
                    except Exception:
                        return val  # keep as string if neither works
            return val

        try:
            df_parameter = pd.read_csv(read_path + 'parameter.csv', usecols=[0, 1])
            df_parameter[ParameterHeader.VALUE] = df_parameter[ParameterHeader.VALUE].apply(auto_cast)

            parameter_dict = dict(zip(df_parameter.iloc[:, 0], df_parameter.iloc[:, 1]))
            self.raw_data[InputDataName.PARAMETERS_DICT] = parameter_dict
        except FileNotFoundError:
            logger.error("Error: Cannot find parameter.csv")
            raise
        
        # read the branch data from data folder
        try:
            df_branch = pd.read_csv(read_path + 'branch.csv', usecols=[BranchHeader.FROM_NODE, BranchHeader.TO_NODE, BranchHeader.R, BranchHeader.X, BranchHeader.INIT_STATE, BranchHeader.POLE_NUMBER])
            self.raw_data[InputDataName.BRANCH_DF] = df_branch
        except FileNotFoundError:
            logger.error("Error: Cannot find branch.csv")
            raise
        
        # read the node data from data folder
        try:
            df_node = pd.read_csv(read_path + 'node.csv')
            self.raw_data[InputDataName.NODE_DF] = df_node
        except FileNotFoundError:
            logger.error("Error: Cannot find node.csv")
            raise
        
        # read the power curve (power demand variation in 24h) from data folder
        try:
            df_power_curve = pd.read_csv(read_path + 'power_curve.csv')
            self.raw_data[InputDataName.POWER_CURVE_DF] = df_power_curve
        except FileNotFoundError:
            logger.error("Error: Cannot find power_curve.csv")
            raise
        
        # read power of each node in each scenario
        try:
            df_scenario_node_load = pd.read_csv(read_path + 's_load_P3715kW_system.csv')
            self.raw_data[InputDataName.SCENARIO_NODE_LOAD_DF] = df_scenario_node_load
        except FileNotFoundError:
            logger.error("Error: Cannot find s_load.csv")
            raise
        
        # read scenario probability
        try:
            # choose one of following:
            # scenario_df = pd.read_csv(read_path + 'scenario_reduced_probability.csv')
            scenario_df = pd.read_csv(read_path + 's_probability.csv')
            self.raw_data[InputDataName.SCENARIO_PROB_DF] = scenario_df
        except FileNotFoundError:
            logger.error("Error: Cannot find s_probability.csv")
            raise
        
        # read scenario (line status in each scenario, under different hardening methods) - for line hardening plan, line status can change in one scenario
        try:
            df_scenario = pd.read_csv(read_path + 's_line_state_w_o_harden.csv')
            self.raw_data[InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF] = df_scenario
        except FileNotFoundError:
            logger.error("Error: Cannot find s_line_state_w_o_harden.csv")
            raise

        logger.info('raw data reading finished')
        
        return self.raw_data
