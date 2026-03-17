# -*- coding: utf-8 -*-
# @Time     : 2025/10/04
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import sys
import os

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from util.headers import *
from util.names import *
from util.project_logger import init_logger
from util.topo_visual import net_topo_preview_with_results
from dao.data_reader import DataReader
from dao.data_processor import DataProcessor
from model.model_combined import ModelCombined

import math
import logging


logger = logging.getLogger(__name__)


def run_combined_formulation_test(
    test_file_path: str = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file',
    data_set_name: str = 'function_test_fixed_rated_p',
    scenario_list_assigned: list = None,
    time_list_assigned: list = None,
    output_path: str = None,
    generate_plot: bool = True
):
    """Run the combined formulation test.
    
    Args:
        test_file_path: Path to the test data folder.
        data_set_name: Name of the dataset folder.
        scenario_list_assigned: List of scenario names to process. Default: ['s_1', 's_2', 's_3'].
        time_list_assigned: List of time indices to process. Default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].
        output_path: Optional path to save the visualization. If provided, the directory will be created if it doesn't exist.
        generate_plot: Whether to generate and display the plot. Default: True.
    """
    if scenario_list_assigned is None:
        scenario_list_assigned = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6']
    if time_list_assigned is None:
        time_list_assigned = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    test_read_method = InputMethodName.LOCAL_CSV

    r = DataReader(
        read_method=test_read_method,
        local_file_path=test_file_path,
        data_set_name=data_set_name
    )

    r.read()
    print("CSV read success:")

    DataProcessorModule = DataProcessor(r.raw_data)
    processed_data = DataProcessorModule.data_process(
        scenario_list_assigned=scenario_list_assigned,
        time_list_assigned=time_list_assigned
    )

    print('Data processing completed.')

    m_combined = ModelCombined(model_name='m_combined_test', model_data=processed_data)
    m_combined.build_model_all_obj_terms()

    m_combined.set_parameters(
        param_dict={
            'MIPGap': 0.01
        }
    )

    m_combined.solve()
    m_combined.cal_detailed_obj()
    m_combined.get_result(
        [VarName.DG_INSTALL,
         VarName.LINE_HARDEN,
         VarName.DG_INSTALL_TYPE]
    )

    print('Solved')
    
    # Visualize results if requested
    if generate_plot:
        folder_path = os.path.join(test_file_path, data_set_name)
        node_result = m_combined.result[VarName.DG_INSTALL]
        line_result = m_combined.result[VarName.LINE_HARDEN]
        
        # Create output directory if output_path is provided
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            save_path = os.path.join(output_path, 'topology_result.png')
        else:
            save_path = None
        
        net_topo_preview_with_results(
            folder_path=folder_path,
            node_result=node_result,
            line_result=line_result,
            save_path=save_path
        )
    
    return m_combined


if __name__ == "__main__":
    run_combined_formulation_test()