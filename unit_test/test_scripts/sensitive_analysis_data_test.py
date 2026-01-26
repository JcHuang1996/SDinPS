# -*- coding: utf-8 -*-
# @Time     : 2026/01/26
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import sys
import os

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from datetime import datetime
import pandas as pd
from util.names import VarName
from util.topo_visual import net_topo_preview_with_results
from unit_test.test_scripts.combined_formulation_test import run_combined_formulation_test


def run_sensitive_analysis_test(
    test_dataset_path: str,
    scenario_list: list = None
):
    """Run sensitive analysis test across multiple datasets.
    
    Args:
        test_dataset_path: Path to folder containing subfolders to test 
                          (e.g., 'unit_test/test_local_csv_file/sensitive_load/10_1_0.1')
        scenario_list: List of scenario names to process. Passed as scenario_list_assigned 
                      to run_combined_formulation_test. Default: ['s_1', 's_2', 's_3'].
    """
    if scenario_list is None:
        scenario_list = ['s_1', 's_2', 's_3']
    
    # Get absolute path
    test_dataset_path = os.path.abspath(test_dataset_path)
    
    # Extract dataset name (basename of the path)
    dataset_name = os.path.basename(test_dataset_path)
    
    # Get parent directory for test_file_path
    test_file_path = os.path.dirname(test_dataset_path)
    
    # Generate timestamp and output folder
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', f'{timestamp}_{dataset_name}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plot subfolder
    plot_dir = os.path.join(output_dir, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get all subdirectories in test_dataset_path
    subfolders = []
    for item in os.listdir(test_dataset_path):
        item_path = os.path.join(test_dataset_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    
    # Sort subfolders for consistent ordering
    subfolders.sort()
    
    print(f'Found {len(subfolders)} subfolders to test: {subfolders}')
    
    # Storage for results
    dg_results = {}  # {subfolder_name: {node_id: value}}
    line_results = {}  # {subfolder_name: {line_id: value}}
    obj_results = {}  # {subfolder_name: {obj_term_name: value}}
    all_nodes = set()
    all_lines = set()
    
    # Iterate through each subfolder
    for subfolder_name in subfolders:
        print(f'\nProcessing subfolder: {subfolder_name}')
        
        # Construct data_set_name
        data_set_name = os.path.join(dataset_name, subfolder_name)
        
        try:
            # Run the combined formulation test
            m_combined = run_combined_formulation_test(
                test_file_path=test_file_path,
                data_set_name=data_set_name,
                scenario_list_assigned=scenario_list,
                output_path=None,  # We'll handle plotting separately
                generate_plot=False  # Skip plotting here, we'll do it with custom paths
            )
            
            # Extract results
            node_result = m_combined.result[VarName.DG_INSTALL]
            line_result = m_combined.result[VarName.LINE_HARDEN]
            
            # Store results
            dg_results[subfolder_name] = node_result.copy()
            line_results[subfolder_name] = line_result.copy()
            obj_results[subfolder_name] = m_combined.obj_term_value.copy()
            
            # Collect all nodes and lines
            all_nodes.update(node_result.keys())
            all_lines.update(line_result.keys())
            
            # Generate plot for this dataset
            folder_path = os.path.join(test_file_path, data_set_name)
            plot_save_path = os.path.join(plot_dir, f'{subfolder_name}_topology_result.png')
            
            net_topo_preview_with_results(
                folder_path=folder_path,
                node_result=node_result,
                line_result=line_result,
                save_path=plot_save_path
            )
            
            print(f'Completed processing {subfolder_name}')
            
        except Exception as e:
            print(f'Error processing {subfolder_name}: {e}')
            continue
    
    # Generate DG_result.csv
    print('\nGenerating DG_result.csv...')
    dg_data = {'node_id': sorted(all_nodes)}
    for subfolder_name in subfolders:
        if subfolder_name in dg_results:
            dg_data[subfolder_name] = [
                dg_results[subfolder_name].get(node_id, 0) 
                for node_id in sorted(all_nodes)
            ]
        else:
            dg_data[subfolder_name] = [0] * len(all_nodes)
    
    dg_df = pd.DataFrame(dg_data)
    dg_csv_path = os.path.join(output_dir, 'DG_result.csv')
    dg_df.to_csv(dg_csv_path, index=False)
    print(f'Saved DG_result.csv to {dg_csv_path}')
    
    # Generate Line_harden_result.csv
    print('Generating Line_harden_result.csv...')
    # Convert line tuples to string format: "node_1-node_2"
    # Sort lines as tuples first, then convert to strings
    sorted_line_tuples = sorted(all_lines)
    line_id_list = [f'{from_node}-{to_node}' for (from_node, to_node) in sorted_line_tuples]
    line_data = {'line_id': line_id_list}
    
    for subfolder_name in subfolders:
        if subfolder_name in line_results:
            line_data[subfolder_name] = [
                line_results[subfolder_name].get(line_tuple, 0)
                for line_tuple in sorted_line_tuples
            ]
        else:
            line_data[subfolder_name] = [0] * len(line_id_list)
    
    line_df = pd.DataFrame(line_data)
    line_csv_path = os.path.join(output_dir, 'Line_harden_result.csv')
    line_df.to_csv(line_csv_path, index=False)
    print(f'Saved Line_harden_result.csv to {line_csv_path}')
    
    # Generate objective_details.csv
    print('Generating objective_details.csv...')
    obj_term_names = []
    if obj_results:
        all_terms = set()
        for subfolder_name in subfolders:
            if subfolder_name in obj_results:
                all_terms.update(obj_results[subfolder_name].keys())
        obj_term_names = sorted(all_terms)
    
    obj_data = {'obj_term_name': obj_term_names}
    for subfolder_name in subfolders:
        if subfolder_name in obj_results:
            obj_data[subfolder_name] = [
                obj_results[subfolder_name].get(term_name, 0.0)
                for term_name in obj_term_names
            ]
        else:
            obj_data[subfolder_name] = [0.0] * len(obj_term_names)
    
    obj_df = pd.DataFrame(obj_data)
    # Add Total row: sum of all objective terms per column
    total_row = ['Total'] + [
        sum(obj_results[sf].get(term_name, 0.0) for term_name in obj_term_names)
        if sf in obj_results else 0.0
        for sf in subfolders
    ]
    obj_df.loc[len(obj_df)] = total_row
    
    obj_csv_path = os.path.join(output_dir, 'objective_details.csv')
    obj_df.to_csv(obj_csv_path, index=False)
    print(f'Saved objective_details.csv to {obj_csv_path}')
    
    print(f'\nSensitive analysis test completed. Results saved to: {output_dir}')


if __name__ == "__main__":
    # Example usage
    test_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/sensitive_load/10_1_0.1'
    run_sensitive_analysis_test(test_dataset_path=test_path, scenario_list=['s_1'])
