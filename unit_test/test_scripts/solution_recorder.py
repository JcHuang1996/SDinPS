# -*- coding: utf-8 -*-
# @Time     : 2026/01/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import sys
import os

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pandas as pd
from unit_test.test_scripts.combined_formulation_test import run_combined_formulation_test


def record_solution(
    dataset_path: str,
    scenario_list: list = None
):
    """Record solution for a dataset by running combined formulation test.
    
    Args:
        dataset_path: Path to the dataset folder (e.g., 'unit_test/test_local_csv_file/function_test')
        scenario_list: List of scenario names to process. Passed as scenario_list_assigned 
                      to run_combined_formulation_test. Default: ['s_1', 's_2', 's_3'].
    """
    if scenario_list is None:
        scenario_list = ['s_1', 's_2', 's_3']
    
    # Get absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # Extract test_file_path (parent directory) and data_set_name (basename)
    test_file_path = os.path.dirname(dataset_path)
    data_set_name = os.path.basename(dataset_path)
    
    # Run the combined formulation test
    print(f'Running combined formulation test for dataset: {data_set_name}')
    print(f'Scenario list: {scenario_list}')
    
    m_combined = run_combined_formulation_test(
        test_file_path=test_file_path,
        data_set_name=data_set_name,
        scenario_list_assigned=scenario_list,
        output_path=None,
        generate_plot=False  # Skip plotting for solution recording
    )
    
    # Get objective term values
    obj_term_value = m_combined.obj_term_value.copy()
    
    # Prepare the new row data
    # Convert scenario_list to string representation for storage
    sce_str = str(scenario_list)
    
    # Create row data: 'sce' column + objective term columns
    new_row = {'sce': sce_str}
    for term_name in sorted(obj_term_value.keys()):
        new_row[term_name] = obj_term_value[term_name]
    
    # Calculate total objective value
    total_obj = sum(obj_term_value.values())
    new_row['Total'] = total_obj
    
    # Check if solution reference file exists
    solution_ref_path = os.path.join(dataset_path, 'solution reference.csv')
    
    if os.path.exists(solution_ref_path):
        # Read existing file
        df = pd.read_csv(solution_ref_path)
        
        # Ensure all columns from new_row exist in df (add as 0.0 for numeric columns)
        for col in new_row.keys():
            if col not in df.columns:
                if col == 'sce':
                    df[col] = ''
                else:
                    df[col] = 0.0
        
        # Ensure all columns from df exist in new_row (fill with 0.0 for missing numeric terms, '' for sce)
        for col in df.columns:
            if col not in new_row:
                if col == 'sce':
                    new_row[col] = ''
                else:
                    new_row[col] = 0.0
        
        # Append new row
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        print('Appended new row to existing solution reference file')
    else:
        # Create new file with the row
        df = pd.DataFrame([new_row])
        print('Created new solution reference file')
    
    # Save the CSV file
    df.to_csv(solution_ref_path, index=False)
    print(f'Saved solution reference to: {solution_ref_path}')
    
    return df


if __name__ == "__main__":
    # Example usage
    dataset_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/IEEE123bus'
    scenario_list = ['s_6']
    # scenario_list = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6']
    # scenario_list = ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10']
    for s in scenario_list:
        s_list = [s]
        record_solution(dataset_path=dataset_path, scenario_list=s_list)
