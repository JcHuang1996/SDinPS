# -*- coding: utf-8 -*-
# @Time     : 2026/01/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import sys
import os
import json

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from unit_test.test_scripts.combined_formulation_test import run_combined_formulation_test


def _to_json_serializable(obj):
    """Convert nested dict/values to JSON-serializable form (e.g. numpy scalars -> float/int, tuple keys -> str)."""
    if isinstance(obj, dict):
        return {_json_key(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _json_key(k):
    """Convert dict key to JSON-allowed type (str, int, float, bool, None). Tuples become strings."""
    if isinstance(k, tuple):
        return str(k)
    if isinstance(k, (str, int, float, bool)) or k is None:
        return k
    return str(k)


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

    m_result = m_combined.result

    # Get objective term values
    obj_term_value = m_combined.obj_term_value.copy()

    # Prepare the new row data
    # Convert scenario_list to string representation for storage
    sce_str = str(scenario_list)

    # Output result as JSON under dataset_path/solutions/, named by sce_str
    solutions_dir = os.path.join(dataset_path, 'solutions')
    os.makedirs(solutions_dir, exist_ok=True)
    # Sanitize sce_str to a valid filename (e.g. "['s_1', 's_2']" -> "s_1_s_2.json")
    filename_safe = sce_str.replace("'", "").replace("[", "").replace("]", "").replace(", ", "_").replace(" ", "_").strip() + ".json"
    result_json_path = os.path.join(solutions_dir, filename_safe)
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(_to_json_serializable(m_result), f, indent=2)
    print(f'Saved result JSON to: {result_json_path}')
    
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
    dataset_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test_fixed_rated_p'
    s_list_list = [
        ['s_1'],
        ['s_2'],
        ['s_3'],
        ['s_4'],
        ['s_5'],
        ['s_6'],
        ['s_1', 's_2', 's_3'],
        ['s_1', 's_3', 's_5'],
        ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
    ]
    for s_list in s_list_list:
        record_solution(dataset_path=dataset_path, scenario_list=s_list)
