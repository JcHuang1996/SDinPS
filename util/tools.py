# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import pandas as pd
import os
import random
import shutil
from typing import Any, Dict, List, Optional, Tuple

from util.names import VarName


def iter_general_csv(ite_obj_value_dict=None, output_dir=None):
    """Write or overwrite iter_general_info.csv from ite_obj_value_dict. Handles empty dict (header-only)."""
    general_records = []
    for ite_num in ite_obj_value_dict:
        incumbent = ite_obj_value_dict[ite_num].get('sub_obj(best_incumbent)', {}).get('sub_p_total', None)
        bound_data = ite_obj_value_dict[ite_num].get('main_obj(bound)', {})
        bound_total = bound_data.get('total', None)
        detail_dict = bound_data.get('detail', {})

        record = {
            'ite_num': ite_num,
            'best_incumbent_obj_value': incumbent,
            'best_bound_objective_value': bound_total
        }
        # merge cost components
        record.update(detail_dict)
        general_records.append(record)

    if not general_records:
        df_general = pd.DataFrame(columns=['ite_num', 'best_incumbent_obj_value', 'best_bound_objective_value'])
    else:
        df_general = pd.DataFrame(general_records)
    csv_general_path = os.path.join(output_dir, 'iter_general_info.csv')
    df_general.to_csv(csv_general_path, index=False)

def iter_sub_prob_info(ite_obj_value_dict=None, output_dir=None, scenario_list=None):
    """Write or overwrite iter_subproblem_info.csv from ite_obj_value_dict. Handles empty dict (header-only)."""
    sub_records = []
    for ite_num in ite_obj_value_dict:
        for s in scenario_list:
            sub_info = ite_obj_value_dict[ite_num].get(tuple([s]), {})
            sub_total = sub_info.get('total', None)
            sub_detail = sub_info.get('detail', {})
            record = {'ite_num': ite_num, 'scenario': s}
            record.update(sub_detail)
            record['sub_obj_total'] = sub_total
            sub_records.append(record)

    if not sub_records:
        df_sub = pd.DataFrame(columns=['ite_num', 'scenario', 'sub_obj_total'])
    else:
        df_sub = pd.DataFrame(sub_records)
    csv_sub_path = os.path.join(output_dir, 'iter_subproblem_info.csv')
    df_sub.to_csv(csv_sub_path, index=False)


def write_power_usage_capacity_ratio(
    best_sub_results: Optional[Dict[str, Any]],
    output_dir: str,
    detailed_csv_name: str = "power_usage_capacity_ratio_detailed.csv",
    averaged_csv_name: str = "power_usage_capacity_ratio_averaged.csv",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute and write power usage capacity ratio from best-iteration sub results.

    best_sub_results: dict with VarName.DG_ACTIVE_POWER -> {(j,t,s): value},
        VarName.DG_RATED_POWER -> {(j,s): value}. If None or empty, no files are written.
    output_dir: directory to write CSV files.
    detailed_csv_name / averaged_csv_name: output file names.

    Power usage capacity ratio = active_power / rated_power (per node, scenario, time).
    Averaged ratio = mean over timeslots for each (node, scenario).

    Returns (path_to_detailed, path_to_averaged), or (None, None) if no data.
    """
    if not best_sub_results:
        return None, None
    pg = best_sub_results.get(VarName.DG_ACTIVE_POWER, {})
    pgrt = best_sub_results.get(VarName.DG_RATED_POWER, {})
    if not pg or not pgrt:
        return None, None

    # Detailed: node, scenario, time, active_power, rated_power, ratio
    detailed_rows = []
    for (j, t, s), active_power in pg.items():
        rated_power = pgrt.get((j, s), None)
        if rated_power is None:
            continue
        ratio = (active_power / rated_power) if rated_power != 0 else float("nan")
        detailed_rows.append({
            "node": j,
            "scenario": s,
            "time": t,
            "active_power": active_power,
            "rated_power": rated_power,
            "ratio": ratio,
        })
    if not detailed_rows:
        return None, None
    df_detailed = pd.DataFrame(detailed_rows)
    os.makedirs(output_dir, exist_ok=True)
    path_detailed = os.path.join(output_dir, detailed_csv_name)
    df_detailed.to_csv(path_detailed, index=False)

    # Averaged: node, scenario, rated_power, averaged_ratio
    df = df_detailed.copy()
    df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")
    agg = df.groupby(["node", "scenario"], as_index=False).agg(
        rated_power=("rated_power", "first"),
        averaged_ratio=("ratio", "mean"),
    )
    path_averaged = os.path.join(output_dir, averaged_csv_name)
    agg.to_csv(path_averaged, index=False)

    return path_detailed, path_averaged


def write_decomp_run_parameters(
    output_dir: str,
    params: dict,
    file_name: str = "decomp_run_parameters.txt",
) -> str:
    """
    Write a text file in output_dir with one line per key-value pair in params.
    Returns the path of the written file.
    """
    path = os.path.join(output_dir, file_name)
    lines = [f"{k}: {v}" for k, v in params.items()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def load_warm_start(path):
    df = pd.read_csv(path)
    out = {}
    for i, row in df.iterrows():
        xg = {c: int(row[c]) for c in df.columns if c.startswith("node_") and "-" not in c}
        xl = {(a, b): int(row[f"{a}-{b}"])
              for c in df.columns if "-" in c
              for a, b in [c.split("-")]}
        out[f"w_{i+1}"] = {
            "xg": xg,
            # "xl": xl
        }
    return out


def data_perturbation(file_path, perturb_type, sim_num, node_num=1, load_percent=0.1, line_num=1):
    """
    Generate random perturbations on CSV datasets in the origin folder.
    
    Args:
        file_path: Path to the folder containing the 'origin' subfolder
        perturb_type: Type of perturbation ('load' or 'broken')
        sim_num: Number of perturbed datasets to generate
        node_num: Number of nodes to perturb per scenario for 'load' type (default: 1)
        load_percent: Percentage range for load perturbation for 'load' type (default: 0.1)
        line_num: Number of lines to perturb per scenario for 'broken' type (default: 1)
    
    Returns:
        None. Creates a folder containing subfolders p_1, p_2, ..., p_sim_num with modified datasets.
        For 'load' type: folder named 'sim_num_node_num_load_percent'
        For 'broken' type: folder named 'line_num'
    """
    origin_path = os.path.join(file_path, 'origin')
    
    # Get all files in origin folder to copy
    origin_files = [f for f in os.listdir(origin_path) if os.path.isfile(os.path.join(origin_path, f))]
    
    # Determine intermediate folder name based on perturbation type
    if perturb_type == 'load':
        intermediate_folder = os.path.join(file_path, f'{sim_num}_{node_num}_{load_percent}')
        # Read the original load CSV
        load_csv_name = 's_load_P3715kW_system.csv'
        load_csv_path = os.path.join(origin_path, load_csv_name)
        df_load = pd.read_csv(load_csv_path)
    elif perturb_type == 'broken':
        intermediate_folder = os.path.join(file_path, str(line_num))
        # Read the original line state CSV
        line_state_csv_name = 's_line_state_w_o_harden.csv'
        line_state_csv_path = os.path.join(origin_path, line_state_csv_name)
        df_line_state = pd.read_csv(line_state_csv_path)
    else:
        raise ValueError(f"Unsupported perturb_type: {perturb_type}")
    
    os.makedirs(intermediate_folder, exist_ok=True)
    
    # Generate sim_num perturbed datasets
    for sim_idx in range(1, sim_num + 1):
        # Create output folder inside the intermediate folder
        output_folder = os.path.join(intermediate_folder, f'p_{sim_idx}')
        os.makedirs(output_folder, exist_ok=True)
        
        # Copy all files from origin to output folder
        for file_name in origin_files:
            src_path = os.path.join(origin_path, file_name)
            dst_path = os.path.join(output_folder, file_name)
            shutil.copy2(src_path, dst_path)
        
        # Apply perturbation based on type
        if perturb_type == 'load':
            _apply_load_perturbation(df_load, output_folder, load_csv_name, node_num, load_percent)
        elif perturb_type == 'broken':
            _apply_broken_perturbation(df_line_state, output_folder, line_state_csv_name, line_num)


def _apply_load_perturbation(df_load, output_folder, load_csv_name, node_num, load_percent):
    """Apply load perturbation to the dataset."""
    # Create a copy of the load dataframe for modification
    df_perturbed = df_load.copy()
    
    # Group by scenario_id
    for scenario_id, group in df_perturbed.groupby('scenario_id'):
        # Get all nodes for this scenario, excluding node_1
        available_nodes = group[group['node'] != 'node_1']['node'].unique().tolist()
        
        # Randomly select node_num nodes (or all available if fewer)
        num_to_select = min(node_num, len(available_nodes))
        selected_nodes = random.sample(available_nodes, num_to_select)
        
        # Apply perturbation to selected nodes
        for node in selected_nodes:
            # Generate random multiplier between (1 - load_percent) and (1 + load_percent)
            multiplier = random.uniform(1 - load_percent, 1 + load_percent)
            
            # Apply multiplier to both P and Q load for this node in this scenario
            mask = (df_perturbed['scenario_id'] == scenario_id) & (df_perturbed['node'] == node)
            df_perturbed.loc[mask, 'P_LOAD_MW'] *= multiplier
            df_perturbed.loc[mask, 'Q_LOAD_MW'] *= multiplier
    
    # Save the perturbed load CSV
    perturbed_load_path = os.path.join(output_folder, load_csv_name)
    df_perturbed.to_csv(perturbed_load_path, index=False)


def _apply_broken_perturbation(df_line_state, output_folder, line_state_csv_name, line_num):
    """Apply broken line perturbation to the dataset."""
    # Create a copy of the line state dataframe for modification
    df_perturbed = df_line_state.copy()
    
    # Group by scenario_id
    for scenario_id, group in df_perturbed.groupby('scenario_id'):
        # Get unique lines (from_bus, to_bus pairs) for this scenario as tuples
        unique_lines = [tuple(row) for row in group[['from_bus', 'to_bus']].drop_duplicates().values]
        
        # Randomly select line_num lines (or all available if fewer)
        num_to_select = min(line_num, len(unique_lines))
        selected_lines = random.sample(unique_lines, num_to_select)
        
        # Apply perturbation to selected lines
        for selected_line in selected_lines:
            from_bus, to_bus = selected_line
            
            # Find adjacent lines (lines that share a node with the selected line)
            adjacent_lines = []
            for line in unique_lines:
                other_from, other_to = line
                # Check if lines share a node (adjacent) and are different
                if (line != selected_line and 
                    (from_bus == other_from or from_bus == other_to or 
                     to_bus == other_from or to_bus == other_to)):
                    adjacent_lines.append(line)
            
            # If no adjacent lines found, skip this line
            if not adjacent_lines:
                continue
            
            # Randomly pick one adjacent line
            adjacent_line = random.choice(adjacent_lines)
            adj_from, adj_to = adjacent_line
            
            # Exchange state_no_harden and state_harden between the two lines
            # Get all rows for both lines in this scenario
            mask_line1 = ((df_perturbed['scenario_id'] == scenario_id) & 
                         (df_perturbed['from_bus'] == from_bus) & 
                         (df_perturbed['to_bus'] == to_bus))
            mask_line2 = ((df_perturbed['scenario_id'] == scenario_id) & 
                         (df_perturbed['from_bus'] == adj_from) & 
                         (df_perturbed['to_bus'] == adj_to))
            
            # Exchange the states
            temp_no_harden = df_perturbed.loc[mask_line1, 'state_no_harden'].copy()
            temp_harden = df_perturbed.loc[mask_line1, 'state_harden'].copy()
            
            df_perturbed.loc[mask_line1, 'state_no_harden'] = df_perturbed.loc[mask_line2, 'state_no_harden'].values
            df_perturbed.loc[mask_line1, 'state_harden'] = df_perturbed.loc[mask_line2, 'state_harden'].values
            
            df_perturbed.loc[mask_line2, 'state_no_harden'] = temp_no_harden.values
            df_perturbed.loc[mask_line2, 'state_harden'] = temp_harden.values
    
    # Save the perturbed line state CSV
    perturbed_line_state_path = os.path.join(output_folder, line_state_csv_name)
    df_perturbed.to_csv(perturbed_line_state_path, index=False)


if __name__ == '__main__':
    load_perturb_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/sensitive_load'
    broken_perturb_path = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/sensitive_broken'
    sim_num = 10
    node_num = 1
    load_percent = 0.1
    line_num = 2

    data_perturbation(load_perturb_path, 'load', sim_num, node_num=node_num, load_percent=load_percent)
    data_perturbation(broken_perturb_path, 'broken', sim_num, line_num=line_num)

