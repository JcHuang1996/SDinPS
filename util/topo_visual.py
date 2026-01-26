# -*- coding: utf-8 -*-
# @Time     : 2026/01/24
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

# Add project root to path for imports when running as script
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

from util.headers import NodeHeader, BranchHeader, NodeCoordinateHeader, ScenarioLineStateHeader


def net_topo_preview(folder_path: str):
    """
    Visualize network topology from CSV files in the given folder.
    
    Args:
        folder_path: Absolute path to the folder containing node.csv, branch.csv, 
                     and node_coordinate.csv files.
    """
    # Read CSV files
    node_df = pd.read_csv(os.path.join(folder_path, 'node.csv'))
    branch_df = pd.read_csv(os.path.join(folder_path, 'branch.csv'))
    
    # Try to read coordinate file, if it doesn't exist, use empty DataFrame
    coord_file = os.path.join(folder_path, 'node_coordinate.csv')
    if os.path.exists(coord_file):
        coord_df = pd.read_csv(coord_file)
    else:
        coord_df = pd.DataFrame(columns=[NodeCoordinateHeader.NODE_ID, 
                                         NodeCoordinateHeader.X, 
                                         NodeCoordinateHeader.Y])
    
    # Get all valid node IDs from node.csv
    valid_nodes = set(node_df[NodeHeader.NODE_ID].unique())
    
    # Filter branches to only include edges where both nodes are in node.csv
    branch_df = branch_df[
        (branch_df[BranchHeader.FROM_NODE].isin(valid_nodes)) &
        (branch_df[BranchHeader.TO_NODE].isin(valid_nodes))
    ].copy()
    
    # Create graph object
    G = nx.Graph()
    
    # Add all valid nodes to the graph
    for node_id in valid_nodes:
        G.add_node(node_id)
    
    # Build position dictionary from coordinates
    pos = {}
    coord_dict = {}
    if not coord_df.empty:
        for _, row in coord_df.iterrows():
            node_id = row[NodeCoordinateHeader.NODE_ID]
            if node_id in valid_nodes:
                x = row[NodeCoordinateHeader.X]
                y = row[NodeCoordinateHeader.Y]
                pos[node_id] = (x, y)
                coord_dict[node_id] = (x, y)
    
    # Add edges to graph and separate by init_state
    solid_edges = []
    dash_edges = []
    
    for _, row in branch_df.iterrows():
        from_node = row[BranchHeader.FROM_NODE]
        to_node = row[BranchHeader.TO_NODE]
        init_state = row[BranchHeader.INIT_STATE]
        
        if from_node in valid_nodes and to_node in valid_nodes:
            G.add_edge(from_node, to_node)
            if init_state == 1:
                solid_edges.append((from_node, to_node))
            elif init_state == 0:
                dash_edges.append((from_node, to_node))
    
    # Auto-position nodes without coordinates using spring layout
    nodes_without_coords = [n for n in valid_nodes if n not in pos]
    if nodes_without_coords:
        # Use spring layout for nodes without coordinates
        if len(coord_dict) > 0:
            # Use the existing positions as fixed positions for spring layout
            # This keeps nodes with coordinates in place and positions others relative to them
            spring_pos = nx.spring_layout(G, pos=coord_dict, fixed=list(coord_dict.keys()), 
                                         k=0.3, iterations=50)
            # Only update positions for nodes without coordinates
            for node in nodes_without_coords:
                if node in spring_pos:
                    pos[node] = spring_pos[node]
        else:
            # If no coordinates at all, use standard spring layout for all nodes
            spring_pos = nx.spring_layout(G, k=0.3, iterations=50)
            pos.update(spring_pos)
    
    # Setup the plot
    plt.figure(figsize=(15, 11))
    ax = plt.gca()
    
    # Define colors
    node_color = 'black'
    solid_edge_color = 'blue'
    dash_edge_color = 'gray'
    label_color = 'darkred'
    
    # Draw solid edges
    if solid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                              edge_color=solid_edge_color, 
                              style='solid', width=1.0, alpha=0.6)
    
    # Draw dash edges
    if dash_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dash_edges, 
                              edge_color=dash_edge_color, 
                              style='dashed', width=1.0, alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_color)
    
    # Draw node labels with small y-offset
    label_pos = {node: (coords[0], coords[1] - 0.02) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=7, font_color=label_color, 
                           verticalalignment='center')
    
    # Add grid and labels
    ax.set_xlabel('x (layout.json units)')
    ax.set_ylabel('y (layout.json units)')
    ax.set_title('Network Topology Preview')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Invert Y-axis to match typical coordinate system
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def net_topo_preview_with_results(folder_path: str, node_result: dict = None, line_result: dict = None, save_path: str = None):
    """
    Visualize network topology from CSV files with highlighted nodes and lines.
    Based on net_topo_preview, but marks nodes and lines with value 1 in the result dicts.
    
    Args:
        folder_path: Absolute path to the folder containing node.csv, branch.csv, 
                     and node_coordinate.csv files.
        node_result: Dict mapping node names (str) to binary values (0 or 1).
                     Nodes with value 1 will be highlighted.
        line_result: Dict mapping line names (tuple of (from_node, to_node)) to binary values (0 or 1).
                     Lines with value 1 will be highlighted.
        save_path: Optional path to save the figure. If provided, the figure will be saved before showing.
    """
    # Default to empty dicts if not provided
    if node_result is None:
        node_result = {}
    if line_result is None:
        line_result = {}
    
    # Read CSV files
    node_df = pd.read_csv(os.path.join(folder_path, 'node.csv'))
    branch_df = pd.read_csv(os.path.join(folder_path, 'branch.csv'))
    
    # Try to read coordinate file, if it doesn't exist, use empty DataFrame
    coord_file = os.path.join(folder_path, 'node_coordinate.csv')
    if os.path.exists(coord_file):
        coord_df = pd.read_csv(coord_file)
    else:
        coord_df = pd.DataFrame(columns=[NodeCoordinateHeader.NODE_ID, 
                                         NodeCoordinateHeader.X, 
                                         NodeCoordinateHeader.Y])
    
    # Get all valid node IDs from node.csv
    valid_nodes = set(node_df[NodeHeader.NODE_ID].unique())
    
    # Filter branches to only include edges where both nodes are in node.csv
    branch_df = branch_df[
        (branch_df[BranchHeader.FROM_NODE].isin(valid_nodes)) &
        (branch_df[BranchHeader.TO_NODE].isin(valid_nodes))
    ].copy()
    
    # Create graph object
    G = nx.Graph()
    
    # Add all valid nodes to the graph
    for node_id in valid_nodes:
        G.add_node(node_id)
    
    # Build position dictionary from coordinates
    pos = {}
    coord_dict = {}
    if not coord_df.empty:
        for _, row in coord_df.iterrows():
            node_id = row[NodeCoordinateHeader.NODE_ID]
            if node_id in valid_nodes:
                x = row[NodeCoordinateHeader.X]
                y = row[NodeCoordinateHeader.Y]
                pos[node_id] = (x, y)
                coord_dict[node_id] = (x, y)
    
    # Add edges to graph and separate by init_state and line_result
    solid_edges = []
    dash_edges = []
    highlighted_edges = []
    
    for _, row in branch_df.iterrows():
        from_node = row[BranchHeader.FROM_NODE]
        to_node = row[BranchHeader.TO_NODE]
        init_state = row[BranchHeader.INIT_STATE]
        edge_key = (from_node, to_node)
        edge_key_reverse = (to_node, from_node)
        
        if from_node in valid_nodes and to_node in valid_nodes:
            G.add_edge(from_node, to_node)
            
            # Check if edge is highlighted in line_result
            is_highlighted = (edge_key in line_result and line_result[edge_key] == 1) or \
                            (edge_key_reverse in line_result and line_result[edge_key_reverse] == 1)
            
            if is_highlighted:
                highlighted_edges.append((from_node, to_node))
            elif init_state == 1:
                solid_edges.append((from_node, to_node))
            elif init_state == 0:
                dash_edges.append((from_node, to_node))
    
    # Auto-position nodes without coordinates using spring layout
    nodes_without_coords = [n for n in valid_nodes if n not in pos]
    if nodes_without_coords:
        # Use spring layout for nodes without coordinates
        if len(coord_dict) > 0:
            # Use the existing positions as fixed positions for spring layout
            # This keeps nodes with coordinates in place and positions others relative to them
            spring_pos = nx.spring_layout(G, pos=coord_dict, fixed=list(coord_dict.keys()), 
                                         k=0.3, iterations=50)
            # Only update positions for nodes without coordinates
            for node in nodes_without_coords:
                if node in spring_pos:
                    pos[node] = spring_pos[node]
        else:
            # If no coordinates at all, use standard spring layout for all nodes
            spring_pos = nx.spring_layout(G, k=0.3, iterations=50)
            pos.update(spring_pos)
    
    # Separate nodes into regular and highlighted
    regular_nodes = []
    highlighted_nodes = []
    for node_id in valid_nodes:
        if node_id in node_result and node_result[node_id] == 1:
            highlighted_nodes.append(node_id)
        else:
            regular_nodes.append(node_id)
    
    # Setup the plot
    plt.figure(figsize=(15, 11))
    ax = plt.gca()
    
    # Define colors
    node_color = 'black'
    highlighted_node_color = 'red'
    solid_edge_color = 'blue'
    dash_edge_color = 'gray'
    highlighted_edge_color = 'magenta'
    label_color = 'darkred'
    
    # Draw solid edges
    if solid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                              edge_color=solid_edge_color, 
                              style='solid', width=1.0, alpha=0.6)
    
    # Draw dash edges
    if dash_edges:
        nx.draw_networkx_edges(G, pos, edgelist=dash_edges, 
                              edge_color=dash_edge_color, 
                              style='dashed', width=1.0, alpha=0.6)
    
    # Draw highlighted edges (on top, with thicker width)
    if highlighted_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, 
                              edge_color=highlighted_edge_color, 
                              style='solid', width=2.5, alpha=0.9)
    
    # Draw regular nodes
    if regular_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                              node_size=20, node_color=node_color)
    
    # Draw highlighted nodes (on top, with larger size)
    if highlighted_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, 
                              node_size=50, node_color=highlighted_node_color)
    
    # Draw node labels with small y-offset
    label_pos = {node: (coords[0], coords[1] - 0.02) for node, coords in pos.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=7, font_color=label_color, 
                           verticalalignment='center')
    
    # Add grid and labels
    ax.set_xlabel('x (layout.json units)')
    ax.set_ylabel('y (layout.json units)')
    ax.set_title('Network Topology Preview with Results')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Invert Y-axis to match typical coordinate system
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')
    
    plt.show()


def _load_network_data(folder_path: str):
    """
    Helper function to load network topology data (nodes, branches, coordinates).
    
    Returns:
        tuple: (node_df, branch_df, coord_df, valid_nodes, G, pos)
    """
    # Read CSV files
    node_df = pd.read_csv(os.path.join(folder_path, 'node.csv'))
    branch_df = pd.read_csv(os.path.join(folder_path, 'branch.csv'))
    
    # Try to read coordinate file
    coord_file = os.path.join(folder_path, 'node_coordinate.csv')
    if os.path.exists(coord_file):
        coord_df = pd.read_csv(coord_file)
    else:
        coord_df = pd.DataFrame(columns=[NodeCoordinateHeader.NODE_ID, 
                                         NodeCoordinateHeader.X, 
                                         NodeCoordinateHeader.Y])
    
    # Get all valid node IDs from node.csv
    valid_nodes = set(node_df[NodeHeader.NODE_ID].unique())
    
    # Filter branches to only include edges where both nodes are in node.csv
    branch_df = branch_df[
        (branch_df[BranchHeader.FROM_NODE].isin(valid_nodes)) &
        (branch_df[BranchHeader.TO_NODE].isin(valid_nodes))
    ].copy()
    
    # Create graph object
    G = nx.Graph()
    
    # Add all valid nodes to the graph
    for node_id in valid_nodes:
        G.add_node(node_id)
    
    # Build position dictionary from coordinates
    pos = {}
    coord_dict = {}
    if not coord_df.empty:
        for _, row in coord_df.iterrows():
            node_id = row[NodeCoordinateHeader.NODE_ID]
            if node_id in valid_nodes:
                x = row[NodeCoordinateHeader.X]
                y = row[NodeCoordinateHeader.Y]
                pos[node_id] = (x, y)
                coord_dict[node_id] = (x, y)
    
    # Auto-position nodes without coordinates using spring layout
    nodes_without_coords = [n for n in valid_nodes if n not in pos]
    if nodes_without_coords:
        if len(coord_dict) > 0:
            spring_pos = nx.spring_layout(G, pos=coord_dict, fixed=list(coord_dict.keys()), 
                                         k=0.3, iterations=50)
            for node in nodes_without_coords:
                if node in spring_pos:
                    pos[node] = spring_pos[node]
        else:
            spring_pos = nx.spring_layout(G, k=0.3, iterations=50)
            pos.update(spring_pos)
    
    return node_df, branch_df, coord_df, valid_nodes, G, pos


def _get_state_color_mapping():
    """
    Returns color mapping for the 4 state combinations.
    
    Returns:
        dict: {(state_no_harden, state_harden): color}
    """
    return {
        (0, 0): 'red',      # Failed without hardening, Failed with hardening
        (0, 1): 'orange',   # Failed without hardening, Working with hardening
        (1, 0): 'yellow',   # Working without hardening, Failed with hardening
        (1, 1): 'green'     # Working without hardening, Working with hardening
    }


def _get_state_label(state_no_harden: int, state_harden: int):
    """
    Returns label for state combination.
    """
    labels = {
        (0, 0): 'Failed (no harden), Failed (harden)',
        (0, 1): 'Failed (no harden), Working (harden)',
        (1, 0): 'Working (no harden), Failed (harden)',
        (1, 1): 'Working (no harden), Working (harden)'
    }
    return labels.get((state_no_harden, state_harden), 'Unknown')


def sto_status_preview(folder_path: str, line_state_file: str):
    """
    Visualize stochastic line status for each scenario-time combination.
    Generates static plots saved to folder_path/stochastic_preview/.
    
    Args:
        folder_path: Absolute path to the folder containing node.csv, branch.csv, 
                     and node_coordinate.csv files.
        line_state_file: Absolute path to s_line_stage_w_o_harden.csv file.
    """
    # Load network data
    node_df, branch_df, coord_df, valid_nodes, G, pos = _load_network_data(folder_path)
    
    # Read line state file
    line_state_df = pd.read_csv(line_state_file)
    
    # Get color mapping
    color_map = _get_state_color_mapping()
    
    # Create output directory structure
    output_base = os.path.join(folder_path, 'stochastic_preview')
    os.makedirs(output_base, exist_ok=True)
    
    # Group by scenario_id and time_idx
    grouped = line_state_df.groupby([ScenarioLineStateHeader.SCENARIO_ID, 
                                     ScenarioLineStateHeader.TIME_IDX])
    
    for (scenario_id, time_idx), group_df in grouped:
        # Create scenario folder
        scenario_folder = os.path.join(output_base, str(scenario_id))
        os.makedirs(scenario_folder, exist_ok=True)
        
        # Create edge state dictionary for this scenario-time
        edge_states = {}
        for _, row in group_df.iterrows():
            from_node = row[ScenarioLineStateHeader.FROM_NODE]
            to_node = row[ScenarioLineStateHeader.TO_NODE]
            state_no_harden = int(row[ScenarioLineStateHeader.STATE_NO_HARDEN])
            state_harden = int(row[ScenarioLineStateHeader.STATE_HARDEN])
            
            if from_node in valid_nodes and to_node in valid_nodes:
                edge_key = (from_node, to_node)
                edge_states[edge_key] = (state_no_harden, state_harden)
        
        # Group edges by state for plotting
        edges_by_state = {state: [] for state in color_map.keys()}
        for edge_key, state in edge_states.items():
            if state in edges_by_state:
                edges_by_state[state].append(edge_key)
        
        # Setup the plot
        plt.figure(figsize=(15, 11))
        ax = plt.gca()
        
        # Draw edges grouped by state
        for state, edges in edges_by_state.items():
            if edges:
                color = color_map[state]
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                      edge_color=color, 
                                      style='solid', width=1.5, alpha=0.7)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='black')
        
        # Draw node labels
        label_pos = {node: (coords[0], coords[1] - 0.02) for node, coords in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=7, font_color='darkred', 
                               verticalalignment='center')
        
        # Create legend
        legend_elements = [Patch(facecolor=color, label=_get_state_label(state[0], state[1]))
                          for state, color in color_map.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add grid and labels
        ax.set_xlabel('x (layout.json units)')
        ax.set_ylabel('y (layout.json units)')
        ax.set_title(f'Scenario {scenario_id} - Time {time_idx}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.invert_yaxis()
        
        # Save plot
        filename = f'{scenario_id} - {time_idx}.png'
        filepath = os.path.join(scenario_folder, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Static plots saved to {output_base}")


if __name__ == '__main__':
    # Example usage: visualize IEEE123bus network topology
    test_folder = '/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file/function_test'
    
    # Option 1: Basic network topology preview
    net_topo_preview(test_folder)
    
    # Option 2: Stochastic status preview (static plots)
    # line_state_file = os.path.join(test_folder, 's_line_stage_w_o_harden.csv')
    # sto_status_preview(test_folder, line_state_file)