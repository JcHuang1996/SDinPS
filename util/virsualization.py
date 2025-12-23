# -*- coding: utf-8 -*-
# @Time     : 2025/10/26
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_iter_obj_curves(ite_obj_value_dict, output_dir=None, real_objective_value: float = None):
    """
    Visualize iteration convergence curves and export as PNG (Matplotlib version).

    Parameters
    ----------
    ite_obj_value_dict : dict
        The iteration objective dictionary (same as used in iter_general_csv).
    output_dir : str or None
        Path to output folder. If provided, saves the plot as 'iter_obj_plot.png'.
    real_objective_value : float, optional
        If provided, draws a horizontal reference line at this value.

    Notes
    -----
    - Plots two curves: best incumbent and best bound objective values.
    - Iteration keys that are non-numeric are ignored.
    - Uses Matplotlib for compatibility with macOS (no Plotly dependency).
    """
    ite_nums, incumbent_vals, bound_vals = [], [], []

    for ite_key, ite_data in ite_obj_value_dict.items():
        try:
            ite_num = int(ite_key)
        except (TypeError, ValueError):
            continue  # skip non-numeric iteration keys

        incumbent = ite_data.get('sub_obj(best_incumbent)', {}).get('sub_p_total', None)
        bound = ite_data.get('main_obj(bound)', {}).get('total', None)

        ite_nums.append(ite_num)
        incumbent_vals.append(incumbent)
        bound_vals.append(bound)

    if not ite_nums:
        print("No numeric iteration data found — plot not generated.")
        return

    # Sort by iteration number
    sorted_data = sorted(zip(ite_nums, incumbent_vals, bound_vals), key=lambda x: x[0])
    ite_nums, incumbent_vals, bound_vals = zip(*sorted_data)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(ite_nums, pd.Series(incumbent_vals).interpolate().tolist(), marker='.', markersize=4, label='Best Incumbent Objective')
    plt.plot(ite_nums, bound_vals, marker='.', markersize=4, label='Best Bound Objective')

    if real_objective_value is not None:
        plt.axhline(y=real_objective_value, color='r', linestyle='--', label='Real Objective Value')

    plt.xlabel('Iteration Number')
    plt.ylabel('Objective Value')
    plt.title('Iteration Objective Convergence')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --- Save and show ---
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'iter_obj_plot.png')
        plt.savefig(save_path, dpi=200)
        print(f"Plot saved to {save_path}")

    plt.show()
    plt.close()

# def plot_iter_obj_curves(ite_obj_value_dict, output_dir: str = None, real_objective_value: float = None):
#     """
#     Visualize iteration convergence curves from the in-memory result dictionary.
#
#     Parameters
#     ----------
#     ite_obj_value_dict : dict
#         The iteration objective dictionary (same as used in iter_general_csv).
#     real_objective_value : float, optional
#         If provided, draws a horizontal reference line at this value.
#
#     Notes
#     -----
#     - Plots two curves: best incumbent and best bound objective values.
#     - Iteration keys that are non-numeric are ignored.
#     """
#     ite_nums, incumbent_vals, bound_vals = [], [], []
#
#     for ite_key, ite_data in ite_obj_value_dict.items():
#         try:
#             ite_num = int(ite_key)
#         except (TypeError, ValueError):
#             continue  # skip non-numeric iteration keys
#
#         incumbent = ite_data.get('sub_obj(best_incumbent)', {}).get('sub_p_total', None)
#         bound = ite_data.get('main_obj(bound)', {}).get('total', None)
#
#         ite_nums.append(ite_num)
#         incumbent_vals.append(incumbent)
#         bound_vals.append(bound)
#
#     # Sort by iteration number for clean plotting
#     sorted_data = sorted(zip(ite_nums, incumbent_vals, bound_vals), key=lambda x: x[0])
#     ite_nums, incumbent_vals, bound_vals = zip(*sorted_data)
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=ite_nums,
#         y=incumbent_vals,
#         mode='lines+markers',
#         name='Best Incumbent Objective'
#     ))
#     fig.add_trace(go.Scatter(
#         x=ite_nums,
#         y=bound_vals,
#         mode='lines+markers',
#         name='Best Bound Objective'
#     ))
#
#     if real_objective_value is not None:
#         fig.add_hline(
#             y=real_objective_value,
#             line=dict(color='red', dash='dash'),
#             annotation_text='Real Objective',
#             annotation_position='top left'
#         )
#
#     fig.update_layout(
#         title='Iteration Objective Convergence',
#         xaxis_title='Iteration Number',
#         yaxis_title='Objective Value',
#         template='plotly_white'
#     )
#
#     # Save plot if output_dir is given
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         save_path = os.path.join(output_dir, 'iter_obj_plot.png')
#         fig.write_image(save_path)
#         print(f"Plot saved to {save_path}")
#
#     fig.show()


# def plot_iter_obj_curves_with_csv(output_dir=None, csv_path: str, real_objective_value: float = None):
#     """
#     Visualize iteration convergence curves from CSV.
#
#     Parameters
#     ----------
#     csv_path : str
#         Path to the CSV file generated by iter_general_csv.
#     real_objective_value : float, optional
#         If provided, draws a horizontal line at this objective value.
#
#     Notes
#     -----
#     - Plots two curves: best incumbent and best bound objective values.
#     - Iteration numbers that are non-numeric are ignored.
#     """
#     df = pd.read_csv(csv_path)
#
#     # keep only numeric iteration numbers
#     df = df[df['ite_num'].apply(lambda x: str(x).isdigit())].copy()
#     df['ite_num'] = df['ite_num'].astype(int)
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=df['ite_num'],
#         y=df['best_incumbent_obj_value'],
#         mode='lines+markers',
#         name='Best Incumbent Objective'
#     ))
#     fig.add_trace(go.Scatter(
#         x=df['ite_num'],
#         y=df['best_bound_objective_value'],
#         mode='lines+markers',
#         name='Best Bound Objective'
#     ))
#
#     if real_objective_value is not None:
#         fig.add_hline(
#             y=real_objective_value,
#             line=dict(color='red', dash='dash'),
#             annotation_text='Real Objective',
#             annotation_position='top left'
#         )
#
#     fig.update_layout(
#         title='Iteration Objective Convergence',
#         xaxis_title='Iteration Number',
#         yaxis_title='Objective Value',
#         template='plotly_white'
#     )
#
#
#     # Save plot if output_dir is given
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         save_path = os.path.join(output_dir, 'iter_obj_plot.png')
#         fig.write_image(save_path)
#         print(f"Plot saved to {save_path}")
#
#     # fig.show()
