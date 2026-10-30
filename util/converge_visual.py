# -*- coding: utf-8 -*-
# @Time     : 2025/10/26
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_iter_obj_curves(ite_obj_value_dict, output_dir=None, real_objective_value: float = None,
                         y_limits=None, show_plot=False):
    """
    Plot the incumbent and bound trajectories of one run.

    Certified bounds and master diagnostics are separate series. Missing certificates stay missing;
    neither guidance values nor interpolated values are presented as global lower bounds.
    """
    ite_nums, incumbent_vals, bound_vals, diagnostic_vals = [], [], [], []

    for ite_key, ite_data in ite_obj_value_dict.items():
        try:
            ite_num = int(ite_key)
        except (TypeError, ValueError):
            continue

        incumbent = ite_data.get('sub_obj(best_incumbent)', {}).get('sub_p_total', None)
        bound_data = ite_data.get('main_obj(bound)', {})
        valid_bound = bound_data.get('valid_lower_bound', None)

        ite_nums.append(ite_num)
        incumbent_vals.append(incumbent)
        bound_vals.append(valid_bound)
        diagnostic_vals.append(bound_data.get('total') if valid_bound is None else None)

    if not ite_nums:
        print("No numeric iteration data found — plot not generated.")
        return

    sorted_data = sorted(zip(ite_nums, incumbent_vals, bound_vals, diagnostic_vals), key=lambda x: x[0])
    ite_nums, incumbent_vals, bound_vals, diagnostic_vals = zip(*sorted_data)

    inc_s = pd.to_numeric(pd.Series(incumbent_vals), errors="coerce").cummin().ffill()
    bnd_s = pd.to_numeric(pd.Series(bound_vals), errors="coerce")
    diagnostic_s = pd.to_numeric(pd.Series(diagnostic_vals), errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ite_nums, inc_s.tolist(), marker='.', markersize=4, label='Best Incumbent Objective')
    if bnd_s.notna().any():
        ax.plot(ite_nums, bnd_s.tolist(), marker='.', markersize=4, label='Valid global lower bound')
    if diagnostic_s.notna().any():
        ax.plot(ite_nums, diagnostic_s.tolist(), marker='.', markersize=4,
                label='Master diagnostic (not a global lower bound)')

    if real_objective_value is not None:
        ax.axhline(y=real_objective_value, color='r', linestyle='--', label='Real Objective Value')

    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Objective Value')
    ax.set_title('Iteration Objective Convergence')
    ax.legend()
    ax.grid(True)

    # ---- y-axis: scale to the data, with an optional explicit override ----
    # A hardcoded 0 .. 1.15e7 window used to be forced here, which made the plot unreadable on any
    # instance of a different magnitude (the micro_bench instances, for example).
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        finite_values = [
            value for value in list(inc_s) + list(bnd_s) + list(diagnostic_s)
            + ([real_objective_value] if real_objective_value is not None else [])
            if value is not None and np.isfinite(value)
        ]
        if finite_values:
            low, high = min(finite_values), max(finite_values)
            pad = 0.05 * (high - low) if high > low else (abs(high) * 0.05 or 1.0)
            ax.set_ylim(min(0.0, low - pad), high + pad)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # -----------------------------------------------------------------------

    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'iter_obj_plot.png')
        fig.savefig(save_path, dpi=200)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close(fig)


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
