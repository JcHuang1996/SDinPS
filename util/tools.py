# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import pandas as pd
import os


def iter_general_csv(ite_obj_value_dict=None, output_dir=None):
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

    df_general = pd.DataFrame(general_records)
    csv_general_path = os.path.join(output_dir, 'iter_general_info.csv')
    df_general.to_csv(csv_general_path, index=False)

def iter_sub_prob_info(ite_obj_value_dict=None, output_dir=None, scenario_list=None):
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

    df_sub = pd.DataFrame(sub_records)
    csv_sub_path = os.path.join(output_dir, 'iter_subproblem_info.csv')
    df_sub.to_csv(csv_sub_path, index=False)


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
