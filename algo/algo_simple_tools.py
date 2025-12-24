# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

# the function to divide and record the var names and var keys by their binary values

def bi_var_counter(var_result_dict):

    zero_var_idx, one_var_idx = {}, {}

    for var_class_name in sorted(var_result_dict.keys()):
        zero_var_idx[var_class_name], one_var_idx[var_class_name] = [], []
        for var_key in sorted(var_result_dict[var_class_name].keys()):
            if var_result_dict[var_class_name][var_key] == 0:
                zero_var_idx[var_class_name].append(var_key)
            elif var_result_dict[var_class_name][var_key] == 1:
                one_var_idx[var_class_name].append(var_key)
            else:
                raise ValueError('the value of variable must be binary for this function')

    return zero_var_idx, one_var_idx
