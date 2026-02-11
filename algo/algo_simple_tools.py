# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import logging
from typing import Tuple, Optional

from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.environ as pyo

logger = logging.getLogger(__name__)

# Default tolerance for comparing RHS coefficients and constant terms (same = within this tolerance)
DEFAULT_RHS_COMPARE_TOL = 1e-9

# Name used in comparison output for the constant term
CONSTANT_TERM_LABEL = 'constant term'


def _parse_rhs_to_constant_and_linear(rhs_expr):
    """
    Parse a linear Pyomo expression (e.g. RHS from gen_sce_bds_opt_cut / gen_sce_strengthen_bds_cut)
    into a constant term and a dict of variable name -> coefficient.
    :param rhs_expr: Pyomo expression (linear)
    :return: (constant: float, var_coef: dict[str, float]). If expression is not linear, returns (None, None).
    """
    try:
        repn = generate_standard_repn(rhs_expr, compute_values=True)
    except Exception:
        return None, None
    if repn.nonlinear_expr is not None:
        return None, None
    constant = 0.0
    if repn.constant is not None:
        constant = float(pyo.value(repn.constant))
    var_coef = {}
    if repn.linear_vars is not None and repn.linear_coefs is not None:
        for v, c in zip(repn.linear_vars, repn.linear_coefs):
            key = getattr(v, 'name', None)
            if key is None:
                key = str(v)
            var_coef[key] = float(pyo.value(c))
    return constant, var_coef


def compare_rhs(
    rhs1,
    rhs2,
    tol: Optional[float] = None,
    show_in_log: bool = True,
    show_in_console: bool = False,
) -> Tuple[bool, float, str]:
    """
    Compare two RHS expressions (e.g. from gen_sce_bds_opt_cut or gen_sce_strengthen_bds_cut).
    'Same' means: same set of variables, and for each variable and the constant term,
    the coefficients differ by at most tol.

    :param rhs1: First RHS (Pyomo expression).
    :param rhs2: Second RHS (Pyomo expression).
    :param tol: Tolerance for numeric equality. If None, uses DEFAULT_RHS_COMPARE_TOL.
    :param show_in_log: If True, log the comparison result (logger.info).
    :param show_in_console: If True, print the comparison result to stdout.
    :return: (is_same: bool, max_abs_diff: float, max_diff_location: str)
        max_diff_location is the variable name or CONSTANT_TERM_LABEL where the largest difference occurs.
    """
    if tol is None:
        tol = DEFAULT_RHS_COMPARE_TOL

    c1, d1 = _parse_rhs_to_constant_and_linear(rhs1)
    c2, d2 = _parse_rhs_to_constant_and_linear(rhs2)

    if (c1 is None and d1 is None) or (c2 is None and d2 is None):
        msg = "compare_rhs: one or both expressions could not be parsed (non-linear or error)."
        if show_in_log:
            logger.warning(msg)
        if show_in_console:
            print(msg)
        return False, float('inf'), 'parse error'

    # Same set of variables required
    set1 = set(d1.keys())
    set2 = set(d2.keys())
    if set1 != set2:
        only1 = set1 - set2
        only2 = set2 - set1
        msg = (
            "compare_rhs: different variables. Only in first: %s; only in second: %s."
            % (only1, only2)
        )
        if show_in_log:
            logger.info(msg)
        if show_in_console:
            print(msg)
        # Max diff is undefined; use inf and a descriptive location
        loc = "variable set mismatch (only in first: %s; only in second: %s)" % (only1, only2)
        return False, float('inf'), loc

    max_abs_diff = 0.0
    max_diff_location = CONSTANT_TERM_LABEL

    diff_const = abs(c1 - c2)
    if diff_const > max_abs_diff:
        max_abs_diff = diff_const
        max_diff_location = CONSTANT_TERM_LABEL

    for var_name in set1:
        coef1 = d1[var_name]
        coef2 = d2[var_name]
        d = abs(coef1 - coef2)
        if d > max_abs_diff:
            max_abs_diff = d
            max_diff_location = var_name

    is_same = max_abs_diff <= tol
    msg = (
        "compare_rhs: is_same=%s, max_abs_diff=%.6e at '%s' (tol=%.6e)."
        % (is_same, max_abs_diff, max_diff_location, tol)
    )
    if show_in_log:
        logger.info(msg)
    if show_in_console:
        print(msg)

    return is_same, max_abs_diff, max_diff_location


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
