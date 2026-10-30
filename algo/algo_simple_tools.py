# -*- coding: utf-8 -*-
# @Time     : 2025/10/27
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import ast
import json
import logging
import math
import os
from typing import Dict, List, Sequence, Tuple, Optional, Any

import pandas as pd
from pyomo.repn.standard_repn import generate_standard_repn
import pyomo.environ as pyo
from util.names import VarName

logger = logging.getLogger(__name__)

# Default tolerance for comparing RHS coefficients and constant terms (same = within this tolerance)
DEFAULT_RHS_COMPARE_TOL = 1e-9

# Name used in comparison output for the constant term
CONSTANT_TERM_LABEL = 'constant term'

# Default tolerance for cut-signature quantization.
DEFAULT_CUT_SIGNATURE_TOL = 1e-8


def _canonicalize_nested_obj(obj: Any):
    """Convert nested structures to a deterministic, hashable representation."""
    if isinstance(obj, dict):
        items = [(_canonicalize_nested_obj(k), _canonicalize_nested_obj(v)) for k, v in obj.items()]
        return ("dict", tuple(sorted(items, key=lambda x: repr(x[0]))))
    if isinstance(obj, (list, tuple)):
        return ("seq", tuple(_canonicalize_nested_obj(v) for v in obj))
    if isinstance(obj, set):
        vals = [_canonicalize_nested_obj(v) for v in obj]
        return ("set", tuple(sorted(vals, key=repr)))
    return ("atom", obj)


def _quantize_float_for_signature(val: float, tol: float = DEFAULT_CUT_SIGNATURE_TOL) -> float:
    if tol <= 0:
        return float(val)
    return round(float(val) / tol) * tol


def _quantize_obj_for_signature(obj: Any, tol: float = DEFAULT_CUT_SIGNATURE_TOL):
    if isinstance(obj, dict):
        return {k: _quantize_obj_for_signature(v, tol=tol) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_quantize_obj_for_signature(v, tol=tol) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_quantize_obj_for_signature(v, tol=tol) for v in obj)
    if isinstance(obj, float):
        return _quantize_float_for_signature(obj, tol=tol)
    return obj


def build_cut_signature_from_row(
    cut_type: str,
    row: dict,
    tol: float = DEFAULT_CUT_SIGNATURE_TOL,
    include_cut_type: bool = True,
    theta_weights: Optional[dict] = None,
    sub_sce_list: Optional[Sequence] = None,
):
    """
    Build a deterministic signature for a cut row independent of cut name.
    """
    payload = {
        "x_coef": row.get("x_coef", {}),
        "rhs": float(row.get("rhs", 0.0)),
    }
    # Equal right-hand sides on different theta variables are different inequalities.
    if theta_weights is not None:
        payload['theta_coef'] = theta_weights
    else:
        payload['scenario_group'] = tuple(sorted(sub_sce_list or (), key=repr))
    if include_cut_type:
        payload["cut_type"] = cut_type
    return _canonicalize_nested_obj(_quantize_obj_for_signature(payload, tol=tol))


def build_linear_expr_signature(
    expr,
    cut_type: str,
    tol: float = DEFAULT_CUT_SIGNATURE_TOL,
    include_cut_type: bool = True,
):
    """
    Build a deterministic signature for a linear Pyomo expression.
    Returns None when the expression is not linear / parseable.
    """
    const, var_coef = _parse_rhs_to_constant_and_linear(expr)
    if const is None and var_coef is None:
        return None
    payload = {
        "constant": float(const),
        "var_coef": var_coef,
    }
    if include_cut_type:
        payload["cut_type"] = cut_type
    return _canonicalize_nested_obj(_quantize_obj_for_signature(payload, tol=tol))


class CutRepetitionTracker:
    """Track repeated cuts across iterations using deterministic signatures."""

    def __init__(self):
        self.seen_signatures = {}
        self.records = []

    @staticmethod
    def _normalize_iteration_label(ite_name):
        if isinstance(ite_name, int):
            return ite_name
        try:
            return int(ite_name)
        except (TypeError, ValueError):
            return ite_name

    def _record(self, signature, cut_type: str, ite_name, cut_name: str, sub_sce_list: Optional[List] = None) -> dict:
        current_iteration = self._normalize_iteration_label(ite_name)
        first_record = self.seen_signatures.get(signature)
        is_repeated = first_record is not None
        record = {
            "cut_type": cut_type,
            "cut_name": cut_name,
            "iteration": current_iteration,
            "scenario_group": list(sub_sce_list) if sub_sce_list is not None else None,
            "is_repeated": is_repeated,
            "first_seen_iteration": first_record["iteration"] if is_repeated else None,
            "first_seen_cut_name": first_record["cut_name"] if is_repeated else None,
        }
        self.records.append(record)
        if not is_repeated:
            self.seen_signatures[signature] = {
                "iteration": current_iteration,
                "cut_name": cut_name,
            }
        return record

    def _build_peek_record(self, first_record) -> dict:
        return {
            "is_repeated": first_record is not None,
            "first_seen_iteration": first_record["iteration"] if first_record is not None else None,
            "first_seen_cut_name": first_record["cut_name"] if first_record is not None else None,
        }

    def peek_by_row(
        self,
        cut_type: str,
        row: dict,
        include_cut_type: bool = True,
        theta_weights: Optional[dict] = None,
        sub_sce_list: Optional[Sequence] = None,
    ) -> dict:
        signature = build_cut_signature_from_row(
            cut_type=cut_type,
            row=row,
            include_cut_type=include_cut_type,
            theta_weights=theta_weights,
            sub_sce_list=sub_sce_list,
        )
        return self._build_peek_record(self.seen_signatures.get(signature))

    def peek_by_expr(
        self,
        cut_type: str,
        expr,
        include_cut_type: bool = True,
    ) -> dict:
        signature = build_linear_expr_signature(
            expr=expr,
            cut_type=cut_type,
            include_cut_type=include_cut_type,
        )
        if signature is None:
            signature = ("fallback", cut_type if include_cut_type else None, str(expr))
        return self._build_peek_record(self.seen_signatures.get(signature))

    def record_by_row(
        self,
        cut_type: str,
        ite_name,
        cut_name: str,
        row: dict,
        sub_sce_list: Optional[List] = None,
        include_cut_type: bool = True,
        theta_weights: Optional[dict] = None,
    ) -> dict:
        return self._record(
            signature=build_cut_signature_from_row(
                cut_type=cut_type,
                row=row,
                include_cut_type=include_cut_type,
                theta_weights=theta_weights,
                sub_sce_list=sub_sce_list,
            ),
            cut_type=cut_type,
            ite_name=ite_name,
            cut_name=cut_name,
            sub_sce_list=sub_sce_list,
        )

    def record_by_expr(
        self,
        cut_type: str,
        ite_name,
        cut_name: str,
        expr,
        sub_sce_list: Optional[List] = None,
        include_cut_type: bool = True,
    ) -> dict:
        signature = build_linear_expr_signature(
            expr=expr,
            cut_type=cut_type,
            include_cut_type=include_cut_type,
        )
        if signature is None:
            signature = ("fallback", cut_type if include_cut_type else None, str(expr))
        return self._record(
            signature=signature,
            cut_type=cut_type,
            ite_name=ite_name,
            cut_name=cut_name,
            sub_sce_list=sub_sce_list,
        )

    def get_repetitive_records(self) -> List[dict]:
        return [record for record in self.records if record.get("is_repeated")]


def detect_and_record_repetition(
    value: Any,
    seen_signatures: Dict[Any, int],
    iteration: int,
    tol: Optional[float] = None,
) -> Tuple[bool, Optional[int]]:
    """
    Detect repetition online (during an iteration) and update seen_signatures.

    Returns:
        is_repeated: True if value has appeared before.
        first_seen_iteration: The first iteration index where it appeared, or None for first occurrence.
    """
    signature_obj = _quantize_obj_for_signature(value, tol=tol) if tol is not None else value
    signature = _canonicalize_nested_obj(signature_obj)
    if signature in seen_signatures:
        return True, seen_signatures[signature]
    seen_signatures[signature] = iteration
    return False, None


def _parse_rhs_to_constant_and_linear(rhs_expr):
    """
    Parse a linear Pyomo expression (e.g. RHS from build_multiplier_cut / gen_sce_strengthen_bds_cut)
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
    Compare two RHS expressions (e.g. from build_multiplier_cut or gen_sce_strengthen_bds_cut).
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


def flatten_ordered_var_values(var_value_dict: dict, ordered_var_keys: Sequence[Tuple[str, object]]) -> Tuple[int, ...]:
    values = tuple(
        float(var_value_dict[var_name][var_key])
        for var_name, var_key in ordered_var_keys
    )
    if any(not math.isfinite(v) or min(abs(v), abs(v - 1.0)) > 1e-6 for v in values):
        raise ValueError('CGLP state values must be binary within solver tolerance.')
    # Truncating 0.999999999 would silently register a different point.
    return tuple(round(v) for v in values)


def get_prefix_tuple(binary_point: Sequence[int], prefix_len: int) -> Tuple[int, ...]:
    return tuple(binary_point[:prefix_len])


def flip_prefix_tail(prefix_tuple: Sequence[int]) -> Tuple[int, ...]:
    prefix_list = list(prefix_tuple)
    prefix_list[-1] = 1 - int(prefix_list[-1])
    return tuple(prefix_list)


def prefix_to_w_vector(prefix_tuple: Sequence[int], total_dim: int) -> Tuple[int, ...]:
    prefix = tuple(prefix_tuple)
    return prefix + (0,) * (total_dim - len(prefix))


def extract_affine_terms_over_vars(expr, var_id_map: Dict[int, Any]) -> Tuple[float, Dict[Any, float]]:
    """
    Reduce an affine Pyomo expression to ``(constant, {key: coefficient})`` over selected variables.

    ``var_id_map`` maps ``id(var_object)`` to the key the caller wants back; variables outside the
    map are ignored, so the same expression can be projected onto theta variables or onto master
    state variables. A constant expression (int/float) is handled directly.

    Used by the cut-validity harness to replay a cut offline as plain numbers.
    """
    if isinstance(expr, (int, float)):
        return float(expr), {}

    repn = generate_standard_repn(expr, compute_values=True)

    constant_term = 0.0 if repn.constant is None else float(pyo.value(repn.constant))

    coef_by_key: Dict[Any, float] = {}
    if repn.linear_vars is not None:
        for linear_var, linear_coef in zip(repn.linear_vars, repn.linear_coefs):
            key = var_id_map.get(id(linear_var))
            if key is None:
                continue
            coef_by_key[key] = coef_by_key.get(key, 0.0) + float(pyo.value(linear_coef))

    return constant_term, coef_by_key


def flatten_nested_var_coef_dict(var_coef_dict: dict) -> Dict[Tuple[str, object], float]:
    return {
        (var_name, var_key): float(coef)
        for var_name in sorted(var_coef_dict.keys())
        for var_key, coef in var_coef_dict[var_name].items()
    }


def normalize_affine_lower_bound_row(cut_name: str, constant_term: float, var_coef_dict: dict) -> dict:
    """
    Normalize theta >= constant_term + sum(coef * x) into row form a x - theta <= b.
    """
    return {
        'name': cut_name,
        'x_coef': flatten_nested_var_coef_dict(var_coef_dict),
        'rhs': -float(constant_term),
    }


def normalize_cglp_lower_bound_row(row: dict, x_var_keys: Sequence[Tuple[str, object]]) -> dict:
    """Convert a lower-bound cut into the dense row format expected by ModelCGLP."""
    x_coef = row.get('x_coef', {})
    if x_coef and isinstance(next(iter(x_coef.values())), dict):
        x_coef = flatten_nested_var_coef_dict(x_coef)

    return {
        'name': row['name'],
        'x_coef': {
            x_key: float(x_coef.get(x_key, 0.0))
            for x_key in x_var_keys
        },
        'rhs': float(row['rhs']),
    }


def flat_to_nested_coef_dict(flat_coef_dict: Dict[Tuple[str, object], float]) -> dict:
    nested_coef_dict = {}
    for (var_name, var_key), coef_value in flat_coef_dict.items():
        if var_name not in nested_coef_dict:
            nested_coef_dict[var_name] = {}
        nested_coef_dict[var_name][var_key] = float(coef_value)
    return nested_coef_dict


def get_affine_constant_from_point_and_coef(base_value: float, var_coef_dict: dict, ref_point: dict) -> float:
    return float(base_value) - sum(
        float(var_coef_dict[var_name][var_key]) * float(ref_point[var_name][var_key])
        for var_name in sorted(var_coef_dict.keys())
        for var_key in sorted(var_coef_dict[var_name].keys())
    )


def build_cglp_lower_bound_row_from_alpha_eta(cut_name: str, alpha: Dict[Tuple[str, object], float], eta: float) -> dict:
    return normalize_affine_lower_bound_row(
        cut_name=cut_name,
        constant_term=-float(eta),
        var_coef_dict=flat_to_nested_coef_dict(
            {
                var_key: -float(alpha[var_key])
                for var_key in alpha
            }
        ),
    )


def build_cglp_lower_bound_row_from_multiplier_cut(
    cut_name: str,
    base_value: float,
    multiplier_info: dict,
    constr_var_map: dict,
    ref_point: dict,
) -> dict:
    var_coef_dict = {}
    for constr_name in sorted(multiplier_info.keys()):
        var_name, var_key = constr_var_map[constr_name]
        if var_name not in var_coef_dict:
            var_coef_dict[var_name] = {}
        if var_key not in var_coef_dict[var_name]:
            var_coef_dict[var_name][var_key] = 0.0
        var_coef_dict[var_name][var_key] += float(multiplier_info[constr_name])

    return normalize_affine_lower_bound_row(
        cut_name=cut_name,
        constant_term=get_affine_constant_from_point_and_coef(
            base_value=base_value,
            var_coef_dict=var_coef_dict,
            ref_point=ref_point,
        ),
        var_coef_dict=var_coef_dict,
    )


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


def _parse_key(k: str):
    """If *k* looks like a string-repr of a tuple, e.g. \"('a', 'b')\", return the actual tuple; else return *k*."""
    if k.startswith('(') and k.endswith(')'):
        try:
            val = ast.literal_eval(k)
            if isinstance(val, tuple):
                return val
        except (ValueError, SyntaxError):
            pass
    return k


def _restore_keys(d: dict) -> dict:
    """Recursively convert string-encoded tuple keys back to real tuples in a two-layer solution dict."""
    return {var_name: {_parse_key(k): v for k, v in inner.items()} for var_name, inner in d.items()}


def _sce_list_to_filename(scenario_list: list) -> str:
    """Convert a scenario list to its solution JSON filename (mirrors solution_recorder.py)."""
    sce_str = str(scenario_list)
    return (sce_str.replace("'", "").replace("[", "").replace("]", "")
                   .replace(", ", "_").replace(" ", "_").strip() + ".json")


def analyze_scenario_solutions(dataset_path: str, scenario_list: list, hat: float):
    """
    Compute weighted-average (expectation) of solution variables across scenarios,
    then partition variable keys by whether their expectation exceeds *hat*.

    When the overall solution for the full *scenario_list* exists in the solutions
    folder, also computes the "true" partition (keys with value 1 vs 0 in that
    overall solution), the corresponding expectation sums, and a suggested hat
    via grid search that best matches the true partition.

    Returns:
        expectation      – dict {var_name: {key: weighted_avg, ...}, ...}
        keys_above       – dict {var_name: sorted list of keys where expectation > hat}
        keys_below       – dict {var_name: sorted list of keys where expectation <= hat}
        sum_above        – dict {var_name: sum of expectations for keys in keys_above}
        sum_below        – dict {var_name: sum of expectations for keys in keys_below}
        true_keys_above  – (or None) dict {var_name: sorted keys with value 1 in overall solution}
        true_keys_below  – (or None) dict {var_name: sorted keys with value != 1 in overall solution}
        true_sum_above   – (or None) dict {var_name: sum of expectations for keys in true_keys_above}
        true_sum_below   – (or None) dict {var_name: sum of expectations for keys in true_keys_below}
        suggested_hat    – (or None) float, hat that best reproduces the true partition
    """
    dataset_path = os.path.abspath(dataset_path)
    solutions_dir = os.path.join(dataset_path, 'solutions')

    # Read scenario probabilities and normalise to the selected subset
    prob_df = pd.read_csv(os.path.join(dataset_path, 's_probability.csv'))
    prob_map = dict(zip(prob_df['scenario_id'], prob_df['probability']))
    raw_weights = {s: prob_map[s] for s in scenario_list}
    total_weight = sum(raw_weights.values())
    weights = {s: w / total_weight for s, w in raw_weights.items()}

    # Load individual-scenario solution JSONs
    solutions = {}
    for s in scenario_list:
        filepath = os.path.join(solutions_dir, _sce_list_to_filename([s]))
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Solution file not found for scenario '{s}': {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            solutions[s] = _restore_keys(json.load(f))

    # Compute expectation (weighted average) keeping the same nested-dict structure
    template = solutions[scenario_list[0]]
    expectation = {}
    for var_name in template:
        expectation[var_name] = {}
        for key in template[var_name]:
            expectation[var_name][key] = sum(
                weights[s] * solutions[s][var_name][key] for s in scenario_list
            )

    # Partition keys into above / below hat, and compute grouped sums
    keys_above, keys_below = {}, {}
    sum_above, sum_below = {}, {}
    for var_name, key_vals in expectation.items():
        above = sorted(k for k, v in key_vals.items() if v > hat)
        below = sorted(k for k, v in key_vals.items() if v <= hat)
        keys_above[var_name] = above
        keys_below[var_name] = below
        sum_above[var_name] = sum(key_vals[k] for k in above)
        sum_below[var_name] = sum(key_vals[k] for k in below)

    # --- True partition & suggested hat (only when overall solution exists) ---
    overall_path = os.path.join(solutions_dir, _sce_list_to_filename(scenario_list))
    if not os.path.exists(overall_path):
        return (expectation, keys_above, keys_below, sum_above, sum_below,
                None, None, None, None, None)

    with open(overall_path, 'r', encoding='utf-8') as f:
        overall_sol = _restore_keys(json.load(f))

    # True partition: value == 1 → above, otherwise → below
    true_keys_above, true_keys_below = {}, {}
    true_sum_above, true_sum_below = {}, {}
    true_above_sets = {}
    for var_name in expectation:
        t_above = sorted(k for k, v in overall_sol[var_name].items() if v > 0.9)
        t_below = sorted(k for k, v in overall_sol[var_name].items() if v <= 0.9)
        true_keys_above[var_name] = t_above
        true_keys_below[var_name] = t_below
        true_above_sets[var_name] = set(t_above)
        true_sum_above[var_name] = sum(expectation[var_name][k] for k in t_above)
        true_sum_below[var_name] = sum(expectation[var_name][k] for k in t_below)

    # Grid search: test every distinct expectation value (+ one below min) as candidate hat.
    # The partition {k : exp[k] > hat} only changes at these breakpoints.
    all_vals = sorted({v for kv in expectation.values() for v in kv.values()})
    candidates = [all_vals[0] - 1.0] + all_vals

    best_hat, best_diff, best_sum_gap = None, float('inf'), float('inf')
    for c in candidates:
        diff, sum_gap = 0, 0.0
        for var_name, key_vals in expectation.items():
            c_above = {k for k, v in key_vals.items() if v > c}
            diff += len(c_above.symmetric_difference(true_above_sets[var_name]))
            sum_gap += abs(sum(key_vals[k] for k in c_above) - true_sum_above[var_name])
        if diff < best_diff or (diff == best_diff and sum_gap < best_sum_gap):
            best_hat, best_diff, best_sum_gap = c, diff, sum_gap

    return (expectation, keys_above, keys_below, sum_above, sum_below,
            true_keys_above, true_keys_below, true_sum_above, true_sum_below, best_hat)
