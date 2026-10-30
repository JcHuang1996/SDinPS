# -*- coding: utf-8 -*-
# @Time     : 2026/03/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import os
import sys
import tempfile

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    import pyomo.environ as pyo
    from pyomo.core.base.label import cpxlp_label_from_name

    from model.model_base import ModelBase
except ModuleNotFoundError as exc:
    raise SystemExit(
        'This script requires a Python environment with `pyomo` installed. '
        'In this project, `conda run -n DailyOpt python unit_test/test_scripts/model_base_lp_export_test.py` works.'
    ) from exc


def _read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_ob:
        return file_ob.read()


def test_named_constraint_labels_and_existing_constraint_ops():
    model = ModelBase(model_name='lp_export_smoke')

    model.var['x'] = {0: model.add_var(domain=pyo.NonNegativeReals, name='x')}
    model.var['y'] = {0: model.add_var(domain=pyo.NonNegativeReals, name='y')}

    x_var = model.var['x'][0]
    y_var = model.var['y'][0]

    named_upper = 'named[row] 1'
    named_lower = 'named{row}_2'

    model.add_constr(x_var + y_var <= 5, name=named_upper)
    model.add_constr(x_var - y_var >= 1, name=named_lower)
    model.add_constr(x_var <= 4)

    if model.get_constr_idx_by_name(named_upper) != 1:
        raise AssertionError('Named upper constraint index lookup failed.')
    if model.get_constr_idx_by_name(named_lower) != 2:
        raise AssertionError('Named lower constraint index lookup failed.')

    model.update_constr_by_name(named_upper, x_var + y_var <= 6)
    updated_upper = model.get_constr_item_by_name(named_upper)
    if pyo.value(updated_upper.upper) != 6:
        raise AssertionError('Named constraint update did not change the upper bound.')

    model.remove_constr_by_name(named_lower)
    if model.get_constr_idx_by_name(named_lower) is not None:
        raise AssertionError('Named constraint removal left a stale name-map entry.')
    model.add_constr(x_var - y_var >= 1, name=named_lower)

    model.set_objective(x_var + y_var, sense=pyo.minimize)

    with tempfile.TemporaryDirectory() as tmp_dir:
        lp_path = os.path.join(tmp_dir, 'model.lp')
        model.write_file(lp_path)
        lp_text = _read_text(lp_path)

    expected_upper_label = f'c_u_{cpxlp_label_from_name(named_upper)}_:'
    expected_lower_label = f'c_l_{cpxlp_label_from_name(named_lower)}_:'

    if expected_upper_label not in lp_text:
        raise AssertionError(f'Missing named LP row label: {expected_upper_label}')
    if expected_lower_label not in lp_text:
        raise AssertionError(f'Missing named LP row label: {expected_lower_label}')
    if 'c_u__constr_list(3)_' not in lp_text:
        raise AssertionError('Unnamed constraint no longer falls back to the default ConstraintList label.')


def test_lp_name_collision_raises_clear_error():
    model = ModelBase(model_name='lp_export_collision')

    model.var['x'] = {0: model.add_var(domain=pyo.NonNegativeReals, name='x')}
    x_var = model.var['x'][0]

    model.add_constr(x_var <= 1, name='dup[row]')
    model.add_constr(x_var <= 2, name='dup(row)')
    model.set_objective(x_var, sense=pyo.minimize)

    with tempfile.TemporaryDirectory() as tmp_dir:
        lp_path = os.path.join(tmp_dir, 'collision.lp')
        try:
            model.write_file(lp_path)
        except ValueError as exc:
            msg = str(exc)
            if 'Constraint names conflict after LP sanitization' not in msg:
                raise AssertionError(f'Unexpected collision error message: {msg}')
        else:
            raise AssertionError('Expected LP export collision check to raise ValueError.')


def test_duplicate_raw_constraint_name_raises_clear_error():
    model = ModelBase(model_name='duplicate_raw_name')
    model.var['x'] = {0: model.add_var(domain=pyo.NonNegativeReals, name='x')}
    x_var = model.var['x'][0]
    model.add_constr(x_var <= 1, name='same_name')
    try:
        model.add_constr(x_var <= 2, name='same_name')
    except ValueError as exc:
        if 'Duplicate constraint name' not in str(exc):
            raise AssertionError(f'Unexpected duplicate-name error: {exc}')
    else:
        raise AssertionError('Duplicate raw constraint names must be rejected at insertion time.')


def main():
    test_named_constraint_labels_and_existing_constraint_ops()
    test_lp_name_collision_raises_clear_error()
    test_duplicate_raw_constraint_name_raises_clear_error()


if __name__ == '__main__':
    main()
