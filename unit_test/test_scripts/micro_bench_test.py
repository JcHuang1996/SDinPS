# -*- coding: utf-8 -*-
# @Time     : 2026/09/17
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
Correctness check of the combined (extensive-form) model on the tiny 'micro_bench' instances.

Each instance folder in unit_test/test_local_csv_file/micro_bench/ carries a reference.json with
the unique optimal objective value, the unique optimal first stage and the unique optimal
load-shedding / line-connection pattern. This script rebuilds every instance with ModelCombined,
solves it to a zero MIP gap and reports every deviation.

Run:
    python unit_test/test_scripts/micro_bench_test.py
    python unit_test/test_scripts/micro_bench_test.py --solver appsi_highs
    python unit_test/test_scripts/micro_bench_test.py --instances m1_dg_site_type,m6_two_scenario_hedging
"""

import argparse
import json
import os
import sys
import traceback

# Add project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from util.names import InputMethodName, ModelStatus, VarName
from util.project_logger import init_logger
from dao.data_reader import DataReader
from dao.data_processor import DataProcessor
from model.model_combined import ModelCombined
from model.solver_config import SolverConfig

import logging

logger = logging.getLogger(__name__)

MICRO_BENCH_DIR_NAME = 'micro_bench'
VALUE_TOL = 1e-6


def _key_str(key):
    if isinstance(key, tuple):
        return '|'.join(str(part) for part in key)
    return str(key)


def _close(a, b, tol=VALUE_TOL):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def _compare_map(label, actual, expected, failures, tol=VALUE_TOL):
    """Compare {model key: value} against {'a|b': value}; every expected key must match."""
    actual_str = {_key_str(k): v for k, v in actual.items()}
    for key, exp in expected.items():
        got = actual_str.get(key)
        if got is None:
            failures.append(f'{label}: key {key} missing in model result')
        elif not _close(got, exp, tol):
            failures.append(f'{label}: {key} expected {exp}, got {got}')
    return actual_str


def run_one_instance(local_file_path, instance_name, solver_config):
    folder = os.path.join(local_file_path, MICRO_BENCH_DIR_NAME, instance_name)
    with open(os.path.join(folder, 'reference.json'), 'r', encoding='utf-8') as f:
        ref = json.load(f)

    reader = DataReader(
        read_method=InputMethodName.LOCAL_CSV,
        local_file_path=local_file_path,
        data_set_name=f'{MICRO_BENCH_DIR_NAME}/{instance_name}',
    )
    reader.read()

    processor = DataProcessor(reader.raw_data)
    processed_data = processor.data_process(
        scenario_list_assigned=ref['scenario_list'],
        time_list_assigned=ref['time_list'],
    )

    model = ModelCombined(
        model_name=f'micro_{instance_name}',
        model_data=processed_data,
        solver_config=solver_config,
    )
    model.build_model_all_obj_terms()
    model.solve()
    model.cal_detailed_obj()
    model.get_result([
        VarName.DG_INSTALL,
        VarName.DG_INSTALL_TYPE,
        VarName.LINE_HARDEN,
        VarName.LOAD_SHED_RATIO,
        VarName.LINE_CONNECTED,
        VarName.VIRTUAL_SOURCE_INDICATOR,
    ])

    failures = []
    obj_value = model.get_obj_value()
    term_value = dict(model.obj_term_value)
    term_sum = sum(term_value.values())

    print(f'\n--- {instance_name} ---')
    print(f'  description          : {ref["description"]}')
    print(f'  solve status         : {model.solve_status}')
    print(f'  objective  reference : {ref["objective"]}')
    print(f'  objective  model     : {obj_value!r}')
    print(f'  sum of obj terms     : {term_sum!r}')
    print(f'  2nd best design obj  : {ref["second_best_design_obj"]} '
          f'(uniqueness margin {ref["second_best_design_obj"] - ref["objective"]})')

    if model.solve_status != ModelStatus.SOLVED_OPT:
        failures.append(f'solve status is {model.solve_status}, expected an optimal solve')
    if not _close(obj_value, ref['objective']):
        failures.append(f'objective expected {ref["objective"]}, got {obj_value}')
    if not _close(term_sum, obj_value):
        failures.append(f'sum of objective terms {term_sum} != model objective {obj_value}')

    print('  objective terms:')
    for name in sorted(set(list(ref['objective_terms'].keys()) + list(term_value.keys()))):
        exp = ref['objective_terms'].get(name)
        got = term_value.get(name)
        flag = ''
        if exp is None:
            flag = ' <- not in reference'
        elif got is None:
            flag = ' <- missing in model'
            failures.append(f'objective term {name} missing in model')
        elif not _close(got, exp):
            flag = ' <- MISMATCH'
            failures.append(f'objective term {name}: expected {exp}, got {got}')
        print(f'    {name:<24} reference={exp!r:<12} model={got!r}{flag}')

    _compare_map('x^G', model.result[VarName.DG_INSTALL], ref['xg'], failures)

    xg_type_actual = {_key_str(k): v for k, v in model.result[VarName.DG_INSTALL_TYPE].items()}
    for key, exp in ref['xg_type'].items():
        if xg_type_actual.get(key) != exp:
            failures.append(f'x^GT: {key} expected {exp}, got {xg_type_actual.get(key)}')
    for key, got in xg_type_actual.items():
        if key not in ref['xg_type'] and got != 0:
            failures.append(f'x^GT: {key} expected 0, got {got}')

    _compare_map('x^L', model.result[VarName.LINE_HARDEN], ref['xl'], failures)
    _compare_map('y^L', model.result[VarName.LOAD_SHED_RATIO], ref['yl'], failures)
    _compare_map('y^C', model.result[VarName.LINE_CONNECTED], ref['yc'], failures)

    if 'sum_virtual_source' in ref:
        us_actual = model.result[VarName.VIRTUAL_SOURCE_INDICATOR]
        for suffix, exp in ref['sum_virtual_source'].items():
            got = sum(v for k, v in us_actual.items() if _key_str(k).endswith('|' + suffix))
            if got != exp:
                failures.append(f'sum u^S for ({suffix}) expected {exp}, got {got}')

    xg_on = [k for k, v in model.result[VarName.DG_INSTALL].items() if v]
    xgt_on = [_key_str(k) for k, v in model.result[VarName.DG_INSTALL_TYPE].items() if v]
    xl_on = [_key_str(k) for k, v in model.result[VarName.LINE_HARDEN].items() if v]
    yl_nz = {_key_str(k): round(v, 8)
             for k, v in model.result[VarName.LOAD_SHED_RATIO].items() if abs(v) > 1e-8}
    yc_all = {_key_str(k): v for k, v in model.result[VarName.LINE_CONNECTED].items()}
    us_sum = sum(model.result[VarName.VIRTUAL_SOURCE_INDICATOR].values())

    print(f'  x^G = 1 (model)      : {xg_on}')
    print(f'  x^GT = 1 (model)     : {xgt_on}')
    print(f'  x^L = 1 (model)      : {xl_on}')
    print(f'  y^L != 0 (model)     : {yl_nz}')
    print(f'  y^C (model)          : {yc_all}')
    print(f'  sum u^S (model)      : {us_sum}')

    if failures:
        print(f'  RESULT               : FAIL ({len(failures)} deviation(s))')
        for msg in failures:
            print(f'      * {msg}')
    else:
        print('  RESULT               : PASS')

    return {
        'instance': instance_name,
        'status': str(model.solve_status),
        'objective_reference': ref['objective'],
        'objective_model': obj_value,
        'objective_terms_model': term_value,
        'x_g': {_key_str(k): v for k, v in model.result[VarName.DG_INSTALL].items()},
        'x_gt': {_key_str(k): v for k, v in model.result[VarName.DG_INSTALL_TYPE].items()},
        'x_l': {_key_str(k): v for k, v in model.result[VarName.LINE_HARDEN].items()},
        'y_l': {_key_str(k): v for k, v in model.result[VarName.LOAD_SHED_RATIO].items()},
        'y_c': {_key_str(k): v for k, v in model.result[VarName.LINE_CONNECTED].items()},
        'failures': failures,
    }


def run_micro_bench_test(local_file_path=None, instance_name_list=None, report_path=None,
                         solver_name='gurobi_direct', fallback_solver_name=None, time_limit=120,
                         tee=False):
    if local_file_path is None:
        local_file_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'test_local_csv_file'))

    micro_root = os.path.join(local_file_path, MICRO_BENCH_DIR_NAME)
    if instance_name_list is None:
        instance_name_list = sorted(
            name for name in os.listdir(micro_root)
            if os.path.isdir(os.path.join(micro_root, name))
            and os.path.exists(os.path.join(micro_root, name, 'reference.json'))
        )

    # mip_gap = 0: these instances pin an exact objective, so a gap would blur the comparison.
    solver_config = SolverConfig(
        solver_name=solver_name,
        fallback_solver_name=fallback_solver_name,
        mip_gap=0.0,
        time_limit=time_limit,
        tee=tee,
    )

    print('=' * 88)
    print('micro_bench correctness test (combined / extensive-form model)')
    print(f'data folder : {micro_root}')
    print(f'instances   : {instance_name_list}')
    print(f'solver      : {solver_config.solver_name}, MIPGap={solver_config.mip_gap}')
    print('=' * 88)

    reports, failed = [], []
    for instance_name in instance_name_list:
        try:
            report = run_one_instance(local_file_path, instance_name, solver_config)
        except Exception as exc:  # keep going so one broken instance does not hide the others
            print(f'\n--- {instance_name} ---')
            print(f'  RESULT               : ERROR {exc}')
            traceback.print_exc()
            report = {'instance': instance_name, 'error': f'{type(exc).__name__}: {exc}'}
        reports.append(report)
        if report.get('failures') or report.get('error'):
            failed.append(instance_name)

    print('\n' + '=' * 88)
    print('SUMMARY')
    for report in reports:
        if report.get('error'):
            print(f'  {report["instance"]:<34} ERROR  {report["error"]}')
        else:
            mark = 'FAIL' if report['failures'] else 'PASS'
            print(f'  {report["instance"]:<34} {mark}   '
                  f'reference={report["objective_reference"]}  model={report["objective_model"]}')
    print(f'  {len(reports) - len(failed)}/{len(reports)} instances matched the reference solution.')
    print('=' * 88)

    if report_path is None:
        report_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'output', 'micro_bench_report.json'))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, default=str)
    print(f'full report written to: {report_path}')

    return reports


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', default='gurobi_direct',
                        help='Pyomo solver name, e.g. gurobi_direct, appsi_highs, cbc, glpk')
    parser.add_argument('--fallback-solver', default=None)
    parser.add_argument('--instances', default=None,
                        help='comma-separated instance folder names; default is every instance')
    parser.add_argument('--time-limit', type=float, default=120)
    parser.add_argument('--report-path', default=None)
    parser.add_argument('--tee', action='store_true')
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    init_logger(enable_file_output=False)
    logging.getLogger().setLevel(logging.WARNING)

    reports = run_micro_bench_test(
        instance_name_list=(
            [part.strip() for part in args.instances.split(',') if part.strip()]
            if args.instances else None
        ),
        report_path=args.report_path,
        solver_name=args.solver,
        fallback_solver_name=args.fallback_solver,
        time_limit=args.time_limit,
        tee=args.tee,
    )
    sys.exit(1 if any(r.get('failures') or r.get('error') for r in reports) else 0)
