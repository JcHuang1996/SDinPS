# -*- coding: utf-8 -*-
# @Time     : 2026/06/23
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import itertools
import math
import os
import sys
from types import SimpleNamespace

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
    import pyomo.environ as pyo
    from pyomo.opt import TerminationCondition

    from dao.data_processor import DataProcessor
    from dao.data_reader import DataReader
    from algo.interfaces import (
        CutOutcome,
        CutRequest,
        DualInfo,
        GeneratedCut,
        IterationContext,
        MainSolution,
        ScenarioGroup,
        SubProblemResult,
    )
    from algo.two_stage_decomp_redo import TwoStageDecompRedo
    from algo.algo_simple_tools import CutRepetitionTracker
    from algo.algo_simple_tools import extract_affine_terms_over_vars
    from algo.runner_helpers import (
        IncumbentTracker,
        summarize_incumbent_records,
        validate_bound_mode,
    )
    from model.model_base import ModelBase, ModelSolveError
    from model.model_cglp import ModelCGLP
    from model.model_combined import ModelCombined
    from model.model_lagrangian import ModelLagrangianMultiplierHeuristic
    from model.model_main import ModelMain
    from model.model_sub import ModelSub
    from model.solver_config import SolverConfig, SolverConfigSet
    from util.names import (
        ConstrName,
        CutTypeName,
        DataName,
        DECOMPOSITION_STATE_VAR_NAMES,
        InputMethodName,
        MainProblemModelTypeName,
        ModelStatus,
        SolverRoleName,
        VarName,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        'This script requires a Python environment with `pyomo` installed. '
        'In this project, `conda run -n DailyOpt python unit_test/test_scripts/model_build_smoke_test.py` works.'
    ) from exc


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_FILE_PATH = os.path.join(PROJECT_ROOT, 'unit_test', 'test_local_csv_file')
DATA_SET_NAME = 'function_test_fixed_rated_p'
SCENARIOS = ['s_1', 's_2']
TIMES = [1, 2]


class DummySolver:

    def __init__(self, termination_condition=TerminationCondition.optimal, problem_lower_bound=None,
                 solution_count=0, solve_exception=None):
        self.options = {}
        self.solve_calls = []
        self.termination_condition = termination_condition
        self.problem_lower_bound = problem_lower_bound
        self.solution_count = solution_count
        self.solve_exception = solve_exception

    def available(self, exception_flag=False):
        return True

    def solve(self, pyomo_model, tee=True, load_solutions=True):
        self.solve_calls.append(
            {
                'model': pyomo_model,
                'tee': tee,
                'load_solutions': load_solutions,
                'options': self.options.copy(),
            }
        )
        if self.solve_exception is not None:
            raise self.solve_exception
        return SimpleNamespace(
            solver=SimpleNamespace(termination_condition=self.termination_condition),
            problem=SimpleNamespace(
                lower_bound=self.problem_lower_bound,
                upper_bound=None,
            ),
            solution=[None] * self.solution_count,
        )


def _read_raw_data():
    reader = DataReader(
        read_method=InputMethodName.LOCAL_CSV,
        local_file_path=TEST_FILE_PATH,
        data_set_name=DATA_SET_NAME,
    )
    return reader.read()


def _process_data(raw_data, main_problem_model_type=MainProblemModelTypeName.COLLAPSED_NO_TIME):
    processor = DataProcessor(raw_data=raw_data)
    processor.main_problem_model_type = main_problem_model_type
    return processor.data_process(
        scenario_list_assigned=SCENARIOS,
        time_list_assigned=TIMES,
    )


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label}: expected {expected}, got {actual}')


def _assert_set_equal(actual, expected, label):
    actual_set = set(actual)
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise AssertionError(f'{label}: missing {missing}, extra {extra}')


def _assert_objective(model, label):
    obj = model.model.find_component(model._objective_name)
    if obj is None or not obj.active:
        raise AssertionError(f'{label}: objective was not created.')


def _assert_var_bounds(var_ob, lb, ub, label):
    actual_lb = None if var_ob.lb is None else pyo.value(var_ob.lb)
    actual_ub = None if var_ob.ub is None else pyo.value(var_ob.ub)
    if actual_lb != lb or actual_ub != ub:
        raise AssertionError(f'{label}: expected bounds {(lb, ub)}, got {(actual_lb, actual_ub)}')


def _line_time_scenario_keys(data):
    return {
        (i, j, t, s)
        for (i, j) in data[DataName.LIST_LINE]
        for t in data[DataName.LIST_TIME]
        for s in data[DataName.LIST_SCENARIO]
    }


def _node_time_scenario_keys(data):
    return {
        (j, t, s)
        for j in data[DataName.LIST_NODE]
        for t in data[DataName.LIST_TIME]
        for s in data[DataName.LIST_SCENARIO]
    }


def _expected_total_power(data):
    return max(
        sum(
            data[DataName.DICT_DEMAND_ACTIVE][j, t, s]
            for j in data[DataName.LIST_NODE]
        )
        for t in data[DataName.LIST_TIME]
        for s in data[DataName.LIST_SCENARIO]
    )


def _assert_processed_data(data, expected_scenarios=SCENARIOS, expected_times=TIMES):
    required_keys = {
        DataName.LIST_NODE,
        DataName.LIST_LINE,
        DataName.LIST_TIME,
        DataName.LIST_SCENARIO,
        DataName.LIST_DG_TYPE,
        DataName.NUM_DG_UB,
        DataName.DICT_DEMAND_ACTIVE,
        DataName.DICT_DEMAND_REACTIVE,
        DataName.DICT_LINE_HEALTHY_NH,
        DataName.DICT_LINE_HEALTHY_H,
        DataName.DICT_SC_PROB,
        DataName.NUM_TOTAL_POWER,
        DataName.SLACK_NODE_IDX,
    }
    missing_keys = sorted(required_keys - set(data.keys()))
    if missing_keys:
        raise AssertionError(f'Processed data missing required keys: {missing_keys}')

    _assert_set_equal(data[DataName.LIST_SCENARIO], expected_scenarios, 'scenario list')
    _assert_set_equal(data[DataName.LIST_TIME], expected_times, 'time list')
    _assert_set_equal(data[DataName.DICT_SC_PROB].keys(), expected_scenarios, 'scenario probability keys')
    if not math.isclose(sum(data[DataName.DICT_SC_PROB].values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise AssertionError('Scenario probabilities are not normalized over the selected scenario list.')

    demand_keys = _node_time_scenario_keys(data)
    _assert_set_equal(data[DataName.DICT_DEMAND_ACTIVE].keys(), demand_keys, 'active-demand keys')
    _assert_set_equal(data[DataName.DICT_DEMAND_REACTIVE].keys(), demand_keys, 'reactive-demand keys')

    line_state_keys = _line_time_scenario_keys(data)
    _assert_set_equal(data[DataName.DICT_LINE_HEALTHY_NH].keys(), line_state_keys, 'non-hardened line-state keys')
    _assert_set_equal(data[DataName.DICT_LINE_HEALTHY_H].keys(), line_state_keys, 'hardened line-state keys')

    if not math.isclose(data[DataName.NUM_TOTAL_POWER], _expected_total_power(data), rel_tol=0, abs_tol=1e-9):
        raise AssertionError('NUM_TOTAL_POWER does not match the selected scenario/time demand maximum.')


def _count_vars(model):
    return sum(len(var_dict) for var_dict in model.var.values())


def _count_constraints(model):
    return len(model.model._constr_list)


def _dim(data):
    return (
        len(data[DataName.LIST_NODE]),
        len(data[DataName.LIST_LINE]),
        len(data[DataName.LIST_TIME]),
        len(data[DataName.LIST_SCENARIO]),
        len(data[DataName.LIST_DG_TYPE]),
    )


def _expected_combined_var_count(data):
    n, line_count, time_count, scenario_count, type_count = _dim(data)
    time_scenario_count = time_count * scenario_count
    return 2 * n + n * type_count + line_count + 4 * line_count * time_scenario_count + 6 * n * time_scenario_count


def _expected_combined_constr_count(data):
    n, line_count, time_count, scenario_count, _ = _dim(data)
    time_scenario_count = time_count * scenario_count
    return 1 + 2 * n + 8 * n * time_scenario_count + 15 * line_count * time_scenario_count + 2 * time_scenario_count


def _expected_main_var_count(data, model_type):
    n, line_count, time_count, scenario_count, type_count = _dim(data)
    if model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
        return 8 * n + n * type_count + 5 * line_count + scenario_count
    if model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
        return 2 * n + n * type_count + line_count + 4 * line_count * time_count + 6 * n * time_count + scenario_count
    if model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
        return 2 * n + n * type_count + line_count + 4 * line_count * scenario_count + 6 * n * scenario_count + scenario_count
    raise AssertionError(f'Unsupported model type in smoke test: {model_type}')


def _expected_main_constr_count(data, model_type):
    n, line_count, time_count, scenario_count, _ = _dim(data)
    if model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
        return 3 + 10 * n + 15 * line_count
    if model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
        return 1 + 2 * n + 8 * n * time_count + 15 * line_count * time_count + 2 * time_count
    if model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
        return 1 + 2 * n + 8 * n * scenario_count + 15 * line_count * scenario_count + 2 * scenario_count
    raise AssertionError(f'Unsupported model type in smoke test: {model_type}')


def _expected_sub_redo_constr_count(data, main_result):
    n, line_count, time_count, scenario_count, _ = _dim(data)
    time_scenario_count = time_count * scenario_count
    local_copy_count = sum(len(var_dict) for var_dict in main_result.values())
    return n + 8 * n * time_scenario_count + 15 * line_count * time_scenario_count + 2 * time_scenario_count + local_copy_count


def _zero_main_result(data):
    return {
        VarName.DG_INSTALL: {
            j: 0
            for j in data[DataName.LIST_NODE]
        },
        VarName.LINE_HARDEN: {
            (i, j): 0
            for (i, j) in data[DataName.LIST_LINE]
        },
        VarName.DG_INSTALL_TYPE: {
            (j, typ): 0
            for j in data[DataName.LIST_NODE]
            for typ in data[DataName.LIST_DG_TYPE]
        },
    }


def _assert_common_design_vars(model, data):
    _assert_set_equal(model.var[VarName.DG_INSTALL].keys(), data[DataName.LIST_NODE], VarName.DG_INSTALL)
    _assert_set_equal(model.var[VarName.LINE_HARDEN].keys(), data[DataName.LIST_LINE], VarName.LINE_HARDEN)
    _assert_set_equal(
        model.var[VarName.DG_INSTALL_TYPE].keys(),
        itertools.product(data[DataName.LIST_NODE], data[DataName.LIST_DG_TYPE]),
        VarName.DG_INSTALL_TYPE,
    )

    sample_node = data[DataName.LIST_NODE][0]
    sample_line = data[DataName.LIST_LINE][0]
    sample_type = data[DataName.LIST_DG_TYPE][0]
    if not model.var[VarName.DG_INSTALL][sample_node].is_binary():
        raise AssertionError(f'{VarName.DG_INSTALL} sample variable is not binary.')
    if not model.var[VarName.LINE_HARDEN][sample_line].is_binary():
        raise AssertionError(f'{VarName.LINE_HARDEN} sample variable is not binary.')
    if not model.var[VarName.DG_INSTALL_TYPE][sample_node, sample_type].is_binary():
        raise AssertionError(f'{VarName.DG_INSTALL_TYPE} sample variable is not binary.')


def _assert_combined_model(data):
    model = ModelCombined(model_name='combined_build_smoke', model_data=data)
    model.build_model_all_obj_terms()

    _assert_processed_data(data)
    _assert_common_design_vars(model, data)
    _assert_set_equal(model.var[VarName.LINE_CONNECTED].keys(), _line_time_scenario_keys(data), VarName.LINE_CONNECTED)
    _assert_set_equal(model.var[VarName.LOAD_SHED_RATIO].keys(), _node_time_scenario_keys(data), VarName.LOAD_SHED_RATIO)

    sample_key = next(iter(model.var[VarName.LOAD_SHED_RATIO].keys()))
    _assert_var_bounds(model.var[VarName.LOAD_SHED_RATIO][sample_key], 0, 1, VarName.LOAD_SHED_RATIO)

    _assert_equal(_count_vars(model), _expected_combined_var_count(data), 'combined variable count')
    _assert_equal(_count_constraints(model), _expected_combined_constr_count(data), 'combined constraint count')
    _assert_objective(model, 'combined model')


def _assert_main_model(data, model_type):
    model = ModelMain(model_name=f'main_build_smoke_{model_type}', model_data=data)
    model.build_main_model()

    _assert_common_design_vars(model, data)
    _assert_set_equal(model.var[VarName.SUB_OBJ_EST].keys(), data[DataName.LIST_SCENARIO], VarName.SUB_OBJ_EST)

    if model_type == MainProblemModelTypeName.COLLAPSED_NO_TIME:
        _assert_set_equal(model.var[VarName.MAIN_LINE_CONNECTED].keys(), data[DataName.LIST_LINE], VarName.MAIN_LINE_CONNECTED)
        _assert_set_equal(model.var[VarName.MAIN_LOAD_SHED_RATIO].keys(), data[DataName.LIST_NODE], VarName.MAIN_LOAD_SHED_RATIO)
    elif model_type == MainProblemModelTypeName.COLLAPSED_BY_TIME:
        _assert_set_equal(
            model.var[VarName.MAIN_LINE_CONNECTED].keys(),
            {
                (i, j, t)
                for (i, j) in data[DataName.LIST_LINE]
                for t in data[DataName.LIST_TIME]
            },
            VarName.MAIN_LINE_CONNECTED,
        )
    elif model_type == MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS:
        _assert_set_equal(
            model.var[VarName.MAIN_LINE_CONNECTED].keys(),
            {
                (i, j, s)
                for (i, j) in data[DataName.LIST_LINE]
                for s in data[DataName.LIST_SCENARIO]
            },
            VarName.MAIN_LINE_CONNECTED,
        )

    _assert_equal(_count_vars(model), _expected_main_var_count(data, model_type), f'{model_type} variable count')
    _assert_equal(_count_constraints(model), _expected_main_constr_count(data, model_type), f'{model_type} constraint count')
    _assert_objective(model, f'{model_type} main model')

    model.set_added_obj_term_weight(2.0)
    try:
        model.get_iteration_bound_value()
    except ValueError:
        pass
    else:
        raise AssertionError('A guidance-objective solve must not expose an iteration lower bound.')
    model.set_added_obj_term_weight(0.0)

    if ConstrName.DG_UPPERBOUND not in model.get_decomposition_structural_constr_name_list():
        raise AssertionError('Main model did not record the design structural constraint list.')


def _assert_sub_model_redo(data):
    main_result = _zero_main_result(data)
    model = ModelSub(
        model_name='sub_build_smoke',
        model_data=data,
        main_result=main_result,
    )
    model.build_sub_model_canonical()

    _assert_common_design_vars(model, data)
    _assert_set_equal(model.var[VarName.LINE_CONNECTED].keys(), _line_time_scenario_keys(data), VarName.LINE_CONNECTED)
    _assert_equal(_count_vars(model), _expected_combined_var_count(data), 'subproblem variable count')
    _assert_equal(_count_constraints(model), _expected_sub_redo_constr_count(data, main_result), 'subproblem constraint count')
    _assert_objective(model, 'subproblem')

    expected_local_copy_count = sum(len(var_dict) for var_dict in main_result.values())
    _assert_equal(len(model.local_copy_constr_info), expected_local_copy_count, 'local-copy constraint count')

    for constr_name, (var_name, var_key) in model.local_copy_constr_info.items():
        if constr_name not in model._constr_name_map:
            raise AssertionError(f'Local-copy constraint is missing from the constraint map: {constr_name}')
        if var_name not in main_result or var_key not in main_result[var_name]:
            raise AssertionError(f'Local-copy map points to an unknown main-result key: {constr_name}')

    sample_constr_name = sorted(model.local_copy_constr_info.keys())[0]
    sample_constr = model.get_constr_item_by_name(sample_constr_name)
    if sample_constr.lower is None or sample_constr.upper is None:
        raise AssertionError('Local-copy constraint is not an equality row.')
    if not math.isclose(pyo.value(sample_constr.lower), pyo.value(sample_constr.upper), abs_tol=1e-9):
        raise AssertionError('Local-copy equality row has mismatched lower/upper bounds.')

    model.model_relax = model.model.clone()
    pyo.TransformationFactory('core.relax_integer_vars').apply_to(model.model_relax)
    model.model_relax.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    for constr_name in sorted(model.local_copy_constr_info.keys())[:5]:
        model.get_constr_item_in_relax_by_name(constr_name)
    if not hasattr(model.model_relax, 'dual'):
        raise AssertionError('Relaxed subproblem clone did not get a dual suffix.')


def _assert_cglp_build():
    main_model = ModelBase(model_name='cglp_dummy_main')
    main_model.var['x'] = {
        1: main_model.add_var(domain=pyo.Binary, name='x_1'),
        2: main_model.add_var(domain=pyo.Binary, name='x_2'),
    }
    main_model.var[VarName.SUB_OBJ_EST] = {
        0: main_model.add_var(domain=pyo.Reals, lb=0, name='theta_0'),
    }
    main_model.set_objective(0.0, sense=pyo.minimize)

    cglp_model = ModelCGLP(
        main_model=main_model,
        x_var_keys=[('x', 1), ('x', 2)],
        z_var_keys=[],
        structural_constr_names=[],
        theta_keys=[0],
        lower_bound_rows=[{'name': 'lb0', 'x_coef': {}, 'rhs': 0.0}],
        model_name='cglp_build_smoke',
        exact_point_data=[((1, 0), 2.0)],
    )
    cglp_model.build_model()

    required_var_blocks = {'alpha', 'eta', 'sigma', 'rho', 'varphi', 'nu', 'psi'}
    missing_var_blocks = sorted(required_var_blocks - set(cglp_model.var.keys()))
    if missing_var_blocks:
        raise AssertionError(f'CGLP missing variable blocks: {missing_var_blocks}')
    _assert_equal(len(cglp_model.var['alpha']), 2, 'CGLP alpha variable count')
    _assert_equal(len(cglp_model.var['psi']), 1, 'CGLP psi variable count')
    _assert_equal(_count_constraints(cglp_model), 9, 'CGLP constraint count')
    _assert_objective(cglp_model, 'CGLP model')


def test_solver_config_and_param_dict():
    model = ModelBase(
        model_name='solver_config_param_smoke',
        solver_config=SolverConfig(mip_gap=0.1, time_limit=12, tee=False),
    )
    dummy_solver = DummySolver()
    model.solver = dummy_solver
    model.set_objective(0.0, sense=pyo.minimize)
    model.solve(param_dict={'MIPGap': 0.005, 'TimeLimit': 9, 'CustomFlag': 1})

    _assert_equal(dummy_solver.options['MIPGap'], 0.005, 'solve param_dict MIPGap')
    _assert_equal(dummy_solver.options['TimeLimit'], 9, 'solve param_dict TimeLimit')
    _assert_equal(dummy_solver.options['CustomFlag'], 1, 'solve param_dict custom option')
    _assert_equal(dummy_solver.solve_calls[0]['tee'], False, 'solver tee option')

    model_with_override = ModelBase(
        model_name='solver_config_override_smoke',
        solver_config=SolverConfig(mip_gap=0.1, time_limit=12, tee=False),
    )
    dummy_solver = DummySolver()
    model_with_override.solver = dummy_solver
    model_with_override.set_objective(0.0, sense=pyo.minimize)
    model_with_override.set_parameters({'MIPGap': 0.03})
    model_with_override.solve()

    _assert_equal(dummy_solver.options['MIPGap'], 0.03, 'set_parameters override MIPGap')
    _assert_equal(dummy_solver.options['TimeLimit'], 12, 'configured TimeLimit default')

    highs_model = ModelBase(
        model_name='solver_config_highs_smoke',
        solver_config=SolverConfig(solver_name='highs', mip_gap=0.04, time_limit=7, tee=False),
    )
    dummy_solver = DummySolver()
    highs_model.solver = dummy_solver
    highs_model.set_objective(0.0, sense=pyo.minimize)
    highs_model.solve()

    _assert_equal(dummy_solver.options['mip_rel_gap'], 0.04, 'HiGHS mip gap option name')
    _assert_equal(dummy_solver.options['time_limit'], 7, 'HiGHS time limit option name')


def test_data_process_returns_independent_data_objects():
    raw_data = _read_raw_data()
    processor = DataProcessor(raw_data=raw_data)

    first_data = processor.data_process(
        scenario_list_assigned=['s_1'],
        time_list_assigned=TIMES,
    )
    second_data = processor.data_process(
        scenario_list_assigned=['s_2'],
        time_list_assigned=TIMES,
    )
    third_data = processor.data_process(
        scenario_list_assigned=['s_1'],
        time_list_assigned=TIMES,
    )

    if first_data is second_data or first_data is third_data:
        raise AssertionError('DataProcessor.data_process returned the same data object across calls.')

    _assert_processed_data(first_data, expected_scenarios=['s_1'])
    _assert_processed_data(second_data, expected_scenarios=['s_2'])
    _assert_processed_data(third_data, expected_scenarios=['s_1'])

    if first_data[DataName.DICT_SC_PROB] is second_data[DataName.DICT_SC_PROB]:
        raise AssertionError('Scenario probability dictionaries are shared across processed data objects.')

    if first_data[DataName.DICT_DEMAND_ACTIVE] != third_data[DataName.DICT_DEMAND_ACTIVE]:
        raise AssertionError('Repeated processing for the same scenario group is not deterministic.')


def test_algorithm_data_contracts():
    scenario_group = ScenarioGroup(['s_1', 's_2'])
    main_solution = MainSolution(
        values={
            VarName.DG_INSTALL: {1: 1},
            VarName.LINE_HARDEN: {(1, 2): 0},
        },
        name='current',
    )
    fractional_solution = MainSolution(
        values={
            VarName.DG_INSTALL: {1: 0.5},
            VarName.LINE_HARDEN: {(1, 2): 0.0},
        },
        name='fractional',
        is_relaxed=True,
    )
    dual_info = DualInfo(
        dual_values={'local_xg_1': 3.0},
        var_map={'local_xg_1': (VarName.DG_INSTALL, 1)},
        local_copy_constraint_names=['local_xg_1'],
        source='current',
    )
    fractional_dual_info = DualInfo(
        dual_values={'local_xg_1': 4.0},
        var_map={'local_xg_1': (VarName.DG_INSTALL, 1)},
        local_copy_constraint_names=['local_xg_1'],
        source='fractional',
    )
    sub_result = SubProblemResult(
        scenario_group=scenario_group,
        main_solution=main_solution,
        relaxed_obj_value=12.0,
        benders_dual=dual_info,
        exact_obj_value=15.0,
        exact_obj_bound=14.5,
        fractional_solution=fractional_solution,
        fractional_relaxed_obj_value=13.0,
        fractional_dual=fractional_dual_info,
    )

    _assert_equal(scenario_group.key, ('s_1', 's_2'), 'ScenarioGroup key')
    _assert_equal(scenario_group.as_list(), ['s_1', 's_2'], 'ScenarioGroup list conversion')
    if not math.isclose(scenario_group.probability_weight({'s_1': 0.25, 's_2': 0.75}), 1.0):
        raise AssertionError('ScenarioGroup probability weight is incorrect.')

    context = IterationContext(
        ite_name='0',
        iteration_number=0,
        current_solution=main_solution,
        fractional_solution=fractional_solution,
        candidate_cut_types=['benders', 'strengthen_benders'],
    )
    request = CutRequest(
        scenario_group=scenario_group,
        ite_name=context.ite_name,
        anchor_solution=main_solution,
        dual_info=dual_info,
        relaxed_obj=sub_result.relaxed_obj_value,
    )
    _assert_equal(context.candidate_cut_types, ('benders', 'strengthen_benders'), 'IterationContext cut types')
    _assert_equal(request.dual_info.constr_var_map, dual_info.var_map, 'CutRequest dual map')
    _assert_equal(
        sub_result.local_copy_constraint_names,
        ('local_xg_1',),
        'SubProblemResult local-copy constraint names',
    )
    _assert_equal(
        sub_result.local_copy_var_map,
        {'local_xg_1': (VarName.DG_INSTALL, 1)},
        'SubProblemResult local-copy var map',
    )
    _assert_equal(
        DECOMPOSITION_STATE_VAR_NAMES,
        (VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE),
        'canonical decomposition state variable list',
    )

    generated_cut = GeneratedCut(
        cut_type='benders',
        cut_name='bd_s_0',
        scenario_group=scenario_group,
        ite_name='0',
        lhs='lhs',
        rhs='rhs',
        lower_bound_row={'name': 'bd_s_0', 'x_coef': {}, 'rhs': 0.0},
    )
    _assert_equal(generated_cut.cut_type, 'benders', 'GeneratedCut cut type')


def test_cut_generation_registry_build_only():
    main_model = ModelBase(model_name='cut_registry_dummy_main')
    main_model.var[VarName.SUB_OBJ_EST] = {
        's_1': main_model.add_var(domain=pyo.Reals, lb=0, name='theta_s_1'),
    }
    main_model.var[VarName.DG_INSTALL] = {
        'node_1': main_model.add_var(domain=pyo.Binary, name='xg_node_1'),
    }
    main_model.set_objective(0.0, sense=pyo.minimize)

    def _add_custom_cut(cut_name, cut_lhs, cut_rhs):
        main_model.add_constr(cut_lhs >= cut_rhs, name=cut_name)

    main_model.add_custom_cut = _add_custom_cut
    # The cut left-hand side is sum_{s in G} P_s * theta_s; this stub keeps P_s = 1 so the expected
    # row numbers below stay readable. The weighted case is covered by
    # test_group_theta_lhs_uses_master_weights.
    main_model.get_theta_weight = lambda scenario: 1.0
    main_model.get_weighted_theta_expr = lambda scenario_list: pyo.quicksum(
        main_model.get_theta_weight(s) * main_model.var[VarName.SUB_OBJ_EST][s]
        for s in scenario_list
    )

    decomp = TwoStageDecompRedo.__new__(TwoStageDecompRedo)
    decomp.model_main = main_model
    decomp.cglp_lower_bound_row_dict = {}
    decomp.cut_repetition_tracker = CutRepetitionTracker()
    decomp.cut_metadata_records = []
    decomp.cut_generator_registry = {}
    decomp.enable_cut_algebra_record = False
    decomp.cglp_x_var_key_list = [(VarName.DG_INSTALL, 'node_1')]
    TwoStageDecompRedo._register_cut_generators(decomp)

    forward_sol = {VarName.DG_INSTALL: {'node_1': 1}}
    dual_values = {'local_xg_node_1': 2.0}
    var_map = {'local_xg_node_1': (VarName.DG_INSTALL, 'node_1')}

    first_outcome, first_cut = decomp.dispatch_cut(
        CutTypeName.BENDERS,
        CutRequest(
            scenario_group=ScenarioGroup(['s_1']),
            ite_name='0',
            anchor_solution=MainSolution(forward_sol),
            dual_info=DualInfo(dual_values, var_map),
            relaxed_obj=5.0,
        ),
    )
    _assert_equal(first_outcome, CutOutcome.ADDED, 'registry first cut outcome')
    _assert_equal(first_cut.cut_name, "bd_s_['s_1']_0", 'registry Benders cut name')
    _assert_equal(
        len(decomp.cglp_lower_bound_row_dict[('s_1',)]),
        2,
        'CGLP lower-bound rows after first registry insert',
    )
    row = decomp.cglp_lower_bound_row_dict[('s_1',)][1]
    _assert_equal(row['x_coef'][(VarName.DG_INSTALL, 'node_1')], 2.0, 'registry row coefficient')
    _assert_equal(row['rhs'], -3.0, 'registry row rhs')
    _assert_equal(len(decomp.cut_metadata_records), 1, 'registry cut metadata count')

    second_outcome, second_cut = decomp.dispatch_cut(
        CutTypeName.BENDERS,
        CutRequest(
            scenario_group=ScenarioGroup(['s_1']),
            ite_name='1',
            anchor_solution=MainSolution(forward_sol),
            dual_info=DualInfo(dual_values, var_map),
            relaxed_obj=5.0,
        ),
    )
    _assert_equal(second_outcome, CutOutcome.REPEATED, 'registry repeated cut outcome')
    _assert_equal(second_cut.cut_name, "bd_s_['s_1']_1", 'registry repeated Benders cut name')
    _assert_equal(len(decomp.get_repetitive_cut_records()), 1, 'registry repeated-cut count')

    not_applicable, missing_cut = decomp.dispatch_cut(
        CutTypeName.BENDERS,
        CutRequest(
            scenario_group=ScenarioGroup(['s_1']),
            ite_name='2',
            anchor_solution=MainSolution(forward_sol),
        ),
    )
    _assert_equal(not_applicable, CutOutcome.NOT_APPLICABLE, 'missing cut inputs outcome')
    _assert_equal(missing_cut, None, 'not-applicable request has no generated cut')


def test_lagrangian_inner_approximation_names_are_unique():
    """Each multiplier-master iteration must append a distinctly named approximation row."""
    main_result = {'x': {0: 1.0}}
    constr_dual_info = {'local_x_0': 0.0}
    constr_var_map = {'local_x_0': ('x', 0)}
    model = ModelLagrangianMultiplierHeuristic(
        model_name='lagrangian_name_smoke',
        main_result=main_result,
        constr_dual_info=constr_dual_info,
        constr_var_map=constr_var_map,
    )
    model.build_ini_model(
        ini_multiplier=constr_dual_info,
        ini_stabilization_param=0.01,
    )

    model.add_constr_for_regularized_model(sub_obj_value=1.0, main_sol={'x': {0: 0.0}})
    model.add_constr_for_regularized_model(sub_obj_value=2.0, main_sol={'x': {0: 1.0}})

    _assert_equal(
        sorted(model._constr_name_map),
        ['inner_approx_2', 'inner_approx_3'],
        'Lagrangian approximation row names',
    )
    _assert_equal(_count_constraints(model), 2, 'Lagrangian approximation row count')


# =========================================================================================
# P0-P3 regression checks that need no solver
# =========================================================================================

def test_solver_config_set_roles():
    """Every role gets its own config, and the LP roles never carry a MIP gap."""
    config_set = SolverConfigSet.build(
        solver_name='gurobi_direct',
        master_mip_gap=0.01,
        master_time_limit=30,
        subproblem_exact_mip_gap=0.0,
        inner_min_mip_gap=0.0,
        inner_min_time_limit=5,
    )

    _assert_set_equal(config_set.as_dict().keys(), SolverRoleName.ALL_ROLES, 'solver role coverage')
    _assert_equal(config_set.get(SolverRoleName.MASTER).mip_gap, 0.01, 'master mip gap')
    _assert_equal(config_set.get(SolverRoleName.MASTER).time_limit, 30, 'master time limit')
    _assert_equal(config_set.get(SolverRoleName.SUBPROBLEM_EXACT).mip_gap, 0.0, 'exact subproblem mip gap')
    _assert_equal(config_set.get(SolverRoleName.INNER_MIN).time_limit, 5, 'inner min time limit')
    _assert_equal(config_set.get(SolverRoleName.SUBPROBLEM_LP).mip_gap, None, 'LP role carries no mip gap')
    _assert_equal(config_set.get(SolverRoleName.CGLP).mip_gap, None, 'CGLP role carries no mip gap')

    try:
        config_set.get('not_a_role')
    except ValueError:
        pass
    else:
        raise AssertionError('SolverConfigSet.get accepted an unknown role.')


def test_solve_option_merge_and_override_scope():
    """param_dict is a per-call override merged on top of the config, and never persists."""
    model = ModelBase(
        model_name='option_scope_smoke',
        solver_config=SolverConfig(mip_gap=0.1, time_limit=12, tee=False),
    )
    dummy_solver = DummySolver()
    model.solver = dummy_solver
    model.set_objective(0.0, sense=pyo.minimize)

    model.solve(param_dict={'MIPGap': 0.005})
    _assert_equal(dummy_solver.solve_calls[0]['options']['MIPGap'], 0.005, 'per-call MIPGap override')
    # The config default must survive a single-option override (previously it was suppressed).
    _assert_equal(dummy_solver.solve_calls[0]['options']['TimeLimit'], 12, 'config TimeLimit survives override')

    model.solve()
    _assert_equal(dummy_solver.solve_calls[1]['options']['MIPGap'], 0.1, 'per-call override does not persist')

    model.set_parameters({'MIPGap': 0.03})
    model.solve()
    _assert_equal(dummy_solver.solve_calls[2]['options']['MIPGap'], 0.03, 'persistent override applies')
    model.clear_solver_overrides()
    model.solve()
    _assert_equal(dummy_solver.solve_calls[3]['options']['MIPGap'], 0.1, 'clear_solver_overrides restores config')


def test_fallback_solver_is_scoped_to_one_call():
    model = ModelBase(
        model_name='fallback_scope_smoke',
        solver_config=SolverConfig(
            solver_name='gurobi_direct',
            fallback_solver_name='appsi_highs',
            mip_gap=0.1,
            tee=False,
        ),
    )
    model.set_objective(0.0, sense=pyo.minimize)
    primary = DummySolver(solve_exception=RuntimeError('primary failed'))
    primary.options['preexisting'] = 7
    fallback = DummySolver()
    model.solver_name = 'gurobi_direct'
    model.solver = primary

    def switch_solver(solver_name):
        model.solver_name = solver_name
        model.solver = fallback
        model._applied_solver_option_keys = set()

    model._switch_solver = switch_solver
    model.solve()

    if model.solver is not primary or model.solver_name != 'gurobi_direct':
        raise AssertionError('Fallback solver persisted beyond the failed primary solve.')
    _assert_equal(dict(primary.options), {'preexisting': 7}, 'primary solver options restored')


def test_solve_failure_raises_and_bound_reads_dual_value():
    """An unusable solve raises instead of fabricating variable values; get_obj_bound reads the dual bound."""
    model = ModelBase(model_name='solve_failure_smoke', solver_config=SolverConfig(tee=False))
    model.set_objective(0.0, sense=pyo.minimize)
    model.solver = DummySolver(
        termination_condition=TerminationCondition.infeasible,
    )

    try:
        model.solve()
    except ModelSolveError:
        pass
    else:
        raise AssertionError('An infeasible solve must raise ModelSolveError unless allow_failure=True.')

    model.solve(allow_failure=True)
    _assert_equal(model.solve_status, ModelStatus.INFEASIBLE, 'infeasible status recorded')

    bound_model = ModelBase(model_name='bound_smoke', solver_config=SolverConfig(tee=False))
    bound_model.set_objective(0.0, sense=pyo.minimize)
    bound_model.solver = DummySolver(problem_lower_bound=41.0)
    bound_model.solve()
    _assert_equal(bound_model.get_obj_bound(), 41.0, 'get_obj_bound reads the reported dual bound')


def test_group_theta_lhs_uses_master_weights():
    """A group cut must be written on sum_{s in G} P_s * theta_s, for any grouping (risk R3.4)."""
    main_model = ModelBase(model_name='theta_weight_dummy_main')
    main_model.var[VarName.SUB_OBJ_EST] = {
        's_1': main_model.add_var(domain=pyo.Reals, lb=0, name='theta_s_1'),
        's_2': main_model.add_var(domain=pyo.Reals, lb=0, name='theta_s_2'),
    }
    weights = {'s_1': 0.25, 's_2': 0.75}
    main_model.get_theta_weight = lambda scenario: weights[scenario]
    main_model.get_weighted_theta_expr = lambda scenario_list: pyo.quicksum(
        weights[s] * main_model.var[VarName.SUB_OBJ_EST][s] for s in scenario_list
    )

    decomp = TwoStageDecompRedo.__new__(TwoStageDecompRedo)
    decomp.model_main = main_model

    theta_id_map = {
        id(var_ob): scenario
        for scenario, var_ob in main_model.var[VarName.SUB_OBJ_EST].items()
    }
    _, coef = extract_affine_terms_over_vars(decomp.build_group_theta_lhs(['s_1', 's_2']), theta_id_map)

    if not math.isclose(coef['s_1'], 0.25) or not math.isclose(coef['s_2'], 0.75):
        raise AssertionError(f'Group theta left-hand side is not probability weighted: {coef}')


def test_unconditional_scenario_probability():
    """
    DICT_SC_PROB stays conditional on the processed subset; DICT_SC_PROB_UNCOND is normalized over
    the run-level universe, so master and subproblems agree on P_s.
    """
    raw_data = _read_raw_data()

    processor = DataProcessor(raw_data=raw_data)
    processor.reference_scenario_list = SCENARIOS
    single_scenario_data = processor.data_process(
        scenario_list_assigned=[SCENARIOS[0]],
        time_list_assigned=TIMES,
    )

    conditional = single_scenario_data[DataName.DICT_SC_PROB]
    unconditional = single_scenario_data[DataName.DICT_SC_PROB_UNCOND]

    if not math.isclose(conditional[SCENARIOS[0]], 1.0, abs_tol=1e-12):
        raise AssertionError('A single-scenario subset must have conditional probability 1.')
    if unconditional[SCENARIOS[0]] >= 1.0:
        raise AssertionError(
            'The unconditional probability of one scenario out of several must be below 1; '
            f'got {unconditional[SCENARIOS[0]]}.'
        )

    full_processor = DataProcessor(raw_data=raw_data)
    full_processor.reference_scenario_list = SCENARIOS
    full_data = full_processor.data_process(
        scenario_list_assigned=SCENARIOS,
        time_list_assigned=TIMES,
    )
    if not math.isclose(sum(full_data[DataName.DICT_SC_PROB_UNCOND].values()), 1.0, abs_tol=1e-9):
        raise AssertionError('Unconditional probabilities over the whole universe must sum to 1.')
    for scenario in SCENARIOS:
        if not math.isclose(
            full_data[DataName.DICT_SC_PROB][scenario],
            full_data[DataName.DICT_SC_PROB_UNCOND][scenario],
            abs_tol=1e-12,
        ):
            raise AssertionError(
                'On the full scenario universe the conditional and unconditional distributions '
                'must coincide.'
            )

    first_group_data = full_processor.data_process(
        scenario_list_assigned=[SCENARIOS[0]],
        time_list_assigned=TIMES,
    )
    second_group_data = full_processor.data_process(
        scenario_list_assigned=[SCENARIOS[1]],
        time_list_assigned=TIMES,
    )
    expected_big_m = full_data[DataName.NUM_TOTAL_POWER]
    if (
        first_group_data[DataName.NUM_TOTAL_POWER] != expected_big_m
        or second_group_data[DataName.NUM_TOTAL_POWER] != expected_big_m
    ):
        raise AssertionError('Master and scenario subproblems must use the same NUM_TOTAL_POWER.')

    # A scenario outside the declared universe is a configuration error, not a silent rescale.
    bad_processor = DataProcessor(raw_data=raw_data)
    bad_processor.reference_scenario_list = [SCENARIOS[0]]
    try:
        bad_processor.data_process(scenario_list_assigned=SCENARIOS, time_list_assigned=TIMES)
    except ValueError:
        pass
    else:
        raise AssertionError('Scenarios outside the reference universe must be rejected.')


def test_incumbent_summary_and_bound_validity():
    """Incumbent records sum without an extra probability factor; a penalized master has no valid bound."""
    records = [
        {
            'scenario_group': ['s_1'],
            'weighted_sub_obj': 30.0,
            'main_stage_obj': 100.0,
            'group_probability_weight': 0.25,
            'sub_result': {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}},
        },
        {
            'scenario_group': ['s_2'],
            'weighted_sub_obj': 45.0,
            'main_stage_obj': 100.0,
            'group_probability_weight': 0.75,
            'sub_result': {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}},
        },
    ]
    incumbent_obj, _ = summarize_incumbent_records(incumbent_records=records)
    if not math.isclose(incumbent_obj, 175.0):
        raise AssertionError(f'Incumbent objective should be 100 + 30 + 45 = 175, got {incumbent_obj}')

    inconsistent_records = [dict(records[0]), dict(records[1])]
    inconsistent_records[0]['main_stage_obj'] = 0.0
    inconsistent_records[1]['main_stage_obj'] = 1.0
    try:
        summarize_incumbent_records(incumbent_records=inconsistent_records)
    except ValueError:
        pass
    else:
        raise AssertionError('Incumbent records with different design costs must be rejected.')

    tracker = IncumbentTracker()
    tracker.best_obj = 175.0
    if tracker.has_converged(main_bound=None):
        raise AssertionError('Convergence must never be claimed without a valid lower bound.')
    if not tracker.has_converged(main_bound=174.0, rel_gap=0.01):
        raise AssertionError('A valid bound within the gap should report convergence.')

    try:
        validate_bound_mode('not_a_mode')
    except ValueError:
        pass
    else:
        raise AssertionError('validate_bound_mode accepted an unknown mode.')


def test_build_only_power_models():
    raw_data = _read_raw_data()

    no_time_data = _process_data(
        raw_data=raw_data,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_NO_TIME,
    )
    _assert_combined_model(no_time_data)
    _assert_main_model(no_time_data, MainProblemModelTypeName.COLLAPSED_NO_TIME)
    _assert_sub_model_redo(no_time_data)

    by_time_data = _process_data(
        raw_data=raw_data,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_BY_TIME,
    )
    _assert_main_model(by_time_data, MainProblemModelTypeName.COLLAPSED_BY_TIME)

    by_scenario_data = _process_data(
        raw_data=raw_data,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS,
    )
    _assert_main_model(by_scenario_data, MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS)

    _assert_cglp_build()


def main():
    test_solver_config_and_param_dict()
    test_solver_config_set_roles()
    test_solve_option_merge_and_override_scope()
    test_fallback_solver_is_scoped_to_one_call()
    test_solve_failure_raises_and_bound_reads_dual_value()
    test_group_theta_lhs_uses_master_weights()
    test_unconditional_scenario_probability()
    test_incumbent_summary_and_bound_validity()
    test_data_process_returns_independent_data_objects()
    test_algorithm_data_contracts()
    test_cut_generation_registry_build_only()
    test_lagrangian_inner_approximation_names_are_unique()
    test_build_only_power_models()
    print('PASS -> solver configuration and build-only model structure checks completed without solving.')


if __name__ == '__main__':
    main()
