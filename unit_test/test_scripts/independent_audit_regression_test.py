"""Counterexamples and contracts for audit B in docs/development_record.md (test_fN = B-FN; B-F2 deferred)."""

import copy
import itertools
import json
import pickle
import random
from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.repn.standard_repn import generate_standard_repn

from algo.algo_simple_tools import CutRepetitionTracker, flatten_ordered_var_values
from algo.interfaces import CutRequest, MainSolution, ScenarioGroup, StaticSolutionGenerator
from algo.runner import CutSchedule, DecompositionRunner
from algo.two_stage_decomp_redo import TwoStageDecompRedo
from analysis.out_sample_main_stage_eval import load_main_solution_from_result_dir
from dao.data_processor import DataProcessor
from model.model_base import ModelBase, ModelSolveError
from model.model_cglp import ModelCGLP
from model.model_lagrangian import ModelLagrangianMultiplierHeuristic
from model.model_sub import ModelSub
from model.solver_config import SolverConfig, SolverConfigSet
from unit_test.test_scripts import decomp_validity_harness as harness
from util.headers import ParameterKey, ScenarioNodePowerHeader
from util.names import (
    BoundModeName, CutTypeName, DataName, InputDataName, MainProblemModelTypeName,
    ModelStatus, ResultName, SolverRoleName, VarName,
)
from util.tools import enrich_iter_general_dataframe, iter_sub_prob_info


def _config(solver_name):
    return SolverConfigSet.build(solver_name=solver_name, master_mip_gap=0, master_time_limit=None)


def _raw(local_file_path):
    return harness.load_raw_data(local_file_path, 'm1_dg_site_type')


def _design(data, install=True):
    design = harness.enumerate_first_stage_designs(data, max_designs=1)[0]
    if install:
        design[VarName.DG_INSTALL]['node_1'] = 1
        design[VarName.DG_INSTALL_TYPE]['node_1', 'T2'] = 1
    return design


def _sub(data, design, config):
    sub = ModelSub(model_data=data, main_result=design, sub_model_sce_list=['s_1'], solver_config=config)
    sub.build_sub_model_canonical()
    return sub


def test_f1_signature_includes_theta_expression():
    tracker = CutRepetitionTracker()
    row = {'x_coef': {}, 'rhs': -10.0}
    for index, weights in enumerate(({'s_1': .5}, {'s_2': .5}, {'s_1': 1.0}, {'s_1': .5})):
        record = tracker.record_by_row('benders', str(index), str(index), row, theta_weights=weights)
        assert record['is_repeated'] == (index == 3)
    scoped = CutRepetitionTracker()
    scoped.record_by_row('benders', '0', 'a', row, sub_sce_list=['s_1'])
    assert not scoped.peek_by_row('benders', row, sub_sce_list=['s_2'])['is_repeated']
    assert scoped.peek_by_row('benders', row, sub_sce_list=['s_1'])['is_repeated']


@pytest.mark.requires_solver
def test_f1_identical_scenarios_converge(local_file_path, solver_name):
    raw = _raw(local_file_path)
    for name in (InputDataName.SCENARIO_NODE_LOAD_DF, InputDataName.SCENARIO_LINE_STATE_WO_HARDEN_DF):
        other = raw[name].copy()
        other['scenario_id'] = 's_2'
        raw[name] = pd.concat([raw[name], other], ignore_index=True)
    raw[InputDataName.SCENARIO_PROB_DF] = pd.DataFrame(
        {'scenario_id': ['s_1', 's_2'], 'probability': [.5, .5]},
    )
    config = _config(solver_name)
    reference = harness.solve_extensive_form(raw, ['s_1', 's_2'], [1], config.subproblem_exact)
    module, bounds = harness.run_decomposition(
        raw, ['s_1', 's_2'], [1], [['s_1'], ['s_2']], config, 20,
        [CutTypeName.BENDERS, CutTypeName.INTEGER_OPT], 0.0,
        BoundModeName.RESOLVE_AT_ZERO_WEIGHT, MainProblemModelTypeName.BASE,
    )
    assert reference['objective'] == pytest.approx(2870)
    assert bounds[-1]['valid_lower_bound'] == pytest.approx(reference['objective'])
    assert bounds[-1]['incumbent_upper_bound'] == pytest.approx(reference['objective'])
    assert {tuple(r['scenario_group']) for r in module.cut_metadata_records if r['added']} == {
        ('s_1',), ('s_2',),
    }


@pytest.mark.requires_solver
@pytest.mark.parametrize('output', [False, True])
def test_f3_generated_incumbent_design_roundtrip(local_file_path, solver_name, tmp_path, output):
    config = _config(solver_name)
    module = TwoStageDecompRedo(_raw(local_file_path), [1], ['s_1'], solver_config_set=config)
    module.build_main_stage_model()
    candidate = _design(module.main_model_data)
    expected = copy.deepcopy(candidate)
    result = DecompositionRunner(
        module, [['s_1']], 1, cut_schedule=CutSchedule({CutTypeName.BENDERS: (0, 0)}),
        solution_generator=StaticSolutionGenerator([MainSolution(candidate)]),
        enable_result_output=output, output_dir=str(tmp_path),
    ).run()
    assert result[ResultName.BEST_INCUMBENT_OBJ] == pytest.approx(2870)
    candidate[VarName.DG_INSTALL]['node_1'] = 0
    assert result[ResultName.BEST_MAIN_SOLUTION] == expected
    assert '_heur_' in result[ResultName.BEST_INCUMBENT_SOURCE]
    if output:
        loaded, source = load_main_solution_from_result_dir(str(tmp_path), 'best')
        assert loaded == expected
        assert source == result[ResultName.BEST_INCUMBENT_SOURCE]
        sub = _sub(module.main_model_data, loaded, config.subproblem_exact)
        sub.solve()
        actual = sub.get_obj_value() + module.model_main.evaluate_design_objective_for_solution(loaded)
        assert actual == pytest.approx(result[ResultName.BEST_INCUMBENT_OBJ])
    else:
        assert not (tmp_path / ResultName.BEST_INCUMBENT_FILE).exists()


def test_f3_legacy_history_loading_and_unrecoverable_generated_candidate(tmp_path):
    design = {VarName.DG_INSTALL: {'n': 1}, VarName.DG_INSTALL_TYPE: {('n', 'T'): 1}, VarName.LINE_HARDEN: {}}
    (tmp_path / 'main_result_repetition_report.pkl').write_bytes(pickle.dumps({'curr_main_result_history': [design]}))
    csv_path = tmp_path / 'iter_general_info.csv'
    pd.DataFrame([{'ite_num': '0', 'best_incumbent_obj_value': 100}]).to_csv(csv_path, index=False)
    assert load_main_solution_from_result_dir(str(tmp_path), 'best') == (design, 0)
    pd.DataFrame([
        {'ite_num': '0', 'best_incumbent_obj_value': 100},
        {'ite_num': '0_heur_0', 'best_incumbent_obj_value': 50},
    ]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='no saved incumbent design'):
        load_main_solution_from_result_dir(str(tmp_path), 'best')
    assert load_main_solution_from_result_dir(str(tmp_path), 'latest') == (design, 0)


@pytest.mark.requires_solver
def test_f4_reactive_heavy_demand_is_not_clipped(local_file_path, solver_name):
    raw = _raw(local_file_path)
    raw[InputDataName.PARAMETERS_DICT].pop(ParameterKey.TOTAL_POWER)
    raw[InputDataName.PARAMETERS_DICT][ParameterKey.DG_PF_LB] = .5
    loads = raw[InputDataName.SCENARIO_NODE_LOAD_DF]
    loads[ScenarioNodePowerHeader.P_LOAD_MW] = [0., .1, 0.]
    loads[ScenarioNodePowerHeader.Q_LOAD_MW] = [0., .15, 0.]
    data = DataProcessor(raw).data_process(['s_1'], [1])
    assert data[DataName.NUM_TOTAL_POWER] == pytest.approx(.1)
    sub = _sub(data, _design(data), _config(solver_name).subproblem_exact)
    sub.solve()
    assert sub.get_obj_value() == pytest.approx(10.)


@pytest.mark.requires_solver
def test_f4_zero_demand_open_lines_allow_voltage_extrema(local_file_path, solver_name):
    data = DataProcessor(_raw(local_file_path)).data_process(['s_1'], [1])
    data[DataName.NUM_TOTAL_POWER] = 0.
    for name in (DataName.DICT_DEMAND_ACTIVE, DataName.DICT_DEMAND_REACTIVE,
                 DataName.DICT_LINE_HEALTHY_NH, DataName.DICT_LINE_HEALTHY_H):
        data[name] = dict.fromkeys(data[name], 0.)
    sub = _sub(data, _design(data, install=False), _config(solver_name).subproblem_exact)
    sub.var[VarName.BUS_VOLTAGE]['node_2', 1, 's_1'].fix(data[DataName.NUM_VOLTAGE_LB])
    sub.var[VarName.BUS_VOLTAGE]['node_3', 1, 's_1'].fix(data[DataName.NUM_VOLTAGE_UB])
    sub.solve()
    assert sub.get_obj_value() == pytest.approx(0.)


@pytest.mark.requires_solver
def test_f5_repeated_and_improved_cglp_points(solver_name):
    main = ModelBase(solver_config=_config(solver_name).cglp)
    main.var['x'] = {i: main.add_var(domain=pyo.Binary) for i in range(2)}
    main.var[VarName.SUB_OBJ_EST] = {0: main.add_var(lb=0)}
    cg = ModelCGLP(main, [('x', 0), ('x', 1)], [], [], [0],
                   [{'name': 'zero', 'x_coef': {}, 'rhs': 0.}], solver_config=_config(solver_name).cglp)
    cg.build_model()
    table = {(0, 0): 10., (0, 1): 4., (1, 0): 5., (1, 1): 1.}
    for point, value in table.items():
        cg.update_with_exact_point(point, value - .5)
    cg.update_with_exact_point((0, 0), 10.)
    cg.update_with_exact_point((0, 0), 8.)
    assert cg.exact_point_value_dict[0, 0] == 10.
    assert cg.current_tight_point == (0, 0)
    assert len(cg.exact_point_row_name_dict) == 4
    alpha, eta = cg.solve_cglp()
    assert -eta == pytest.approx(10.)
    for point, value in table.items():
        assert -eta - sum(alpha['x', i] * point[i] for i in range(2)) <= value + 1e-7
    cg.sync_lower_bound_rows([
        {'name': 'zero', 'x_coef': {}, 'rhs': 0.}, cg.build_lower_bound_row('generated', alpha, eta),
    ])
    assert len(generate_standard_repn(cg.get_constr_item_by_name('psi_normalization').body).linear_vars) == 2
    for invalid in ((.2, 1), (1,), (float('nan'), 0)):
        with pytest.raises(ValueError, match='binary state'):
            cg.update_with_exact_point(invalid, 10.)


def test_f5_binary_point_keys_round_solver_tolerance():
    keys = [('x', 0), ('x', 1)]
    assert flatten_ordered_var_values({'x': {0: 1. - 1e-9, 1: -1e-9}}, keys) == (1, 0)
    with pytest.raises(ValueError, match='binary'):
        flatten_ordered_var_values({'x': {0: .4, 1: 0}}, keys)


@pytest.mark.requires_solver
def test_f5_seeded_cglp_vertex_validity_with_revisits(solver_name):
    rng = random.Random(20261029)
    config = _config(solver_name).cglp
    for n in range(1, 5):
        main = ModelBase(solver_config=config)
        main.var['x'] = {i: main.add_var(domain=pyo.Binary) for i in range(n)}
        keys = [('x', i) for i in range(n)]
        points = list(itertools.product((0, 1), repeat=n))
        rows = [{'name': 'zero', 'x_coef': {}, 'rhs': 0.}]
        table = {point: rng.uniform(1., 10.) for point in points}
        cg = ModelCGLP(main, keys, [], [], [0], rows, solver_config=config)
        cg.build_model()
        rng.shuffle(points)
        visited = points[:5]
        for index, point in enumerate(visited + visited[::-1]):
            cg.update_with_exact_point(point, table[point])
            cg.sync_lower_bound_rows(rows)
            alpha, eta = cg.solve_cglp()
            for vertex in points:
                cut_value = -eta - sum(alpha[key] * vertex[i] for i, key in enumerate(keys))
                assert cut_value <= table[vertex] + 1e-6
                if vertex == point:
                    assert cut_value == pytest.approx(table[vertex])
            rows.append(cg.build_lower_bound_row(f'cut_{index}', alpha, eta))


@pytest.mark.requires_solver
def test_f5_module_retargets_cached_anchor(local_file_path, solver_name):
    config = _config(solver_name)
    module = TwoStageDecompRedo(_raw(local_file_path), [1], ['s_1'], solver_config_set=config)
    module.build_main_stage_model()
    designs = [_design(module.main_model_data, install=False), _design(module.main_model_data)]
    requests = []
    for i, design in enumerate(designs):
        sub = _sub(module.main_model_data, design, config.subproblem_exact)
        sub.solve()
        requests.append(CutRequest(
            scenario_group=ScenarioGroup(('s_1',)), ite_name=str(i), anchor_solution=MainSolution(design),
            exact_obj_bound=sub.get_obj_bound(), theta_values={'s_1': 0.},
        ))
    for request in requests + requests[:1]:
        cut = module.generate_cut(CutTypeName.CGLP, request)
        point = tuple(request.anchor_solution.values[name][key] for name, key in module.cglp_x_var_key_list)
        cg = module.cglp_model_dict['s_1',]
        assert cg.current_tight_point == point
        assert cut is not None
        value = -cut.metadata['eta'] - sum(cut.metadata['alpha'][key] * point[i]
                                           for i, key in enumerate(module.cglp_x_var_key_list))
        assert value == pytest.approx(request.exact_obj_bound)


@pytest.mark.parametrize('fail', [False, True])
def test_f6_lp_role_replaces_and_restores_solver_configuration(monkeypatch, fail):
    exact_config = SolverConfig(solver_name='gurobi_direct', time_limit=.001, tee=False)
    model = ModelBase(solver_config=exact_config)
    model.set_objective(0.)
    model.set_parameters({'TimeLimit': .002})
    exact_solver = model.solver
    original_options = dict(exact_solver.options)
    calls = []
    lp_solver = SimpleNamespace(options={}, available=lambda exception_flag=False: True)

    def solve(pyomo_model, **kwargs):
        calls.append((model.solver_name, dict(lp_solver.options), kwargs))
        if fail:
            raise RuntimeError('LP failure')
        return SimpleNamespace(solver=SimpleNamespace(termination_condition=pyo.TerminationCondition.optimal),
                               solution=[], problem=SimpleNamespace(lower_bound=0, upper_bound=0))

    lp_solver.solve = solve
    monkeypatch.setattr('model.model_base.SolverFactory', lambda name: lp_solver)
    config = SolverConfig(solver_name='appsi_highs', time_limit=None, mip_gap=None, tee=False)
    if fail:
        with pytest.raises(RuntimeError, match='LP failure'):
            model.solve_relaxed(solver_config=config)
    else:
        model.solve_relaxed(solver_config=config)
        assert model.get_relaxed_obj_bound() == 0
    assert calls == [('appsi_highs', {}, {'tee': False, 'load_solutions': True})]
    assert model.solver is exact_solver and model.solver_config is exact_config
    assert model.solver_name == 'gurobi_direct'
    assert dict(exact_solver.options) == original_options
    assert model._resolve_solve_options()['TimeLimit'] == .002


@pytest.mark.requires_solver
def test_f6_module_lp_ignores_exact_role_time_limit(local_file_path, solver_name):
    config = _config(solver_name).with_role(SolverRoleName.SUBPROBLEM_EXACT,
        SolverConfig(solver_name=solver_name, time_limit=0., tee=False))
    module = TwoStageDecompRedo(_raw(local_file_path), [1], ['s_1'], solver_config_set=config)
    module.build_main_stage_model()
    module.build_sub_model(['s_1'], _design(module.main_model_data))
    module.solve_relaxed_sub_model()
    assert module.current_sub_model.solve_status_relax == ModelStatus.SOLVED_OPT
    assert module.current_sub_model.solver_config.time_limit == 0.


def test_f7_report_uses_only_certified_bounds_and_g1_group_export(tmp_path):
    df = pd.DataFrame([
        {'ite_num': '0', 'best_incumbent_obj_value': 100., 'best_bound_objective_value': 90., 'valid_lower_bound': 20.},
        {'ite_num': '1', 'best_incumbent_obj_value': 100., 'best_bound_objective_value': 95., 'valid_lower_bound': None},
    ])
    result = enrich_iter_general_dataframe(df)
    assert result.iloc[0]['gap'] == pytest.approx(.8)
    assert pd.isna(result.iloc[1]['gap'])
    assert result.iloc[-1]['gap'] == pytest.approx(.8)
    assert result.iloc[-1]['valid_lower_bound'] == 20.
    assert result.iloc[-1]['best_bound_objective_value'] == 95.
    assert enrich_iter_general_dataframe(df.drop(columns='valid_lower_bound'))['gap'].isna().all()
    iter_sub_prob_info({'0': {('s_1', 's_2'): {'total': 30., 'detail': {'operation': 30.}}}},
                       str(tmp_path), ['s_1', 's_2'])
    rows = pd.read_csv(tmp_path / 'iter_subproblem_info.csv')
    assert len(rows) == 1
    assert json.loads(rows.iloc[0]['scenario']) == ['s_1', 's_2']
    assert rows.iloc[0]['sub_obj_total'] == 30.


@pytest.mark.requires_solver
def test_f8_fix_recover_invalidates_bounds_and_preserves_original_fixed_state(solver_name):
    model = ModelBase(solver_config=_config(solver_name).master)
    x = model.add_var(domain=pyo.Binary)
    y = model.add_var(lb=-1, ub=2)
    y.fix(.5)
    model.var['x'] = {0: x, 1: y}
    model.set_objective(x)
    model.fix_variable_value({'x': {0: 1, 1: 1}})
    model.solve()
    model.solve_relaxed()
    assert model.get_obj_bound() == pytest.approx(1.)
    model.fix_variable_value({'x': {1: 2}})
    with pytest.raises(ModelSolveError):
        model.get_obj_bound()
    model.recover_variable_from_fixed()
    assert not x.fixed
    assert y.fixed and y.value == .5 and y.lb == -1 and y.ub == 2
    with pytest.raises(ModelSolveError):
        model.get_relaxed_obj_bound()
    model.solve()
    assert model.get_obj_bound() == pytest.approx(0.)


def test_r2_multiplier_initialization_honors_supplied_center():
    model = ModelLagrangianMultiplierHeuristic(
        main_result={'x': {0: 1.}}, constr_dual_info={'local_x': 0.},
        constr_var_map={'local_x': ('x', 0)},
    )
    model.build_ini_model(ini_multiplier={'local_x': 3.}, ini_stabilization_param=1.)
    model.var[VarName.LIFT_VALUE].set_value(0.)
    model.var[VarName.LAG_MULTIPLIER]['local_x'].set_value(3.)
    assert model.get_obj_value() == pytest.approx(0.)
