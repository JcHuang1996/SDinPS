"""Regression coverage for the 2026-10-28 maintenance of the pre-course implementation."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from algo.interfaces import CutRequest, MainSolution, NoOpSolutionGenerator, ScenarioGroup, StaticSolutionGenerator
from algo.runner import CutSchedule, DecompositionRunner
from algo.runner_helpers import IncumbentTracker
from algo.two_stage_decomp_redo import TwoStageDecompRedo
from model.model_base import ModelSolveError
from model.solver_config import SolverConfigSet
from unit_test.test_scripts import decomp_validity_harness as harness
from unit_test.test_scripts.model_build_smoke_test import DummySolver
from util.names import (
    BoundModeName, ConstrName, CutTypeName, DataName, MainProblemModelTypeName,
    ModelStatus, SolverRoleName, VarName,
)


def _module(local_file_path, instance='m1_dg_site_type', solver='gurobi_direct'):
    reference = harness.load_instance_reference(local_file_path, instance)
    module = TwoStageDecompRedo(
        raw_data=harness.load_raw_data(local_file_path, instance),
        time_list=reference['time_list'],
        scenario_list=reference['scenario_list'],
        solver_config_set=SolverConfigSet.build(
            solver_name=solver, tee=False, master_mip_gap=0.0, master_time_limit=None,
        ),
    )
    module.build_main_stage_model()
    return module


def _runner(module, **kwargs):
    return DecompositionRunner(
        module, [[s] for s in module.scenario_list], max_iterations=3,
        cut_schedule=CutSchedule(ranges={
            CutTypeName.BENDERS: (0, 2), CutTypeName.STRENGTHEN_BENDERS: (0, 2),
            CutTypeName.INTEGER_OPT: (0, 2), CutTypeName.CGLP: (0, 2),
        }), **kwargs,
    )


def _candidate(module):
    return harness.enumerate_first_stage_designs(module.main_model_data, max_designs=1)[0]


@pytest.mark.parametrize('groups', [
    [['s_1']], [['s_1'], ['s_1', 's_2']], [['s_1', 's_2'], []],
    [['s_1'], ['unknown']], [],
])
def test_runner_rejects_invalid_scenario_partitions(local_file_path, groups):
    module = _module(local_file_path, 'm6_two_scenario_hedging')
    with pytest.raises(ValueError, match='partition'):
        DecompositionRunner(module, groups, max_iterations=1)


def test_objective_change_discards_old_bound(local_file_path):
    model = _module(local_file_path).model_main
    model.solver = DummySolver(problem_lower_bound=2940.0)
    model.set_added_obj_term_weight(2.0)
    model.solve()
    model.set_added_obj_term_weight(0.0)
    for read in (model.get_obj_bound, model.get_valid_lower_bound, model.get_iteration_bound_value):
        with pytest.raises(ModelSolveError):
            read()
    model.solve()
    model.set_added_obj_term_weight(0.0)
    assert model.get_valid_lower_bound() == 2940.0  # unchanged objective retains the solve


def test_pretrained_region_never_certifies_a_global_bound(local_file_path):
    module = _module(local_file_path)
    model = module.model_main
    model.set_pretrained_cut(
        above_var={VarName.DG_INSTALL: ['node_2']}, above_sum=1.0,
        below_var={VarName.DG_INSTALL: ['node_1', 'node_3']}, below_sum=0.0,
    )
    model.solver = DummySolver(problem_lower_bound=2970.0)
    model.solve()
    with pytest.raises(ValueError, match='global'):
        model.get_valid_lower_bound()
    assert module.get_main_stage_valid_lower_bound() is None
    runner = _runner(module)
    for weight in (0.0, 2.0):
        assert runner._resolve_iteration_lower_bound('0', weight) == (None, False)
    assert len(model.solver.solve_calls) == 1
    model.remove_constr_by_name(ConstrName.PRETRAINED_ABOVE)
    model.remove_constr_by_name(ConstrName.PRETRAINED_BELOW)
    with pytest.raises(ModelSolveError):
        model.get_valid_lower_bound()


@pytest.mark.parametrize('status', [ModelStatus.SOLVED_FEASIBLE, ModelStatus.SOLVED_TIMEOUT_WS])
def test_unproven_lp_is_not_a_benders_anchor(status):
    model = SimpleNamespace(model_name='unproven_lp', solve_status_relax=status)
    with pytest.raises(ModelSolveError, match='optimal LP duals'):
        TwoStageDecompRedo._require_optimal_relaxation(model, 'current')


@pytest.mark.parametrize('value', [-1, 2, float('nan'), float('inf')])
def test_generated_design_values_are_validated(local_file_path, value):
    module = _module(local_file_path)
    values = _candidate(module)
    values[VarName.DG_INSTALL]['node_1'] = value
    runner = _runner(module)
    assert not runner._is_binary_design(values)
    with pytest.raises(ValueError, match='bounds'):
        runner._validate_heuristic_solution(MainSolution(values))


@pytest.mark.parametrize('bad_block', ['missing_state', 'missing_key', 'extra_key', 'theta_keys'])
def test_generated_solution_keys_are_validated(local_file_path, bad_block):
    module = _module(local_file_path)
    values = _candidate(module)
    if bad_block == 'missing_state':
        del values[VarName.DG_INSTALL_TYPE]
    elif bad_block == 'missing_key':
        del values[VarName.DG_INSTALL]['node_1']
    elif bad_block == 'extra_key':
        values[VarName.DG_INSTALL]['unknown'] = 0
    else:
        values[VarName.SUB_OBJ_EST] = {'unknown': 0.0}
    with pytest.raises(ValueError):
        _runner(module)._validate_heuristic_solution(MainSolution(values))


def test_fractional_candidate_cannot_request_integer_cuts(local_file_path):
    module = _module(local_file_path)
    values = _candidate(module)
    values[VarName.DG_INSTALL]['node_1'] = 0.5
    values[VarName.SUB_OBJ_EST] = {'s_1': 0.0}
    runner = _runner(module)
    solution = runner._validate_heuristic_solution(MainSolution(values))
    assert solution.is_relaxed
    permitted = runner._filter_heuristic_cut_types(solution, CutTypeName.ALL_TYPES)
    assert CutTypeName.CGLP not in permitted
    assert CutTypeName.INTEGER_OPT not in permitted
    assert CutTypeName.BENDERS in permitted


@pytest.mark.parametrize('family', [CutTypeName.CGLP, CutTypeName.INTEGER_OPT])
def test_integer_cut_dispatch_checks_values_not_only_candidate_flag(local_file_path, family):
    module = _module(local_file_path)
    values = _candidate(module)
    values[VarName.DG_INSTALL]['node_1'] = 0.5
    request = CutRequest(
        ScenarioGroup(['s_1']), '0', MainSolution(values, is_relaxed=False),
        exact_obj_bound=0.0, theta_values={'s_1': 0.0},
    )
    assert not module._request_is_applicable(family, request)


def test_convergence_acceptance_is_explicit():
    records = [{'iteration': 0, 'valid_lower_bound': 4.0,
                'is_bound_valid': True, 'incumbent_upper_bound': 9.0}]
    assert not harness.check_bound_trajectory(records, 5.0)[0]
    assert harness.check_bound_trajectory(records, 5.0, require_convergence=True)[0]
    records[0].update(valid_lower_bound=5.0, incumbent_upper_bound=5.0)
    assert not harness.check_bound_trajectory(records, 5.0, require_convergence=True)[0]


def test_requested_golden_must_exist(tmp_path):
    report = {'bound_records': [], 'cut_algebra': []}
    path = str(tmp_path / 'missing.json')
    assert harness._handle_golden_file(report, path, False)
    assert not harness._handle_golden_file(report, path, True)
    assert not harness._handle_golden_file(report, path, False)


def test_unproven_oracle_value_is_rejected():
    model = SimpleNamespace(model_name='oracle', solve_status=ModelStatus.SOLVED_FEASIBLE)
    with pytest.raises(RuntimeError, match='proven optimality'):
        harness.require_exact_objective(model)
    model.solve_status = ModelStatus.SOLVED_OPT
    model.get_obj_value = lambda: 10.0
    model.get_obj_bound = lambda: 5.0
    with pytest.raises(RuntimeError, match='unresolved objective gap'):
        harness.require_exact_objective(model)


def test_oracle_uses_exact_solves_and_common_big_m(local_file_path):
    module = _module(local_file_path, 'm6_two_scenario_hedging')
    config = module.get_solver_config(SolverRoleName.SUBPROBLEM_EXACT)
    oracle = harness.ScenarioRecourseOracle(
        module.data_processor_module.raw_data, module.scenario_list, module.time_list,
        replace(config, mip_gap=0.5, time_limit=1), _candidate(module),
    )
    for scenario, model in oracle._model_by_scenario.items():
        assert model.solver_config.mip_gap == 0.0
        assert model.solver_config.time_limit is None
        assert model.data[DataName.NUM_TOTAL_POWER] == module.main_model_data[DataName.NUM_TOTAL_POWER]
        assert model.data[DataName.DICT_SC_PROB_UNCOND] == {scenario: 1.0}


def test_incumbent_records_preserve_best_value():
    module = SimpleNamespace(ite_obj_value_dict={})
    tracker = IncumbentTracker()
    tracker.update_from_value(module, '0', 100.0, {})
    tracker.update_from_value(module, '1', 120.0, {})
    assert module.ite_obj_value_dict['1']['sub_obj(best_incumbent)']['sub_p_total'] == 100.0
    assert not tracker.has_converged(110.0)


def test_plot_separates_certificates_from_diagnostics(monkeypatch):
    from util import converge_visual
    converge_visual.plt.switch_backend('Agg')
    close = converge_visual.plt.close
    monkeypatch.setattr(converge_visual.plt, 'close', lambda figure: None)
    records = {
        '0': {'main_obj(bound)': {'total': 20.0}, 'sub_obj(best_incumbent)': {'sub_p_total': 100.0}},
        '1': {'main_obj(bound)': {'total': 30.0, 'valid_lower_bound': 25.0},
              'sub_obj(best_incumbent)': {'sub_p_total': 90.0}},
        '2': {'main_obj(bound)': {'total': 40.0}, 'sub_obj(best_incumbent)': {'sub_p_total': 95.0}},
    }
    try:
        converge_visual.plot_iter_obj_curves(records)
        lines = {line.get_label(): list(line.get_ydata())
                 for line in converge_visual.plt.gcf().axes[0].lines}
        assert lines['Best Incumbent Objective'] == [100.0, 90.0, 90.0]
        bounds = lines['Valid global lower bound']
        assert math.isnan(bounds[0]) and bounds[1] == 25.0 and math.isnan(bounds[2])
        diagnostics = lines['Master diagnostic (not a global lower bound)']
        assert diagnostics[0] == 20.0 and math.isnan(diagnostics[1]) and diagnostics[2] == 40.0
    finally:
        close(converge_visual.plt.gcf())


@pytest.mark.requires_solver
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_runner_frontends_only_load_pretrained_when_requested(
    local_file_path, solver_name, monkeypatch, dynamic, enabled
):
    from unit_test.test_scripts import dynamic_decomposition_test, two_stage_decomposition_test
    frontend = dynamic_decomposition_test if dynamic else two_stage_decomposition_test
    calls = []

    def pretrained(two_stage_decomp_module, **kwargs):
        calls.append(True)
        two_stage_decomp_module.model_main.set_pretrained_cut(
            above_var={VarName.DG_INSTALL: ['node_2']}, above_sum=1.0,
            below_var={VarName.DG_INSTALL: ['node_1', 'node_3']}, below_sum=0.0,
        )
        return {'added': True}

    monkeypatch.setattr(frontend, 'setup_pretrained_cut', pretrained)
    options = {'enable_pretrained_cut': True} if enabled else {}
    result = frontend.run_decomp_module_test(
        enable_result_output=False, test_file_path=local_file_path,
        data_set_name='micro_bench/m1_dg_site_type', scenario_list=['s_1'], time_list=[1],
        max_iterations=1, solver_config_set=SolverConfigSet.build(solver_name=solver_name), **options,
    )
    assert bool(calls) == enabled
    assert result['iteration_bound_records'][0]['is_bound_valid'] == (not enabled)


@pytest.mark.requires_solver
@pytest.mark.parametrize('weight', [0.0, 2.0])
def test_restricted_run_reports_no_global_certificate(local_file_path, solver_name, weight):
    module = _module(local_file_path, solver=solver_name)
    module.model_main.set_pretrained_cut(
        above_var={VarName.DG_INSTALL: ['node_2']}, above_sum=1.0,
        below_var={VarName.DG_INSTALL: ['node_1', 'node_3']}, below_sum=0.0,
    )
    result = DecompositionRunner(
        module, [['s_1']], max_iterations=4,
        cut_schedule=CutSchedule(ranges={
            CutTypeName.BENDERS: (0, 3), CutTypeName.INTEGER_OPT: (0, 3),
        }), added_obj_term_weight=weight,
    ).run()
    assert result['best_incumbent_obj'] == pytest.approx(2970.0)
    assert len(result['iteration_bound_records']) == 4
    assert all(not r['is_bound_valid'] and r['valid_lower_bound'] is None
               for r in result['iteration_bound_records'])


@pytest.mark.requires_solver
@pytest.mark.parametrize('fractional', [False, True])
def test_generated_theta_candidate_runs_without_exact_fractional_solve(local_file_path, solver_name, fractional):
    module = _module(local_file_path, solver=solver_name)
    values = _candidate(module)
    values[VarName.SUB_OBJ_EST] = {'s_1': 0.0}
    if fractional:
        values[VarName.DG_INSTALL]['node_1'] = 0.5
        dg_type = module.main_model_data[DataName.LIST_DG_TYPE][0]
        values[VarName.DG_INSTALL_TYPE]['node_1', dg_type] = 0.5
    result = _runner(module, solution_generator=StaticSolutionGenerator([
        MainSolution(values, name='theta_candidate'),
    ])).run()
    assert math.isfinite(result['best_incumbent_obj'])
    assert result['iteration_branch_cut_records'][0]['heuristic_records']


@pytest.mark.requires_solver
def test_no_op_generator_preserves_cut_algebra_and_bound_trajectory(local_file_path, solver_name):
    module = _module(local_file_path, 'm6_two_scenario_hedging', solver_name)
    kwargs = dict(
        raw_data=module.data_processor_module.raw_data, scenario_list=module.scenario_list,
        time_list=module.time_list, sce_group_list=[[s] for s in module.scenario_list],
        solver_config_set=module.solver_config_set, max_iterations=4,
        families=[CutTypeName.BENDERS, CutTypeName.INTEGER_OPT], added_obj_term_weight=0.0,
        bound_mode=BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_NO_TIME,
    )
    baseline, baseline_bounds = harness.run_decomposition(**kwargs)
    noop, noop_bounds = harness.run_decomposition(**kwargs, solution_generator=NoOpSolutionGenerator())
    assert baseline_bounds == noop_bounds
    assert [r['cut_algebra'] for r in baseline.cut_metadata_records] == [
        r['cut_algebra'] for r in noop.cut_metadata_records
    ]


@pytest.mark.requires_solver
def test_unchecked_cut_fails_the_harness(local_file_path, solver_name, tmp_path, monkeypatch):
    monkeypatch.setattr(TwoStageDecompRedo, 'extract_cut_algebra', lambda self, cut: None)
    report = harness.run_validity_harness(
        instance_name='m1_dg_site_type', local_file_path=local_file_path,
        solver_name=solver_name, families=[CutTypeName.BENDERS], max_iterations=1,
        report_path=str(tmp_path / 'unchecked.json'),
    )
    assert report['failures'] and all('was not checked' in item for item in report['failures'])
