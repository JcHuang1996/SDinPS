"""OOS expectations must use the optimizer's normalized scenario-weight convention."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from analysis import out_sample_main_stage_eval as evaluation
from dao.data_processor import DataProcessor
from model.model_sub import ModelSub
from model.solver_config import SolverConfig
from unit_test.test_scripts import decomp_validity_harness as harness
from util.names import VarName


@pytest.fixture
def sampled_evaluation(tmp_path, monkeypatch):
    samples = pd.DataFrame([
        {'scenario_id': scenario, 'sample_id': sample, 'time_idx': 1,
         'from_bus': 'n1', 'to_bus': 'n2'}
        for scenario, sample in [('s_1', 0), ('s_1', 1), ('s_2', 0), ('s_3', 0)]
    ])
    samples.to_csv(tmp_path / 'samples.csv', index=False)
    costs = {('s_1', 0): 10., ('s_1', 1): 30., ('s_2', 0): 80., ('s_3', 0): 1000.}

    def sample_result(sample_line_state_df, **kwargs):
        row = sample_line_state_df.iloc[0]
        return {'scenario_id': row['scenario_id'], 'solve_status': 'opt',
                'sample_objective_value': costs[row['scenario_id'], row['sample_id']]}

    monkeypatch.setattr(evaluation, 'load_base_raw_data', lambda **kwargs: {})
    monkeypatch.setattr(evaluation, 'load_main_solution', lambda **kwargs: ({}, 'json', None))
    monkeypatch.setattr(evaluation, 'evaluate_design_cost', lambda **kwargs: 100.)
    monkeypatch.setattr(evaluation, 'evaluate_one_sample', sample_result)
    return tmp_path


@pytest.mark.parametrize('scale', [1., 7.])
@pytest.mark.parametrize('scenarios, expected', [
    (None, 718.), (['s_1', 's_2'], 60.), (['s_1'], 20.),
])
def test_oos_normalizes_evaluated_scenarios_and_preserves_design_cost(
    sampled_evaluation, scenarios, expected, scale,
):
    root = sampled_evaluation
    pd.DataFrame({'scenario_id': ['s_1', 's_2', 's_3'],
                  'probability': [scale * .1, scale * .2, scale * .7]}).to_csv(
        root / 's_probability.csv', index=False,
    )
    result = evaluation.evaluate_main_solution_out_of_sample(
        str(root / 'samples.csv'), str(root), scenario_list=scenarios,
    )
    assert result['expected_subproblem_objective'] == pytest.approx(expected)
    assert result['design_cost'] == 100.
    assert result['expected_total_objective'] == pytest.approx(100. + expected)
    summary = pd.read_csv(result['summary_output_path'])
    assert summary.iloc[:-1]['probability'].sum() == pytest.approx(1.)
    assert summary.iloc[-1]['probability'] == pytest.approx(1.)
    assert summary.iloc[-1]['weighted_objective_contribution'] == pytest.approx(expected)
    # Average samples within a scenario first: two s_1 samples must not double its weight.
    assert summary.iloc[0]['sample_objective_mean'] == 20.
    assert summary.iloc[0]['sample_count'] == 2


@pytest.mark.parametrize('weights', [[0., 0.], [-.1, .4], [float('inf'), .2]])
def test_oos_rejects_invalid_normalization_weights(sampled_evaluation, weights):
    root = sampled_evaluation
    pd.DataFrame({'scenario_id': ['s_1', 's_2'], 'probability': weights}).to_csv(
        root / 's_probability.csv', index=False,
    )
    with pytest.raises(ValueError, match='probabilities'):
        evaluation.evaluate_main_solution_out_of_sample(
            str(root / 'samples.csv'), str(root), scenario_list=['s_1', 's_2'],
        )


@pytest.mark.requires_solver
def test_oos_probability_scale_matches_extensive_form(local_file_path, solver_name, tmp_path, monkeypatch):
    data_dir = tmp_path / 'm1'
    data_dir.mkdir()
    for source in (Path(local_file_path) / 'micro_bench' / 'm1_dg_site_type').glob('*.csv'):
        shutil.copyfile(source, data_dir / source.name)
    pd.DataFrame({'scenario_id': ['s_1'], 'probability': [.051]}).to_csv(
        data_dir / 's_probability.csv', index=False,
    )
    raw = evaluation.load_base_raw_data(str(data_dir))
    data = DataProcessor(raw).data_process(['s_1'], [1])
    design = harness.enumerate_first_stage_designs(data, max_designs=1)[0]
    design[VarName.DG_INSTALL]['node_1'] = 1
    design[VarName.DG_INSTALL_TYPE]['node_1', 'T2'] = 1
    solution_path = tmp_path / 'solution.json'
    solution_path.write_text(json.dumps({
        name: {str(key): value for key, value in block.items()} for name, block in design.items()
    }))
    samples = pd.read_csv(data_dir / 's_line_state_w_o_harden.csv')
    samples['sample_id'] = 0
    sample_path = tmp_path / 'samples.csv'
    samples.to_csv(sample_path, index=False)

    config = SolverConfig(solver_name=solver_name, mip_gap=0., tee=False)

    def configured_sub_model(*args, **kwargs):
        return ModelSub(*args, solver_config=config, **kwargs)

    monkeypatch.setattr('model.model_sub.ModelSub', configured_sub_model)
    result = evaluation.evaluate_main_solution_out_of_sample(
        str(sample_path), str(data_dir), main_solution_json_path=str(solution_path),
    )
    reference = harness.solve_extensive_form(raw, ['s_1'], [1], config)
    assert result['expected_subproblem_objective'] == pytest.approx(70.)
    assert result['design_cost'] == pytest.approx(2800.)
    assert result['expected_total_objective'] == pytest.approx(2870.)
    assert result['expected_total_objective'] == pytest.approx(reference['objective'])
