# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime
from typing import Optional

from algo.interfaces import (
    AggregationPolicy,
    MainSolution,
    SolutionGenerator,
    StaticSolutionGenerator,
)
from algo.runner import DecompositionRunner, PhasePolicy
from algo.runner_helpers import setup_pretrained_cut, write_iteration_branch_cut_report
from algo.two_stage_decomp_redo import TwoStageDecompRedo
from dao.data_reader import DataReader
from model.solver_config import SolverConfigSet
from util.converge_visual import plot_iter_obj_curves
from util.names import (
    BoundModeName,
    InputMethodName,
    MainProblemHatDataMethodName,
    MainProblemModelTypeName,
    PhaseName,
    ResultName,
    RunParameterName,
)
from util.project_logger import init_logger
from util.tools import (
    load_main_stage_solution_json,
    report_repetitive_cuts,
    write_decomp_run_parameters,
    write_main_result_repetition_report,
    write_power_usage_capacity_ratio,
)


logger = logging.getLogger(__name__)


def run_decomp_module_test(
    enable_log_output=False,
    enable_result_output=True,
    scenario_list=None,
    time_list=None,
    test_read_method=None,
    test_file_path=None,
    data_set_name='function_test_fixed_rated_p',
    max_iterations=4,
    output_label=None,
    phase_1_end_iteration: Optional[int] = None,
    phase_2_end_iteration: Optional[int] = None,
    phase_3_end_iteration: Optional[int] = None,
    reference_solution_path: Optional[str] = None,
    record_incumbent_every_k: Optional[int] = None,
    use_aggregated_cuts: bool = False,
    main_problem_model_type: str = MainProblemModelTypeName.COLLAPSED_NO_TIME,
    main_problem_hat_data_method: str = MainProblemHatDataMethodName.WEIGHTED_AVERAGE,
    added_obj_term_weight: float = 0.0,
    phase_1_added_obj_term_weight: Optional[float] = None,
    phase_2_added_obj_term_weight: Optional[float] = None,
    phase_3_added_obj_term_weight: Optional[float] = None,
    bound_mode: str = BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
    convergence_rel_gap: float = 0.01,
    solver_config_set: Optional[SolverConfigSet] = None,
    reference_objective_value: Optional[float] = None,
    solution_generator: Optional[SolutionGenerator] = None,
    enable_pretrained_cut: bool = False,
):
    """Configure and run the phase-driven decomposition algorithm."""
    scenario_list = scenario_list or ['s_1', 's_2', 's_3']
    time_list = time_list or list(range(1, 12))
    test_read_method = test_read_method or InputMethodName.LOCAL_CSV
    test_file_path = test_file_path or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'test_local_csv_file')
    )
    solver_config_set = solver_config_set or SolverConfigSet.build()

    if phase_1_end_iteration is None and phase_2_end_iteration is None and phase_3_end_iteration is None:
        phase_1_end_iteration = 0
        phase_2_end_iteration = max(0, max_iterations - 4)
        phase_3_end_iteration = max_iterations - 1
    if reference_solution_path is not None:
        if solution_generator is not None:
            raise ValueError(
                'Pass either reference_solution_path or solution_generator, not both.'
            )
        solution_generator = StaticSolutionGenerator([
            MainSolution(
                values=load_main_stage_solution_json(reference_solution_path),
                name='reference_solution',
            )
        ])

    init_logger(enable_file_output=enable_log_output)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'output', timestamp)
    )
    if output_label:
        output_dir = f'{output_dir}_{output_label}'
    if enable_result_output:
        os.makedirs(output_dir, exist_ok=True)

    reader = DataReader(
        read_method=test_read_method,
        local_file_path=test_file_path,
        data_set_name=data_set_name,
    )
    reader.read()

    module = TwoStageDecompRedo(
        raw_data=reader.raw_data,
        time_list=time_list,
        scenario_list=scenario_list,
        solver_config_set=solver_config_set,
    )
    module.data_processor_module.main_problem_model_type = main_problem_model_type
    module.data_processor_module.main_problem_hat_data_method = main_problem_hat_data_method
    module.build_main_stage_model()

    pretrained_info = {'added': False}
    if enable_pretrained_cut:
        pretrained_info = setup_pretrained_cut(
            two_stage_decomp_module=module,
            dataset_path=os.path.join(test_file_path, data_set_name),
            scenario_list=scenario_list,
            hat=0.3334,
            use_fallback_partition=True,
            logger=logger,
        )
    logger.info('Pretrained restrictions enabled=%s, added=%s',
                enable_pretrained_cut, pretrained_info['added'])

    phase_policy = PhasePolicy(
        phase_1_end_iteration=phase_1_end_iteration,
        phase_2_end_iteration=phase_2_end_iteration,
        phase_3_end_iteration=phase_3_end_iteration,
        default_weight=added_obj_term_weight,
        phase_weights={
            PhaseName.PHASE_1: phase_1_added_obj_term_weight,
            PhaseName.PHASE_2: phase_2_added_obj_term_weight,
            PhaseName.PHASE_3: phase_3_added_obj_term_weight,
        },
    )
    runner = DecompositionRunner(
        two_stage_decomp_module=module,
        scenario_groups=[[scenario] for scenario in scenario_list],
        max_iterations=max_iterations,
        phase_policy=phase_policy,
        aggregation_policy=(
            AggregationPolicy.AGGREGATE if use_aggregated_cuts else AggregationPolicy.PER_GROUP
        ),
        added_obj_term_weight=added_obj_term_weight,
        bound_mode=bound_mode,
        convergence_rel_gap=convergence_rel_gap,
        record_incumbent_every_k=record_incumbent_every_k,
        enable_result_output=enable_result_output,
        output_dir=output_dir,
        solution_generator=solution_generator,
    )
    run_result = runner.run()

    params = {
        RunParameterName.ENABLE_PRETRAINED_CUT: enable_pretrained_cut,
        RunParameterName.PRETRAINED_CUT_ADDED: pretrained_info['added'],
        'scenario_list': scenario_list,
        'data_set_name': data_set_name,
        'max_iterations': max_iterations,
        'phase_1_end_iteration': phase_1_end_iteration,
        'phase_2_end_iteration': phase_2_end_iteration,
        'phase_3_end_iteration': phase_3_end_iteration,
        'reference_solution_path': reference_solution_path,
        'record_incumbent_every_k': record_incumbent_every_k,
        'use_aggregated_cuts': use_aggregated_cuts,
        'main_problem_model_type': main_problem_model_type,
        'main_problem_hat_data_method': main_problem_hat_data_method,
        'added_obj_term_weight': added_obj_term_weight,
        'phase_1_added_obj_term_weight': phase_1_added_obj_term_weight,
        'phase_2_added_obj_term_weight': phase_2_added_obj_term_weight,
        'phase_3_added_obj_term_weight': phase_3_added_obj_term_weight,
        'bound_mode': bound_mode,
        'convergence_rel_gap': convergence_rel_gap,
        'solver_config_set': solver_config_set.as_dict(),
        'iteration_bound_records': run_result['iteration_bound_records'],
    }
    if enable_result_output:
        write_decomp_run_parameters(output_dir=output_dir, params=params)

    repetitive_cut_records = module.get_repetitive_cut_records()
    report_repetitive_cuts(
        repetitive_cut_records=repetitive_cut_records,
        enable_result_output=enable_result_output,
        output_dir=output_dir,
        logger=logger,
    )
    if enable_result_output:
        if run_result['best_sub_results'] is not None:
            write_power_usage_capacity_ratio(
                best_sub_results=run_result['best_sub_results'],
                output_dir=output_dir,
            )
        write_main_result_repetition_report(
            output_dir=output_dir,
            curr_main_result_history=run_result['current_solution_history'],
            frac_main_result_history=run_result['fractional_solution_history'],
            curr_main_repetition_records=run_result['current_repetition_records'],
            frac_main_repetition_records=run_result['fractional_repetition_records'],
        )
        write_iteration_branch_cut_report(
            output_dir=output_dir,
            iteration_branch_cut_records=run_result['iteration_branch_cut_records'],
        )
        plot_iter_obj_curves(
            ite_obj_value_dict=module.ite_obj_value_dict,
            output_dir=output_dir,
            real_objective_value=reference_objective_value,
        )

    return {
        'repetitive_cut_count': len(repetitive_cut_records),
        'repetitive_cut_records': repetitive_cut_records,
        'iteration_bound_records': run_result['iteration_bound_records'],
        ResultName.BEST_INCUMBENT_OBJ: run_result[ResultName.BEST_INCUMBENT_OBJ],
        ResultName.BEST_MAIN_SOLUTION: run_result[ResultName.BEST_MAIN_SOLUTION],
        ResultName.BEST_INCUMBENT_SOURCE: run_result[ResultName.BEST_INCUMBENT_SOURCE],
        'two_stage_decomp_module': module,
    }


if __name__ == '__main__':
    max_iter = 55
    run_decomp_module_test(
        enable_log_output=False,
        enable_result_output=True,
        scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        data_set_name='function_test_fixed_rated_p',
        max_iterations=max_iter,
        output_label='new_m_agg_t',
        phase_1_end_iteration=0,
        phase_2_end_iteration=max_iter - 4,
        phase_3_end_iteration=max_iter - 1,
        record_incumbent_every_k=2,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_BY_SCENARIOS,
        added_obj_term_weight=1.0,
        phase_3_added_obj_term_weight=0,
    )
