# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


import os
import sys

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from algo.two_stage_decomp_redo import TwoStageDecompRedo
from algo.interfaces import AggregationPolicy, SolutionGenerator
from algo.runner import CutSchedule, DecompositionRunner
from algo.runner_helpers import (
    setup_pretrained_cut,
    validate_bound_mode as _validate_bound_mode,
)
from model.solver_config import SolverConfigSet

from util.project_logger import init_logger
from util.names import (
    BoundModeName,
    CutTypeName,
    InputMethodName,
    MainProblemHatDataMethodName,
    MainProblemModelTypeName,
    ResultName,
    RunParameterName,
)
from dao.data_reader import DataReader

from util.converge_visual import plot_iter_obj_curves

from util.tools import (
    report_repetitive_cuts,
    write_decomp_run_parameters,
    write_main_result_repetition_report,
    write_power_usage_capacity_ratio,
)

from datetime import datetime
import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def _check_iter_range(name: str, val: Tuple[int, int], max_iterations: int) -> None:
    """Raise ValueError if (a, b) does not satisfy 0 <= a <= b <= max_iterations."""
    if len(val) != 2:
        raise ValueError(f"{name} must be a 2-tuple (first_iter, last_iter), got {val}")
    a, b = val
    if not (0 <= a <= b <= max_iterations):
        raise ValueError(
            f"{name} (first_iter, last_iter) must satisfy 0 <= first <= last <= max_iterations "
            f"(max_iterations={max_iterations}), got ({a}, {b})"
        )


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
    benders_cut_iter_range: Optional[Tuple[int, int]] = None,
    strengthen_benders_cut_iter_range: Optional[Tuple[int, int]] = None,
    lagrangian_cut_iter_range: Optional[Tuple[int, int]] = None,
    cglp_cut_iter_range: Optional[Tuple[int, int]] = None,
    integer_opt_cut_iter_range: Optional[Tuple[int, int]] = None,
    record_incumbent_every_k: Optional[int] = None,
    use_aggregated_cuts: bool = False,
    main_problem_model_type: str = MainProblemModelTypeName.COLLAPSED_NO_TIME,
    main_problem_hat_data_method: str = MainProblemHatDataMethodName.WEIGHTED_AVERAGE,
    added_obj_term_weight: float = 0.0,
    bound_mode: str = BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
    convergence_rel_gap: float = 0.01,
    solver_config_set: Optional[SolverConfigSet] = None,
    reference_objective_value: Optional[float] = None,
    solution_generator: Optional[SolutionGenerator] = None,
    enable_pretrained_cut: bool = False,
):
    """Run the decomposition module test.

    Args:
        enable_log_output: Whether to write running logs to log output folder (default: False)
        enable_result_output: Whether to write result output (default: True)
        scenario_list: List of scenarios to test (default: ['s_1', 's_2', 's_3'])
        time_list: List of time periods (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        test_read_method: Input method for reading data (default: InputMethodName.LOCAL_CSV)
        test_file_path: Path to test data files (default: <repo>/unit_test/test_local_csv_file)
        data_set_name: Name of the data set (default: 'function_test_fixed_rated_p')
        max_iterations: Maximum number of iterations to run (default: 10)
        benders_cut_iter_range: (first_iter, last_iter) to add Benders cuts; None to skip.
        strengthen_benders_cut_iter_range: (first_iter, last_iter) to add strengthened Benders cuts; None to skip.
        lagrangian_cut_iter_range: (first_iter, last_iter) to add Lagrangian cuts; None to skip.
        cglp_cut_iter_range: (first_iter, last_iter) to add CGLP cuts; None to skip.
        integer_opt_cut_iter_range: (first_iter, last_iter) to add integer optimality cuts; None to skip.
        record_incumbent_every_k: If a positive integer k, record incumbent and run convergence check only every k
            iterations (0, k, 2k, ...). If None, record every iteration. When not recording, solve_sub_model() is
            skipped and Benders cut uses relaxed LP objective (get_relaxed_obj_value()); strengthen/Lagrangian
            cuts are unchanged (they use curr_sub_frac_model only, not current_sub_model).
        use_aggregated_cuts: Add the sum of the already probability-weighted group cuts.
        enable_pretrained_cut: Restrict the search to the predicted region; disables global bounds.
        main_problem_model_type: Selected main-problem model variant.
        main_problem_hat_data_method: Method used to collapse time/scenario-dependent data for the enriched main model.
        added_obj_term_weight: Scalar weight of the added master-only operating objective terms.
    """
    # Set default values
    if scenario_list is None:
        scenario_list = [
            's_1',
            's_2',
            's_3',
            # 's_4',
            # 's_5',
            # 's_6',
            # 's_7',
            # 's_8',
            # 's_9',
            # 's_10'
        ]
    if time_list is None:
        time_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if test_read_method is None:
        test_read_method = InputMethodName.LOCAL_CSV
    if test_file_path is None:
        test_file_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'test_local_csv_file')
        )
    if data_set_name is None:
        data_set_name = 'function_test_fixed_rated_p'
    if enable_result_output is None:
        enable_result_output = False

    for name, val in [
        ("benders_cut_iter_range", benders_cut_iter_range),
        ("strengthen_benders_cut_iter_range", strengthen_benders_cut_iter_range),
        ("lagrangian_cut_iter_range", lagrangian_cut_iter_range),
        ("cglp_cut_iter_range", cglp_cut_iter_range),
        ("integer_opt_cut_iter_range", integer_opt_cut_iter_range),
    ]:
        if val is not None:
            _check_iter_range(name, val, max_iterations)

    if record_incumbent_every_k is not None and (not isinstance(record_incumbent_every_k, int) or record_incumbent_every_k < 1):
        raise ValueError("record_incumbent_every_k must be a positive integer or None")
    init_logger(enable_file_output=enable_log_output)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', timestamp)
    if output_label is not None:
        output_dir = output_dir + '_' + output_label
    if enable_result_output:
        os.makedirs(output_dir, exist_ok=True)

    r = DataReader(
        read_method=test_read_method,
        local_file_path=test_file_path,
        data_set_name=data_set_name
    )

    r.read()
    print("CSV read success:")

    data_read_path = test_file_path + '/' + data_set_name + '/'

    if solver_config_set is None:
        solver_config_set = SolverConfigSet.build()
    _validate_bound_mode(bound_mode)

    two_stage_decomp_module = TwoStageDecompRedo(
        raw_data=r.raw_data,
        time_list=time_list,
        scenario_list=scenario_list,
        solver_config_set=solver_config_set,
    )
    two_stage_decomp_module.data_processor_module.main_problem_model_type = main_problem_model_type
    two_stage_decomp_module.data_processor_module.main_problem_hat_data_method = main_problem_hat_data_method

    # build main model
    two_stage_decomp_module.build_main_stage_model()
    two_stage_decomp_module.model_main.set_added_obj_term_weight(added_obj_term_weight)

    pretrained_info = {'added': False}
    if enable_pretrained_cut:
        pretrained_info = setup_pretrained_cut(
            two_stage_decomp_module=two_stage_decomp_module,
            dataset_path=data_read_path,
            scenario_list=scenario_list,
            hat=0.3334,
            use_fallback_partition=False,
            logger=logger,
        )
    logger.info('Pretrained restrictions enabled=%s, added=%s',
                enable_pretrained_cut, pretrained_info['added'])

    # decide how to group and iterate the scenarios
    sce_group_list = [
        [s] for s in scenario_list
    ]

    if enable_result_output:
        two_stage_decomp_module.write_iteration_csvs(output_dir=output_dir, scenario_list=scenario_list)
    runner = DecompositionRunner(
        two_stage_decomp_module=two_stage_decomp_module,
        scenario_groups=sce_group_list,
        max_iterations=max_iterations,
        cut_schedule=CutSchedule(ranges={
            CutTypeName.BENDERS: benders_cut_iter_range,
            CutTypeName.STRENGTHEN_BENDERS: strengthen_benders_cut_iter_range,
            CutTypeName.LAGRANGIAN: lagrangian_cut_iter_range,
            CutTypeName.CGLP: cglp_cut_iter_range,
            CutTypeName.INTEGER_OPT: integer_opt_cut_iter_range,
        }),
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
    incumbent_tracker = runner.incumbent
    iteration_bound_records = run_result['iteration_bound_records']
    curr_main_result_history = run_result['current_solution_history']
    frac_main_result_history = run_result['fractional_solution_history']
    curr_main_repetition_records = run_result['current_repetition_records']
    frac_main_repetition_records = run_result['fractional_repetition_records']

    if enable_result_output:
        write_decomp_run_parameters(
            output_dir=output_dir,
            params={
                RunParameterName.ENABLE_PRETRAINED_CUT: enable_pretrained_cut,
                RunParameterName.PRETRAINED_CUT_ADDED: pretrained_info['added'],
                "scenario_list": scenario_list,
                "data_set_name": data_set_name,
                "max_iterations": max_iterations,
                "benders_cut_iter_range": benders_cut_iter_range,
                "strengthen_benders_cut_iter_range": strengthen_benders_cut_iter_range,
                "lagrangian_cut_iter_range": lagrangian_cut_iter_range,
                "cglp_cut_iter_range": cglp_cut_iter_range,
                "integer_opt_cut_iter_range": integer_opt_cut_iter_range,
                "record_incumbent_every_k": record_incumbent_every_k,
                "use_aggregated_cuts": use_aggregated_cuts,
                "main_problem_model_type": main_problem_model_type,
                "main_problem_hat_data_method": main_problem_hat_data_method,
                "added_obj_term_weight": added_obj_term_weight,
            },
        )

    if enable_result_output and incumbent_tracker.best_sub_results is not None:
        write_power_usage_capacity_ratio(
            best_sub_results=incumbent_tracker.best_sub_results,
            output_dir=output_dir,
        )

    repetitive_cut_records = two_stage_decomp_module.get_repetitive_cut_records()
    report_repetitive_cuts(
        repetitive_cut_records=repetitive_cut_records,
        enable_result_output=enable_result_output,
        output_dir=output_dir,
        logger=logger,
    )
    if enable_result_output:
        write_main_result_repetition_report(
            output_dir=output_dir,
            curr_main_result_history=curr_main_result_history,
            frac_main_result_history=frac_main_result_history,
            curr_main_repetition_records=curr_main_repetition_records,
            frac_main_repetition_records=frac_main_repetition_records,
        )

    # after finishing the algorithm
    if enable_result_output:
        plot_iter_obj_curves(
            ite_obj_value_dict=two_stage_decomp_module.ite_obj_value_dict,
            output_dir=output_dir,
            real_objective_value=reference_objective_value,
        )
    return {
        "repetitive_cut_count": len(repetitive_cut_records),
        "repetitive_cut_records": repetitive_cut_records,
        "iteration_bound_records": iteration_bound_records,
        ResultName.BEST_INCUMBENT_OBJ: incumbent_tracker.best_obj,
        ResultName.BEST_MAIN_SOLUTION: run_result[ResultName.BEST_MAIN_SOLUTION],
        ResultName.BEST_INCUMBENT_SOURCE: run_result[ResultName.BEST_INCUMBENT_SOURCE],
        "two_stage_decomp_module": two_stage_decomp_module,
    }



if __name__ == "__main__":
    _max_iter = 30
    run_decomp_module_test(
        enable_log_output=False,
        enable_result_output=True,
        # scenario_list=['s_1', 's_2', 's_3', 's_4', 's_5', 's_6'],
        scenario_list=['s_1', 's_3', 's_5'],
        time_list=None,
        test_read_method=None,
        test_file_path=None,
        data_set_name='function_test_fixed_rated_p',
        max_iterations=_max_iter,
        output_label='test_new_m_all_c',
        benders_cut_iter_range=(0, _max_iter - 1),
        strengthen_benders_cut_iter_range=(0, _max_iter - 1),
        # lagrangian_cut_iter_range=(0, _max_iter - 1),
        # cglp_cut_iter_range=(0, _max_iter - 1),
        record_incumbent_every_k=2,
        use_aggregated_cuts=False,
        main_problem_model_type=MainProblemModelTypeName.COLLAPSED_NO_TIME,
        main_problem_hat_data_method=MainProblemHatDataMethodName.WEIGHTED_AVERAGE,
        added_obj_term_weight= 2.0,
    )
