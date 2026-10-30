# -*- coding: utf-8 -*-

import copy
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from algo.algo_simple_tools import analyze_scenario_solutions
from util.names import BoundModeName, DECOMPOSITION_STATE_VAR_NAMES, ResultName, VarName


def validate_bound_mode(bound_mode: str) -> str:
    if bound_mode not in {BoundModeName.RESOLVE_AT_ZERO_WEIGHT, BoundModeName.HEURISTIC}:
        raise ValueError(
            f'Unsupported bound_mode: {bound_mode}. '
            f'Use BoundModeName.RESOLVE_AT_ZERO_WEIGHT or BoundModeName.HEURISTIC.'
        )
    return bound_mode


def write_iteration_branch_cut_report(
    output_dir: str,
    iteration_branch_cut_records: List[dict],
    file_name: str = 'iteration_branch_cut_report.json',
) -> str:
    path = os.path.join(output_dir, file_name)
    with open(path, 'w', encoding='utf-8') as file_ob:
        json.dump({'iteration_branch_cut_records': iteration_branch_cut_records}, file_ob, indent=2)
    return path


def summarize_incumbent_records(incumbent_records: List[dict]) -> Tuple[float, dict]:
    """Combine probability-weighted group incumbents into one feasible objective value."""
    incumbent_obj_value = 0.0
    main_stage_obj_value = None
    sub_results = {VarName.DG_ACTIVE_POWER: {}, VarName.DG_RATED_POWER: {}}

    for record in incumbent_records:
        scenario_group = record['scenario_group']
        sub_result = record['sub_result']
        incumbent_obj_value += record['weighted_sub_obj']

        record_main_stage_obj = record.get('main_stage_obj') or 0.0
        if (
            main_stage_obj_value is not None
            and abs(record_main_stage_obj - main_stage_obj_value) > 1e-9
        ):
            raise ValueError(
                'Incumbent records disagree on the first-stage objective value: '
                f'{main_stage_obj_value} vs {record_main_stage_obj}.'
            )
        main_stage_obj_value = record_main_stage_obj

        sub_results[VarName.DG_ACTIVE_POWER].update(sub_result[VarName.DG_ACTIVE_POWER])
        for node, value in sub_result[VarName.DG_RATED_POWER].items():
            for scenario in scenario_group:
                sub_results[VarName.DG_RATED_POWER][node, scenario] = value

    return incumbent_obj_value + (main_stage_obj_value or 0.0), sub_results


class IncumbentTracker:

    def __init__(self):
        self.best_obj = float('inf')
        self.best_sub_results = None
        self.best_main_solution = None
        self.best_source = None

    def update_from_records(
        self,
        two_stage_decomp_module,
        ite_name: str,
        incumbent_records: List[dict],
        sce_prob: dict = None,
        main_solution: Optional[dict] = None,
    ):
        incumbent_obj_value, sub_results = summarize_incumbent_records(incumbent_records)
        return self.update_from_value(
            two_stage_decomp_module,
            ite_name,
            incumbent_obj_value,
            sub_results,
            main_solution=main_solution,
        )

    def update_from_value(
        self,
        two_stage_decomp_module,
        ite_name: str,
        incumbent_obj_value: float,
        sub_results: dict,
        main_solution: Optional[dict] = None,
    ):
        iteration_record = two_stage_decomp_module.ite_obj_value_dict.setdefault(ite_name, {})
        new_best = incumbent_obj_value < self.best_obj
        if new_best:
            self.best_obj = incumbent_obj_value
            self.best_sub_results = copy.deepcopy(sub_results)
            self.best_main_solution = (
                copy.deepcopy({name: main_solution[name] for name in DECOMPOSITION_STATE_VAR_NAMES})
                if main_solution is not None else None
            )
            self.best_source = ite_name
        iteration_record['sub_obj(best_incumbent)'] = {'sub_p_total': self.best_obj}
        return incumbent_obj_value, sub_results, new_best

    def write_best_solution(self, output_dir: str):
        """Export the incumbent design with its objective and source, including generated candidates."""
        if self.best_main_solution is None:
            return
        payload = {
            ResultName.BEST_INCUMBENT_OBJ: self.best_obj,
            ResultName.BEST_INCUMBENT_SOURCE: self.best_source,
            ResultName.BEST_MAIN_SOLUTION: {
                name: {str(key): value for key, value in block.items()}
                for name, block in self.best_main_solution.items()
            },
        }
        path = os.path.join(output_dir, ResultName.BEST_INCUMBENT_FILE)
        with open(path, 'w', encoding='utf-8') as file_ob:
            json.dump(payload, file_ob, indent=2, allow_nan=False)

    def has_converged(self, main_bound: Optional[float], rel_gap: float = 0.01) -> bool:
        if main_bound is None or not math.isfinite(main_bound):
            return False
        return (
            self.best_obj < float('inf')
            and main_bound <= self.best_obj + 1e-6 * max(1.0, abs(self.best_obj))
            and self.best_obj - main_bound <= rel_gap * self.best_obj
        )


def setup_pretrained_cut(
    two_stage_decomp_module,
    dataset_path: str,
    scenario_list: List,
    hat: float = 0.3334,
    use_fallback_partition: bool = True,
    logger=None,
) -> Dict[str, Any]:
    try:
        (
            expectation,
            keys_above,
            keys_below,
            sum_above,
            sum_below,
            true_keys_above,
            true_keys_below,
            true_sum_above,
            true_sum_below,
            best_hat,
        ) = analyze_scenario_solutions(
            dataset_path=dataset_path,
            scenario_list=scenario_list,
            hat=hat,
        )
    except FileNotFoundError as exc:
        if logger is not None:
            logger.info('Skipping pretrained cut: %s', exc)
        return {'added': False, 'reason': str(exc)}

    if all(value is not None for value in (
        true_keys_above,
        true_keys_below,
        true_sum_above,
        true_sum_below,
    )):
        above_var = true_keys_above
        below_var = true_keys_below
        above_sum = sum(true_sum_above.values())
        below_sum = sum(true_sum_below.values())
        source = 'combined_scenario_solution'
    elif use_fallback_partition:
        above_var = keys_above
        below_var = keys_below
        above_sum = sum(sum_above.values())
        below_sum = sum(sum_below.values())
        source = 'hat_partition'
    else:
        if logger is not None:
            logger.info(
                'Skipping pretrained cut because no combined scenario solution was found for %s.',
                scenario_list,
            )
        return {'added': False, 'expectation': expectation, 'best_hat': best_hat}

    two_stage_decomp_module.model_main.set_pretrained_cut(
        above_var=above_var,
        below_var=below_var,
        above_sum=above_sum,
        below_sum=below_sum,
    )
    return {
        'added': True,
        'source': source,
        'expectation': expectation,
        'best_hat': best_hat,
        'above_sum': above_sum,
        'below_sum': below_sum,
    }
