# -*- coding: utf-8 -*-

import copy
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo

from algo.algo_simple_tools import detect_and_record_repetition
from algo.interfaces import (
    AggregationPolicy,
    CutOutcome,
    IterationContext,
    MainSolution,
    NoOpSolutionGenerator,
    SolutionGenerator,
)
from algo.runner_helpers import IncumbentTracker
from util.names import (
    BendersAnchorName,
    BoundModeName,
    BranchName,
    CutTypeName,
    DataName,
    PhaseName,
    ResultName,
    VarName,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CutSchedule:
    """Iteration ranges for the non-dynamic runner."""

    ranges: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    benders_anchor: str = BendersAnchorName.INTEGER

    def cut_types(self, iteration: int) -> List[str]:
        return [
            cut_type
            for cut_type, iteration_range in self.ranges.items()
            if iteration_range is not None
            and iteration_range[0] <= iteration <= iteration_range[1]
        ]


@dataclass(frozen=True)
class PhasePolicy:
    """Single source of truth for dynamic phases, branches, cuts, and guidance weights."""

    phase_1_end_iteration: int
    phase_2_end_iteration: int
    phase_3_end_iteration: int
    default_weight: float = 0.0
    phase_weights: Dict[str, Optional[float]] = field(default_factory=dict)

    def __post_init__(self):
        if not (
            0 <= self.phase_1_end_iteration
            <= self.phase_2_end_iteration
            <= self.phase_3_end_iteration
        ):
            raise ValueError('Phase endpoints must be nondecreasing and start at or after zero.')

    def phase_name(self, iteration: int) -> str:
        if iteration <= self.phase_1_end_iteration:
            return PhaseName.PHASE_1
        if iteration <= self.phase_2_end_iteration:
            return PhaseName.PHASE_2
        if iteration <= self.phase_3_end_iteration:
            return PhaseName.PHASE_3
        raise ValueError(f'Iteration {iteration} is outside configured phase ranges.')

    def branch_name(self, iteration: int, frac_repeated: bool, current_repeated: bool) -> str:
        phase_name = self.phase_name(iteration)
        if phase_name == PhaseName.PHASE_1:
            return (
                BranchName.PHASE_1_REPEATED_FRAC
                if frac_repeated else BranchName.PHASE_1_NEW_FRAC
            )
        if phase_name == PhaseName.PHASE_2:
            return (
                BranchName.PHASE_2_REPEATED_FRAC
                if frac_repeated else BranchName.PHASE_2_NEW_FRAC
            )
        if not frac_repeated:
            return BranchName.PHASE_3_NEW_FRAC
        if not current_repeated:
            return BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR
        return BranchName.PHASE_3_REPEATED_FRAC_REPEATED_CURR

    @staticmethod
    def candidate_cut_types(branch_name: str) -> List[str]:
        return {
            BranchName.PHASE_1_NEW_FRAC: [CutTypeName.BENDERS],
            BranchName.PHASE_1_REPEATED_FRAC: [],
            BranchName.PHASE_2_NEW_FRAC: [
                CutTypeName.BENDERS,
                CutTypeName.STRENGTHEN_BENDERS,
            ],
            BranchName.PHASE_2_REPEATED_FRAC: [],
            BranchName.PHASE_3_NEW_FRAC: [
                CutTypeName.BENDERS,
                CutTypeName.STRENGTHEN_BENDERS,
                CutTypeName.CGLP,
                CutTypeName.LAGRANGIAN,
            ],
            BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR: [],
            BranchName.PHASE_3_REPEATED_FRAC_REPEATED_CURR: list(CutTypeName.ALL_TYPES),
        }[branch_name]

    def added_obj_term_weight(self, iteration: int) -> float:
        phase_name = self.phase_name(iteration)
        phase_weight = self.phase_weights.get(phase_name)
        return float(self.default_weight if phase_weight is None else phase_weight)

    @staticmethod
    def should_add_integer_fallback(branch_name: str, any_repeated: bool) -> bool:
        if branch_name in {
            BranchName.PHASE_1_REPEATED_FRAC,
            BranchName.PHASE_2_REPEATED_FRAC,
            BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR,
        }:
            return True
        if branch_name in {
            BranchName.PHASE_1_NEW_FRAC,
            BranchName.PHASE_2_NEW_FRAC,
            BranchName.PHASE_3_NEW_FRAC,
        }:
            return any_repeated
        return False

    @staticmethod
    def should_try_post_integer_cglp(branch_name: str) -> bool:
        return branch_name == BranchName.PHASE_3_REPEATED_FRAC_NEW_CURR


class DecompositionRunner:
    """Own the decomposition loop, policies, incumbent, convergence, and iteration reporting."""

    def __init__(
        self,
        two_stage_decomp_module,
        scenario_groups: Sequence[Sequence],
        max_iterations: int,
        cut_schedule: Optional[CutSchedule] = None,
        phase_policy: Optional[PhasePolicy] = None,
        aggregation_policy: AggregationPolicy = AggregationPolicy.PER_GROUP,
        added_obj_term_weight: float = 0.0,
        bound_mode: str = BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
        convergence_rel_gap: float = 0.01,
        record_incumbent_every_k: Optional[int] = None,
        solution_generator: Optional[SolutionGenerator] = None,
        max_lagrangian_iterations: int = 10,
        enable_result_output: bool = False,
        output_dir: Optional[str] = None,
    ):
        if cut_schedule is not None and phase_policy is not None:
            raise ValueError('Choose either CutSchedule or PhasePolicy, not both.')
        if phase_policy is not None and phase_policy.phase_3_end_iteration != max_iterations - 1:
            raise ValueError('phase_3_end_iteration must equal max_iterations - 1.')
        if record_incumbent_every_k is not None and record_incumbent_every_k < 1:
            raise ValueError('record_incumbent_every_k must be positive or None.')
        if bound_mode not in {BoundModeName.RESOLVE_AT_ZERO_WEIGHT, BoundModeName.HEURISTIC}:
            raise ValueError(f'Unsupported bound mode: {bound_mode}')
        if enable_result_output and output_dir is None:
            raise ValueError('output_dir is required when enable_result_output is True.')

        self.module = two_stage_decomp_module
        self.scenario_groups = [list(group) for group in scenario_groups]
        scenarios = list(self.module.scenario_list)
        flat_scenarios = [scenario for group in self.scenario_groups for scenario in group]
        if (
            not scenarios
            or len(set(scenarios)) != len(scenarios)
            or any(not group for group in self.scenario_groups)
            or Counter(flat_scenarios) != Counter(scenarios)
        ):
            raise ValueError('Scenario groups must be a nonempty, duplicate-free partition of scenario_list.')
        self.max_iterations = max_iterations
        self.cut_schedule = cut_schedule or CutSchedule()
        self.phase_policy = phase_policy
        self.aggregation_policy = AggregationPolicy(aggregation_policy)
        self.added_obj_term_weight = float(added_obj_term_weight)
        self.bound_mode = bound_mode
        self.convergence_rel_gap = convergence_rel_gap
        self.record_incumbent_every_k = record_incumbent_every_k
        self.solution_generator = solution_generator or NoOpSolutionGenerator()
        self.max_lagrangian_iterations = max_lagrangian_iterations
        self.enable_result_output = enable_result_output
        self.output_dir = output_dir

        self.incumbent = IncumbentTracker()
        self.iteration_bound_records = []
        self.iteration_branch_cut_records = []
        self.current_solution_history = []
        self.fractional_solution_history = []
        self.current_repetition_records = []
        self.fractional_repetition_records = []
        self._current_seen = {}
        self._fractional_seen = {}
        self._heuristic_anchor_cut_keys = set()

    def _iteration_policy(self, iteration, frac_repeated, current_repeated):
        if self.phase_policy is None:
            return {
                'phase_name': None,
                'branch_name': None,
                'cut_types': self.cut_schedule.cut_types(iteration),
                'weight': self.added_obj_term_weight,
                'benders_anchor': self.cut_schedule.benders_anchor,
            }
        phase_name = self.phase_policy.phase_name(iteration)
        branch_name = self.phase_policy.branch_name(
            iteration,
            frac_repeated=frac_repeated,
            current_repeated=current_repeated,
        )
        return {
            'phase_name': phase_name,
            'branch_name': branch_name,
            'cut_types': self.phase_policy.candidate_cut_types(branch_name),
            'weight': self.phase_policy.added_obj_term_weight(iteration),
            'benders_anchor': BendersAnchorName.FRACTIONAL,
        }

    def _resolve_iteration_lower_bound(self, ite_name: str, current_weight: float):
        if self.module.model_main.has_pretrained_restrictions():
            return None, False
        if current_weight == 0.0:
            lower_bound = self.module.get_main_stage_valid_lower_bound()
            return lower_bound, lower_bound is not None
        if self.bound_mode == BoundModeName.HEURISTIC:
            return None, False

        self.module.model_main.set_added_obj_term_weight(0.0)
        try:
            self.module.solve_main_stage_model()
            lower_bound = self.module.get_main_stage_valid_lower_bound()
        finally:
            self.module.model_main.set_added_obj_term_weight(current_weight)

        record = self.module.ite_obj_value_dict.get(ite_name, {}).get('main_obj(bound)')
        if record is not None:
            record['valid_lower_bound'] = lower_bound
        return lower_bound, lower_bound is not None

    @staticmethod
    def _anchor_signature(solution: MainSolution, cut_type: str):
        values = tuple(
            (var_name, repr(var_key), round(float(value), 8))
            for var_name in sorted(solution.values)
            if var_name != VarName.SUB_OBJ_EST
            for var_key, value in sorted(solution.values[var_name].items(), key=lambda item: repr(item[0]))
        )
        return values, cut_type

    def _filter_heuristic_cut_types(self, solution: MainSolution, cut_types: Sequence[str]):
        filtered = []
        has_theta = VarName.SUB_OBJ_EST in solution.values
        for cut_type in cut_types:
            if cut_type == CutTypeName.CGLP and (
                not has_theta or not self._is_binary_design(solution.values)
            ):
                continue
            if cut_type == CutTypeName.INTEGER_OPT and not self._is_binary_design(solution.values):
                continue
            signature = self._anchor_signature(solution, cut_type)
            if signature in self._heuristic_anchor_cut_keys:
                continue
            self._heuristic_anchor_cut_keys.add(signature)
            filtered.append(cut_type)
        return filtered

    def _is_binary_design(self, values: dict) -> bool:
        for var_name in (VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE):
            if var_name not in values:
                return False
            if any(
                not math.isfinite(float(value))
                or min(abs(float(value)), abs(float(value) - 1.0)) > 1e-6
                for value in values[var_name].values()
            ):
                return False
        return True

    def _validate_heuristic_solution(self, solution: MainSolution) -> MainSolution:
        """Validate and normalize candidate state before it is cached or sent to subproblems."""
        state_vars = self.module.model_main.get_decomposition_state_var_name_list()
        allowed_vars = set(state_vars) | {VarName.SUB_OBJ_EST}
        if not isinstance(solution.values, dict) or not set(state_vars) <= set(solution.values) <= allowed_vars:
            raise ValueError('Generated solutions require exactly the design-state blocks and optional theta.')
        normalized = {}
        for var_name in solution.values:
            block = solution.values[var_name]
            expected_keys = set(self.module.model_main.var[var_name])
            if not isinstance(block, dict) or set(block) != expected_keys:
                raise ValueError(f'Generated solution keys mismatch for {var_name}.')
            normalized[var_name] = {}
            for key, value in block.items():
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Generated {var_name}[{key}] must be numeric.') from exc
                var = self.module.model_main.var[var_name][key]
                if (
                    not math.isfinite(number)
                    or (var.lb is not None and number < var.lb - 1e-6)
                    or (var.ub is not None and number > var.ub + 1e-6)
                ):
                    raise ValueError(f'Generated {var_name}[{key}] is nonfinite or outside its bounds.')
                if var.is_binary() and min(abs(number), abs(number - 1.0)) <= 1e-6:
                    number = float(round(number))
                elif var.lb is not None:
                    number = max(number, float(var.lb))
                normalized[var_name][key] = number
        return MainSolution(
            values=normalized,
            name=solution.name,
            is_relaxed=not self._is_binary_design(normalized),
        )

    def _is_design_feasible(self, values: dict) -> bool:
        if not self._is_binary_design(values):
            return False
        data = self.module.main_model_data
        if sum(round(values[VarName.DG_INSTALL][j]) for j in data[DataName.LIST_NODE]) > data[DataName.NUM_DG_UB]:
            return False
        for node in data[DataName.LIST_NODE]:
            type_sum = sum(
                round(values[VarName.DG_INSTALL_TYPE][node, dg_type])
                for dg_type in data[DataName.LIST_DG_TYPE]
            )
            if type_sum != round(values[VarName.DG_INSTALL][node]):
                return False
        return True

    def _evaluate_heuristic_solutions(self, solutions, base_context, policy):
        records = []
        for index, solution in enumerate(solutions):
            solution = self._validate_heuristic_solution(solution)
            cut_types = self._filter_heuristic_cut_types(solution, policy['cut_types'])
            if not cut_types:
                continue
            feasible_design = self._is_design_feasible(solution.values)
            incumbent_records = [] if feasible_design else None
            context = IterationContext(
                ite_name=f'{base_context.ite_name}_heur_{index}',
                iteration_number=base_context.iteration_number,
                current_solution=solution,
                fractional_solution=MainSolution(
                    values=solution.values,
                    name=solution.name,
                    is_relaxed=True,
                ),
                main_stage_obj=(
                    self.module.model_main.evaluate_design_objective_for_solution(solution.values)
                    if feasible_design else None
                ),
                candidate_cut_types=cut_types,
            )
            cut_records = self.module.run_iteration_cuts(
                sce_group_list=self.scenario_groups,
                iteration_context=context,
                aggregation_policy=self.aggregation_policy,
                incumbent_records=incumbent_records,
                options={
                    'benders_anchor': BendersAnchorName.INTEGER,
                    'max_lagrangian_ite': self.max_lagrangian_iterations,
                    'theta_values': solution.values.get(VarName.SUB_OBJ_EST),
                    'use_current_master_theta': False,
                },
            )
            if incumbent_records:
                self.incumbent.update_from_records(
                    two_stage_decomp_module=self.module,
                    ite_name=context.ite_name,
                    incumbent_records=incumbent_records,
                    main_solution=solution.values,
                )
            records.append({
                'solution_name': solution.name,
                'feasible_design': feasible_design,
                'cut_records': self._serialize_cut_records(cut_records),
            })
        return records

    @staticmethod
    def _serialize_cut_records(records):
        serialized = []
        for record in records:
            serialized.append({
                'scenario_group': record.get('scenario_group'),
                'aggregated': record.get('aggregated', False),
                'generated_cut_types': list(record.get('generated_cut_types', [])),
                'cut_outcomes': {
                    cut_type: outcome.value
                    for cut_type, outcome in record.get('cut_outcomes', {}).items()
                },
            })
        return serialized

    def _dispatch_followup_cut(
        self,
        cut_type,
        report_key,
        cut_records,
        context,
        theta_values,
        request_options,
    ):
        """Dispatch a policy-triggered cut while preserving the configured aggregation mode."""
        generated_cuts = []
        for record in cut_records:
            cut_result = record.get('subproblem_result')
            if cut_result is None:
                continue
            request = self.module.build_cut_request(
                cut_type=cut_type,
                cut_result=cut_result,
                iteration_context=context,
                theta_values=theta_values,
                options=request_options,
            )
            if self.aggregation_policy == AggregationPolicy.PER_GROUP:
                outcome, _ = self.module.dispatch_cut(
                    cut_type,
                    request,
                    include_cut_type_in_repetition=False,
                )
                record['cut_outcomes'][report_key] = outcome
                continue

            generated_cut = self.module.generate_cut(cut_type, request)
            if generated_cut is None:
                record['cut_outcomes'][report_key] = CutOutcome.NOT_VIOLATED
                continue
            record.setdefault('generated_cut_types', []).append(report_key)
            generated_cuts.append(generated_cut)

        if self.aggregation_policy == AggregationPolicy.AGGREGATE and generated_cuts:
            outcome, _ = self.module.dispatch_aggregated_cuts(
                cut_type=cut_type,
                ite_name=context.ite_name,
                generated_cuts=generated_cuts,
                include_cut_type_in_repetition=False,
                cut_name_prefix=report_key,
            )
            cut_records.append({
                'scenario_group': None,
                'cut_outcomes': {report_key: outcome},
                'aggregated': True,
            })

    def run(self):
        state_vars = list(self.module.model_main.get_decomposition_state_var_name_list())
        for ite_num in range(self.max_iterations):
            ite_name = str(ite_num)
            provisional_weight = (
                self.phase_policy.added_obj_term_weight(ite_num)
                if self.phase_policy is not None else self.added_obj_term_weight
            )
            self.module.model_main.set_added_obj_term_weight(provisional_weight)
            self.module.solve_main_stage_model()
            main_stage_obj, reported_bound = self.module.record_main_stage_model(ite_name=ite_name)
            current_values = copy.deepcopy(self.module.model_main.get_result(state_vars))
            theta_values = {
                scenario: pyo.value(self.module.model_main.var[VarName.SUB_OBJ_EST][scenario])
                for scenario in self.module.scenario_list
            }
            self.module.solve_main_stage_relaxed_model()
            fractional_values = copy.deepcopy(self.module.model_main.get_result_relaxed(state_vars))

            current_repeated, current_first = detect_and_record_repetition(
                current_values, self._current_seen, ite_num, tol=1e-6
            )
            frac_repeated, frac_first = detect_and_record_repetition(
                fractional_values, self._fractional_seen, ite_num, tol=1e-6
            )
            self.current_solution_history.append(copy.deepcopy(current_values))
            self.fractional_solution_history.append(copy.deepcopy(fractional_values))
            self.current_repetition_records.append({
                'iteration': ite_num,
                'is_repeated': current_repeated,
                'first_seen_iteration': current_first,
            })
            self.fractional_repetition_records.append({
                'iteration': ite_num,
                'is_repeated': frac_repeated,
                'first_seen_iteration': frac_first,
            })

            policy = self._iteration_policy(ite_num, frac_repeated, current_repeated)
            context = IterationContext(
                ite_name=ite_name,
                iteration_number=ite_num,
                current_solution=MainSolution(current_values, name='current'),
                fractional_solution=MainSolution(fractional_values, name='fractional', is_relaxed=True),
                main_stage_obj=main_stage_obj,
                candidate_cut_types=policy['cut_types'],
            )

            heuristic_solutions = self.solution_generator.generate(context)
            if not isinstance(heuristic_solutions, list):
                raise TypeError('SolutionGenerator.generate() must return a list.')
            if any(not isinstance(solution, MainSolution) for solution in heuristic_solutions):
                raise TypeError('SolutionGenerator.generate() must return MainSolution objects.')
            heuristic_records = self._evaluate_heuristic_solutions(
                heuristic_solutions, context, policy
            )

            do_record_incumbent = (
                self.record_incumbent_every_k is None
                or ite_num % self.record_incumbent_every_k == 0
            )
            incumbent_records = [] if do_record_incumbent else None
            force_exact = (
                self.phase_policy is not None
                and (frac_repeated or policy['phase_name'] == PhaseName.PHASE_3)
            )
            cut_records = self.module.run_iteration_cuts(
                sce_group_list=self.scenario_groups,
                iteration_context=context,
                aggregation_policy=self.aggregation_policy,
                incumbent_records=incumbent_records,
                options={
                    'benders_anchor': policy['benders_anchor'],
                    'max_lagrangian_ite': self.max_lagrangian_iterations,
                    'theta_values': theta_values,
                    'force_exact_subproblem': force_exact,
                    'include_cut_type_in_repetition': self.phase_policy is None,
                },
            )

            if self.phase_policy is not None:
                any_repeated = any(
                    outcome == CutOutcome.REPEATED
                    for record in cut_records
                    for outcome in record.get('cut_outcomes', {}).values()
                )
                if self.phase_policy.should_add_integer_fallback(policy['branch_name'], any_repeated):
                    self._dispatch_followup_cut(
                        cut_type=CutTypeName.INTEGER_OPT,
                        report_key=CutTypeName.INTEGER_OPT,
                        cut_records=cut_records,
                        context=context,
                        theta_values=theta_values,
                        request_options={'lower_bound': 0.0},
                    )
                if self.phase_policy.should_try_post_integer_cglp(policy['branch_name']):
                    post_integer_key = f'{CutTypeName.CGLP}_post_integer'
                    self._dispatch_followup_cut(
                        cut_type=CutTypeName.CGLP,
                        report_key=post_integer_key,
                        cut_records=cut_records,
                        context=context,
                        theta_values=theta_values,
                        request_options={'cut_name_prefix': post_integer_key},
                    )

            if incumbent_records is not None:
                self.incumbent.update_from_records(
                    two_stage_decomp_module=self.module,
                    ite_name=ite_name,
                    incumbent_records=incumbent_records,
                    main_solution=current_values,
                )

            valid_bound, is_bound_valid = self._resolve_iteration_lower_bound(
                ite_name=ite_name,
                current_weight=policy['weight'],
            )
            self.iteration_bound_records.append({
                'iteration': ite_name,
                'added_obj_term_weight': policy['weight'],
                'reported_bound_at_incumbent': reported_bound,
                'valid_lower_bound': valid_bound,
                'is_bound_valid': is_bound_valid,
            })
            self.iteration_branch_cut_records.append({
                'iteration': ite_name,
                'phase_name': policy['phase_name'],
                'branch_name': policy['branch_name'],
                'candidate_cut_types': list(policy['cut_types']),
                'scenario_group_records': self._serialize_cut_records(cut_records),
                'heuristic_records': heuristic_records,
            })

            if self.enable_result_output:
                self.incumbent.write_best_solution(self.output_dir)
                self.module.write_iteration_csvs(
                    output_dir=self.output_dir,
                    scenario_list=self.module.scenario_list,
                )
            if do_record_incumbent and self.incumbent.has_converged(
                main_bound=valid_bound,
                rel_gap=self.convergence_rel_gap,
            ):
                break

        if self.enable_result_output:
            self.module.model_main.write_file(f'{self.output_dir}/main_model.lp')

        return {
            'iteration_bound_records': self.iteration_bound_records,
            'iteration_branch_cut_records': self.iteration_branch_cut_records,
            ResultName.BEST_INCUMBENT_OBJ: self.incumbent.best_obj,
            ResultName.BEST_MAIN_SOLUTION: copy.deepcopy(self.incumbent.best_main_solution),
            ResultName.BEST_INCUMBENT_SOURCE: self.incumbent.best_source,
            'best_sub_results': self.incumbent.best_sub_results,
            'current_solution_history': self.current_solution_history,
            'fractional_solution_history': self.fractional_solution_history,
            'current_repetition_records': self.current_repetition_records,
            'fractional_repetition_records': self.fractional_repetition_records,
        }
