# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from util.names import BendersAnchorName, CutTypeName, DataName, ModelStatus, ObjName, SolverRoleName, VarName

from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub, ModelLagrangianMultiplierHeuristic, PSInnerMinimizationProblem, ModelCGLP
from model.model_base import ModelSolveError
from model.solver_config import SolverConfigSet
from algo.interfaces import (
    AggregationPolicy,
    CutOutcome,
    CutRequest,
    DualInfo,
    GeneratedCut,
    InnerMinimizationResult,
    IterationContext,
    MainSolution,
    ScenarioGroup,
    SubProblemResult,
)
from algo.algo_simple_tools import (
    CutRepetitionTracker,
    build_cglp_lower_bound_row_from_alpha_eta,
    build_cglp_lower_bound_row_from_multiplier_cut,
    extract_affine_terms_over_vars,
    flatten_ordered_var_values,
    normalize_affine_lower_bound_row,
)
from util.tools import iter_general_csv, iter_sub_prob_info

import pyomo.environ as pyo
import logging
import math


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CutGeneratorSpec:
    """Registry metadata needed to prepare and validate one cut family."""

    generator: Callable[[CutRequest], Optional[GeneratedCut]]
    anchor_source: str
    required_request_fields: Tuple[str, ...]
    requires_exact_subproblem: bool = False
    requires_integer_anchor: bool = False


class TwoStageDecompRedo:

    def __init__(self, raw_data, time_list, scenario_list, solver_config_set: Optional[SolverConfigSet] = None):

        self.data_processor_module = DataProcessor(raw_data=raw_data)
        self.time_list = time_list
        self.scenario_list = scenario_list

        # Role-scoped solver configuration: every model built below asks for the config of its own
        # role instead of relying on a shared default plus inline set_parameters calls.
        self.solver_config_set = solver_config_set or SolverConfigSet.build()

        self.main_model_data, self.sub_model_data_dict = {}, {}

        self.sce_prob_dict = {}

        self.model_main, self.current_sub_model, self.curr_sub_frac_model = None, None, None

        # the dict for collecting objective values
        self.ite_obj_value_dict = {}

        # the dict for restoring inner minimization model
        self.inner_minimization_model_dict = {}

        # persistent CGLP state per scenario group
        self.cglp_model_dict = {}
        self.cglp_lower_bound_row_dict = {}
        self.cglp_exact_point_value_dict = {}
        self.main_stage_structural_constr_name_list = []
        self.cglp_x_var_key_list = []
        self.cglp_z_var_key_list = []

        # global cut repetition tracking over all added cuts
        self.cut_repetition_tracker = CutRepetitionTracker()
        self.cut_metadata_records = []
        self.cut_generator_registry = {}
        self._register_cut_generators()

        # Opt-in: store the affine algebra (theta coefficients, rhs constant and rhs x coefficients)
        # of every inserted cut so the cut-validity harness can replay it offline. Off by default so
        # production runs pay nothing.
        self.enable_cut_algebra_record = False

    def _register_cut_generators(self):
        """Install the cut-type to generator-function mapping used by all insertion paths."""
        self.cut_generator_registry = {
            CutTypeName.BENDERS: CutGeneratorSpec(
                generator=self._generate_benders_cut,
                anchor_source='benders_option',
                required_request_fields=('dual_info', 'relaxed_obj'),
            ),
            CutTypeName.INTEGER_OPT: CutGeneratorSpec(
                generator=self._generate_integer_opt_cut,
                anchor_source='current',
                required_request_fields=('exact_obj_bound',),
                requires_exact_subproblem=True,
                requires_integer_anchor=True,
            ),
            CutTypeName.STRENGTHEN_BENDERS: CutGeneratorSpec(
                generator=self._generate_strengthen_benders_cut,
                anchor_source='fractional',
                required_request_fields=('dual_info', 'relaxed_obj'),
            ),
            CutTypeName.LAGRANGIAN: CutGeneratorSpec(
                generator=self._generate_lagrangian_cut,
                anchor_source='fractional',
                required_request_fields=('dual_info', 'relaxed_obj'),
            ),
            CutTypeName.CGLP: CutGeneratorSpec(
                generator=self._generate_cglp_cut,
                anchor_source='current',
                required_request_fields=('exact_obj_bound', 'theta_values'),
                requires_exact_subproblem=True,
                requires_integer_anchor=True,
            ),
        }

    def get_solver_config(self, role: str):
        """Return the SolverConfig registered for one SolverRoleName."""
        return self.solver_config_set.get(role)

    def get_group_theta_weights(self, sub_sce_list) -> Dict[Any, float]:
        """Return {s: P_s} for one scenario group, taken from the master's own weights."""
        return {s: self.model_main.get_theta_weight(s) for s in sub_sce_list}

    def build_group_theta_lhs(self, sub_sce_list):
        """
        Canonical cut left-hand side for one scenario group: ``sum_{s in G} P_s * theta_s``.

        This matches the master objective term ``sum_s P_s * theta_s`` and the subproblem objective
        ``sum_{s in G} P_s * Q_s(x)``, which is what makes a group cut valid for ANY grouping
        (risk R3.4). For singleton groups it is the previous unweighted cut multiplied by P_s.
        """
        return self.model_main.get_weighted_theta_expr(sub_sce_list)

    def _get_cut_generator(self, cut_type: str):
        """Return the registered generator for a cut type and fail early on unknown names."""
        if cut_type not in self.cut_generator_registry:
            raise ValueError(f'Unknown cut generator type: {cut_type}')
        return self.cut_generator_registry[cut_type].generator

    def build_main_stage_model(self):

        # Pin the run-level scenario universe BEFORE any data is processed, so that every
        # subproblem's DICT_SC_PROB_UNCOND agrees with the master's theta weights.
        self.data_processor_module.reference_scenario_list = list(self.scenario_list)

        # process the data for main model
        self.main_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=self.scenario_list,
            time_list_assigned=self.time_list
        )

        self.sce_prob_dict = self.main_model_data[DataName.DICT_SC_PROB_UNCOND].copy()

        # build the main model with processed main model data
        self.model_main = ModelMain(
            model_name='Main',
            model_data=self.main_model_data,
            solver_config=self.get_solver_config(SolverRoleName.MASTER),
        )
        self.model_main.build_main_model()

        self.main_stage_structural_constr_name_list = self.model_main.get_decomposition_structural_constr_name_list()
        self.cglp_x_var_key_list, self.cglp_z_var_key_list = self.model_main.get_decomposition_var_key_lists_for_cglp()

    def solve_main_stage_model(self):
        # solve the main model, and compute the obj terms.
        # Solver options come from the master role of the SolverConfigSet; there is no inline override.
        self.model_main.solve()
        self.model_main.cal_detailed_obj()

    def solve_main_stage_relaxed_model(self):
        self.model_main.solve_relaxed()

    def get_main_stage_valid_lower_bound(self) -> Optional[float]:
        """
        Return a guaranteed lower bound from the last master solve, or None when none is available.

        None means the master was solved with a penalized (guidance) objective, so no valid bound
        can be read from it; the caller decides whether to re-solve at weight 0 or to declare the
        run heuristic (risk R3.3).
        """
        if (
            not self.model_main.is_bound_objective_active()
            or self.model_main.has_pretrained_restrictions()
        ):
            return None
        return self.model_main.get_valid_lower_bound()

    def record_main_stage_model(self, ite_name=None):
        # show and record the main model objective value (the best bound objective value)
        total_objective_value = self.model_main.get_reported_base_objective_value()
        main_stage_objective_value = self.model_main.get_design_objective_value()
        valid_lower_bound = self.get_main_stage_valid_lower_bound()

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        # record the objective value and values of every terms
        self.ite_obj_value_dict[ite_name]['main_obj(bound)'] = {
            'total': total_objective_value,
            'valid_lower_bound': valid_lower_bound,
            'added_obj_term_weight': self.model_main.added_obj_term_weight,
            'detail': self.model_main.obj_term_value.copy()
        }

        return main_stage_objective_value, total_objective_value

    def build_sub_model(
        self,
        sub_model_sce_list=None,
        given_main_result=None,
        target_slot: str = 'current_sub_model',
        solver_role: str = SolverRoleName.SUBPROBLEM_EXACT,
    ):
        """Build a canonical local-copy subproblem into the requested model slot."""
        if sub_model_sce_list is None:
            raise ValueError('sub_model_sce_list cannot be None')
        if target_slot not in {'current_sub_model', 'curr_sub_frac_model'}:
            raise ValueError(f'Unsupported subproblem target slot: {target_slot}')

        sce_dict_key = tuple(sub_model_sce_list)
        if sce_dict_key not in self.sub_model_data_dict:
            # process the sub model data
            sub_model_data = self.data_processor_module.data_process(
                scenario_list_assigned=sub_model_sce_list,
                time_list_assigned=self.time_list
            )
            self.sub_model_data_dict[sce_dict_key] = sub_model_data
        else:
            sub_model_data = self.sub_model_data_dict[sce_dict_key]

        sub_model = ModelSub(
            model_name=f'{tuple(sub_model_sce_list)}_model',
            model_data=sub_model_data,
            main_result=given_main_result,
            sub_model_sce_list=sub_model_sce_list,
            solver_config=self.get_solver_config(solver_role),
        )
        sub_model.build_sub_model_canonical()
        setattr(self, target_slot, sub_model)
        return sub_model

    def solve_sub_model(self):
        self.current_sub_model.solve()
        self.current_sub_model.cal_detailed_obj()

    def solve_relaxed_sub_model(self):
        self.current_sub_model.solve_relaxed(
            solver_config=self.get_solver_config(SolverRoleName.SUBPROBLEM_LP)
        )

    def solve_sub_frac_model(self):
        self.curr_sub_frac_model.solve_relaxed()

    def get_group_probability_weight(self, sub_sce_list) -> float:
        """Total unconditional probability carried by a scenario group."""
        return sum(self.model_main.get_theta_weight(s) for s in sub_sce_list)

    def record_sub_model(self, main_stage_obj_value=None, ite_name=None):
        """
        Record the exact subproblem incumbent for one scenario group.

        ``sce_obj_value`` is already probability weighted (sum_{s in G} P_s * Q_s), so the design
        cost is added in proportion to the group's own probability weight. Summing 'total' over a
        partition of the scenario universe therefore reproduces the incumbent objective exactly.
        """
        sce_obj_value = self.current_sub_model.get_obj_value()
        group_weight = self.get_group_probability_weight(self.current_sub_model.sub_model_sce_list)
        main_stage_obj_value = 0.0 if main_stage_obj_value is None else main_stage_obj_value
        sce_obj_value_w_main = sce_obj_value + group_weight * main_stage_obj_value

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        detail = self.current_sub_model.obj_term_value.copy()
        detail['group_probability_weight'] = group_weight
        detail['weighted_sub_obj'] = sce_obj_value

        self.ite_obj_value_dict[ite_name][tuple(self.current_sub_model.sub_model_sce_list)] = {
            'total': sce_obj_value_w_main,
            'detail': detail
        }

        return sce_obj_value_w_main, sce_obj_value

    def write_iteration_csvs(self, output_dir: str, scenario_list: Optional[list] = None) -> None:
        """Create or update iteration CSVs (iter_general_info.csv, iter_subproblem_info.csv)."""
        if scenario_list is None:
            scenario_list = self.scenario_list
        iter_general_csv(ite_obj_value_dict=self.ite_obj_value_dict, output_dir=output_dir)
        iter_sub_prob_info(
            ite_obj_value_dict=self.ite_obj_value_dict,
            output_dir=output_dir,
            scenario_list=scenario_list,
        )

    def _register_group_lower_bound_row(self, sub_sce_list: List, row: dict):
        """Append one normalized lower-bound row to the persistent CGLP state for a group."""
        group_key = tuple(sub_sce_list)
        self._ensure_group_lower_bound_rows(sub_sce_list=sub_sce_list)
        self.cglp_lower_bound_row_dict[group_key].append(row)

    def _record_cut_metadata(
        self,
        generated_cut: GeneratedCut,
        outcome: CutOutcome,
        repetition_record: Optional[dict] = None,
    ) -> None:
        """Store lightweight bookkeeping for each generated cut insertion decision."""
        scenario_group = self._get_cut_tracking_scenario_group(generated_cut)
        record = {
            "cut_type": generated_cut.cut_type,
            "cut_name": generated_cut.cut_name,
            "iteration": generated_cut.ite_name,
            "scenario_group": scenario_group,
            "outcome": outcome.value,
            "added": outcome == CutOutcome.ADDED,
            "has_lower_bound_row": generated_cut.lower_bound_row is not None,
            "repetition": repetition_record,
            "metadata": generated_cut.metadata.copy(),
        }
        if self.enable_cut_algebra_record:
            record["cut_algebra"] = self.extract_cut_algebra(generated_cut)
        self.cut_metadata_records.append(record)

    def extract_cut_algebra(self, generated_cut: GeneratedCut) -> dict:
        """
        Reduce one cut to plain numbers: the theta coefficients of its left-hand side and the affine
        form of its right-hand side over the master state variables.

        This is the input of the cut-validity harness (B3): a cut ``sum_s c_s theta_s >= r(x)`` is
        valid iff ``sum_s c_s Q_s(x) >= r(x)`` for every x in the first-stage feasible set, which the
        harness can then check offline without re-entering Pyomo.
        """
        theta_id_map = {
            id(var_ob): scenario
            for scenario, var_ob in self.model_main.var[VarName.SUB_OBJ_EST].items()
        }
        state_id_map = {
            id(self.model_main.var[var_name][var_key]): (var_name, var_key)
            for var_name, var_key in self.cglp_x_var_key_list
        }

        combined_id_map = dict(theta_id_map)
        combined_id_map.update(state_id_map)

        lhs_constant, theta_coef = extract_affine_terms_over_vars(generated_cut.lhs, theta_id_map)
        rhs_constant, rhs_x_coef = extract_affine_terms_over_vars(generated_cut.rhs, state_id_map)

        # Any master variable outside {theta} U {binary state} would make the offline replay
        # incomplete, so it is counted rather than silently dropped.
        _, rhs_all_coef = extract_affine_terms_over_vars(generated_cut.rhs, combined_id_map)

        return {
            "theta_coef": theta_coef,
            "lhs_constant": lhs_constant,
            "rhs_constant": rhs_constant,
            "rhs_x_coef": rhs_x_coef,
            "rhs_theta_coef_count": len(rhs_all_coef) - len(rhs_x_coef),
        }

    @staticmethod
    def _get_cut_tracking_scenario_group(generated_cut: GeneratedCut):
        """Return the scenario-group label used in repetition reports."""
        if generated_cut.metadata.get("is_aggregated", False):
            return None
        return generated_cut.scenario_group.as_list()

    def _insert_generated_cut(
        self,
        generated_cut: GeneratedCut,
        include_cut_type_in_repetition: bool = True,
        register_lower_bound_row: bool = True,
    ) -> CutOutcome:
        """Centralize master insertion, repetition tracking, CGLP row updates, and metadata."""
        tracking_scenario_group = self._get_cut_tracking_scenario_group(generated_cut)
        if generated_cut.lower_bound_row is not None:
            repetition_record = self.cut_repetition_tracker.record_by_row(
                cut_type=generated_cut.cut_type,
                ite_name=generated_cut.ite_name,
                cut_name=generated_cut.cut_name,
                row=generated_cut.lower_bound_row,
                sub_sce_list=tracking_scenario_group,
                include_cut_type=include_cut_type_in_repetition,
                theta_weights=self.get_group_theta_weights(generated_cut.scenario_group.as_list()),
            )
        else:
            repetition_record = self.cut_repetition_tracker.record_by_expr(
                cut_type=generated_cut.cut_type,
                ite_name=generated_cut.ite_name,
                cut_name=generated_cut.cut_name,
                expr=generated_cut.lhs - generated_cut.rhs,
                sub_sce_list=tracking_scenario_group,
                include_cut_type=include_cut_type_in_repetition,
            )

        if repetition_record['is_repeated']:
            self._record_cut_metadata(
                generated_cut=generated_cut,
                outcome=CutOutcome.REPEATED,
                repetition_record=repetition_record,
            )
            return CutOutcome.REPEATED

        self.model_main.add_custom_cut(
            cut_name=generated_cut.cut_name,
            cut_lhs=generated_cut.lhs,
            cut_rhs=generated_cut.rhs,
        )
        if register_lower_bound_row and generated_cut.lower_bound_row is not None:
            self._register_group_lower_bound_row(
                sub_sce_list=generated_cut.scenario_group.as_list(),
                row=generated_cut.lower_bound_row,
            )

        self._record_cut_metadata(
            generated_cut=generated_cut,
            outcome=CutOutcome.ADDED,
            repetition_record=repetition_record,
        )
        return CutOutcome.ADDED

    def get_repetitive_cut_records(self) -> List[dict]:
        """Return recorded cut-insertion attempts that repeated a previous cut signature."""
        return self.cut_repetition_tracker.get_repetitive_records()

    def _ensure_group_lower_bound_rows(self, sub_sce_list: List):
        """Initialize the default zero lower-bound row for a scenario group if needed."""
        group_key = tuple(sub_sce_list)
        if group_key in self.cglp_lower_bound_row_dict:
            return

        self.cglp_lower_bound_row_dict[group_key] = [
            normalize_affine_lower_bound_row(
                cut_name=f'cglp_lb_0_{group_key}',
                constant_term=0.0,
                var_coef_dict={},
            )
        ]

    def _register_exact_point_for_group(self, sub_sce_list: List, forward_sol: dict, exact_sub_obj_value: float):
        """Record one exact evaluated point for later CGLP cut generation."""
        group_key = tuple(sub_sce_list)
        x_point = flatten_ordered_var_values(
            var_value_dict=forward_sol,
            ordered_var_keys=self.cglp_x_var_key_list,
        )

        if group_key not in self.cglp_exact_point_value_dict:
            self.cglp_exact_point_value_dict[group_key] = {}

        point_values = self.cglp_exact_point_value_dict[group_key]
        point_values[x_point] = max(point_values.get(x_point, -math.inf), float(exact_sub_obj_value))

        if group_key in self.cglp_model_dict:
            self.cglp_model_dict[group_key].update_with_exact_point(
                x_point=x_point,
                q_value=point_values[x_point],
                make_tight=False,
            )

        return x_point

    def _generate_benders_cut(self, request: CutRequest) -> GeneratedCut:
        """Generate a classical Benders cut without mutating the master model."""
        scenario_group = request.scenario_group
        dual_info = request.dual_info
        cut_name_prefix = request.options.get('cut_name_prefix', 'bd')
        lhs, rhs = self.build_multiplier_cut(
            base_value=request.relaxed_obj,
            multiplier=dual_info.dual_values,
            constr_var_map=dual_info.var_map,
            anchor_main_result=request.anchor_solution.as_model_result(),
            sub_model_sce_list=scenario_group.as_list(),
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{request.ite_name}"
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=request.relaxed_obj,
            multiplier_info=dual_info.dual_values,
            constr_var_map=dual_info.var_map,
            ref_point=request.anchor_solution.as_model_result(),
        )
        return GeneratedCut(
            cut_type=CutTypeName.BENDERS,
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=request.ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={"base_value": request.relaxed_obj},
        )

    def _generate_integer_opt_cut(self, request: CutRequest) -> GeneratedCut:
        """Generate an integer optimality cut without mutating the master model."""
        scenario_group = request.scenario_group
        scenario_group_list = scenario_group.as_list()
        cut_name_prefix = request.options.get('cut_name_prefix', 'int_opt')
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{request.ite_name}"
        tol = 1e-6
        exact_sub_obj_value = float(request.exact_obj_bound)
        lower_bound = float(request.options.get('lower_bound', 0.0))
        forward_sol = request.anchor_solution.as_model_result()

        if lower_bound > exact_sub_obj_value + tol:
            raise ValueError(
                f"lower_bound ({lower_bound}) cannot exceed exact_sub_obj_value ({exact_sub_obj_value})"
            )

        lhs = self.build_group_theta_lhs(scenario_group_list)

        gap = exact_sub_obj_value - lower_bound
        one_count = 0
        var_coef_dict = {}
        rhs_terms = []

        for var_name, var_key in self.cglp_x_var_key_list:
            var_value = float(forward_sol[var_name][var_key])
            if math.isclose(var_value, 1.0, abs_tol=tol):
                coef = gap
                one_count += 1
            elif math.isclose(var_value, 0.0, abs_tol=tol):
                coef = -gap
            else:
                raise ValueError(
                    f"Integer optimality cut requires binary forward solution; got {var_name}[{var_key}]={var_value}"
                )

            if var_name not in var_coef_dict:
                var_coef_dict[var_name] = {}
            var_coef_dict[var_name][var_key] = coef
            rhs_terms.append(coef * self.model_main.var[var_name][var_key])

        constant_term = exact_sub_obj_value - gap * one_count
        rhs = constant_term + pyo.quicksum(rhs_terms)
        row = normalize_affine_lower_bound_row(
            cut_name=cut_name,
            constant_term=constant_term,
            var_coef_dict=var_coef_dict,
        )

        return GeneratedCut(
            cut_type=CutTypeName.INTEGER_OPT,
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=request.ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "exact_sub_obj_value": exact_sub_obj_value,
                "lower_bound": lower_bound,
                "gap": gap,
                "one_count": one_count,
            },
        )

    def _generate_strengthen_benders_cut(self, request: CutRequest) -> GeneratedCut:
        """Generate a strengthened Benders cut and its normalized row."""
        scenario_group = request.scenario_group
        dual_info = request.dual_info
        given_main_result = request.anchor_solution.as_model_result()
        cut_name_prefix = request.options.get('cut_name_prefix', 'str_bd')
        lhs, rhs, inner_result = self.gen_sce_strengthen_bds_cut(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=given_main_result,
            given_dual_info=dual_info.dual_values,
            constr_var_map=dual_info.var_map,
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{request.ite_name}"
        # Anchor: L_lb + lambda^T (y - y*), so the row constant is L_lb - lambda^T y*.
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=inner_result.lagrangian_lower_bound,
            multiplier_info=dual_info.dual_values,
            constr_var_map=dual_info.var_map,
            ref_point=given_main_result,
        )
        return GeneratedCut(
            cut_type=CutTypeName.STRENGTHEN_BENDERS,
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=request.ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "lagrangian_lower_bound": inner_result.lagrangian_lower_bound,
                "inner_incumbent_sub_obj": inner_result.incumbent_sub_obj,
                "inner_incumbent_main_sol": inner_result.incumbent_main_sol,
                "inner_solve_status": inner_result.solve_status,
            },
        )

    def _generate_lagrangian_cut(self, request: CutRequest) -> GeneratedCut:
        """Generate a Lagrangian cut and its normalized row."""
        scenario_group = request.scenario_group
        dual_info = request.dual_info
        given_main_result = request.anchor_solution.as_model_result()
        max_ite_num = request.options.get('max_lagrangian_ite', 10)
        cut_name_prefix = request.options.get('cut_name_prefix', 'str_lag')
        lhs, rhs, lagrangian_lower_bound, lag_multiplier = self.gen_sub_sce_lag_cut_heuristic(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=given_main_result,
            given_dual_info=dual_info.dual_values,
            constr_var_map=dual_info.var_map,
            max_ite_num=max_ite_num,
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{request.ite_name}"
        # Anchor: L_lb(lambda) + lambda^T (y - y*), with y* the master point the multipliers
        # were built at, so the row constant is L_lb - lambda^T y*.
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=lagrangian_lower_bound,
            multiplier_info=lag_multiplier,
            constr_var_map=dual_info.var_map,
            ref_point=given_main_result,
        )
        return GeneratedCut(
            cut_type=CutTypeName.LAGRANGIAN,
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=request.ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "lagrangian_lower_bound": lagrangian_lower_bound,
                "anchor_main_sol": given_main_result,
                "lag_multiplier": lag_multiplier,
            },
        )

    def _generate_cglp_cut(self, request: CutRequest) -> Optional[GeneratedCut]:
        """Solve/update the CGLP and return a generated cut when the current theta is violated."""
        scenario_group = request.scenario_group
        exact_sub_obj_value = request.exact_obj_bound
        cut_name_prefix = request.options.get('cut_name_prefix', CutTypeName.CGLP)
        group_key = scenario_group.key
        self._ensure_group_lower_bound_rows(sub_sce_list=scenario_group.as_list())

        if group_key not in self.cglp_model_dict:
            cglp_model = ModelCGLP(
                main_model=self.model_main,
                x_var_keys=self.cglp_x_var_key_list,
                z_var_keys=self.cglp_z_var_key_list,
                structural_constr_names=self.main_stage_structural_constr_name_list,
                theta_keys=scenario_group.as_list(),
                lower_bound_rows=self.cglp_lower_bound_row_dict[group_key],
                model_name=f'cglp_{group_key}',
                solver_config=self.get_solver_config(SolverRoleName.CGLP),
                theta_weight_dict=self.get_group_theta_weights(scenario_group.as_list()),
            )
            cglp_model.build_model()
            for x_point, q_value in self.cglp_exact_point_value_dict.get(group_key, {}).items():
                cglp_model.update_with_exact_point(
                    x_point=x_point,
                    q_value=q_value,
                )
            self.cglp_model_dict[group_key] = cglp_model

        cglp_model = self.cglp_model_dict[group_key]
        cglp_model.sync_lower_bound_rows(self.cglp_lower_bound_row_dict[group_key])
        x_point = self._register_exact_point_for_group(
            scenario_group.as_list(), request.anchor_solution.as_model_result(), exact_sub_obj_value,
        )
        cglp_model.set_tight_point(x_point)
        exact_sub_obj_value = self.cglp_exact_point_value_dict[group_key][x_point]
        # The violation test compares like with like: the WEIGHTED master theta sum against the
        # weighted group recourse bound that the cut would enforce.
        theta_value = sum(
            self.model_main.get_theta_weight(sce_idx) * request.theta_values[sce_idx]
            for sce_idx in scenario_group.as_list()
        )

        if theta_value + 1e-6 >= exact_sub_obj_value:
            return None

        alpha, eta, lhs, rhs = cglp_model.solve_and_build_group_cut(model_main=self.model_main)
        cut_value = -float(eta) - sum(alpha[key] * x_point[i] for i, key in enumerate(self.cglp_x_var_key_list))
        if theta_value + 1e-6 >= cut_value:
            return None
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{request.ite_name}"
        row = build_cglp_lower_bound_row_from_alpha_eta(
            cut_name=cut_name,
            alpha=alpha,
            eta=eta,
        )
        row["name"] = cut_name
        return GeneratedCut(
            cut_type=CutTypeName.CGLP,
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=request.ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "alpha": alpha,
                "eta": eta,
                "exact_sub_obj_value": exact_sub_obj_value,
                "theta_value": theta_value,
            },
        )

    def solve_inner_minimization(
        self,
        sub_model_sce_list=None,
        given_main_result=None,
        given_dual_info=None,
        constr_var_map=None,
    ) -> InnerMinimizationResult:
        r"""
        Solve the Lagrangian inner minimization for one scenario group and one multiplier.

        Model (Rahmaniani et al. 2020, MILP (9)):

            L(lambda) = min_{x, y in X_sub}  c^T x - lambda^T (y - y*)

        Two different quantities come out of this solve and they must not be confused:

        * ``lagrangian_lower_bound`` -- the solver's DUAL bound on L(lambda). It satisfies
          ``L_lb <= L(lambda)`` under any MIP gap or time limit, so it is the only value that may
          anchor a cut. This is the fix for risk R3.1: the previous code read the INCUMBENT
          objective, which for a minimization is an UPPER bound on L(lambda), so a truncated solve
          could produce a cut strictly stronger than valid and cut off the true optimum.
        * ``incumbent_sub_obj`` / ``incumbent_main_sol`` -- the (c^T xbar, zbar) pair of the returned
          feasible point. These stay incumbent values on purpose: they feed the multiplier
          heuristic's outer approximation ``eta <= c^T xbar + lambda^T (y* - zbar)``, which is a
          valid over-estimator of L(lambda) for ANY feasible point.
        """
        dict_key = tuple(sub_model_sce_list)
        if dict_key not in self.inner_minimization_model_dict:
            sub_model_data = self.sub_model_data_dict[dict_key]
            inner_model = PSInnerMinimizationProblem(
                model_name=f'{dict_key}_inm',
                model_data=sub_model_data,
                main_result=given_main_result,
                constr_dual_info=given_dual_info,
                constr_var_map=constr_var_map,
                solver_config=self.get_solver_config(SolverRoleName.INNER_MIN),
            )
            inner_model.build_model()
            self.inner_minimization_model_dict[dict_key] = inner_model
        else:
            inner_model = self.inner_minimization_model_dict[dict_key]
            inner_model.main_result = given_main_result
            inner_model.set_objective_function(lag_multiplier=given_dual_info)

        # Solver options come from the inner_min role. A gap or time limit here now only WEAKENS the
        # cut, because the anchor below is the dual bound.
        inner_model.solve()

        main_var_name_list = self.model_main.get_decomposition_state_var_name_list()
        incumbent_main_sol = inner_model.get_result(main_var_name_list)
        inner_model.cal_detailed_obj()
        incumbent_sub_obj = inner_model.obj_term_value[ObjName.SUB_OBJ_FUNCTION]

        # L_lb = dual bound of  min c^T x - lambda^T (y - y*).
        lagrangian_lower_bound = inner_model.get_obj_bound()

        return InnerMinimizationResult(
            scenario_group=self._as_scenario_group(sub_model_sce_list),
            multiplier=dict(given_dual_info),
            lagrangian_lower_bound=float(lagrangian_lower_bound),
            incumbent_sub_obj=float(incumbent_sub_obj),
            incumbent_main_sol=incumbent_main_sol,
            solve_status=inner_model.solve_status,
        )

    def build_multiplier_cut(self, sub_model_sce_list, anchor_main_result, multiplier, constr_var_map, base_value):
        r"""
        Build ``sum_{s in G} P_s theta_s >= base_value + lambda^T (y - y*)``.

        With ``base_value = L_lb <= L(lambda)`` and ``y* = anchor_main_result`` this is exactly the
        Lagrangian cut, valid for every y because ``Q_G(y) >= L(lambda) + lambda^T (y - y*)``.
        """
        constr_list = sorted(multiplier.keys())
        lhs = self.build_group_theta_lhs(sub_model_sce_list)
        rhs = base_value + pyo.quicksum(
            multiplier[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
               - anchor_main_result[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )
        return lhs, rhs

    def gen_sce_strengthen_bds_cut(self, sub_model_sce_list=None, given_main_result=None, given_dual_info=None, constr_var_map=None):
        """
        Generate a strengthened Benders optimality cut for one scenario group at the master point y*.

        Cut format: ``sum_{s in G} P_s theta_s >= L_lb + lambda^T (y - y*)``.

        :return: lhs, rhs, inner_result (InnerMinimizationResult)
        """
        inner_result = self.solve_inner_minimization(
            sub_model_sce_list=sub_model_sce_list,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )

        lhs, rhs = self.build_multiplier_cut(
            sub_model_sce_list=sub_model_sce_list,
            anchor_main_result=given_main_result,
            multiplier=given_dual_info,
            constr_var_map=constr_var_map,
            base_value=inner_result.lagrangian_lower_bound,
        )

        return lhs, rhs, inner_result

    def gen_sub_sce_lag_cut_heuristic(
            self,
            sub_model_sce_list=None,
            given_main_result=None,
            given_dual_info=None,
            constr_var_map=None,
            max_ite_num=10
    ):
        r"""
        Phase 3 Lagrangian cut generation via regularized multiplier updates.

        Cut format: ``sum_{s in G} P_s theta_s >= L_lb(lambda) + lambda^T (y - y*)``.

        The multiplier search keeps two quantities apart, which is what makes the emitted cut valid
        regardless of how the inner MILPs terminate (risk R3.1):

        * the inner approximation of the multiplier master uses the inner minimization's INCUMBENT
          pair ``(c^T xbar, zbar)``, a valid over-estimator of L(lambda) for any feasible point;
        * the incumbency test and the final anchor use the inner minimization's DUAL BOUND
          ``L_lb <= L(lambda)``. The regularized multiplier model supplies a heuristic stopping
          estimate; its lift value is not a certificate of the best possible multiplier.

        :return: lhs, rhs, best_lagrangian_lower_bound, best_multiplier
        """
        curr_lag_multiplier = dict(given_dual_info)

        inner_result = self.solve_inner_minimization(
            sub_model_sce_list=sub_model_sce_list,
            given_main_result=given_main_result,
            given_dual_info=curr_lag_multiplier,
            constr_var_map=constr_var_map,
        )

        best_L = inner_result.lagrangian_lower_bound
        potential_lambda = dict(curr_lag_multiplier)

        lagrangian_multiplier_model = ModelLagrangianMultiplierHeuristic(
            model_name=f'{tuple(sub_model_sce_list)}_lmm',
            main_result=given_main_result,
            constr_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
            solver_config=self.get_solver_config(SolverRoleName.LAGRANGIAN_MASTER),
        )
        lagrangian_multiplier_model.build_ini_model(
            ini_multiplier=given_dual_info,
            ini_stabilization_param=0.01,
        )

        for ite_num in range(1, max_ite_num + 1):

            # The outer approximation row must use the INCUMBENT point of the inner minimization.
            lagrangian_multiplier_model.add_constr_for_regularized_model(
                sub_obj_value=inner_result.incumbent_sub_obj,
                main_sol=inner_result.incumbent_main_sol,
            )
            lagrangian_multiplier_model.set_objective_function(
                stabilization_param=0.01 / (1 + ite_num),
                prev_multiplier=curr_lag_multiplier,
            )
            lagrangian_multiplier_model.solve()

            eta_ub = lagrangian_multiplier_model.obtain_lift_value()

            curr_lag_multiplier = lagrangian_multiplier_model.get_result(
                var_name_list=[VarName.LAG_MULTIPLIER]
            )[VarName.LAG_MULTIPLIER]

            inner_result = self.solve_inner_minimization(
                sub_model_sce_list=sub_model_sce_list,
                given_main_result=given_main_result,
                given_dual_info=curr_lag_multiplier,
                constr_var_map=constr_var_map,
            )

            actual_L = inner_result.lagrangian_lower_bound

            if actual_L > best_L + 1e-6:
                best_L = actual_L
                potential_lambda = dict(curr_lag_multiplier)
                logger.info(f'Lagrangian cut improved: L_lb={actual_L:.6f}, eta_ub={eta_ub:.6f}')

            if eta_ub - best_L <= 0.001 * math.fabs(best_L):
                logger.info(f'Lagrangian cut converged: best_L_lb={best_L:.6f}, eta_ub={eta_ub:.6f}')
                break

        lhs, rhs = self.build_multiplier_cut(
            sub_model_sce_list=sub_model_sce_list,
            anchor_main_result=given_main_result,
            multiplier=potential_lambda,
            constr_var_map=constr_var_map,
            base_value=best_L,
        )
        return lhs, rhs, best_L, potential_lambda

    def _request_is_applicable(self, cut_type: str, request: CutRequest) -> bool:
        spec = self.cut_generator_registry.get(cut_type)
        if spec is None:
            return False
        if any(getattr(request, field_name) is None for field_name in spec.required_request_fields):
            return False
        if spec.requires_integer_anchor:
            return not request.anchor_solution.is_relaxed and all(
                math.isclose(float(request.anchor_solution.values[name][key]), 0.0, abs_tol=1e-6)
                or math.isclose(float(request.anchor_solution.values[name][key]), 1.0, abs_tol=1e-6)
                for name, key in self.cglp_x_var_key_list
            )
        return True

    def generate_cut(self, cut_type: str, request: CutRequest) -> Optional[GeneratedCut]:
        """Dispatch one uniform request through the registered cut generator."""
        if not self._request_is_applicable(cut_type, request):
            return None
        return self._get_cut_generator(cut_type)(request)

    def dispatch_cut(
        self,
        cut_type: str,
        request: CutRequest,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[CutOutcome, Optional[GeneratedCut]]:
        """Generate and insert one cut, returning an exact outcome for reporting."""
        if not self._request_is_applicable(cut_type, request):
            return CutOutcome.NOT_APPLICABLE, None
        generated_cut = self._get_cut_generator(cut_type)(request)
        if generated_cut is None:
            return CutOutcome.NOT_VIOLATED, None
        outcome = self._insert_generated_cut(
            generated_cut=generated_cut,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return outcome, generated_cut

    @staticmethod
    def aggregate_generated_cuts(
        cut_type: str,
        ite_name: str,
        generated_cuts: List[GeneratedCut],
        cut_name_prefix: Optional[str] = None,
    ) -> GeneratedCut:
        """Post-process per-group cuts into their plain sum under the weighted-theta convention."""
        if not generated_cuts:
            raise ValueError('aggregate_generated_cuts requires at least one cut')
        cut_name_prefix = cut_name_prefix or cut_type
        return GeneratedCut(
            cut_type=f'{cut_type}{CutTypeName.AGGREGATED_SUFFIX}',
            cut_name=f'{cut_name_prefix}_agg_{ite_name}',
            scenario_group=ScenarioGroup([]),
            ite_name=ite_name,
            lhs=sum(cut.lhs for cut in generated_cuts),
            rhs=sum(cut.rhs for cut in generated_cuts),
            lower_bound_row=None,
            metadata={
                'is_aggregated': True,
                'source_cut_names': [cut.cut_name for cut in generated_cuts],
            },
        )

    def dispatch_aggregated_cuts(
        self,
        cut_type: str,
        ite_name: str,
        generated_cuts: List[GeneratedCut],
        include_cut_type_in_repetition: bool = True,
        cut_name_prefix: Optional[str] = None,
    ) -> Tuple[CutOutcome, GeneratedCut]:
        """Aggregate already-generated group cuts and insert exactly one master row."""
        aggregated_cut = self.aggregate_generated_cuts(
            cut_type=cut_type,
            ite_name=ite_name,
            generated_cuts=generated_cuts,
            cut_name_prefix=cut_name_prefix,
        )
        outcome = self._insert_generated_cut(
            generated_cut=aggregated_cut,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
            register_lower_bound_row=False,
        )
        return outcome, aggregated_cut

    @staticmethod
    def _as_scenario_group(sub_sce_list: Any) -> ScenarioGroup:
        """Normalize a scenario-list input into a ScenarioGroup contract object."""
        if isinstance(sub_sce_list, ScenarioGroup):
            return sub_sce_list
        return ScenarioGroup(sub_sce_list)

    def build_cut_request(
        self,
        cut_type: str,
        cut_result: SubProblemResult,
        iteration_context: IterationContext,
        theta_values: Optional[Dict[Any, float]],
        options: Dict[str, Any],
    ) -> CutRequest:
        """Translate one solved scenario group into the uniform generator contract."""
        benders_anchor = options.get('benders_anchor', BendersAnchorName.INTEGER)
        if benders_anchor not in {BendersAnchorName.INTEGER, BendersAnchorName.FRACTIONAL}:
            raise ValueError(f'Unsupported Benders anchor: {benders_anchor}')

        spec = self.cut_generator_registry.get(cut_type)
        if spec is None:
            raise ValueError(f'Unknown cut generator type: {cut_type}')
        anchor_source = spec.anchor_source
        if anchor_source == 'benders_option':
            anchor_source = (
                'fractional'
                if benders_anchor == BendersAnchorName.FRACTIONAL
                else 'current'
            )

        if anchor_source == 'fractional':
            anchor_solution = cut_result.fractional_solution
            dual_info = cut_result.fractional_dual
            relaxed_obj = cut_result.fractional_relaxed_obj_value
        elif anchor_source == 'current':
            anchor_solution = cut_result.main_solution
            dual_info = cut_result.benders_dual
            relaxed_obj = cut_result.relaxed_obj_value
        else:
            raise ValueError(f'Unsupported anchor source for {cut_type}: {anchor_source}')

        if anchor_solution is None:
            anchor_solution = cut_result.main_solution

        return CutRequest(
            scenario_group=cut_result.scenario_group,
            ite_name=iteration_context.ite_name,
            anchor_solution=anchor_solution,
            dual_info=dual_info,
            relaxed_obj=relaxed_obj,
            exact_obj_bound=cut_result.exact_obj_bound,
            theta_values=theta_values,
            options=options,
        )

    def _build_current_subproblem(
        self,
        scenario_group: ScenarioGroup,
        main_solution: MainSolution,
    ) -> Tuple[str, ...]:
        """Build the canonical subproblem and return its local-copy constraint names."""
        self.build_sub_model(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=main_solution.as_model_result(),
            target_slot='current_sub_model',
            solver_role=SolverRoleName.SUBPROBLEM_EXACT,
        )
        return tuple(sorted(self.current_sub_model.local_copy_constr_info.keys()))

    @staticmethod
    def _require_optimal_relaxation(sub_model, label: str) -> None:
        """
        A Benders cut is only valid when its multipliers are OPTIMAL LP duals.

        Unlike the MILP anchors, a truncated LP cannot be repaired by reading a dual bound: the
        multipliers themselves would be wrong. So a relaxation that stopped early is an error here,
        not a weaker cut.
        """
        status = sub_model.solve_status_relax
        if status == ModelStatus.SOLVED_OPT:
            return
        raise ModelSolveError(
            model_name=f'{sub_model.model_name} ({label} relaxation)',
            solve_status=status,
            message=(
                f'{label} relaxation of {sub_model.model_name} ended with status {status}. '
                f'Benders cuts need optimal LP duals, so a truncated relaxation is not usable; '
                f'remove the time limit on the subproblem_lp solver role.'
            ),
        )

    def _solve_current_subproblem_relaxation(self) -> float:
        """Solve the LP relaxation of the current canonical subproblem."""
        self.solve_relaxed_sub_model()
        self._require_optimal_relaxation(self.current_sub_model, 'current')
        return self.current_sub_model.get_relaxed_obj_value()

    def _solve_current_subproblem_exact_if_needed(
        self,
        scenario_group: ScenarioGroup,
        main_solution: MainSolution,
        need_exact_sub_obj: bool,
        incumbent_records: Optional[List],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Solve the exact subproblem only when a downstream cut or incumbent path needs it.

        Returns ``(incumbent_value, dual_bound)``. The two are kept apart on purpose (risk R3.2):
        the INCUMBENT is an upper bound on Q_G(x) and belongs to incumbent accounting, while every
        cut anchor -- the integer optimality cut's right-hand side and the CGLP exact points -- must
        use the DUAL BOUND, which underestimates Q_G(x) under any MIP gap or time limit.
        """
        if incumbent_records is None and not need_exact_sub_obj:
            return None, None

        self.solve_sub_model()
        exact_sub_obj_value = self.current_sub_model.get_obj_value()
        exact_sub_obj_bound = self.current_sub_model.get_obj_bound()
        self._register_exact_point_for_group(
            sub_sce_list=scenario_group.as_list(),
            forward_sol=main_solution.as_model_result(),
            exact_sub_obj_value=exact_sub_obj_bound,
        )
        return exact_sub_obj_value, exact_sub_obj_bound

    def _record_current_subproblem_incumbent(
        self,
        scenario_group: ScenarioGroup,
        iteration_context: IterationContext,
        incumbent_records: List,
    ) -> Tuple[float, dict]:
        """
        Record the current exact subproblem solution for incumbent accounting.

        ``weighted_sub_obj`` is ``sum_{s in G} P_s Q_s(x)`` at the subproblem INCUMBENT, so the
        upper bound is ``design(x) + sum over groups of weighted_sub_obj`` with no further
        probability multiplier (see runner_helpers.summarize_incumbent_records).
        """
        sub_obj_w_main, weighted_sub_obj = self.record_sub_model(
            main_stage_obj_value=iteration_context.main_stage_obj,
            ite_name=iteration_context.ite_name,
        )
        sub_res = self.current_sub_model.get_result(
            [VarName.DG_ACTIVE_POWER, VarName.DG_RATED_POWER]
        )
        incumbent_records.append(
            {
                'scenario_group': scenario_group.as_list(),
                'weighted_sub_obj': weighted_sub_obj,
                'main_stage_obj': iteration_context.main_stage_obj,
                'group_probability_weight': self.get_group_probability_weight(scenario_group.as_list()),
                'sub_result': sub_res,
                'sub_obj_with_main': sub_obj_w_main,
            }
        )
        return sub_obj_w_main, sub_res

    @staticmethod
    def _collect_dual_info_from_sub_model(
        sub_model: Any,
        local_copy_constr_name: Tuple[str, ...],
        source: str,
    ) -> DualInfo:
        """Collect dual values and local-copy variable mapping into a DualInfo contract."""
        constr_dual_info, constr_var_map = sub_model.collect_dual_opt_sol(
            constr_to_collect_list=list(local_copy_constr_name)
        )
        return DualInfo(
            dual_values=constr_dual_info,
            var_map=constr_var_map,
            local_copy_constraint_names=local_copy_constr_name,
            source=source,
        )

    def _solve_fractional_subproblem_if_needed(
        self,
        scenario_group: ScenarioGroup,
        fractional_solution: Optional[MainSolution],
        local_copy_constr_name: Tuple[str, ...],
        need_frac_sub_model: bool,
    ) -> Tuple[Optional[float], Optional[DualInfo]]:
        """Build and solve the fractional local-copy subproblem when a cut family needs it."""
        if not need_frac_sub_model:
            return None, None
        if fractional_solution is None:
            raise ValueError('Fractional subproblem requested, but no fractional main result was provided.')

        self.build_sub_model(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=fractional_solution.as_model_result(),
            target_slot='curr_sub_frac_model',
            solver_role=SolverRoleName.SUBPROBLEM_LP,
        )
        self.solve_sub_frac_model()
        self._require_optimal_relaxation(self.curr_sub_frac_model, 'fractional')
        frac_sub_obj_value_for_benders = self.curr_sub_frac_model.get_relaxed_obj_value()
        frac_dual_info = self._collect_dual_info_from_sub_model(
            sub_model=self.curr_sub_frac_model,
            local_copy_constr_name=local_copy_constr_name,
            source='fractional',
        )
        return frac_sub_obj_value_for_benders, frac_dual_info

    def _gather_one_group_cut_result(
        self,
        scenario_group: ScenarioGroup,
        iteration_context: IterationContext,
        need_frac_sub_model: bool = False,
        need_exact_sub_obj: bool = False,
        incumbent_records: Optional[List] = None,
    ) -> SubProblemResult:
        """
        Build and solve one scenario-group subproblem and return algorithm-level cut data.
        Model inputs and outputs remain the existing nested dict shapes; this method wraps them
        at the algorithm boundary for downstream cut generation.
        """
        scenario_group = self._as_scenario_group(scenario_group)
        current_solution = iteration_context.current_solution
        if current_solution is None:
            raise ValueError('Current main solution is required to gather cut data.')

        local_copy_constr_name = self._build_current_subproblem(
            scenario_group=scenario_group,
            main_solution=current_solution,
        )
        sub_obj_value_for_benders = self._solve_current_subproblem_relaxation()

        exact_sub_obj_value, exact_sub_obj_bound = self._solve_current_subproblem_exact_if_needed(
            scenario_group=scenario_group,
            main_solution=current_solution,
            need_exact_sub_obj=need_exact_sub_obj,
            incumbent_records=incumbent_records,
        )

        incumbent_obj_with_main, incumbent_result = None, None
        if incumbent_records is not None:
            incumbent_obj_with_main, incumbent_result = self._record_current_subproblem_incumbent(
                scenario_group=scenario_group,
                iteration_context=iteration_context,
                incumbent_records=incumbent_records,
            )

        benders_dual = self._collect_dual_info_from_sub_model(
            sub_model=self.current_sub_model,
            local_copy_constr_name=local_copy_constr_name,
            source='current',
        )

        frac_sub_obj_value_for_benders, frac_dual_info = self._solve_fractional_subproblem_if_needed(
            scenario_group=scenario_group,
            fractional_solution=iteration_context.fractional_solution,
            local_copy_constr_name=local_copy_constr_name,
            need_frac_sub_model=need_frac_sub_model,
        )

        return SubProblemResult(
            scenario_group=scenario_group,
            main_solution=current_solution,
            relaxed_obj_value=sub_obj_value_for_benders,
            benders_dual=benders_dual,
            exact_obj_value=exact_sub_obj_value,
            exact_obj_bound=exact_sub_obj_bound,
            fractional_solution=iteration_context.fractional_solution,
            fractional_relaxed_obj_value=frac_sub_obj_value_for_benders,
            fractional_dual=frac_dual_info,
            incumbent_obj_with_main=incumbent_obj_with_main,
            incumbent_result=incumbent_result,
        )

    def run_iteration_cuts(
        self,
        sce_group_list: List[List],
        iteration_context: IterationContext,
        aggregation_policy: AggregationPolicy = AggregationPolicy.PER_GROUP,
        incumbent_records: Optional[List] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Solve all group subproblems and dispatch every requested cut through one path."""
        options = dict(options or {})
        cut_types = list(iteration_context.candidate_cut_types)
        unknown_cut_types = sorted(set(cut_types) - set(self.cut_generator_registry))
        if unknown_cut_types:
            raise ValueError(f'Unknown cut types: {unknown_cut_types}')
        if not isinstance(aggregation_policy, AggregationPolicy):
            aggregation_policy = AggregationPolicy(aggregation_policy)

        benders_anchor = options.get('benders_anchor', BendersAnchorName.INTEGER)
        active_specs = [self.cut_generator_registry[cut_type] for cut_type in cut_types]
        need_frac_sub_model = any(
            spec.anchor_source == 'fractional'
            or (
                spec.anchor_source == 'benders_option'
                and benders_anchor == BendersAnchorName.FRACTIONAL
            )
            for spec in active_specs
        )
        need_exact_sub_obj = any(spec.requires_exact_subproblem for spec in active_specs)

        theta_values = options.get('theta_values')
        if VarName.SUB_OBJ_EST in iteration_context.current_solution.values:
            theta_values = iteration_context.current_solution.values[VarName.SUB_OBJ_EST]
        elif theta_values is None and options.get('use_current_master_theta', True):
            theta_values = {
                scenario: pyo.value(self.model_main.var[VarName.SUB_OBJ_EST][scenario])
                for scenario in self.scenario_list
            }

        generated_by_type = {cut_type: [] for cut_type in cut_types}
        records = []
        for sub_sce_list in sce_group_list:
            cut_result = self._gather_one_group_cut_result(
                scenario_group=self._as_scenario_group(sub_sce_list),
                iteration_context=iteration_context,
                need_frac_sub_model=need_frac_sub_model,
                need_exact_sub_obj=(
                    need_exact_sub_obj
                    or options.get('force_exact_subproblem', False)
                ),
                incumbent_records=incumbent_records,
            )
            group_record = {
                'scenario_group': cut_result.scenario_group.as_list(),
                'cut_outcomes': {},
                'generated_cut_types': [],
                'subproblem_result': cut_result,
            }
            for cut_type in cut_types:
                request = self.build_cut_request(
                    cut_type=cut_type,
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    theta_values=theta_values,
                    options=options,
                )
                if aggregation_policy == AggregationPolicy.PER_GROUP:
                    outcome, generated_cut = self.dispatch_cut(
                        cut_type=cut_type,
                        request=request,
                        include_cut_type_in_repetition=options.get(
                            'include_cut_type_in_repetition', True
                        ),
                    )
                    group_record['cut_outcomes'][cut_type] = outcome
                else:
                    if not self._request_is_applicable(cut_type, request):
                        outcome, generated_cut = CutOutcome.NOT_APPLICABLE, None
                    else:
                        generated_cut = self.generate_cut(cut_type, request)
                        outcome = (
                            CutOutcome.NOT_VIOLATED
                            if generated_cut is None
                            else CutOutcome.ADDED
                        )
                    if generated_cut is None:
                        group_record['cut_outcomes'][cut_type] = outcome
                    else:
                        group_record['generated_cut_types'].append(cut_type)
                        generated_by_type[cut_type].append(generated_cut)
            records.append(group_record)

        if aggregation_policy == AggregationPolicy.AGGREGATE:
            for cut_type, generated_cuts in generated_by_type.items():
                if not generated_cuts:
                    continue
                outcome, _ = self.dispatch_aggregated_cuts(
                    cut_type=cut_type,
                    ite_name=iteration_context.ite_name,
                    generated_cuts=generated_cuts,
                    include_cut_type_in_repetition=options.get(
                        'include_cut_type_in_repetition', True
                    ),
                )
                records.append({
                    'scenario_group': None,
                    'cut_outcomes': {cut_type: outcome},
                    'aggregated': True,
                })

        return records
