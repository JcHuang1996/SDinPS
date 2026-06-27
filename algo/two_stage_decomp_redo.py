# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from typing import Any, Callable, Dict, List, Optional, Tuple
from util.names import DataName, ObjName, VarName

from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub, ModelLagrangianMultiplierHeuristic, PSInnerMinimizationProblem, ModelCGLP
from algo.interfaces import (
    CutCandidate,
    DualInfo,
    GeneratedCut,
    IterationContext,
    MainSolution,
    ScenarioGroup,
    SubProblemResult,
)
from algo.algo_simple_tools import (
    CutRepetitionTracker,
    build_cglp_lower_bound_row_from_alpha_eta,
    build_cglp_lower_bound_row_from_multiplier_cut,
    flatten_ordered_var_values,
    normalize_affine_lower_bound_row,
)
from util.tools import iter_general_csv, iter_sub_prob_info

import pyomo.environ as pyo
import logging
import math


logger = logging.getLogger(__name__)


class TwoStageDecompRedo:

    def __init__(self, raw_data, time_list, scenario_list):

        self.data_processor_module = DataProcessor(raw_data=raw_data)
        self.time_list = time_list
        self.scenario_list = scenario_list

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

    def _register_cut_generators(self):
        """Install the cut-type to generator-function mapping used by all insertion paths."""
        self.cut_generator_registry = {
            'benders': self._generate_benders_cut,
            'integer_opt': self._generate_integer_opt_cut,
            'strengthen_benders': self._generate_strengthen_benders_cut,
            'lagrangian': self._generate_lagrangian_cut,
            'cglp': self._generate_cglp_cut,
        }

    def _get_cut_generator(self, cut_type: str):
        """Return the registered generator for a cut type and fail early on unknown names."""
        if cut_type not in self.cut_generator_registry:
            raise ValueError(f'Unknown cut generator type: {cut_type}')
        return self.cut_generator_registry[cut_type]

    def build_main_stage_model(self):

        # process the data for main model
        self.main_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=self.scenario_list,
            time_list_assigned=self.time_list
        )

        self.sce_prob_dict = self.main_model_data[DataName.DICT_SC_PROB].copy()

        # build the main model with processed main model data
        self.model_main = ModelMain(model_name='Main', model_data=self.main_model_data)
        self.model_main.build_main_model()

        self.main_stage_structural_constr_name_list = self.model_main.get_decomposition_structural_constr_name_list()
        self.cglp_x_var_key_list, self.cglp_z_var_key_list = self.model_main.get_decomposition_var_key_lists_for_cglp()

    def solve_main_stage_model(self):
        # solve the main model, and compute the obj terms
        # return the optimal value of the solved main model
        self.model_main.set_parameters(
            param_dict={
                'TimeLimit':120
            }
        )
        self.model_main.solve()
        self.model_main.cal_detailed_obj()

    def solve_main_stage_relaxed_model(self):
        self.model_main.solve_relaxed()

    def record_main_stage_model(self, ite_name=None):
        # show and record the main model objective value (the best bound objective value)
        total_objective_value = self.model_main.get_iteration_bound_value()
        main_stage_objective_value = self.model_main.get_design_objective_value()

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        # record the objective value and values of every terms
        self.ite_obj_value_dict[ite_name]['main_obj(bound)'] = {
            'total': total_objective_value,
            'detail': self.model_main.obj_term_value.copy()
        }

        return main_stage_objective_value, total_objective_value

    def build_sub_model_redo(self, sub_model_sce_list=None, given_main_result=None):
        """
        Build the cached canonical subproblem for one scenario group and master solution.
        """

        # the sub model's scenario cannot be empty
        if sub_model_sce_list is None:
            raise ValueError('sub_model_sce_list cannot be None')

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

        # ============================
        # build and solve sub problem models
        # ============================
        # build the sub model for the given scenario(s) in this iteration
        self.current_sub_model = ModelSub(
            model_name=f'{tuple(sub_model_sce_list)}_model',
            model_data=sub_model_data,
            main_result=given_main_result,
            sub_model_sce_list=sub_model_sce_list
        )
        self.current_sub_model.build_sub_model_canonical()

    def build_sub_frac_model(self, sub_model_sce_list=None, given_main_frac_result=None):
        """
        the function is similar to build_sub_model / build_sub_model_redo, however, only applied for using fractional main
        solution to obtaining dual multiplier. Avoid using current_sub_model to do this task since the sub_model with fractional
        main solution will be infeasible
        :param sub_model_sce_list:
        :param given_main_frac_result:
        :return:
        """
        # the sub model's scenario cannot be empty
        if sub_model_sce_list is None:
            raise ValueError('sub_model_sce_list cannot be None')

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

        # ============================
        # build and solve sub problem models
        # ============================
        # build the sub model for the given scenario(s) in this iteration
        self.curr_sub_frac_model = ModelSub(
            model_name=f'{tuple(sub_model_sce_list)}_model',
            model_data=sub_model_data,
            main_result=given_main_frac_result,
            sub_model_sce_list=sub_model_sce_list
        )
        self.curr_sub_frac_model.build_sub_model_canonical()

    def solve_sub_model(self):
        self.current_sub_model.solve()
        self.current_sub_model.cal_detailed_obj()

    def solve_relaxed_sub_model(self):
        self.current_sub_model.solve_relaxed()

    def solve_sub_frac_model(self):
        self.curr_sub_frac_model.solve_relaxed()

    def record_sub_model(self, main_stage_obj_value=None, ite_name=None):
        # collect the scenario's objective value.
        # Note: the 1st stage objective value is included.
        sce_obj_value = self.current_sub_model.get_obj_value()
        sce_obj_value_w_main = sce_obj_value + main_stage_obj_value

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        self.ite_obj_value_dict[ite_name][tuple(self.current_sub_model.sub_model_sce_list)] = {
            'total': sce_obj_value_w_main,
            'detail': self.current_sub_model.obj_term_value.copy()
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

    def gen_sce_bds_opt_cut(
        self,
        lp_opt_value=None,
        constr_dual_info=None,
        constr_var_map=None,
        forward_sol=None,
        sub_sce_list=None,
    ):
        """
        Idea: generate benders opt cuts components in the format of:
         theta >= Q^{LP}_{obtained_main_solution} + pi^{T} * (main var - obtained main solution)
         Reference: Zou, J., Ahmed, S., & Sun, X. (2019). Stochastic dual dynamic integer programming.
        :param lp_opt_value: the relaxed LP objective value of the corresponding sub model.
        :param constr_dual_info: the dict of constr-dual, for generating benders cuts.
                                key: constraint name; value: dual optimal solution value
        :param constr_var_map: the dict showing constr-var relationship, for generating benders cuts.
        :return: lhs, rhs
        lhs: the left hand side of the cut in above format
        rhs: the right hand side of the cut in above format
        """
        constr_list = sorted(constr_dual_info.keys())

        if sub_sce_list is None:
            sub_sce_list = self.current_sub_model.sub_model_sce_list

        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_sce_list
        )

        rhs = lp_opt_value + pyo.quicksum(
            constr_dual_info[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
             - forward_sol[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )

        # rhs = lp_opt_value + pyo.quicksum(
        #     constr_dual_info[constr_name]
        #     * self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
        #     for constr_name in constr_list
        # )

        return lhs, rhs

    def _register_group_lower_bound_row(self, sub_sce_list: List, row: dict):
        """Append one normalized lower-bound row to the persistent CGLP state for a group."""
        group_key = tuple(sub_sce_list)
        self._ensure_group_lower_bound_rows(sub_sce_list=sub_sce_list)
        self.cglp_lower_bound_row_dict[group_key].append(row)

    def _record_cut_metadata(
        self,
        generated_cut: GeneratedCut,
        added: bool,
        repetition_record: Optional[dict] = None,
    ) -> None:
        """Store lightweight bookkeeping for each generated cut insertion decision."""
        scenario_group = self._get_cut_tracking_scenario_group(generated_cut)
        self.cut_metadata_records.append(
            {
                "cut_type": generated_cut.cut_type,
                "cut_name": generated_cut.cut_name,
                "iteration": generated_cut.ite_name,
                "scenario_group": scenario_group,
                "added": added,
                "has_lower_bound_row": generated_cut.lower_bound_row is not None,
                "repetition": repetition_record,
                "metadata": generated_cut.metadata.copy(),
            }
        )

    @staticmethod
    def _get_cut_tracking_scenario_group(generated_cut: GeneratedCut):
        """Return the scenario-group label used in repetition reports."""
        if generated_cut.metadata.get("is_aggregated", False):
            return None
        return generated_cut.scenario_group.as_list()

    def _insert_generated_cut(
        self,
        generated_cut: GeneratedCut,
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
        register_lower_bound_row: bool = True,
    ) -> bool:
        """Centralize master insertion, repetition tracking, CGLP row updates, and metadata."""
        should_add = True
        if should_add_cut_row is not None and generated_cut.lower_bound_row is not None:
            should_add = should_add_cut_row(generated_cut.lower_bound_row, generated_cut.cut_name)
        elif should_add_cut is not None:
            should_add = should_add_cut(generated_cut.lhs, generated_cut.rhs, generated_cut.cut_name)

        if not should_add:
            self._record_cut_metadata(
                generated_cut=generated_cut,
                added=False,
                repetition_record=None,
            )
            return False

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

        tracking_scenario_group = self._get_cut_tracking_scenario_group(generated_cut)
        if generated_cut.lower_bound_row is not None:
            repetition_record = self.cut_repetition_tracker.record_by_row(
                cut_type=generated_cut.cut_type,
                ite_name=generated_cut.ite_name,
                cut_name=generated_cut.cut_name,
                row=generated_cut.lower_bound_row,
                sub_sce_list=tracking_scenario_group,
                include_cut_type=include_cut_type_in_repetition,
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

        self._record_cut_metadata(
            generated_cut=generated_cut,
            added=True,
            repetition_record=repetition_record,
        )
        return True

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

        if x_point in self.cglp_exact_point_value_dict[group_key]:
            return x_point

        self.cglp_exact_point_value_dict[group_key][x_point] = float(exact_sub_obj_value)

        if group_key in self.cglp_model_dict:
            self.cglp_model_dict[group_key].update_with_exact_point(
                x_point=x_point,
                q_value=exact_sub_obj_value,
            )

        return x_point

    def _generate_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        sub_obj_value,
        constr_dual_info,
        constr_var_map,
        forward_sol,
        cut_name_prefix: str = "bd",
    ) -> GeneratedCut:
        """Generate a classical Benders cut without mutating the master model."""
        scenario_group = self._as_scenario_group(sub_sce_list)
        lhs, rhs = self.gen_sce_bds_opt_cut(
            lp_opt_value=sub_obj_value,
            constr_dual_info=constr_dual_info,
            constr_var_map=constr_var_map,
            forward_sol=forward_sol,
            sub_sce_list=scenario_group.as_list(),
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{ite_name}"
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=sub_obj_value,
            multiplier_info=constr_dual_info,
            constr_var_map=constr_var_map,
            ref_point=forward_sol,
        )
        return GeneratedCut(
            cut_type='benders',
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={"base_value": sub_obj_value},
        )

    def _generate_integer_opt_cut(
        self,
        sub_sce_list,
        ite_name,
        exact_sub_obj_value,
        forward_sol,
        lower_bound: float = 0.0,
        cut_name_prefix: str = "int_opt",
    ) -> GeneratedCut:
        """Generate an integer optimality cut without mutating the master model."""
        scenario_group = self._as_scenario_group(sub_sce_list)
        scenario_group_list = scenario_group.as_list()
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{ite_name}"
        tol = 1e-6
        exact_sub_obj_value = float(exact_sub_obj_value)
        lower_bound = float(lower_bound)

        if lower_bound > exact_sub_obj_value + tol:
            raise ValueError(
                f"lower_bound ({lower_bound}) cannot exceed exact_sub_obj_value ({exact_sub_obj_value})"
            )

        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in scenario_group_list
        )

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
            cut_type='integer_opt',
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=ite_name,
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

    def _generate_strengthen_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        given_main_result,
        given_dual_info,
        constr_var_map,
        cut_name_prefix: str = "str_bd",
    ) -> GeneratedCut:
        """Generate a strengthened Benders cut and its normalized row."""
        scenario_group = self._as_scenario_group(sub_sce_list)
        lhs, rhs, obtained_sub_obj, obtained_main_sol = self.gen_sce_strengthen_bds_cut(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{ite_name}"
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=obtained_sub_obj,
            multiplier_info=given_dual_info,
            constr_var_map=constr_var_map,
            ref_point=obtained_main_sol,
        )
        return GeneratedCut(
            cut_type='strengthen_benders',
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "obtained_sub_obj": obtained_sub_obj,
                "obtained_main_sol": obtained_main_sol,
            },
        )

    def _generate_lagrangian_cut(
        self,
        sub_sce_list,
        ite_name,
        given_main_result,
        given_dual_info,
        constr_var_map,
        max_ite_num: int = 10,
        cut_name_prefix: str = "str_lag",
    ) -> GeneratedCut:
        """Generate a Lagrangian cut and its normalized row."""
        scenario_group = self._as_scenario_group(sub_sce_list)
        lhs, rhs, sub_obj, main_sol, lag_multiplier = self.gen_sub_sce_lag_cut_heuristic(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
            max_ite_num=max_ite_num,
        )
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{ite_name}"
        row = build_cglp_lower_bound_row_from_multiplier_cut(
            cut_name=cut_name,
            base_value=sub_obj,
            multiplier_info=lag_multiplier,
            constr_var_map=constr_var_map,
            ref_point=main_sol,
        )
        return GeneratedCut(
            cut_type='lagrangian',
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=ite_name,
            lhs=lhs,
            rhs=rhs,
            lower_bound_row=row,
            metadata={
                "sub_obj": sub_obj,
                "main_sol": main_sol,
                "lag_multiplier": lag_multiplier,
            },
        )

    def _generate_cglp_cut(
        self,
        sub_sce_list,
        ite_name,
        forward_sol,
        exact_sub_obj_value,
        cut_name_prefix: str = "cglp",
    ) -> Optional[GeneratedCut]:
        """Solve/update the CGLP and return a generated cut when the current theta is violated."""
        scenario_group = self._as_scenario_group(sub_sce_list)
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
        theta_value = sum(
            pyo.value(self.model_main.var[VarName.SUB_OBJ_EST][sce_idx])
            for sce_idx in scenario_group.as_list()
        )

        if theta_value + 1e-6 >= exact_sub_obj_value:
            return None

        alpha, eta, lhs, rhs = cglp_model.solve_and_build_group_cut(model_main=self.model_main)
        cut_name = f"{cut_name_prefix}_s_{str(scenario_group.as_list())}_{ite_name}"
        row = build_cglp_lower_bound_row_from_alpha_eta(
            cut_name=cut_name,
            alpha=alpha,
            eta=eta,
        )
        row["name"] = cut_name
        return GeneratedCut(
            cut_type='cglp',
            cut_name=cut_name,
            scenario_group=scenario_group,
            ite_name=ite_name,
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

    def add_cglp_cut(
        self,
        sub_sce_list,
        ite_name,
        forward_sol,
        exact_sub_obj_value,
        cut_name_prefix: str = "cglp",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Optional[Tuple[str, Any, Any, Dict, float]]:
        """Generate and insert a CGLP cut through the common registry path."""
        generated_cut = self._get_cut_generator('cglp')(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            forward_sol=forward_sol,
            exact_sub_obj_value=exact_sub_obj_value,
            cut_name_prefix=cut_name_prefix,
        )
        if generated_cut is None:
            return None

        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=should_add_cut_row,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return (
            generated_cut.cut_name,
            generated_cut.lhs,
            generated_cut.rhs,
            generated_cut.metadata["alpha"],
            generated_cut.metadata["eta"],
        )

    def gen_sce_strengthen_bds_cut(self, sub_model_sce_list=None, given_main_result=None, given_dual_info=None, constr_var_map=None):
        """
        The function generates a strengthened Benders optimality cut for the given scenario(s) based on the given main result.
        Cut format: theta >= c^T*x + (y-z)^T*lambda.
        :param
        sub_model_sce_list: the list of scenarios considered in this sub model.
        given_main_result: the main stage result for current iteration (y*)
        given_dual_info: the dict of dual information (lambda) for the given scenario(s)
        constr_var_map: the dict mapping constraint names to (var_name, index) for cut generation.
        :return: lhs, rhs, obtained_sub_obj, obtained_main_sol
        lhs: left hand side of the cut (sum of theta over scenarios)
        rhs: right hand side of the cut
        obtained_sub_obj: c^T*x term from the inner minimization objective
        obtained_main_sol: obtained feasible main-stage solution from the inner minimization, i.e. the z
        """
        dict_key = tuple(sub_model_sce_list)
        if dict_key not in self.inner_minimization_model_dict:
            sub_model_data = self.sub_model_data_dict[dict_key]
            inner_model = PSInnerMinimizationProblem(
                model_name=f'{dict_key}_inm',
                model_data=sub_model_data,
                main_result=given_main_result,
                constr_dual_info=given_dual_info,
                constr_var_map=constr_var_map
            )
            inner_model.build_model()
            self.inner_minimization_model_dict[dict_key] = inner_model
        else:
            inner_model = self.inner_minimization_model_dict[dict_key]
            inner_model.main_result = given_main_result
            inner_model.set_objective_function(lag_multiplier=given_dual_info)

        inner_model.set_parameters(
            param_dict={
                'MIPGap': 0.002,
                'TimeLimit': 6,
                # 'MIPGapAbs': 10000
            }
        )

        inner_model.solve()

        main_var_name_list = sorted(given_main_result.keys())
        obtained_main_sol = inner_model.get_result(main_var_name_list)
        inner_model.cal_detailed_obj()
        obtained_sub_obj = inner_model.obj_term_value[ObjName.SUB_OBJ_FUNCTION]

        constr_list = sorted(given_dual_info.keys())
        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_model_sce_list
        )
        rhs = obtained_sub_obj + pyo.quicksum(
            given_dual_info[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
             - obtained_main_sol[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )

        return lhs, rhs, obtained_sub_obj, obtained_main_sol

    def gen_sub_sce_lag_cut_heuristic(
            self,
            sub_model_sce_list=None,
            given_main_result=None,
            given_dual_info=None,
            constr_var_map=None,
            max_ite_num=10
    ):
        """
        Phase 3 Lagrangian cut generation via regularized multiplier updates.
        Cut format: theta >= c^T*x + (y-z)^T*lambda.
        :param sub_model_sce_list: scenarios considered
        :param given_main_result: fixed master solution (y*)
        :param given_dual_info: initial dual/multiplier (lambda^1)
        :param constr_var_map: constraint-to-variable mapping for cuts
        :param max_ite_num: max Phase 3 iterations
        :return: lhs, rhs, cTx, z
        """
        lhs, rhs, cTx, z = self.gen_sce_strengthen_bds_cut(
            sub_model_sce_list=sub_model_sce_list,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )
        curr_lag_multiplier = given_dual_info
        constr_list = sorted(given_dual_info.keys())

        best_L = cTx + sum(
            curr_lag_multiplier[cn] * (
                given_main_result[constr_var_map[cn][0]][constr_var_map[cn][1]]
                - z[constr_var_map[cn][0]][constr_var_map[cn][1]]
            )
            for cn in constr_list
        )
        potential_cTx, potential_z, potential_lambda = cTx, z.copy(), curr_lag_multiplier.copy()

        lagrangian_multiplier_model = ModelLagrangianMultiplierHeuristic(
            model_name=f'{tuple(sub_model_sce_list)}_lmm',
            main_result=given_main_result,
            constr_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )
        lagrangian_multiplier_model.build_ini_model(
            ini_sub_obj_value=cTx,
            ini_main_sol=z,
            ini_multiplier=given_dual_info,
            ini_stabilization_param=0.01,
        )

        for ite_num in range(1, max_ite_num + 1):

            lagrangian_multiplier_model.add_constr_for_regularized_model(sub_obj_value=cTx, main_sol=z)
            lagrangian_multiplier_model.set_objective_function(
                stabilization_param=0.01 / (1 + ite_num),
                prev_multiplier=curr_lag_multiplier,
            )
            lagrangian_multiplier_model.solve()

            eta_ub = lagrangian_multiplier_model.obtain_lift_value()

            curr_lag_multiplier = lagrangian_multiplier_model.get_result(
                var_name_list=[VarName.LAG_MULTIPLIER]
            )[VarName.LAG_MULTIPLIER]

            lhs, rhs, cTx, z = self.gen_sce_strengthen_bds_cut(
                sub_model_sce_list=sub_model_sce_list,
                given_main_result=given_main_result,
                given_dual_info=curr_lag_multiplier,
                constr_var_map=constr_var_map,
            )

            actual_L = cTx + sum(
                curr_lag_multiplier[cn] * (
                    given_main_result[constr_var_map[cn][0]][constr_var_map[cn][1]]
                    - z[constr_var_map[cn][0]][constr_var_map[cn][1]]
                )
                for cn in constr_list
            )

            if actual_L > best_L + 1e-6:
                best_L = actual_L
                potential_cTx, potential_z, potential_lambda = cTx, z.copy(), curr_lag_multiplier.copy()
                logger.info(f'Lagrangian cut improved: L={actual_L:.6f}, eta_ub={eta_ub:.6f}')

            if eta_ub - best_L <= 0.001 * math.fabs(best_L):
                logger.info(f'Lagrangian cut converged: best_L={best_L:.6f}, eta_ub={eta_ub:.6f}')
                break

        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_model_sce_list
        )
        rhs = potential_cTx + pyo.quicksum(
            potential_lambda[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
             - potential_z[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )
        return lhs, rhs, potential_cTx, potential_z, potential_lambda

    def add_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        sub_obj_value,
        constr_dual_info,
        constr_var_map,
        forward_sol,
        cut_name_prefix: str = "bd",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[str, Any, Any]:
        """
        Generate Benders optimality cut (via gen_sce_bds_opt_cut) and add it to the main model.
        Returns (cut_name, lhs, rhs) so callers can compare with other cuts in future logic.
        """
        generated_cut = self._get_cut_generator('benders')(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            sub_obj_value=sub_obj_value,
            constr_dual_info=constr_dual_info,
            constr_var_map=constr_var_map,
            forward_sol=forward_sol,
            cut_name_prefix=cut_name_prefix,
        )
        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=should_add_cut_row,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return generated_cut.cut_name, generated_cut.lhs, generated_cut.rhs

    def add_integer_opt_cut(
        self,
        sub_sce_list,
        ite_name,
        exact_sub_obj_value,
        forward_sol,
        lower_bound: float = 0.0,
        cut_name_prefix: str = "int_opt",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[str, Any, Any]:
        """
        Generate the standard integer optimality cut and optionally add it to the main model.
        Returns (cut_name, lhs, rhs).
        """
        generated_cut = self._get_cut_generator('integer_opt')(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            exact_sub_obj_value=exact_sub_obj_value,
            forward_sol=forward_sol,
            lower_bound=lower_bound,
            cut_name_prefix=cut_name_prefix,
        )
        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=should_add_cut_row,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return generated_cut.cut_name, generated_cut.lhs, generated_cut.rhs

    def add_strengthen_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        given_main_result,
        given_dual_info,
        constr_var_map,
        cut_name_prefix: str = "str_bd",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[str, Any, Any, float, Any]:
        """
        Generate strengthened Benders cut (via gen_sce_strengthen_bds_cut) and optionally add it.
        If should_add_cut is provided, the cut is added only when should_add_cut(lhs, rhs, cut_name) is True.
        Otherwise the cut is always added. Returns (cut_name, lhs, rhs, obtained_sub_obj, obtained_main_sol)
        for possible comparison with other cuts in future functions.
        """
        generated_cut = self._get_cut_generator('strengthen_benders')(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
            cut_name_prefix=cut_name_prefix,
        )
        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=should_add_cut_row,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return (
            generated_cut.cut_name,
            generated_cut.lhs,
            generated_cut.rhs,
            generated_cut.metadata["obtained_sub_obj"],
            generated_cut.metadata["obtained_main_sol"],
        )

    def add_lagrangian_cut(
        self,
        sub_sce_list,
        ite_name,
        given_main_result,
        given_dual_info,
        constr_var_map,
        max_ite_num: int = 10,
        cut_name_prefix: str = "str_lag",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        should_add_cut_row: Optional[Callable[[dict, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[str, Any, Any, float, Any]:
        """
        Generate Lagrangian cut (via gen_sub_sce_lag_cut_heuristic) and optionally add it.
        If should_add_cut is provided, the cut is added only when should_add_cut(lhs, rhs, cut_name) is True.
        Otherwise the cut is always added. Returns (cut_name, lhs, rhs, sub_obj, main_sol)
        for possible comparison with other cuts in future functions.
        """
        generated_cut = self._get_cut_generator('lagrangian')(
            sub_sce_list=sub_sce_list,
            ite_name=ite_name,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
            max_ite_num=max_ite_num,
            cut_name_prefix=cut_name_prefix,
        )
        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=should_add_cut_row,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
        )
        return (
            generated_cut.cut_name,
            generated_cut.lhs,
            generated_cut.rhs,
            generated_cut.metadata["sub_obj"],
            generated_cut.metadata["main_sol"],
        )

    def add_aggregated_cut(
        self,
        cut_name_prefix: str,
        ite_name: str,
        cut_terms: List[Tuple[Any, Any, float]],
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
        include_cut_type_in_repetition: bool = True,
    ) -> Tuple[str, Any, Any]:
        """
        Add a single cut for this iteration using a probability-weighted sum of per-scenario cuts.
        cut_terms: list of (lhs, rhs, weight) for each scenario group; weight is the scenario
        probability (e.g. sum of sce_prob_dict[s] for s in that group).
        The cut added is: sum(weight * lhs) >= sum(weight * rhs).
        Returns (cut_name, agg_lhs, agg_rhs).
        """
        if not cut_terms:
            raise ValueError("add_aggregated_cut requires at least one (lhs, rhs, weight) term")
        agg_lhs = sum(w * lhs for (lhs, rhs, w) in cut_terms)
        agg_rhs = sum(w * rhs for (lhs, rhs, w) in cut_terms)
        cut_name = f"{cut_name_prefix}_agg_{ite_name}"
        generated_cut = GeneratedCut(
            cut_type=f"{cut_name_prefix}_aggregated",
            cut_name=cut_name,
            scenario_group=ScenarioGroup([]),
            ite_name=ite_name,
            lhs=agg_lhs,
            rhs=agg_rhs,
            lower_bound_row=None,
            metadata={"is_aggregated": True},
        )
        self._insert_generated_cut(
            generated_cut=generated_cut,
            should_add_cut=should_add_cut,
            should_add_cut_row=None,
            include_cut_type_in_repetition=include_cut_type_in_repetition,
            register_lower_bound_row=False,
        )
        return cut_name, agg_lhs, agg_rhs

    @staticmethod
    def _as_scenario_group(sub_sce_list: Any) -> ScenarioGroup:
        """Normalize a scenario-list input into a ScenarioGroup contract object."""
        if isinstance(sub_sce_list, ScenarioGroup):
            return sub_sce_list
        return ScenarioGroup(sub_sce_list)

    @staticmethod
    def _as_main_solution(main_result: Any, name: str, is_relaxed: bool) -> MainSolution:
        """Normalize a nested model-result dict into a MainSolution contract object."""
        if isinstance(main_result, MainSolution):
            return main_result
        return MainSolution(
            values=main_result,
            name=name,
            is_relaxed=is_relaxed,
        )

    def _build_iteration_context(
        self,
        ite_name: str,
        curr_main_result: Any,
        frac_main_result: Any,
        main_stage_obj: Optional[float],
        candidate_cut_types: Optional[List[str]] = None,
        ite_num: Optional[int] = None,
    ) -> IterationContext:
        """Build the per-iteration contract passed through subproblem and cut generation."""
        return IterationContext(
            ite_name=ite_name,
            iteration_number=ite_num,
            current_solution=self._as_main_solution(
                main_result=curr_main_result,
                name='current',
                is_relaxed=False,
            ),
            fractional_solution=self._as_main_solution(
                main_result=frac_main_result,
                name='fractional',
                is_relaxed=True,
            ) if frac_main_result is not None else None,
            main_stage_obj=main_stage_obj,
            candidate_cut_types=candidate_cut_types or [],
        )

    @staticmethod
    def _make_cut_candidate(
        cut_type: str,
        cut_result: SubProblemResult,
        iteration_context: IterationContext,
        main_solution: Optional[MainSolution],
        dual_info: Optional[DualInfo] = None,
        sub_obj_value: Optional[float] = None,
        exact_sub_obj_value: Optional[float] = None,
    ) -> CutCandidate:
        """Create one cut-candidate contract from a subproblem result and selected solution."""
        if main_solution is None:
            raise ValueError(f'Main solution is required for {cut_type} cut candidate.')
        return CutCandidate(
            cut_type=cut_type,
            scenario_group=cut_result.scenario_group,
            ite_name=iteration_context.ite_name,
            main_solution=main_solution,
            dual_info=dual_info,
            sub_obj_value=sub_obj_value,
            exact_sub_obj_value=exact_sub_obj_value,
        )

    @staticmethod
    def _active_cut_types(
        add_benders: bool,
        add_strengthen: bool,
        add_lagrangian: bool,
        add_cglp: bool = False,
        add_integer_opt: bool = False,
    ) -> List[str]:
        cut_types = []
        if add_benders:
            cut_types.append('benders')
        if add_strengthen:
            cut_types.append('strengthen_benders')
        if add_lagrangian:
            cut_types.append('lagrangian')
        if add_cglp:
            cut_types.append('cglp')
        if add_integer_opt:
            cut_types.append('integer_opt')
        return cut_types

    def _build_current_subproblem(
        self,
        scenario_group: ScenarioGroup,
        main_solution: MainSolution,
    ) -> Tuple[str, ...]:
        """Build the canonical subproblem and return its local-copy constraint names."""
        self.build_sub_model_redo(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_result=main_solution.as_model_result(),
        )
        return tuple(sorted(self.current_sub_model.local_copy_constr_info.keys()))

    def _solve_current_subproblem_relaxation(self) -> float:
        """Solve the LP relaxation of the current canonical subproblem."""
        self.solve_relaxed_sub_model()
        return self.current_sub_model.get_relaxed_obj_value()

    def _solve_current_subproblem_exact_if_needed(
        self,
        scenario_group: ScenarioGroup,
        main_solution: MainSolution,
        add_cglp: bool,
        need_exact_sub_obj: bool,
        incumbent_records: Optional[List],
    ) -> Optional[float]:
        """Solve the exact subproblem only when a downstream cut or incumbent path needs it."""
        if incumbent_records is None and not add_cglp and not need_exact_sub_obj:
            return None

        self.solve_sub_model()
        exact_sub_obj_value = self.current_sub_model.get_obj_value()
        self._register_exact_point_for_group(
            sub_sce_list=scenario_group.as_list(),
            forward_sol=main_solution.as_model_result(),
            exact_sub_obj_value=exact_sub_obj_value,
        )
        return exact_sub_obj_value

    def _record_current_subproblem_incumbent(
        self,
        scenario_group: ScenarioGroup,
        iteration_context: IterationContext,
        incumbent_records: List,
    ) -> Tuple[float, dict]:
        """Record the current exact subproblem solution for incumbent accounting."""
        sub_obj_w_main, _ = self.record_sub_model(
            main_stage_obj_value=iteration_context.main_stage_obj,
            ite_name=iteration_context.ite_name,
        )
        sub_res = self.current_sub_model.get_result(
            [VarName.DG_ACTIVE_POWER, VarName.DG_RATED_POWER]
        )
        incumbent_records.append((scenario_group.as_list(), sub_obj_w_main, sub_res))
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
        add_strengthen: bool,
        add_lagrangian: bool,
        need_frac_sub_model: bool,
    ) -> Tuple[Optional[float], Optional[DualInfo]]:
        """Build and solve the fractional local-copy subproblem when a cut family needs it."""
        if not (add_strengthen or add_lagrangian or need_frac_sub_model):
            return None, None
        if fractional_solution is None:
            raise ValueError('Fractional subproblem requested, but no fractional main result was provided.')

        self.build_sub_frac_model(
            sub_model_sce_list=scenario_group.as_list(),
            given_main_frac_result=fractional_solution.as_model_result(),
        )
        self.solve_sub_frac_model()
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
        add_strengthen: bool,
        add_lagrangian: bool,
        add_cglp: bool,
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

        exact_sub_obj_value = self._solve_current_subproblem_exact_if_needed(
            scenario_group=scenario_group,
            main_solution=current_solution,
            add_cglp=add_cglp,
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
            add_strengthen=add_strengthen,
            add_lagrangian=add_lagrangian,
            need_frac_sub_model=need_frac_sub_model,
        )

        return SubProblemResult(
            scenario_group=scenario_group,
            main_solution=current_solution,
            relaxed_obj_value=sub_obj_value_for_benders,
            benders_dual=benders_dual,
            exact_obj_value=exact_sub_obj_value,
            fractional_solution=iteration_context.fractional_solution,
            fractional_relaxed_obj_value=frac_sub_obj_value_for_benders,
            fractional_dual=frac_dual_info,
            incumbent_obj_with_main=incumbent_obj_with_main,
            incumbent_result=incumbent_result,
        )

    def _gather_one_group_cut_data(
        self,
        sub_sce_list: List,
        curr_main_result: Any,
        frac_main_result: Any,
        ite_name: str,
        add_strengthen: bool,
        add_lagrangian: bool,
        add_cglp: bool,
        need_frac_sub_model: bool = False,
        need_exact_sub_obj: bool = False,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
    ) -> Tuple[float, Any, Any, Optional[Any], Optional[Any], Optional[float], Optional[float]]:
        """
        Compatibility wrapper for existing decomposition scripts.

        Returns the legacy tuple:
        (sub_obj_value_for_benders, constr_dual_bi, constr_var_map_bi, constr_dual_frac, constr_var_map_frac,
         exact_sub_obj_value, frac_sub_obj_value_for_benders).
        """
        iteration_context = self._build_iteration_context(
            ite_name=ite_name,
            curr_main_result=curr_main_result,
            frac_main_result=frac_main_result,
            main_stage_obj=main_stage_obj,
        )
        cut_result = self._gather_one_group_cut_result(
            scenario_group=self._as_scenario_group(sub_sce_list),
            iteration_context=iteration_context,
            add_strengthen=add_strengthen,
            add_lagrangian=add_lagrangian,
            add_cglp=add_cglp,
            need_frac_sub_model=need_frac_sub_model,
            need_exact_sub_obj=need_exact_sub_obj,
            incumbent_records=incumbent_records,
        )
        return cut_result.as_legacy_tuple()

    def run_iteration_cuts_per_scenario(
        self,
        sce_group_list: List[List],
        ite_name: str,
        ite_num: int,
        curr_main_result: Any,
        frac_main_result: Any,
        benders_cut_iter_range: Optional[Tuple[int, int]],
        strengthen_benders_cut_iter_range: Optional[Tuple[int, int]],
        lagrangian_cut_iter_range: Optional[Tuple[int, int]],
        cglp_cut_iter_range: Optional[Tuple[int, int]],
        integer_opt_cut_iter_range: Optional[Tuple[int, int]] = None,
        *,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
        max_lagrangian_ite: int = 10,
    ) -> None:
        """
        For each scenario group in sce_group_list: gather cut data, then add one Benders / strengthened
        Benders / Lagrangian cut per group (according to iter ranges). Corresponds to use_aggregated_cuts=False.
        """
        add_benders = (
            benders_cut_iter_range is not None
            and benders_cut_iter_range[0] <= ite_num <= benders_cut_iter_range[1]
        )
        add_strengthen = (
            strengthen_benders_cut_iter_range is not None
            and strengthen_benders_cut_iter_range[0] <= ite_num <= strengthen_benders_cut_iter_range[1]
        )
        add_lagrangian = (
            lagrangian_cut_iter_range is not None
            and lagrangian_cut_iter_range[0] <= ite_num <= lagrangian_cut_iter_range[1]
        )
        add_cglp = (
            cglp_cut_iter_range is not None
            and cglp_cut_iter_range[0] <= ite_num <= cglp_cut_iter_range[1]
        )
        add_integer_opt = (
            integer_opt_cut_iter_range is not None
            and integer_opt_cut_iter_range[0] <= ite_num <= integer_opt_cut_iter_range[1]
        )
        iteration_context = self._build_iteration_context(
            ite_name=ite_name,
            ite_num=ite_num,
            curr_main_result=curr_main_result,
            frac_main_result=frac_main_result,
            main_stage_obj=main_stage_obj,
            candidate_cut_types=self._active_cut_types(
                add_benders=add_benders,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=add_cglp,
                add_integer_opt=add_integer_opt,
            ),
        )

        for sub_sce_list in sce_group_list:
            cut_result = self._gather_one_group_cut_result(
                scenario_group=self._as_scenario_group(sub_sce_list),
                iteration_context=iteration_context,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=add_cglp,
                need_exact_sub_obj=add_cglp or add_integer_opt,
                incumbent_records=incumbent_records,
            )
            scenario_group_list = cut_result.scenario_group.as_list()

            if add_benders:
                candidate = self._make_cut_candidate(
                    cut_type='benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    dual_info=cut_result.benders_dual,
                    sub_obj_value=cut_result.relaxed_obj_value,
                )
                self.add_benders_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    sub_obj_value=candidate.sub_obj_value,
                    constr_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    forward_sol=candidate.main_solution.as_model_result(),
                )

            if add_strengthen and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='strengthen_benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                self.add_strengthen_benders_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                )

            if add_lagrangian and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='lagrangian',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                self.add_lagrangian_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    max_ite_num=max_lagrangian_ite,
                )

            if add_cglp and cut_result.exact_obj_value is not None:
                candidate = self._make_cut_candidate(
                    cut_type='cglp',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    exact_sub_obj_value=cut_result.exact_obj_value,
                )
                self.add_cglp_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    forward_sol=candidate.main_solution.as_model_result(),
                    exact_sub_obj_value=candidate.exact_sub_obj_value,
                )

            if add_integer_opt and cut_result.exact_obj_value is not None:
                candidate = self._make_cut_candidate(
                    cut_type='integer_opt',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    exact_sub_obj_value=cut_result.exact_obj_value,
                )
                self.add_integer_opt_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    exact_sub_obj_value=candidate.exact_sub_obj_value,
                    forward_sol=candidate.main_solution.as_model_result(),
                    lower_bound=0.0,
                )

    def run_iteration_cuts_for_given_main_result(
        self,
        sce_group_list: List[List],
        ite_name: str,
        given_main_result: Any,
        add_benders: bool,
        add_strengthen: bool,
        add_lagrangian: bool,
        add_cglp: bool,
        *,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
        max_lagrangian_ite: int = 10,
    ) -> None:
        """
        For each scenario group in sce_group_list: gather cut data using the provided main solution
        as both integer and fractional main result, then add the requested cuts per group.
        """
        iteration_context = self._build_iteration_context(
            ite_name=ite_name,
            curr_main_result=given_main_result,
            frac_main_result=given_main_result,
            main_stage_obj=main_stage_obj,
            candidate_cut_types=self._active_cut_types(
                add_benders=add_benders,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=add_cglp,
            ),
        )

        for sub_sce_list in sce_group_list:
            cut_result = self._gather_one_group_cut_result(
                scenario_group=self._as_scenario_group(sub_sce_list),
                iteration_context=iteration_context,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=add_cglp,
                incumbent_records=incumbent_records,
            )
            scenario_group_list = cut_result.scenario_group.as_list()

            if add_benders:
                candidate = self._make_cut_candidate(
                    cut_type='benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    dual_info=cut_result.benders_dual,
                    sub_obj_value=cut_result.relaxed_obj_value,
                )
                self.add_benders_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    sub_obj_value=candidate.sub_obj_value,
                    constr_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    forward_sol=candidate.main_solution.as_model_result(),
                )

            if add_strengthen and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='strengthen_benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                self.add_strengthen_benders_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                )

            if add_lagrangian and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='lagrangian',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                self.add_lagrangian_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    max_ite_num=max_lagrangian_ite,
                )

            if add_cglp and cut_result.exact_obj_value is not None:
                candidate = self._make_cut_candidate(
                    cut_type='cglp',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    exact_sub_obj_value=cut_result.exact_obj_value,
                )
                self.add_cglp_cut(
                    sub_sce_list=scenario_group_list,
                    ite_name=candidate.ite_name,
                    forward_sol=candidate.main_solution.as_model_result(),
                    exact_sub_obj_value=candidate.exact_sub_obj_value,
                )

    def run_iteration_cuts_aggregated(
        self,
        sce_group_list: List[List],
        ite_name: str,
        ite_num: int,
        curr_main_result: Any,
        frac_main_result: Any,
        benders_cut_iter_range: Optional[Tuple[int, int]],
        strengthen_benders_cut_iter_range: Optional[Tuple[int, int]],
        lagrangian_cut_iter_range: Optional[Tuple[int, int]],
        cglp_cut_iter_range: Optional[Tuple[int, int]] = None,
        *,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
        max_lagrangian_ite: int = 10,
    ) -> None:
        """
        For each scenario group: gather cut data and collect (lhs, rhs, weight). After the loop, add
        exactly one cut per type using probability-weighted lhs and rhs. Corresponds to use_aggregated_cuts=True.
        """
        if cglp_cut_iter_range is not None:
            raise NotImplementedError("CGLP cuts are only implemented for run_iteration_cuts_per_scenario().")

        add_benders = (
            benders_cut_iter_range is not None
            and benders_cut_iter_range[0] <= ite_num <= benders_cut_iter_range[1]
        )
        add_strengthen = (
            strengthen_benders_cut_iter_range is not None
            and strengthen_benders_cut_iter_range[0] <= ite_num <= strengthen_benders_cut_iter_range[1]
        )
        add_lagrangian = (
            lagrangian_cut_iter_range is not None
            and lagrangian_cut_iter_range[0] <= ite_num <= lagrangian_cut_iter_range[1]
        )

        benders_terms: List[Tuple[Any, Any, float]] = []
        strengthen_terms: List[Tuple[Any, Any, float]] = []
        lagrangian_terms: List[Tuple[Any, Any, float]] = []
        iteration_context = self._build_iteration_context(
            ite_name=ite_name,
            ite_num=ite_num,
            curr_main_result=curr_main_result,
            frac_main_result=frac_main_result,
            main_stage_obj=main_stage_obj,
            candidate_cut_types=self._active_cut_types(
                add_benders=add_benders,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
            ),
        )

        for sub_sce_list in sce_group_list:
            cut_result = self._gather_one_group_cut_result(
                scenario_group=self._as_scenario_group(sub_sce_list),
                iteration_context=iteration_context,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                add_cglp=False,
                incumbent_records=incumbent_records,
            )
            scenario_group_list = cut_result.scenario_group.as_list()

            weight = cut_result.scenario_group.probability_weight(self.sce_prob_dict)

            if add_benders:
                candidate = self._make_cut_candidate(
                    cut_type='benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.main_solution,
                    dual_info=cut_result.benders_dual,
                    sub_obj_value=cut_result.relaxed_obj_value,
                )
                bd_lhs, bd_rhs = self.gen_sce_bds_opt_cut(
                    lp_opt_value=candidate.sub_obj_value,
                    constr_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    forward_sol=candidate.main_solution.as_model_result(),
                    sub_sce_list=scenario_group_list,
                )
                benders_terms.append((bd_lhs, bd_rhs, weight))

            if add_strengthen and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='strengthen_benders',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                sbd_lhs, sbd_rhs, _, _ = self.gen_sce_strengthen_bds_cut(
                    sub_model_sce_list=scenario_group_list,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                )
                strengthen_terms.append((sbd_lhs, sbd_rhs, weight))

            if add_lagrangian and cut_result.fractional_dual is not None:
                candidate = self._make_cut_candidate(
                    cut_type='lagrangian',
                    cut_result=cut_result,
                    iteration_context=iteration_context,
                    main_solution=cut_result.fractional_solution,
                    dual_info=cut_result.fractional_dual,
                    sub_obj_value=cut_result.fractional_relaxed_obj_value,
                )
                lg_lhs, lg_rhs, _, _, _ = self.gen_sub_sce_lag_cut_heuristic(
                    sub_model_sce_list=scenario_group_list,
                    given_main_result=candidate.main_solution.as_model_result(),
                    given_dual_info=candidate.dual_info.dual_values,
                    constr_var_map=candidate.dual_info.var_map,
                    max_ite_num=max_lagrangian_ite,
                )
                lagrangian_terms.append((lg_lhs, lg_rhs, weight))

        if benders_terms:
            self.add_aggregated_cut(cut_name_prefix="bd", ite_name=ite_name, cut_terms=benders_terms)
        if strengthen_terms:
            self.add_aggregated_cut(
                cut_name_prefix="str_bd", ite_name=ite_name, cut_terms=strengthen_terms
            )
        if lagrangian_terms:
            self.add_aggregated_cut(
                cut_name_prefix="str_lag", ite_name=ite_name, cut_terms=lagrangian_terms
            )
