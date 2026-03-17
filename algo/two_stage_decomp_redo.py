# -*- coding: utf-8 -*-
# @Time     : 2026/01/28
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from typing import Any, Callable, List, Optional, Tuple
from util.names import DataName, ObjName, VarName

from dao.data_processor import DataProcessor
from model import ModelMain, ModelSub, ModelCombined, ModelLagrangianMultiplierHeuristic, ModelInnerMinimizationProblem, ModelLagrangianCutDeterministic
from util.tools import iter_general_csv, iter_sub_prob_info

import pyomo.environ as pyo
import logging
import os
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

    def build_main_stage_model(self):

        # process the data for main model
        self.main_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=self.scenario_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        self.sce_prob_dict = self.main_model_data[DataName.DICT_SC_PROB].copy()

        # build the main model with processed main model data
        self.model_main = ModelMain(model_name='Main', model_data=self.main_model_data)
        self.model_main.build_main_model()

    def solve_main_stage_model(self):
        # solve the main model, and compute the obj terms
        # return the optimal value of the solved main model
        self.model_main.set_parameters(
            param_dict={
                'TimeLimit': 60
            }
        )
        self.model_main.solve()
        self.model_main.cal_detailed_obj()

    def solve_main_stage_relaxed_model(self):
        self.model_main.solve_relaxed()

    def record_main_stage_model(self, ite_name=None):
        # show and record the main model objective value (the best bound objective value)
        total_objective_value = self.model_main.get_obj_value()
        main_stage_objective_value = (
                self.model_main.obj_term_value[ObjName.DG_FIXED_COST]
                + self.model_main.obj_term_value[ObjName.LINE_HARDEN_COST]
                + self.model_main.obj_term_value[ObjName.DG_VARIANT_COST]
        )

        # create the corresponding dict if
        if ite_name not in self.ite_obj_value_dict:
            self.ite_obj_value_dict[ite_name] = {}

        # record the objective value and values of every terms
        self.ite_obj_value_dict[ite_name]['main_obj(bound)'] = {
            'total': total_objective_value,
            'detail': self.model_main.obj_term_value.copy()
        }

        return main_stage_objective_value, total_objective_value

    def build_sub_model(self, sub_model_sce_list=None, given_main_result=None):
        """
        The function iterates one round on the given scenarios based on the given main result.
        :param
        sub_model_sce_list: the list of scenarios considered in this sub model.
        given_main_result: the main stage result for current iteration
        :return:
        """

        # the sub model's scenario cannot be empty
        if sub_model_sce_list is None:
            raise ValueError('sub_model_sce_list cannot be None')

        # process the sub model data
        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

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
        self.current_sub_model.build_sub_model()

    def build_sub_model_redo(self, sub_model_sce_list=None, given_main_result=None):
        """
        The function iterates one round on the given scenarios based on the given main result.
        :param
        sub_model_sce_list: the list of scenarios considered in this sub model.
        given_main_result: the main stage result for current iteration
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
            self.data_processor_module.clear_existing_data()
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
        self.current_sub_model.build_sub_model_redo()

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
            self.data_processor_module.clear_existing_data()
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
        self.curr_sub_frac_model.build_sub_model_redo()

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

    def gen_sce_bds_opt_cut(self, lp_opt_value=None, constr_dual_info=None, constr_var_map=None, forward_sol=None):
        """
        Idea: generate benders opt cuts components in the format of:
         \theta \geq Q^{LP}_{obtained_main_solution} + pi^{T} * (main var - obtained main solution)
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
            inner_model = ModelInnerMinimizationProblem(
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
        potential_cTx, potential_z, potential_lambda = cTx, z.copy(), curr_lag_multiplier.copy()
        prev_lift_value = 0

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

        constr_list = sorted(given_dual_info.keys())
        for ite_num in range(1, max_ite_num + 1):

            # obtain new lagrangian multiplier (dual multiplier)
            lagrangian_multiplier_model.add_constr_for_regularized_model(sub_obj_value=cTx, main_sol=z)
            lagrangian_multiplier_model.set_objective_function(
                stabilization_param=0.01 / (1 + ite_num),
                prev_multiplier=curr_lag_multiplier,
            )
            lagrangian_multiplier_model.solve()

            # test if the new lagrangian multiplier could lift the cut
            lift_value = lagrangian_multiplier_model.obtain_lift_value()

            if lift_value - prev_lift_value <= 0.001 * math.fabs(prev_lift_value):
                # if the new multiplier cannot lift the cut, then stop
                # giving different feedback in log based on if the cut is improved based on the strengthen benders
                if ite_num == 2:
                    logger.info(f'Fail to generate better cut, lift:{lift_value}, prev:{prev_lift_value}')
                else:
                    logger.info(f'Improvement on generating Lagrangian cut, improve:{lift_value - prev_lift_value}')
                break

            prev_lift_value = lift_value

            # otherwise, i.e.  the previous x, z, and dual multiplier lift the cut, then record these new result
            # note that in the first iteration, the ctx, z, and lambda here form a strengthen benders
            potential_cTx, potential_z, potential_lambda = cTx, z.copy(), curr_lag_multiplier.copy()

            curr_lag_multiplier = lagrangian_multiplier_model.get_result(
                var_name_list=[VarName.LAG_MULTIPLIER]
            )[VarName.LAG_MULTIPLIER]

            # then generate new terms for obtaining new dual multiplier for a new trail
            lhs, rhs, cTx, z = self.gen_sce_strengthen_bds_cut(
                sub_model_sce_list=sub_model_sce_list,
                given_main_result=given_main_result,
                given_dual_info=curr_lag_multiplier,
                constr_var_map=constr_var_map,
            )

        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_model_sce_list
        )
        rhs = potential_cTx + pyo.quicksum(
            potential_lambda[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
             - potential_z[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )
        return lhs, rhs, potential_cTx, potential_z

    def gen_sub_sce_lag_cut_deterministic(
            self,
            sub_model_sce_list=None,
            given_main_result=None,
            given_dual_info=None,
            constr_var_map=None,
        ):

        sub_model_data = self.data_processor_module.data_process(
            scenario_list_assigned=sub_model_sce_list,
            time_list_assigned=self.time_list
        )
        self.data_processor_module.clear_existing_data()

        lagrangian_cut_model = ModelLagrangianCutDeterministic(
            model_name=f'{tuple(sub_model_sce_list)}_lcdm',
            model_data=sub_model_data,
            main_result=given_main_result,
            constr_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )
        lagrangian_cut_model.build_model()
        lagrangian_cut_model.solve()

        main_var_name_list = sorted(given_main_result.keys())
        main_sol = lagrangian_cut_model.get_result(main_var_name_list)
        sub_obj_value = lagrangian_cut_model.get_result([VarName.AR_VAR_OBJ])[VarName.AR_VAR_OBJ]
        lag_multiplier = lagrangian_cut_model.get_result([VarName.LAG_MULTIPLIER])[VarName.LAG_MULTIPLIER]

        constr_list = sorted(given_dual_info.keys())
        lhs = pyo.quicksum(
            self.model_main.var[VarName.SUB_OBJ_EST][sub_sce] for sub_sce in sub_model_sce_list
        )
        rhs = sub_obj_value + pyo.quicksum(
            lag_multiplier[constr_name]
            * (self.model_main.var[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]]
             - main_sol[constr_var_map[constr_name][0]][constr_var_map[constr_name][1]])
            for constr_name in constr_list
        )

        return lhs, rhs, sub_obj_value, main_sol, lag_multiplier

    def add_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        sub_obj_value,
        constr_dual_info,
        constr_var_map,
        forward_sol,
        cut_name_prefix: str = "bd",
    ) -> Tuple[str, Any, Any]:
        """
        Generate Benders optimality cut (via gen_sce_bds_opt_cut) and add it to the main model.
        Returns (cut_name, lhs, rhs) so callers can compare with other cuts in future logic.
        """
        bd_lhs, bd_rhs = self.gen_sce_bds_opt_cut(
            lp_opt_value=sub_obj_value,
            constr_dual_info=constr_dual_info,
            constr_var_map=constr_var_map,
            forward_sol=forward_sol,
        )
        cut_name = f"{cut_name_prefix}_s_{str(sub_sce_list)}_{ite_name}"
        self.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=bd_lhs, cut_rhs=bd_rhs)
        return cut_name, bd_lhs, bd_rhs

    def add_strengthen_benders_cut(
        self,
        sub_sce_list,
        ite_name,
        given_main_result,
        given_dual_info,
        constr_var_map,
        cut_name_prefix: str = "str_bd",
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
    ) -> Tuple[str, Any, Any, float, Any]:
        """
        Generate strengthened Benders cut (via gen_sce_strengthen_bds_cut) and optionally add it.
        If should_add_cut is provided, the cut is added only when should_add_cut(lhs, rhs, cut_name) is True.
        Otherwise the cut is always added. Returns (cut_name, lhs, rhs, obtained_sub_obj, obtained_main_sol)
        for possible comparison with other cuts in future functions.
        """
        sbd_lhs, sbd_rhs, obtained_sub_obj, obtained_main_sol = self.gen_sce_strengthen_bds_cut(
            sub_model_sce_list=sub_sce_list,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
        )
        cut_name = f"{cut_name_prefix}_s_{str(sub_sce_list)}_{ite_name}"
        if should_add_cut is None or should_add_cut(sbd_lhs, sbd_rhs, cut_name):
            self.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=sbd_lhs, cut_rhs=sbd_rhs)
        return cut_name, sbd_lhs, sbd_rhs, obtained_sub_obj, obtained_main_sol

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
    ) -> Tuple[str, Any, Any, float, Any]:
        """
        Generate Lagrangian cut (via gen_sub_sce_lag_cut_heuristic) and optionally add it.
        If should_add_cut is provided, the cut is added only when should_add_cut(lhs, rhs, cut_name) is True.
        Otherwise the cut is always added. Returns (cut_name, lhs, rhs, sub_obj, main_sol)
        for possible comparison with other cuts in future functions.
        """
        lg_lhs, lg_rhs, sub_obj, main_sol = self.gen_sub_sce_lag_cut_heuristic(
            sub_model_sce_list=sub_sce_list,
            given_main_result=given_main_result,
            given_dual_info=given_dual_info,
            constr_var_map=constr_var_map,
            max_ite_num=max_ite_num,
        )
        cut_name = f"{cut_name_prefix}_s_{str(sub_sce_list)}_{ite_name}"
        if should_add_cut is None or should_add_cut(lg_lhs, lg_rhs, cut_name):
            self.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=lg_lhs, cut_rhs=lg_rhs)
        return cut_name, lg_lhs, lg_rhs, sub_obj, main_sol

    def add_aggregated_cut(
        self,
        cut_name_prefix: str,
        ite_name: str,
        cut_terms: List[Tuple[Any, Any, float]],
        should_add_cut: Optional[Callable[[Any, Any, str], bool]] = None,
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
        if should_add_cut is None or should_add_cut(agg_lhs, agg_rhs, cut_name):
            self.model_main.add_custom_cut(cut_name=cut_name, cut_lhs=agg_lhs, cut_rhs=agg_rhs)
        return cut_name, agg_lhs, agg_rhs

    def _gather_one_group_cut_data(
        self,
        sub_sce_list: List,
        curr_main_result: Any,
        frac_main_result: Any,
        ite_name: str,
        add_strengthen: bool,
        add_lagrangian: bool,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
    ) -> Tuple[float, Any, Any, Optional[Any], Optional[Any]]:
        """
        Build sub model for sub_sce_list, solve relaxed, optionally record incumbent, collect duals.
        If add_strengthen or add_lagrangian, also build/solve frac model and return frac duals.
        When incumbent_records is not None, solve sub, record, and append (sub_sce_list, sub_obj_w_main, sub_res).
        Returns (sub_obj_value_for_benders, constr_dual_bi, constr_var_map_bi, constr_dual_frac, constr_var_map_frac).
        constr_dual_frac and constr_var_map_frac are None when both add_strengthen and add_lagrangian are False.
        """
        self.build_sub_model_redo(
            sub_model_sce_list=sub_sce_list,
            given_main_result=curr_main_result,
        )
        local_copy_var_constr_name = sorted(self.current_sub_model.local_copy_constr_info.keys())
        self.solve_relaxed_sub_model()
        sub_obj_value_for_benders = self.current_sub_model.get_relaxed_obj_value()

        if incumbent_records is not None:
            self.solve_sub_model()
            sub_obj_w_main, _ = self.record_sub_model(
                main_stage_obj_value=main_stage_obj,
                ite_name=ite_name,
            )
            sub_res = self.current_sub_model.get_result(
                [VarName.DG_ACTIVE_POWER, VarName.DG_RATED_POWER]
            )
            incumbent_records.append((sub_sce_list, sub_obj_w_main, sub_res))

        constr_dual_bi, constr_var_map_bi = self.current_sub_model.collect_dual_opt_sol(
            constr_to_collect_list=local_copy_var_constr_name
        )

        constr_dual_frac, constr_var_map_frac = None, None
        if add_strengthen or add_lagrangian:
            self.build_sub_frac_model(
                sub_model_sce_list=sub_sce_list,
                given_main_frac_result=frac_main_result,
            )
            self.solve_sub_frac_model()
            constr_dual_frac, constr_var_map_frac = self.curr_sub_frac_model.collect_dual_opt_sol(
                constr_to_collect_list=local_copy_var_constr_name
            )

        return (
            sub_obj_value_for_benders,
            constr_dual_bi,
            constr_var_map_bi,
            constr_dual_frac,
            constr_var_map_frac,
        )

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

        for sub_sce_list in sce_group_list:
            (
                sub_obj_value_for_benders,
                constr_dual_bi,
                constr_var_map_bi,
                constr_dual_frac,
                constr_var_map_frac,
            ) = self._gather_one_group_cut_data(
                sub_sce_list=sub_sce_list,
                curr_main_result=curr_main_result,
                frac_main_result=frac_main_result,
                ite_name=ite_name,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                main_stage_obj=main_stage_obj,
                incumbent_records=incumbent_records,
            )

            if add_benders:
                self.add_benders_cut(
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    sub_obj_value=sub_obj_value_for_benders,
                    constr_dual_info=constr_dual_bi,
                    constr_var_map=constr_var_map_bi,
                    forward_sol=curr_main_result,
                )

            if add_strengthen and constr_dual_frac is not None:
                self.add_strengthen_benders_cut(
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    given_main_result=frac_main_result,
                    given_dual_info=constr_dual_frac,
                    constr_var_map=constr_var_map_frac,
                )

            if add_lagrangian and constr_dual_frac is not None:
                self.add_lagrangian_cut(
                    sub_sce_list=sub_sce_list,
                    ite_name=ite_name,
                    given_main_result=frac_main_result,
                    given_dual_info=constr_dual_frac,
                    constr_var_map=constr_var_map_frac,
                    max_ite_num=max_lagrangian_ite,
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
        *,
        main_stage_obj: Optional[float] = None,
        incumbent_records: Optional[List] = None,
        max_lagrangian_ite: int = 10,
    ) -> None:
        """
        For each scenario group: gather cut data and collect (lhs, rhs, weight). After the loop, add
        exactly one cut per type using probability-weighted lhs and rhs. Corresponds to use_aggregated_cuts=True.
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

        benders_terms: List[Tuple[Any, Any, float]] = []
        strengthen_terms: List[Tuple[Any, Any, float]] = []
        lagrangian_terms: List[Tuple[Any, Any, float]] = []

        for sub_sce_list in sce_group_list:
            (
                sub_obj_value_for_benders,
                constr_dual_bi,
                constr_var_map_bi,
                constr_dual_frac,
                constr_var_map_frac,
            ) = self._gather_one_group_cut_data(
                sub_sce_list=sub_sce_list,
                curr_main_result=curr_main_result,
                frac_main_result=frac_main_result,
                ite_name=ite_name,
                add_strengthen=add_strengthen,
                add_lagrangian=add_lagrangian,
                main_stage_obj=main_stage_obj,
                incumbent_records=incumbent_records,
            )

            weight = sum(self.sce_prob_dict[s] for s in sub_sce_list)

            if add_benders:
                bd_lhs, bd_rhs = self.gen_sce_bds_opt_cut(
                    lp_opt_value=sub_obj_value_for_benders,
                    constr_dual_info=constr_dual_bi,
                    constr_var_map=constr_var_map_bi,
                    forward_sol=curr_main_result,
                )
                benders_terms.append((bd_lhs, bd_rhs, weight))

            if add_strengthen and constr_dual_frac is not None:
                sbd_lhs, sbd_rhs, _, _ = self.gen_sce_strengthen_bds_cut(
                    sub_model_sce_list=sub_sce_list,
                    given_main_result=frac_main_result,
                    given_dual_info=constr_dual_frac,
                    constr_var_map=constr_var_map_frac,
                )
                strengthen_terms.append((sbd_lhs, sbd_rhs, weight))

            if add_lagrangian and constr_dual_frac is not None:
                lg_lhs, lg_rhs, _, _ = self.gen_sub_sce_lag_cut_heuristic(
                    sub_model_sce_list=sub_sce_list,
                    given_main_result=frac_main_result,
                    given_dual_info=constr_dual_frac,
                    constr_var_map=constr_var_map_frac,
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
