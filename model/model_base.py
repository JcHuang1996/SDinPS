# -*- coding: utf-8 -*-
# @Time     : 2025/09/16
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from util.headers import *
from util.names import *
from util.project_logger import init_logger

import logging
from typing import Optional, Any

import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

logger = logging.getLogger(__name__)


class ModelBase:

    def __init__(self, model_name='default_m', model_data=None):

        self.model_name = model_name
        self.data = model_data

        # create a new Pyomo model
        self.model = pyo.ConcreteModel()
        self.model_relax = None

        # imperative-style container: incremental add like addConstr
        self.model._constr_list = pyo.ConstraintList()
        self._constr_name_map = {}

        # var registry: locate the corresponding var in cloned/relaxed models
        self._var_counter = 0
        self._var_comp_name_by_id = {}  # id(var_obj) -> component name on self.model

        # objective name
        self._objective_name = '_obj'

        # default solver: use gurobi through Pyomo
        self.solver = SolverFactory('gurobi_direct')

        # initialize vars and results dict
        self.var, self.result, self.result_relax = {}, {}, {}

        # objective term dict
        self.obj_term, self.obj_term_value = {}, {}

        # status
        self.solve_status = ModelStatus.UNSOLVED

        # record bounds before fixing
        self.var_record_before_fixed = {}

        # last solve results
        self._last_results = None
        self._last_results_relax = None

    @staticmethod
    def _normalize_bounds(lb, ub):
        if lb is not None and lb <= -float('inf'):
            lb = None
        if ub is not None and ub >= float('inf'):
            ub = None
        return lb, ub

    def add_var(self, domain=pyo.Reals, lb=None, ub=None, name: Optional[str] = None):

        self._var_counter += 1
        comp_name = f'_v{self._var_counter}'

        lb, ub = self._normalize_bounds(lb, ub)
        var_comp = pyo.Var(domain=domain, bounds=(lb, ub))
        self.model.add_component(comp_name, var_comp)
        var_obj = getattr(self.model, comp_name)

        self._var_comp_name_by_id[id(var_obj)] = comp_name

        if name is not None:
            if not hasattr(self.model, '_user_var_name'):
                self.model._user_var_name = {}
            self.model._user_var_name[comp_name] = name

        return var_obj

    def add_constr(self, expr, name: Optional[str] = None):

        idx = len(self.model._constr_list) + 1
        self.model._constr_list.add(expr)
        if name is not None:
            self._constr_name_map[name] = idx
        return self.model._constr_list[idx]

    def set_objective(self, expr, sense=pyo.minimize):

        if hasattr(self.model, self._objective_name):
            self.model.del_component(getattr(self.model, self._objective_name))

        self.model.add_component(self._objective_name, pyo.Objective(expr=expr, sense=sense))

    def set_solver_option(self, option_name: str, value: Any):
        self.solver.options[option_name] = value

    @staticmethod
    def _has_solution(results) -> bool:
        return (results is not None) and (len(results.solution) > 0)

    @staticmethod
    def _fill_uninitialized_vars(model):
        for v in model.component_data_objects(pyo.Var, active=True):
            if v.value is None:
                if v.lb is not None:
                    v.set_value(pyo.value(v.lb))
                elif v.ub is not None:
                    v.set_value(pyo.value(v.ub))
                else:
                    v.set_value(0.0)

    def _set_solve_status_from_results(self, results):
        tc = getattr(results.solver, 'termination_condition', None)

        if tc in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal
        }:
            self.solve_status = ModelStatus.SOLVED_OPT
            return

        if tc == TerminationCondition.infeasible:
            self.solve_status = ModelStatus.INFEASIBLE
            return

        if tc in {TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded}:
            self.solve_status = ModelStatus.INF_OR_UNBD
            return

        if tc in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
            self.solve_status = ModelStatus.SOLVED_TIMEOUT_WS if self._has_solution(results) else ModelStatus.SOLVED_TIMEOUT_WOS
            return

        self.solve_status = f'{ModelStatus.UNKNOWN}_{tc}'

    def solve(self):
        logger.info(f'Optimizing model {self.model_name}')

        self._last_results = self.solver.solve(self.model, tee=True, load_solutions=True)
        self._set_solve_status_from_results(self._last_results)
        self._fill_uninitialized_vars(self.model)

        if self.solve_status == ModelStatus.SOLVED_OPT:
            logger.info('Model solved with an optimal solution')
        elif self.solve_status == ModelStatus.INFEASIBLE:
            logger.info('Model infeasible')
        elif self.solve_status == ModelStatus.INF_OR_UNBD:
            logger.info('Model infeasible or unbounded')
        elif self.solve_status in {ModelStatus.SOLVED_TIMEOUT_WS, ModelStatus.SOLVED_TIMEOUT_WOS}:
            logger.info('Model timed out')
        else:
            logger.info(f'Unknown model status: {self.solve_status}')

    def solve_relaxed(self):
        logger.info(f'Optimizing relaxed model of {self.model_name}')

        self.model_relax = self.model.clone()
        pyo.TransformationFactory('core.relax_integer_vars').apply_to(self.model_relax)

        self._last_results_relax = self.solver.solve(self.model_relax, tee=True, load_solutions=True)
        self._fill_uninitialized_vars(self.model_relax)

    def get_result(self, var_name_list):

        logger.info(f'Get result for following variables: {var_name_list}')

        for var_name in var_name_list:
            self.result[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_obj = self.var[var_name][key]
                var_value = pyo.value(var_obj)
                if var_obj.is_binary():
                    var_value = int(round(var_value))
                self.result[var_name][key] = var_value

        return self.result

    def get_result_relaxed(self, var_name_list):

        logger.info(f'Get relaxed result for following variables: {var_name_list}')

        for var_name in var_name_list:
            self.result_relax[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_obj = self.var[var_name][key]
                comp_name = self._var_comp_name_by_id[id(var_obj)]
                relax_var = getattr(self.model_relax, comp_name)
                self.result_relax[var_name][key] = pyo.value(relax_var)

        return self.result_relax

    def clear_result(self):
        self.result = {}

    def cal_detailed_obj(self):
        for key, expr in self.obj_term.items():
            self.obj_term_value[key] = pyo.value(expr)

    def fix_variable_value(self, var_fix_info):

        for var_class_name in sorted(var_fix_info.keys()):
            self.var_record_before_fixed[var_class_name] = {}
            for var_key in sorted(var_fix_info[var_class_name].keys()):

                var_obj = self.var[var_class_name][var_key]
                self.var_record_before_fixed[var_class_name][var_key] = (var_obj.lb, var_obj.ub)

                var_obj.fix(var_fix_info[var_class_name][var_key])

    def recover_variable_from_fixed(self):

        for var_class_name in sorted(self.var_record_before_fixed.keys()):
            for var_key in sorted(self.var_record_before_fixed[var_class_name].keys()):
                origin_lb, origin_ub = self.var_record_before_fixed[var_class_name][var_key]
                var_obj = self.var[var_class_name][var_key]
                var_obj.unfix()
                var_obj.setlb(origin_lb)
                var_obj.setub(origin_ub)

        self.var_record_before_fixed = {}
