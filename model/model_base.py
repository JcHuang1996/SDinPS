# -*- coding: utf-8 -*-
# @Time     : 2025/09/16
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from util.headers import *
from util.names import *
from util.project_logger import init_logger
from model.solver_config import SolverConfig

import logging
import math
from typing import Optional, Any

import pyomo.environ as pyo
from pyomo.core.base.label import cpxlp_label_from_name
from pyomo.opt import SolverFactory, TerminationCondition

logger = logging.getLogger(__name__)


class ModelSolveError(RuntimeError):
    """Raised when a solve does not produce a usable solution and the caller did not allow failure."""

    def __init__(self, model_name, solve_status, message=None):
        self.model_name = model_name
        self.solve_status = solve_status
        super().__init__(
            message or f'Model {model_name} did not produce a usable solution (status: {solve_status}).'
        )


class ModelBase:

    def __init__(self, model_name='default_m', model_data=None, solver_config: Optional[SolverConfig] = None):

        self.model_name = model_name
        self.data = model_data

        # create a new Pyomo model
        self.model = pyo.ConcreteModel()
        self.model_relax = None

        # imperative-style container: incremental add like addConstr
        self.model._constr_list = pyo.ConstraintList()
        self._constr_name_map = {}
        self.model._user_constr_name_by_idx = {}

        # var registry: locate the corresponding var in cloned/relaxed models
        self._var_counter = 0
        self._var_comp_name_by_id = {}  # id(var_ob) -> component name on self.model

        # objective name
        self._objective_name = '_obj'

        # default solver: use gurobi through Pyomo
        self.solver_config = solver_config or SolverConfig()
        self.solver_name = self.solver_config.solver_name
        self.solver = SolverFactory(self.solver_name)
        self._solver_option_overrides = {}
        self._applied_solver_option_keys = set()
        self.solver_options = {}

        # initialize vars and results dict
        self.var, self.result, self.result_relax = {}, {}, {}

        # objective term dict
        self.obj_term, self.obj_term_value = {}, {}

        # status
        self.solve_status = ModelStatus.UNSOLVED
        self.solve_status_relax = ModelStatus.UNSOLVED

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
        var_ob = getattr(self.model, comp_name)

        self._var_comp_name_by_id[id(var_ob)] = comp_name

        if name is not None:
            if not hasattr(self.model, '_user_var_name'):
                self.model._user_var_name = {}
            self.model._user_var_name[comp_name] = name

        return var_ob

    def add_constr(self, expr, name: Optional[str] = None):
        if name is not None and name in self._constr_name_map:
            raise ValueError(f'Duplicate constraint name in {self.model_name}: {name}')

        constr_item = self.model._constr_list.add(expr)
        idx = constr_item.index()
        if name is not None:
            self._constr_name_map[name] = idx
            self.model._user_constr_name_by_idx[idx] = name
        return constr_item

    def get_constr_idx_by_name(self, name):
        return self._constr_name_map.get(name)

    def get_constr_item_by_name(self, name):
        constr_idx = self.get_constr_idx_by_name(name)
        if constr_idx is None:
            raise KeyError(f'Unknown constraint name in {self.model_name}: {name}')
        return self.model._constr_list[constr_idx]

    def get_constr_item_in_relax_by_name(self, name):
        constr_idx = self.get_constr_idx_by_name(name)
        if constr_idx is None:
            raise KeyError(f'Unknown constraint name in {self.model_name}: {name}')
        return self.model_relax._constr_list[constr_idx]

    def remove_constr_by_name(self, name):
        """Remove a named row from the model and both name registries."""
        constr_idx = self.get_constr_idx_by_name(name)
        if constr_idx is None:
            raise KeyError(f'Unknown constraint name in {self.model_name}: {name}')
        constr_item = self.model._constr_list[constr_idx]
        del self.model._constr_list[constr_idx]
        del self._constr_name_map[name]
        self.model._user_constr_name_by_idx.pop(constr_idx, None)
        self._invalidate_solve_results()
        return constr_item

    def activate_constr_by_name(self, name):
        constr_item = self.get_constr_item_by_name(name)
        constr_item.activate()
        return constr_item

    def update_constr_by_name(self, name, expr):
        constr_item = self.get_constr_item_by_name(name)
        constr_item.set_value(expr)
        constr_item.activate()
        self._invalidate_solve_results()
        return constr_item

    def set_objective(self, expr, sense=pyo.minimize):

        if hasattr(self.model, self._objective_name):
            self.model.del_component(getattr(self.model, self._objective_name))

        self.model.add_component(self._objective_name, pyo.Objective(expr=expr, sense=sense))
        self._invalidate_solve_results()

    def _invalidate_solve_results(self):
        """Discard certificates after changing the objective or relaxing/replacing model rows."""
        self._last_results = None
        self._last_results_relax = None
        self.solve_status = ModelStatus.UNSOLVED
        self.solve_status_relax = ModelStatus.UNSOLVED

    def set_solver_option(self, option_name: str, value: Any):
        self.set_parameters({option_name: value})

    def clear_solver_overrides(self):
        """Drop every option previously pushed through set_parameters / set_solver_option."""
        for key in list(self._solver_option_overrides.keys()):
            self.solver.options.pop(key, None)
        self._solver_option_overrides = {}
        self.solver_options = {}

    @staticmethod
    def _solver_available(solver) -> bool:
        try:
            return solver.available(exception_flag=False)
        except Exception:
            return False

    def _resolve_solve_options(self, param_dict=None):
        """Merge config defaults, persistent overrides and this call's overrides, in that order."""
        active_solver_name = getattr(self.solver, 'name', None) or self.solver_name
        options = dict(self.solver_config.to_solver_options(active_solver_name))
        options.update(self._solver_option_overrides)
        if param_dict:
            options.update(param_dict)
        return options

    def _apply_solver_options(self, options):
        """Push a fully resolved option dict onto the active solver."""
        for key in self._applied_solver_option_keys - set(options):
            self.solver.options.pop(key, None)
        for key, value in options.items():
            self.solver.options[key] = value
        self._applied_solver_option_keys = set(options)
        self.solver_options = dict(options)

    def _switch_solver(self, solver_name):
        self.solver_name = solver_name
        self.solver = SolverFactory(self.solver_name)
        self._applied_solver_option_keys = set()

    def _solve_with_solver(self, pyomo_model, options, load_solutions):
        self._apply_solver_options(options)
        return self.solver.solve(
            pyomo_model,
            tee=self.solver_config.tee,
            load_solutions=load_solutions
        )

    def _solve_model(self, pyomo_model, param_dict=None, load_solutions=True, solver_config=None):
        """
        Solve one Pyomo model with the merged option set.

        The fallback solver is scoped to this call: if the primary solver is unavailable or raises,
        the fallback is used for this solve only and the object keeps its configured solver, so one
        transient failure never silently re-routes every later solve.
        """
        if solver_config is not None:
            # A different solve role replaces the whole configuration, including None limits.
            saved_state = (
                self.solver_config, self.solver_name, self.solver,
                self._solver_option_overrides, self._applied_solver_option_keys, self.solver_options,
            )
            try:
                self.solver_config = solver_config
                self._switch_solver(solver_config.solver_name)
                self._solver_option_overrides = {}
                self.solver_options = {}
                return self._solve_model(pyomo_model, param_dict, load_solutions)
            finally:
                (
                    self.solver_config, self.solver_name, self.solver,
                    self._solver_option_overrides, self._applied_solver_option_keys, self.solver_options,
                ) = saved_state

        fallback_solver_name = self.solver_config.fallback_solver_name
        primary_solver_name, primary_solver = self.solver_name, self.solver
        primary_applied_option_keys = set(self._applied_solver_option_keys)
        primary_solver_options = dict(primary_solver.options)

        options = self._resolve_solve_options(param_dict=param_dict)

        use_fallback = (
            fallback_solver_name is not None
            and self.solver_name != fallback_solver_name
            and not self._solver_available(self.solver)
        )
        if use_fallback:
            logger.warning(
                f'Solver {self.solver_name} is unavailable; using fallback solver {fallback_solver_name} '
                f'for this solve of {self.model_name}.'
            )
            self._switch_solver(fallback_solver_name)
            options = self._resolve_solve_options(param_dict=param_dict)

        try:
            return self._solve_with_solver(pyomo_model, options, load_solutions)
        except Exception:
            if fallback_solver_name is None or self.solver_name == fallback_solver_name:
                raise

            logger.warning(
                f'Solver {self.solver_name} failed; retrying with fallback solver {fallback_solver_name} '
                f'for this solve of {self.model_name}.'
            )
            self._switch_solver(fallback_solver_name)
            return self._solve_with_solver(
                pyomo_model,
                self._resolve_solve_options(param_dict=param_dict),
                load_solutions,
            )
        finally:
            if self.solver_name != primary_solver_name:
                self.solver_name, self.solver = primary_solver_name, primary_solver
                for key in set(self.solver.options) - set(primary_solver_options):
                    self.solver.options.pop(key, None)
                for key, value in primary_solver_options.items():
                    self.solver.options[key] = value
                self._applied_solver_option_keys = primary_applied_option_keys

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

    def _derive_solve_status(self, results):
        """Map a Pyomo termination condition onto the project's ModelStatus vocabulary."""
        tc = getattr(results.solver, 'termination_condition', None)

        if tc in {
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.globallyOptimal
        }:
            return ModelStatus.SOLVED_OPT

        if tc == TerminationCondition.infeasible:
            return ModelStatus.INFEASIBLE

        if tc in {TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded}:
            return ModelStatus.INF_OR_UNBD

        if tc in {TerminationCondition.maxTimeLimit, TerminationCondition.maxIterations}:
            return ModelStatus.SOLVED_TIMEOUT_WS if self._has_solution(results) else ModelStatus.SOLVED_TIMEOUT_WOS

        # Several open-source interfaces report 'feasible'/'other' on an otherwise successful solve.
        # Such a point is usable but NOT proven optimal, so it gets its own status rather than
        # being promoted to SOLVED_OPT.
        if self._has_solution(results):
            return ModelStatus.SOLVED_FEASIBLE

        return f'{ModelStatus.UNKNOWN}_{tc}'

    def _set_solve_status_from_results(self, results):
        self.solve_status = self._derive_solve_status(results)

    def set_parameters(self, param_dict):
        """
        :param param_dict: the parameters' name and value to set. Default: Gurobi's parameters
        :return: none
        Register persistent solver-option overrides. These are merged ON TOP of the SolverConfig
        defaults (they no longer suppress them); use clear_solver_overrides() to drop them again.
        """
        if param_dict is None:
            return
        for key, value in param_dict.items():
            self._solver_option_overrides[key] = value
            self.solver.options[key] = value
        self.solver_options.update(param_dict)

    @staticmethod
    def _status_has_usable_solution(solve_status) -> bool:
        return solve_status in {
            ModelStatus.SOLVED_OPT,
            ModelStatus.SOLVED_FEASIBLE,
            ModelStatus.SOLVED_TIMEOUT_WS,
        }

    def _raise_if_solve_failed(self, solve_status, allow_failure):
        if allow_failure or self._status_has_usable_solution(solve_status):
            return
        raise ModelSolveError(model_name=self.model_name, solve_status=solve_status)

    def solve(self, param_dict=None, allow_failure=False):
        """
        Solve the model.

        ``param_dict`` is a PER-CALL override merged on top of the SolverConfig defaults and any
        persistent overrides; it does not persist on the object. Unless ``allow_failure`` is set, a
        solve that yields no usable solution raises ModelSolveError instead of silently returning
        fabricated variable values.
        """
        logger.info(f'Optimizing model {self.model_name}')

        self._last_results = None
        self.solve_status = ModelStatus.UNSOLVED
        self._last_results = self._solve_model(
            self.model,
            param_dict=param_dict,
            load_solutions=True
        )
        self._set_solve_status_from_results(self._last_results)

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

        # Only fabricate values for variables the solver genuinely left untouched, and only when the
        # solve actually produced a solution; otherwise the fabricated point looks like a real one.
        if self._status_has_usable_solution(self.solve_status):
            self._fill_uninitialized_vars(self.model)

        self._raise_if_solve_failed(self.solve_status, allow_failure)

    def get_obj_value(self):
        return pyo.value(self.model.find_component(self._objective_name))

    def _get_objective_sense(self, pyomo_model=None):
        pyomo_model = self.model if pyomo_model is None else pyomo_model
        objective = pyomo_model.find_component(self._objective_name)
        if objective is None:
            return pyo.minimize
        return objective.sense

    @staticmethod
    def _extract_problem_bound(results, sense):
        """Read the solver's dual bound out of a Pyomo results object, or None when unavailable."""
        problem = getattr(results, 'problem', None)
        if problem is None:
            return None

        raw_bound = problem.lower_bound if sense == pyo.minimize else problem.upper_bound
        if raw_bound is None:
            return None

        try:
            bound_value = float(raw_bound)
        except (TypeError, ValueError):
            return None

        if math.isnan(bound_value) or math.isinf(bound_value):
            return None
        return bound_value

    def get_obj_bound(self, allow_incumbent_fallback=False):
        """
        Return the solver's DUAL bound on the objective of the last solve.

        This is the quantity every cut anchor must use: for a minimization it satisfies
        ``bound <= true optimum`` under any MIP gap or time limit, so a truncated solve weakens a
        cut instead of invalidating it. At proven optimality it coincides with the objective value.
        """
        if self._last_results is None:
            raise ModelSolveError(
                model_name=self.model_name,
                solve_status=self.solve_status,
                message=f'Model {self.model_name} has no solve results; cannot read an objective bound.',
            )

        sense = self._get_objective_sense()
        bound_value = self._extract_problem_bound(self._last_results, sense)
        if bound_value is not None:
            return bound_value

        if self.solve_status == ModelStatus.SOLVED_OPT:
            # A proven-optimal solve has dual bound == incumbent value.
            return self.get_obj_value()

        if allow_incumbent_fallback and self._status_has_usable_solution(self.solve_status):
            logger.warning(
                f'No dual bound reported for {self.model_name}; falling back to the incumbent value. '
                f'Cuts anchored on this value are only valid if the solve was exact.'
            )
            return self.get_obj_value()

        raise ModelSolveError(
            model_name=self.model_name,
            solve_status=self.solve_status,
            message=f'Model {self.model_name} reported no usable objective bound.',
        )

    def _prepare_relaxed_model(self, relaxed_model):
        """Hook for subclasses that need extra components on the relaxation (e.g. a dual Suffix)."""
        return

    def solve_relaxed(self, param_dict=None, allow_failure=False, solver_config=None):
        logger.info(f'Optimizing relaxed model of {self.model_name}')

        self._last_results_relax = None
        self.solve_status_relax = ModelStatus.UNSOLVED
        self.model_relax = self.model.clone()
        pyo.TransformationFactory('core.relax_integer_vars').apply_to(self.model_relax)
        self._prepare_relaxed_model(self.model_relax)

        self._last_results_relax = self._solve_model(
            self.model_relax,
            param_dict=param_dict,
            load_solutions=True,
            solver_config=solver_config,
        )
        self.solve_status_relax = self._derive_solve_status(self._last_results_relax)

        if self._status_has_usable_solution(self.solve_status_relax):
            self._fill_uninitialized_vars(self.model_relax)

        if not allow_failure and not self._status_has_usable_solution(self.solve_status_relax):
            raise ModelSolveError(
                model_name=f'{self.model_name} (relaxed)',
                solve_status=self.solve_status_relax,
            )

    def get_relaxed_obj_value(self):
        return pyo.value(self.model_relax.find_component(self._objective_name))

    def get_relaxed_obj_bound(self, allow_incumbent_fallback=False):
        """Dual bound of the relaxation; equals the LP optimum when the relaxation solved exactly."""
        if self._last_results_relax is None:
            raise ModelSolveError(
                model_name=f'{self.model_name} (relaxed)',
                solve_status=self.solve_status_relax,
                message=f'Relaxed model of {self.model_name} has no solve results.',
            )

        sense = self._get_objective_sense(self.model_relax)
        bound_value = self._extract_problem_bound(self._last_results_relax, sense)
        if bound_value is not None:
            return bound_value

        if self.solve_status_relax == ModelStatus.SOLVED_OPT:
            return self.get_relaxed_obj_value()

        if allow_incumbent_fallback and self._status_has_usable_solution(self.solve_status_relax):
            return self.get_relaxed_obj_value()

        raise ModelSolveError(
            model_name=f'{self.model_name} (relaxed)',
            solve_status=self.solve_status_relax,
            message=f'Relaxed model of {self.model_name} reported no usable objective bound.',
        )

    def get_result(self, var_name_list):

        logger.info(f'Get result for following variables: {var_name_list}')

        for var_name in var_name_list:
            self.result[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_ob = self.var[var_name][key]
                var_value = pyo.value(var_ob)
                if var_ob.is_binary():
                    var_value = int(round(var_value))
                self.result[var_name][key] = var_value

        return self.result

    def get_result_relaxed(self, var_name_list):

        logger.info(f'Get relaxed result for following variables: {var_name_list}')

        for var_name in var_name_list:
            self.result_relax[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_ob = self.var[var_name][key]
                comp_name = self._var_comp_name_by_id[id(var_ob)]
                relax_var = getattr(self.model_relax, comp_name)
                self.result_relax[var_name][key] = pyo.value(relax_var)

        return self.result_relax

    def clear_result(self):
        self.result = {}

    def cal_detailed_obj(self):
        for key, expr in self.obj_term.items():
            self.obj_term_value[key] = pyo.value(expr)

    def _get_lp_label_name(self, obj):
        if getattr(obj, 'ctype', None) is pyo.Constraint and obj.parent_component() is self.model._constr_list:
            user_name = self.model._user_constr_name_by_idx.get(obj.index())
            if user_name is not None:
                return cpxlp_label_from_name(user_name)
        return cpxlp_label_from_name(obj.getname(True))

    def _validate_lp_constraint_labels(self):
        sanitized_name_map = {}
        for idx, raw_name in self.model._user_constr_name_by_idx.items():
            constr_item = self.model._constr_list[idx]
            if not constr_item.active:
                continue

            sanitized_name = cpxlp_label_from_name(raw_name)
            existing_name = sanitized_name_map.get(sanitized_name)
            if existing_name is not None and existing_name != raw_name:
                raise ValueError(
                    'Constraint names conflict after LP sanitization: '
                    f'"{existing_name}" and "{raw_name}" both map to "{sanitized_name}".'
                )
            sanitized_name_map[sanitized_name] = raw_name

    def write_file(self, file_name):
        self._validate_lp_constraint_labels()
        self.model.write(file_name, io_options={'labeler': self._get_lp_label_name})

    def fix_variable_value(self, var_fix_info):
        self._invalidate_solve_results()
        for var_class_name in sorted(var_fix_info.keys()):
            saved = self.var_record_before_fixed.setdefault(var_class_name, {})
            for var_key in sorted(var_fix_info[var_class_name].keys()):

                var_ob = self.var[var_class_name][var_key]
                saved.setdefault(var_key, (var_ob.lb, var_ob.ub, var_ob.fixed, var_ob.value))

                var_ob.fix(var_fix_info[var_class_name][var_key])

    def recover_variable_from_fixed(self):
        self._invalidate_solve_results()
        for var_class_name in sorted(self.var_record_before_fixed.keys()):
            for var_key in sorted(self.var_record_before_fixed[var_class_name].keys()):
                origin_lb, origin_ub, was_fixed, original_value = self.var_record_before_fixed[var_class_name][var_key]
                var_ob = self.var[var_class_name][var_key]
                var_ob.unfix()
                var_ob.setlb(origin_lb)
                var_ob.setub(origin_ub)
                if was_fixed:
                    var_ob.fix(original_value)

        self.var_record_before_fixed = {}
