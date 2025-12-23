# -*- coding: utf-8 -*-
# @Time     : 2025/09/16
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from util.headers import *
from util.names import *
from util.project_logger import init_logger

from gurobipy import GRB
import gurobipy as gp
import math
import os
import datetime
import logging

logger = logging.getLogger(__name__)


class ModelBase:

    def __init__(self, model_name='default_m', model_data=None):

        if model_data is None:
            logger.error("Error: no model data provided")
            raise

        self.model_name = model_name
        self.data = model_data

        # create a new gurobi model
        self.model = gp.Model(self.model_name)
        self.model_relax = None                 # incase that we want to solve the relax model

        # initialize vars and results dict
        self.var, self.result, self.result_relax = {}, {}, {}

        # generate dict to save specific terms of objective functions
        self.obj_term, self.obj_term_value = {}, {}

        # initialize model status
        self.solve_status = ModelStatus.UNSOLVED

        # the var lb & ub before they are fixed
        self.var_record_before_fixed = {}

    def solve(self):
        logger.info(f'Optimizing model {self.model_name}')
        self.model.optimize()

        if self.model.status == GRB.Status.OPTIMAL:
            self.solve_status = ModelStatus.SOLVED_OPT
            logger.info("Model solved with an optimal solution")

        elif self.model.status == GRB.Status.INFEASIBLE:
            self.solve_status = ModelStatus.INFEASIBLE
            logger.info("Model infeasible, IIS available")

        elif self.model.status in {GRB.Status.INF_OR_UNBD, GRB.Status.UNBOUNDED}:
            self.solve_status = ModelStatus.INF_OR_UNBD
            logger.info("Model infeasible or unbounded, and IIS not guaranteed")

        elif self.model.status == GRB.Status.TIME_LIMIT:
            if self.model.SolCount > 0:
                self.solve_status = ModelStatus.SOLVED_TIMEOUT_WS
                logger.info("Model timed out, feasible solution found")
            else:
                self.solve_status = ModelStatus.SOLVED_TIMEOUT_WOS
                logger.info("Model timed out, no feasible solution found")

        else:
            self.solve_status = f'{ModelStatus.UNKNOWN}_{self.model.status}'
            logger.info(f'An Unknown model status: {self.model.status}')
            raise

    def solve_relaxed(self):
        logger.info(f'Optimizing relaxed model of {self.model_name}')
        self.model_relax = self.model.relax()
        self.model_relax.optimize()

        if self.model_relax.status == GRB.Status.OPTIMAL:
            logger.info("Relaxed Model solved with an optimal solution")

        elif self.model_relax.status == GRB.Status.INFEASIBLE:
            logger.info("Relaxed Model infeasible, IIS available")

        elif self.model_relax.status in {GRB.Status.INF_OR_UNBD, GRB.Status.UNBOUNDED}:
            logger.info("Relaxed Model infeasible or unbounded, and IIS not guaranteed")

        else:
            logger.info(f'An Unknown model status for relaxed model: {self.model.status}')
            raise

    def get_result(self, var_name_list):

        # todo: check if the var names in the input var name list are valid

        logger.info(f'Get result for following variables: {var_name_list}')

        if len(var_name_list) == 0:
            logger.info('No variable name assigned')
            raise

        for var_name in var_name_list:
            self.result[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_value = self.var[var_name][key].X
                if self.var[var_name][key].VType == GRB.BINARY:
                    var_value = int(var_value)
                self.result[var_name][key] = var_value

        return self.result

    def get_result_relaxed(self, var_name_list):

        logger.info(f'Get relaxed result for following variables: {var_name_list}')

        if len(var_name_list) == 0:
            logger.info('No variable name assigned')
            raise

        for var_name in var_name_list:
            self.result_relax[var_name] = {}
            for key in sorted(self.var[var_name].keys()):
                var_accurate_name = self.var[var_name][key].VarName
                self.result_relax[var_name][key] = self.model_relax.getVarByName(var_accurate_name).X

        return self.result_relax

    def reset_model(self):
        self.model.reset()

    def update_the_model(self):
        """
         Note: the model should be updated before any operations except for solve.

         For example, self.build_model() would add all var, constr, obj, etc.
         However, if one executed 'solve_relaxed()' right after build the model without updating,
         then the relaxed model would be empty, as the un-updated model cannot show the added vars, constrs, obj.
        """
        self.model.update()

    def clear_result(self):
        self.result = {}

    def cal_detailed_obj(self):
        for key, lin_expr in self.obj_term.items():
            self.obj_term_value[key] = lin_expr.getValue()

    def fix_variable_value(self, var_fix_info):
        """
        The function is for fixing variable values.
        :param var_fix_info: the dict recording the variable class name, key name and value to fix,
        having the same structure as self.var
        """
        # update the model before fixing to ensure the model is valid.
        self.update_the_model()

        for var_class_name in sorted(var_fix_info.keys()):
            self.var_record_before_fixed[var_class_name] = {}
            for var_key in sorted(var_fix_info[var_class_name].keys()):

                # record the ub and lb of variables before they are fixed
                origin_lb = self.var[var_class_name][var_key].LB
                origin_ub = self.var[var_class_name][var_key].UB
                self.var_record_before_fixed[var_class_name][var_key] = (origin_lb, origin_ub)

                # fix the variable's value by changing the lb and ub
                var_value_fix = var_fix_info[var_class_name][var_key]
                self.var[var_class_name][var_key].LB = var_value_fix
                self.var[var_class_name][var_key].UB = var_value_fix

        # update the model after fixed it
        self.update_the_model()

    def recover_variable_from_fixed(self):
        """
        The function is for releasing the variable's fix based on the recording.
        If there is no recording, it should raise an error.
        :return:
        """
        if len(self.var_record_before_fixed) == 0:
            raise Exception('No variable fixed info')

        for var_class_name in sorted(self.var_record_before_fixed.keys()):
            for var_key in sorted(self.var_record_before_fixed[var_class_name].keys()):
                origin_lb, origin_ub = self.var_record_before_fixed[var_class_name][var_key]
                self.var[var_class_name][var_key].LB = origin_lb
                self.var[var_class_name][var_key].UB = origin_ub

        # the record should be clear after the fix is released
        self.var_record_before_fixed = {}

        # update the model after releasing the fix
        self.update_the_model()

