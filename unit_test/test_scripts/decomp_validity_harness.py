# -*- coding: utf-8 -*-
# @Time     : 2026/09/17
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

r"""
Baseline verification harness for the decomposition (docs/development_record.md, phase P0).

It is deliberately a general INSTRUMENT, not a fixed test: every check below is stated in terms of
quantities the algorithm already produces, so it keeps working when a cut family is added or a
convention changes.

Checks
------
B2  ground truth      solve ModelCombined on the instance -> z*, then assert
                      ``LB_k <= z* + eps`` and ``UB_k >= z* - eps``.
                      With --require-convergence, also require the final LB and UB to reach z*.
B3  cut validity      the general instrument. Every cut is reduced to plain numbers
                      ``sum_s c_s theta_s >= r(x)``. Because any feasible (x, theta) of the true
                      problem satisfies ``theta_s >= Q_s(x)`` and every ``c_s >= 0``, the cut is
                      valid iff

                          for all x in X:   sum_s c_s Q_s(x)  >=  r(x)

                      which the harness checks by enumerating X and solving each ``Q_s(x)`` exactly.
                      This catches R3.1, R3.2, R3.5 and any future cut family.
B4  monotonicity      the recorded valid lower bounds never decrease (valid while no cut is removed).
B6  golden file       cut algebra + bound trajectory, refreshed deliberately with --write-golden.

Q_s(x) is the CONDITIONAL, unweighted recourse cost of scenario s. The oracle uses unit objective
weight for that scenario and the same run-level big-M as the decomposition.

Run
---
    python unit_test/test_scripts/decomp_validity_harness.py --solver gurobi_direct
    python unit_test/test_scripts/decomp_validity_harness.py --solver appsi_highs \
        --instance m6_two_scenario_hedging --families benders,strengthen_benders

    # the P3 acceptance criterion: one non-singleton group with unequal probabilities
    python unit_test/test_scripts/decomp_validity_harness.py --instance m6_two_scenario_hedging \
        --scenario-groups "s_1+s_2"

    # the P1 acceptance criterion: cut validity must survive a deliberately starved inner solve
    python unit_test/test_scripts/decomp_validity_harness.py --instance m6_two_scenario_hedging \
        --families strengthen_benders,lagrangian --inner-min-time-limit 1 --inner-min-mip-gap 0.5
"""

import argparse
import itertools
import json
import math
import os
import sys
import traceback
from dataclasses import replace

if __name__ == "__main__":
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

import logging

from algo.interfaces import AggregationPolicy
from algo.runner import CutSchedule, DecompositionRunner
from algo.two_stage_decomp_redo import TwoStageDecompRedo
from dao.data_processor import DataProcessor
from dao.data_reader import DataReader
from model.model_combined import ModelCombined
from model.model_sub import ModelSub
from model.solver_config import SolverConfigSet
from util.headers import ScenarioProbHeader
from util.names import (
    BoundModeName,
    ConstrName,
    CutTypeName,
    DataName,
    InputDataName,
    InputMethodName,
    MainProblemModelTypeName,
    ModelStatus,
    SolverRoleName,
    VarName,
)
from util.project_logger import init_logger

logger = logging.getLogger(__name__)

MICRO_BENCH_DIR_NAME = 'micro_bench'
DEFAULT_INSTANCE = 'm6_two_scenario_hedging'

#: Absolute tolerance for a cut-validity violation. A cut is reported as INVALID only when it cuts
#: off a feasible point by more than this, so solver noise is not mistaken for a defect.
CUT_VALIDITY_ABS_TOL = 1e-5
#: Relative companion of the above, scaled by the magnitude of the compared values.
CUT_VALIDITY_REL_TOL = 1e-7
#: Tolerance for the bound checks.
BOUND_ABS_TOL = 1e-5
BOUND_REL_TOL = 1e-6

STATE_VAR_NAME_LIST = [VarName.DG_INSTALL, VarName.LINE_HARDEN, VarName.DG_INSTALL_TYPE]


# =========================================================================================
# Data loading
# =========================================================================================

def default_local_file_path():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test_local_csv_file'))


def load_raw_data(local_file_path, instance_name):
    reader = DataReader(
        read_method=InputMethodName.LOCAL_CSV,
        local_file_path=local_file_path,
        data_set_name=f'{MICRO_BENCH_DIR_NAME}/{instance_name}',
    )
    reader.read()
    return reader.raw_data


def json_key(key):
    """
    Render one dict key as a JSON object name.

    Several of the structures reported here are keyed by tuples -- ``xl`` by ``(from_bus, to_bus)``,
    the cut algebra by ``(var_name, var_key)``. ``json.dump``'s ``default=`` hook only ever applies
    to VALUES, so a tuple key raises TypeError and truncates the file. The '|' separator matches the
    convention already used by micro_bench_test.
    """
    if isinstance(key, tuple):
        # Flatten nested tuples too, so ('xl', ('node_1', 'node_2')) reads as 'xl|node_1|node_2'.
        return '|'.join(str(json_key(part)) for part in key)
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    return str(key)


def json_safe(value):
    """Recursively convert a report structure into something json.dump always accepts."""
    if isinstance(value, dict):
        return {json_key(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_instance_reference(local_file_path, instance_name):
    """Read reference.json when the instance ships one; it is optional."""
    path = os.path.join(local_file_path, MICRO_BENCH_DIR_NAME, instance_name, 'reference.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# =========================================================================================
# B2 -- ground truth
# =========================================================================================

def solve_extensive_form(raw_data, scenario_list, time_list, solver_config):
    """Solve ModelCombined on the full scenario set and return (objective, obj terms, x*)."""
    processor = DataProcessor(raw_data=raw_data)
    processor.reference_scenario_list = list(scenario_list)
    processed_data = processor.data_process(
        scenario_list_assigned=scenario_list,
        time_list_assigned=time_list,
    )

    model = ModelCombined(
        model_name='reference_extensive_form',
        model_data=processed_data,
        solver_config=solver_config,
    )
    model.build_model_all_obj_terms()
    model.solve()
    require_exact_objective(model)
    model.cal_detailed_obj()
    model.get_result(STATE_VAR_NAME_LIST)

    return {
        'objective': model.get_obj_value(),
        'objective_bound': model.get_obj_bound(),
        'objective_terms': dict(model.obj_term_value),
        'solve_status': model.solve_status,
        'first_stage': {
            var_name: dict(model.result[var_name])
            for var_name in STATE_VAR_NAME_LIST
        },
    }


# =========================================================================================
# B3 -- the recourse oracle and the first-stage enumeration
# =========================================================================================

def require_exact_objective(model):
    """Reject a feasible point or an unresolved MIP gap as an exact reference value."""
    if model.solve_status != ModelStatus.SOLVED_OPT:
        raise RuntimeError(f'{model.model_name} must be solved to proven optimality for verification.')
    objective, bound = model.get_obj_value(), model.get_obj_bound()
    if not math.isfinite(objective) or not math.isclose(
        objective, bound, rel_tol=BOUND_REL_TOL, abs_tol=BOUND_ABS_TOL
    ):
        raise RuntimeError(f'{model.model_name} has an unresolved objective gap: {objective} vs {bound}.')
    return objective


class ScenarioRecourseOracle:
    r"""
    Exact evaluator of ``Q_s(x)``, the conditional recourse cost of ONE scenario at a given design.

    One ModelSub is built per scenario, with unit objective weight and the shared run-level big-M.
    Its objective is ``Q_s(x)`` in theta_s units, independent of decomposition group weights.

    Re-evaluating at a new x only rewrites the local-copy rows, so enumerating the design space costs
    one small MIP solve per (scenario, design) and no model rebuilds.
    """

    def __init__(self, raw_data, scenario_list, time_list, solver_config, initial_main_result):
        self.scenario_list = list(scenario_list)
        self.solver_config = replace(solver_config, mip_gap=0.0, time_limit=None)
        self._model_by_scenario = {}
        self._cache = {}

        for scenario in self.scenario_list:
            processor = DataProcessor(raw_data=raw_data)
            processor.reference_scenario_list = self.scenario_list
            scenario_data = processor.data_process(
                scenario_list_assigned=[scenario],
                time_list_assigned=time_list,
            )
            scenario_data[DataName.DICT_SC_PROB_UNCOND] = {scenario: 1.0}

            model = ModelSub(
                model_name=f'oracle_{scenario}',
                model_data=scenario_data,
                main_result=initial_main_result,
                sub_model_sce_list=[scenario],
                solver_config=self.solver_config,
            )
            model.build_sub_model_canonical()
            self._model_by_scenario[scenario] = model

    @staticmethod
    def _local_copy_constr_name(var_name, var_key):
        return f'{ConstrName.VAR_LOCAL_COPY}_{var_name}_{var_key}'

    @staticmethod
    def design_key(main_result):
        return tuple(
            (var_name, var_key, int(round(main_result[var_name][var_key])))
            for var_name in STATE_VAR_NAME_LIST
            for var_key in sorted(main_result[var_name].keys())
        )

    def evaluate(self, main_result):
        """Return ``{scenario: Q_s(x)}`` for one design, solved exactly."""
        cache_key = self.design_key(main_result)
        if cache_key in self._cache:
            return self._cache[cache_key]

        value_by_scenario = {}
        for scenario, model in self._model_by_scenario.items():
            for var_name in STATE_VAR_NAME_LIST:
                for var_key in sorted(main_result[var_name].keys()):
                    constr_name = self._local_copy_constr_name(var_name, var_key)
                    model.update_constr_by_name(
                        constr_name,
                        model.var[var_name][var_key] == main_result[var_name][var_key],
                    )
            model.solve()
            value_by_scenario[scenario] = require_exact_objective(model)

        self._cache[cache_key] = value_by_scenario
        return value_by_scenario


def enumerate_first_stage_designs(model_data, max_designs=None):
    r"""
    Enumerate the structural first-stage set

        ``sum_j x^G_j <= N^G``,  ``sum_typ x^GT_{j,typ} == x^G_j``,  ``x^L`` free binary.

    This is a SUPERSET of the master's feasible set (the collapsed SP block may exclude some of
    these designs), which is the conservative direction: a cut must not cut off any point of the
    true epigraph over this set either. Violations found only at designs the master excludes are
    reported as such by the caller.
    """
    node_list = list(model_data[DataName.LIST_NODE])
    line_list = list(model_data[DataName.LIST_LINE])
    dg_type_list = list(model_data[DataName.LIST_DG_TYPE])
    dg_ub = model_data[DataName.NUM_DG_UB]

    # per node: not installed, or installed with exactly one type
    node_options = [[None] + list(dg_type_list) for _ in node_list]

    designs = []
    for node_choice in itertools.product(*node_options):
        installed_count = sum(1 for choice in node_choice if choice is not None)
        if installed_count > dg_ub:
            continue

        xg = {
            node: (0 if node_choice[idx] is None else 1)
            for idx, node in enumerate(node_list)
        }
        xg_type = {
            (node, typ): (1 if node_choice[idx] == typ else 0)
            for idx, node in enumerate(node_list)
            for typ in dg_type_list
        }

        for harden_choice in itertools.product([0, 1], repeat=len(line_list)):
            xl = {line: harden_choice[idx] for idx, line in enumerate(line_list)}
            designs.append({
                VarName.DG_INSTALL: xg,
                VarName.DG_INSTALL_TYPE: xg_type,
                VarName.LINE_HARDEN: xl,
            })
            if max_designs is not None and len(designs) >= max_designs:
                return designs

    return designs


def evaluate_cut_rhs(cut_algebra, main_result):
    """Evaluate the affine right-hand side ``r(x)`` of one cut at a design."""
    value = float(cut_algebra['rhs_constant'])
    for (var_name, var_key), coef in cut_algebra['rhs_x_coef'].items():
        value += float(coef) * float(main_result[var_name][var_key])
    return value


def check_cut_validity(cut_records, designs, oracle):
    r"""
    Run the B3 instrument over every recorded cut.

    For each cut ``sum_s c_s theta_s >= r(x)`` and each enumerated design x, compare
    ``sum_s c_s Q_s(x)`` with ``r(x)``. The worst (most negative) slack over all designs is reported.
    """
    if not designs:
        raise ValueError('Cut validity requires at least one enumerated design.')
    # Q_s(x) is design-dependent only, so evaluate once per design and reuse across all cuts.
    recourse_by_design = []
    for design in designs:
        recourse_by_design.append((design, oracle.evaluate(design)))

    findings = []
    for record in cut_records:
        cut_algebra = record.get('cut_algebra')
        if cut_algebra is None:
            findings.append({
                'cut_name': record['cut_name'],
                'cut_type': record['cut_type'],
                'iteration': record['iteration'],
                'status': 'UNCHECKED',
                'reason': 'no cut algebra recorded (enable_cut_algebra_record was off)',
            })
            continue

        theta_coef = cut_algebra['theta_coef']
        negative_theta_coef = {
            str(scenario): coef for scenario, coef in theta_coef.items() if coef < -CUT_VALIDITY_ABS_TOL
        }
        if negative_theta_coef or cut_algebra.get('rhs_theta_coef_count'):
            findings.append({
                'cut_name': record['cut_name'],
                'cut_type': record['cut_type'],
                'iteration': record['iteration'],
                'status': 'UNCHECKED',
                'reason': 'offline replay requires nonnegative lhs theta weights and no rhs theta terms',
            })
            continue

        worst_slack, worst_design = None, None
        for design, recourse in recourse_by_design:
            lhs_min = sum(
                float(coef) * float(recourse[scenario])
                for scenario, coef in theta_coef.items()
            )
            slack = lhs_min - evaluate_cut_rhs(cut_algebra, design)
            if worst_slack is None or slack < worst_slack:
                worst_slack, worst_design = slack, design

        tolerance = CUT_VALIDITY_ABS_TOL + CUT_VALIDITY_REL_TOL * abs(cut_algebra['rhs_constant'])
        is_valid = worst_slack is None or worst_slack >= -tolerance

        finding = {
            'cut_name': record['cut_name'],
            'cut_type': record['cut_type'],
            'iteration': record['iteration'],
            'scenario_group': record['scenario_group'],
            'added': record['added'],
            'status': 'VALID' if is_valid else 'INVALID',
            'worst_slack': worst_slack,
            'tolerance': tolerance,
        }
        if cut_algebra.get('rhs_theta_coef_count'):
            finding['warning'] = (
                'the right-hand side references theta variables; the offline replay is incomplete'
            )
        if negative_theta_coef:
            finding['warning_negative_theta_coef'] = negative_theta_coef
        if not is_valid:
            finding['violating_design'] = {
                var_name: {str(k): v for k, v in worst_design[var_name].items()}
                for var_name in STATE_VAR_NAME_LIST
            }
        findings.append(finding)

    return findings


# =========================================================================================
# The decomposition driver
# =========================================================================================

def parse_scenario_groups(scenario_groups_arg, scenario_list):
    """'s_1+s_2,s_3' -> [['s_1', 's_2'], ['s_3']]; None -> singleton groups."""
    if scenario_groups_arg is None:
        return [[s] for s in scenario_list]

    groups = [
        [part.strip() for part in chunk.split('+') if part.strip()]
        for chunk in scenario_groups_arg.split(',')
        if chunk.strip()
    ]
    flat = [s for group in groups for s in group]
    if sorted(flat) != sorted(scenario_list):
        raise ValueError(
            f'Scenario groups {groups} are not a partition of the scenario list {scenario_list}.'
        )
    return groups


def _iter_range_for(family, families, max_iterations):
    return (0, max_iterations - 1) if family in families else None


def run_decomposition(
    raw_data,
    scenario_list,
    time_list,
    sce_group_list,
    solver_config_set,
    max_iterations,
    families,
    added_obj_term_weight,
    bound_mode,
    main_problem_model_type,
    solution_generator=None,
):
    """Run the decomposition and return the module plus the recorded bound trajectory."""
    module = TwoStageDecompRedo(
        raw_data=raw_data,
        time_list=time_list,
        scenario_list=scenario_list,
        solver_config_set=solver_config_set,
    )
    module.data_processor_module.main_problem_model_type = main_problem_model_type
    module.enable_cut_algebra_record = True

    module.build_main_stage_model()
    module.model_main.set_added_obj_term_weight(added_obj_term_weight)

    runner = DecompositionRunner(
        two_stage_decomp_module=module,
        scenario_groups=sce_group_list,
        max_iterations=max_iterations,
        cut_schedule=CutSchedule(
            ranges={
                cut_type: _iter_range_for(cut_type, families, max_iterations)
                for cut_type in CutTypeName.ALL_TYPES
            }
        ),
        aggregation_policy=AggregationPolicy.PER_GROUP,
        added_obj_term_weight=added_obj_term_weight,
        bound_mode=bound_mode,
        convergence_rel_gap=-1.0,
        solution_generator=solution_generator,
    )
    run_result = runner.run()
    bound_records = [
        {
            **record,
            'iteration': int(record['iteration']),
            'incumbent_upper_bound': module.ite_obj_value_dict[record['iteration']][
                'sub_obj(best_incumbent)'
            ]['sub_p_total'],
            'main_stage_obj': module.model_main.evaluate_design_objective_for_solution(
                run_result['current_solution_history'][int(record['iteration'])]
            ),
        }
        for record in run_result['iteration_bound_records']
    ]
    return module, bound_records


# =========================================================================================
# B2 / B4 assertions over the recorded trajectory
# =========================================================================================

def check_bound_trajectory(bound_records, reference_objective, require_convergence=False):
    failures = []
    valid_bounds = [
        (record['iteration'], record['valid_lower_bound'])
        for record in bound_records
        if record['is_bound_valid'] and record['valid_lower_bound'] is not None
    ]

    # B2: no lower bound may exceed the true optimum.
    tolerance = BOUND_ABS_TOL + BOUND_REL_TOL * abs(reference_objective)
    for iteration, bound in valid_bounds:
        if bound > reference_objective + tolerance:
            failures.append(
                f'B2: LB at iteration {iteration} is {bound}, which exceeds z* = {reference_objective} '
                f'by more than {tolerance}. The master has been cut off from the optimum.'
            )

    # B2: the final upper bound must not fall below the true optimum either.
    upper_bounds = [
        record['incumbent_upper_bound'] for record in bound_records
        if record['incumbent_upper_bound'] is not None
    ]
    if upper_bounds:
        best_upper_bound = min(upper_bounds)
        if best_upper_bound < reference_objective - tolerance:
            failures.append(
                f'B2: best incumbent UB {best_upper_bound} is BELOW z* = {reference_objective}; '
                f'the incumbent accounting is under-counting the recourse cost.'
            )

    if require_convergence:
        final_bound = bound_records[-1].get('valid_lower_bound') if bound_records else None
        if (
            not upper_bounds
            or min(upper_bounds) > reference_objective + tolerance
            or final_bound is None
            or final_bound < reference_objective - tolerance
        ):
            failures.append('B2: convergence to the reference optimum was requested but not reached.')

    # B4: the valid lower bounds never decrease while cuts are only added.
    for (prev_iteration, prev_bound), (iteration, bound) in zip(valid_bounds, valid_bounds[1:]):
        if bound < prev_bound - tolerance:
            failures.append(
                f'B4: LB decreased from {prev_bound} (iteration {prev_iteration}) to {bound} '
                f'(iteration {iteration}) although no cut was removed.'
            )

    return failures, valid_bounds


# =========================================================================================
# Entry point
# =========================================================================================

def run_validity_harness(
    instance_name=DEFAULT_INSTANCE,
    local_file_path=None,
    solver_name='gurobi_direct',
    fallback_solver_name=None,
    families=None,
    max_iterations=4,
    scenario_groups=None,
    added_obj_term_weight=0.0,
    bound_mode=BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
    main_problem_model_type=MainProblemModelTypeName.COLLAPSED_NO_TIME,
    inner_min_time_limit=None,
    inner_min_mip_gap=0.0,
    subproblem_exact_mip_gap=0.0,
    max_designs=None,
    tee=False,
    report_path=None,
    golden_path=None,
    write_golden=False,
    require_convergence=False,
):
    if local_file_path is None:
        local_file_path = default_local_file_path()
    families = list(families) if families else [CutTypeName.BENDERS]

    unknown_families = [family for family in families if family not in CutTypeName.ALL_TYPES]
    if unknown_families:
        raise ValueError(f'Unknown cut families: {unknown_families}. Known: {list(CutTypeName.ALL_TYPES)}')

    raw_data = load_raw_data(local_file_path, instance_name)
    reference = load_instance_reference(local_file_path, instance_name)

    if reference and 'scenario_list' in reference:
        scenario_list = list(reference['scenario_list'])
    else:
        scenario_list = sorted(
            raw_data[InputDataName.SCENARIO_PROB_DF][ScenarioProbHeader.SCENARIO_ID].unique().tolist()
        )
    time_list = list(reference['time_list']) if reference and 'time_list' in reference else None
    sce_group_list = parse_scenario_groups(scenario_groups, scenario_list)

    solver_config_set = SolverConfigSet.build(
        solver_name=solver_name,
        fallback_solver_name=fallback_solver_name,
        tee=tee,
        master_mip_gap=0.0,
        master_time_limit=None,
        subproblem_exact_mip_gap=subproblem_exact_mip_gap,
        inner_min_mip_gap=inner_min_mip_gap,
        inner_min_time_limit=inner_min_time_limit,
    )

    print('=' * 92)
    print('decomposition validity harness (docs/development_record.md, P0)')
    print(f'instance         : {instance_name}')
    print(f'scenarios        : {scenario_list}   groups: {sce_group_list}')
    print(f'cut families     : {families}')
    print(f'solver           : {solver_name}  (fallback: {fallback_solver_name})')
    print(f'iterations       : {max_iterations}')
    print(f'w (penalty)      : {added_obj_term_weight}   bound mode: {bound_mode}')
    print(f'inner_min        : mip_gap={inner_min_mip_gap}, time_limit={inner_min_time_limit}')
    print('=' * 92)

    # ---- B2 ground truth -----------------------------------------------------------------
    ground_truth = solve_extensive_form(
        raw_data=raw_data,
        scenario_list=scenario_list,
        time_list=time_list,
        solver_config=solver_config_set.get(SolverRoleName.MASTER),
    )
    reference_objective = ground_truth['objective']
    print(f'\n[B2] extensive-form optimum z* = {reference_objective!r} (status {ground_truth["solve_status"]})')
    reference_failures = []
    if reference is not None and 'objective' in reference:
        stored = reference['objective']
        matches = abs(stored - reference_objective) <= BOUND_ABS_TOL + BOUND_REL_TOL * abs(stored)
        print(f'      reference.json objective = {stored}  -> {"MATCH" if matches else "MISMATCH"}')
        if not matches:
            reference_failures.append('B2: extensive-form optimum differs from reference.json.')

    # ---- run the decomposition ------------------------------------------------------------
    module, bound_records = run_decomposition(
        raw_data=raw_data,
        scenario_list=scenario_list,
        time_list=time_list,
        sce_group_list=sce_group_list,
        solver_config_set=solver_config_set,
        max_iterations=max_iterations,
        families=families,
        added_obj_term_weight=added_obj_term_weight,
        bound_mode=bound_mode,
        main_problem_model_type=main_problem_model_type,
    )

    print('\n[bounds] iteration / valid LB / incumbent UB')
    for record in bound_records:
        print(
            f'   {record["iteration"]:>3}  '
            f'LB={record["valid_lower_bound"]!r:>22}  '
            f'UB={record["incumbent_upper_bound"]!r:>22}  '
            f'valid={record["is_bound_valid"]}'
        )

    bound_failures, valid_bounds = check_bound_trajectory(
        bound_records, reference_objective, require_convergence=require_convergence
    )

    # ---- B3 cut validity ------------------------------------------------------------------
    designs = enumerate_first_stage_designs(module.main_model_data, max_designs=max_designs)
    print(f'\n[B3] enumerated {len(designs)} first-stage designs; '
          f'solving {len(designs) * len(scenario_list)} exact recourse problems')

    oracle = ScenarioRecourseOracle(
        raw_data=raw_data,
        scenario_list=scenario_list,
        time_list=time_list,
        solver_config=solver_config_set.get(SolverRoleName.SUBPROBLEM_EXACT),
        initial_main_result=designs[0],
    )
    cut_findings = check_cut_validity(module.cut_metadata_records, designs, oracle)

    invalid_cuts = [finding for finding in cut_findings if finding['status'] == 'INVALID']
    unchecked_cuts = [finding for finding in cut_findings if finding['status'] == 'UNCHECKED']

    print(f'      cuts checked: {len(cut_findings)}   invalid: {len(invalid_cuts)}   '
          f'unchecked: {len(unchecked_cuts)}')
    for finding in invalid_cuts:
        print(f'      * INVALID {finding["cut_type"]:<20} {finding["cut_name"]}  '
              f'worst slack {finding["worst_slack"]:.6g} (tol {finding["tolerance"]:.3g})')

    by_family = {}
    for finding in cut_findings:
        entry = by_family.setdefault(finding['cut_type'], {'total': 0, 'invalid': 0, 'worst_slack': None})
        entry['total'] += 1
        if finding['status'] == 'INVALID':
            entry['invalid'] += 1
        slack = finding.get('worst_slack')
        if slack is not None and (entry['worst_slack'] is None or slack < entry['worst_slack']):
            entry['worst_slack'] = slack

    print('\n[B3] per family')
    for family in sorted(by_family):
        entry = by_family[family]
        worst = entry['worst_slack']
        print(f'      {family:<24} cuts={entry["total"]:<4} invalid={entry["invalid"]:<4} '
              f'worst slack={worst if worst is None else round(worst, 9)}')

    # ---- summary ---------------------------------------------------------------------------
    report = {
        'instance': instance_name,
        'scenario_list': scenario_list,
        'scenario_groups': sce_group_list,
        'cut_families': families,
        'solver_name': solver_name,
        'solver_config_set': solver_config_set.as_dict(),
        'added_obj_term_weight': added_obj_term_weight,
        'bound_mode': bound_mode,
        'require_convergence': require_convergence,
        'ground_truth': ground_truth,
        'bound_records': bound_records,
        'design_count': len(designs),
        'cut_findings': cut_findings,
        'bound_failures': bound_failures,
        'cut_algebra': [
            {
                'cut_name': record['cut_name'],
                'cut_type': record['cut_type'],
                'iteration': record['iteration'],
                'added': record['added'],
                'algebra': record.get('cut_algebra'),
            }
            for record in module.cut_metadata_records
        ],
    }

    golden_failures = _handle_golden_file(report, golden_path, write_golden)

    all_failures = reference_failures + bound_failures + [
        f'B3: cut {finding["cut_name"]} ({finding["cut_type"]}) is INVALID; '
        f'worst slack {finding["worst_slack"]}'
        for finding in invalid_cuts
    ] + [
        f'B3: cut {finding["cut_name"]} was not checked: {finding["reason"]}'
        for finding in unchecked_cuts
    ] + golden_failures
    report['failures'] = all_failures

    print('\n' + '=' * 92)
    if all_failures:
        print(f'RESULT: FAIL ({len(all_failures)} finding(s))')
        for message in all_failures:
            print(f'   * {message}')
    else:
        print('RESULT: PASS')
    print('=' * 92)

    if report_path is None:
        # One file per configuration: a pytest session runs this harness ten times, and a single
        # shared path would leave only the last run's evidence on disk.
        report_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'output',
            f'decomp_validity_{instance_name}__{_run_slug(families, sce_group_list, added_obj_term_weight)}.json'))

    # Writing the report must never change the verdict: the checks above have already run, so a
    # reporting failure is reported and swallowed rather than allowed to mask a PASS or a FAIL.
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe(report), f, indent=2)
        print(f'full report written to: {report_path}')
    except Exception as exc:  # noqa: BLE001 - reporting must not shadow the result
        print(f'WARNING: could not write the report to {report_path}: {type(exc).__name__}: {exc}')
        print('         The verdict above still stands; only the JSON file is missing.')

    return report


def _run_slug(families, sce_group_list, added_obj_term_weight):
    """Short filesystem-safe tag identifying one harness configuration."""
    family_tag = '-'.join(sorted(families)) or 'nocuts'
    group_tag = 'singleton' if all(len(group) == 1 for group in sce_group_list) else (
        'g' + '-'.join('+'.join(str(s) for s in group) for group in sce_group_list)
    )
    weight_tag = f'w{added_obj_term_weight:g}'
    slug = f'{family_tag}__{group_tag}__{weight_tag}'
    return ''.join(ch if (ch.isalnum() or ch in '-_+.') else '_' for ch in slug)


def _golden_payload(report):
    """The subset of a report that the B6 golden file pins down."""
    return {
        'bound_records': [
            {
                'iteration': record['iteration'],
                'valid_lower_bound': record['valid_lower_bound'],
                'incumbent_upper_bound': record['incumbent_upper_bound'],
            }
            for record in report['bound_records']
        ],
        'cut_algebra': [
            {
                'cut_name': item['cut_name'],
                'cut_type': item['cut_type'],
                'algebra': item['algebra'],
            }
            for item in report['cut_algebra']
        ],
    }


def _handle_golden_file(report, golden_path, write_golden):
    """B6: compare against, or refresh, the stored golden file."""
    if golden_path is None:
        return []

    payload = json.loads(json.dumps(json_safe(_golden_payload(report))))

    if write_golden:
        os.makedirs(os.path.dirname(golden_path) or '.', exist_ok=True)
        with open(golden_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.write('\n')
        print(f'\n[B6] golden file refreshed: {golden_path}')
        return []

    if not os.path.exists(golden_path):
        return [f'B6: requested golden file does not exist: {golden_path}']

    with open(golden_path, 'r', encoding='utf-8') as f:
        stored = json.load(f)

    if stored == payload:
        print(f'\n[B6] golden file matches: {golden_path}')
        return []

    return [
        f'B6: output differs from the golden file {golden_path}. '
        f'If the change is intended, refresh it with --write-golden.'
    ]


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--instance', default=DEFAULT_INSTANCE, help='micro_bench instance folder name')
    parser.add_argument('--local-file-path', default=None, help='root of unit_test/test_local_csv_file')
    parser.add_argument('--solver', default='gurobi_direct',
                        help='Pyomo solver name, e.g. gurobi_direct, appsi_highs, cbc, glpk')
    parser.add_argument('--fallback-solver', default=None, help='solver used when the primary is unavailable')
    parser.add_argument('--families', default=CutTypeName.BENDERS,
                        help=f'comma-separated cut families out of {",".join(CutTypeName.ALL_TYPES)}')
    parser.add_argument('--max-iterations', type=int, default=4)
    parser.add_argument('--require-convergence', action='store_true',
                        help='also require final bounds to reach the extensive-form optimum')
    parser.add_argument('--scenario-groups', default=None,
                        help="scenario partition, e.g. 's_1+s_2' or 's_1,s_2'; default is singletons")
    parser.add_argument('--added-obj-term-weight', type=float, default=0.0)
    parser.add_argument('--bound-mode', default=BoundModeName.RESOLVE_AT_ZERO_WEIGHT,
                        choices=[BoundModeName.RESOLVE_AT_ZERO_WEIGHT, BoundModeName.HEURISTIC])
    parser.add_argument('--main-problem-model-type', default=MainProblemModelTypeName.COLLAPSED_NO_TIME)
    parser.add_argument('--inner-min-time-limit', type=float, default=None,
                        help='deliberately starve the Lagrangian inner MILP; validity must survive it')
    parser.add_argument('--inner-min-mip-gap', type=float, default=0.0)
    parser.add_argument('--subproblem-exact-mip-gap', type=float, default=0.0)
    parser.add_argument('--max-designs', type=int, default=None,
                        help='cap the enumerated first-stage designs (default: full enumeration)')
    parser.add_argument('--tee', action='store_true', help='show solver output')
    parser.add_argument('--report-path', default=None)
    parser.add_argument('--golden-path', default=None, help='B6 golden file to compare against')
    parser.add_argument('--write-golden', action='store_true', help='refresh the golden file instead of comparing')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    init_logger(enable_file_output=False)
    logging.getLogger().setLevel(logging.WARNING)

    try:
        report = run_validity_harness(
            instance_name=args.instance,
            local_file_path=args.local_file_path,
            solver_name=args.solver,
            fallback_solver_name=args.fallback_solver,
            families=[part.strip() for part in args.families.split(',') if part.strip()],
            max_iterations=args.max_iterations,
            scenario_groups=args.scenario_groups,
            added_obj_term_weight=args.added_obj_term_weight,
            bound_mode=args.bound_mode,
            main_problem_model_type=args.main_problem_model_type,
            inner_min_time_limit=args.inner_min_time_limit,
            inner_min_mip_gap=args.inner_min_mip_gap,
            subproblem_exact_mip_gap=args.subproblem_exact_mip_gap,
            max_designs=args.max_designs,
            tee=args.tee,
            report_path=args.report_path,
            golden_path=args.golden_path,
            write_golden=args.write_golden,
            require_convergence=args.require_convergence,
        )
    except Exception:
        traceback.print_exc()
        return 2

    return 1 if report['failures'] else 0


if __name__ == "__main__":
    sys.exit(main())
