# -*- coding: utf-8 -*-
# @Time     : 2025/10/04
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
Main entry point for running tests.
Run this from the project root: python main.py <test_name> [options]

For decomposition_algo test, additional options are available:
  --enable-log-output          Enable log file output (default: False)
  --disable-result-output      Disable result output (default: True)
  --scenario-list SCENARIOS    Comma-separated list of scenarios (default: s_1,s_2,s_3)
  --time-list TIMES            Comma-separated list of time periods (default: 1,2,3,4,5,6,7,8,9,10,11)
  --test-file-path PATH        Path to test data files
  --data-set-name NAME         Name of the data set (default: 'function_test_fixed_rated_p')
  --max-iterations N            Maximum number of iterations (default: 4)
"""

import argparse
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from unit_test.test_scripts.combined_formulation_test import run_combined_formulation_test
from unit_test.test_scripts.dynamic_decomposition_test import run_decomp_module_test
from unit_test.test_scripts.micro_bench_test import run_micro_bench_test
from model.solver_config import SolverConfig, SolverConfigSet
from util.names import RunParameterName


# Registry of available tests
TEST_REGISTRY = {
    'combined_formulation': run_combined_formulation_test,
    # 'decomposition_algo' now points at the live runner (TwoStageDecompRedo). The legacy
    # algo/TwoStageDecomposition.py path was removed; see docs/development_record.md (P4).
    'decomposition_algo': run_decomp_module_test,
    'micro_bench': run_micro_bench_test,
    # Add more tests here as they become available
}


def parse_list_arg(value, item_type=int):
    """Parse a comma-separated string into a list, converting items to the specified type."""
    if value is None:
        return None
    items = [item.strip() for item in value.split(',') if item.strip()]
    # Use the type as a converter function
    try:
        return [item_type(item) for item in items]
    except (ValueError, TypeError):
        # If conversion fails, return as strings
        return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for the SDinPS project')
    parser.add_argument('test_name', help='Name of the test to run')
    parser.add_argument('--solver', default='gurobi_direct', help='Pyomo solver name')
    parser.add_argument('--enable-pretrained-cut', action='store_true',
                        help='Restrict decomposition to a predicted region; disables global certification')
    
    # Arguments specific to decomposition_algo test
    parser.add_argument('--enable-log-output', action='store_true',
                       help='Enable log file output (for decomposition_algo)')
    parser.add_argument('--disable-result-output', action='store_true',
                       help='Disable result output (for decomposition_algo)')
    parser.add_argument('--scenario-list', type=str,
                       help='Comma-separated list of scenarios, e.g., s_1,s_2,s_3 (for decomposition_algo)')
    parser.add_argument('--time-list', type=str,
                       help='Comma-separated list of time periods, e.g., 1,2,3,4,5 (for decomposition_algo)')
    parser.add_argument('--test-file-path', type=str,
                       help='Path to test data files (for decomposition_algo)')
    parser.add_argument('--data-set-name', type=str,
                       help='Name of the data set (for decomposition_algo)')
    parser.add_argument('--max-iterations', type=int,
                       help='Maximum number of iterations (for decomposition_algo, default: 4)')
    
    args = parser.parse_args()
    
    if args.test_name not in TEST_REGISTRY:
        parser.error(f"Unknown test name: {args.test_name}. Available tests: {', '.join(TEST_REGISTRY.keys())}")
    
    # Call the test function
    if args.test_name == 'decomposition_algo':
        # Build kwargs for decomposition_algo test
        kwargs = {
            'solver_config_set': SolverConfigSet.build(solver_name=args.solver),
            RunParameterName.ENABLE_PRETRAINED_CUT: args.enable_pretrained_cut,
        }
        
        if args.enable_log_output:
            kwargs['enable_log_output'] = True
        
        if args.disable_result_output:
            kwargs['enable_result_output'] = False
        
        if args.scenario_list:
            kwargs['scenario_list'] = parse_list_arg(args.scenario_list, item_type=str)
        
        if args.time_list:
            kwargs['time_list'] = parse_list_arg(args.time_list, item_type=int)
        
        if args.test_file_path:
            kwargs['test_file_path'] = args.test_file_path
        
        if args.data_set_name:
            kwargs['data_set_name'] = args.data_set_name
        
        if args.max_iterations is not None:
            kwargs['max_iterations'] = args.max_iterations
        
        run_decomp_module_test(**kwargs)
    elif args.test_name == 'micro_bench':
        run_micro_bench_test(solver_name=args.solver)
    else:
        run_combined_formulation_test(
            solver_config=SolverConfig(solver_name=args.solver, mip_gap=0.01, time_limit=600),
        )

