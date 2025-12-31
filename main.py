# -*- coding: utf-8 -*-
# @Time     : 2025/10/04
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

"""
Main entry point for running tests.
Run this from the project root: python main.py <test_name>
"""

import argparse

from unit_test.test_scripts.combined_formulation_test import run_combined_formulation_test


# Registry of available tests
TEST_REGISTRY = {
    'combined_formulation': run_combined_formulation_test,
    # Add more tests here as they become available
    # Example:
    # 'another_test': run_another_test,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for the SDinPS project')
    parser.add_argument('test_name', help='Name of the test to run')
    args = parser.parse_args()
    
    TEST_REGISTRY[args.test_name]()

