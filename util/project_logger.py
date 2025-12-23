# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com

import logging
import sys
import os
from datetime import datetime


def init_logger(log_dir='logs', task_name=None, enable_file_output: bool = True):
    """
    Initializes the logger with both console and optional file output.

    Parameters
    ----------
    log_dir : str
        Directory where log files will be stored (default 'logs').
    task_name : str or None
        Optional prefix for the log filename.
    enable_file_output : bool
        If False, no log directory or file will be created at all.
    """
    root_logger = logging.getLogger()

    # Always clear all existing handlers to ensure clean reconfiguration
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.setLevel(logging.INFO)

    # --- Console handler (always active) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # --- Optional file handler ---
    if enable_file_output:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.abspath(os.path.join(current_file_dir, os.pardir))
        log_dir = os.path.join(project_root_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_name = f"{task_name}_{timestamp}.log" if task_name else f"{timestamp}.log"
        log_path = os.path.join(log_dir, base_name)

        # delay=True → file is not opened until the first log record is emitted
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8', delay=True)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)