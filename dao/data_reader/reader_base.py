# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from __future__ import annotations
from typing import Any, Optional

class DataReaderBase(object):

    local_file_path: Optional[str]
    data_set_name: Optional[str]
    raw_data: Any

    """
    The base for the module for reading data for the model.
    self.raw_data is the data being ready for model processing.
    Methods may vary, but the form should follow the given format, being the same as the local_csv method.
    """
    def __init__(self, read_method, local_file_path, data_set_name):
        self.read_method = read_method
        self.local_file_path = local_file_path
        self.data_set_name = data_set_name

        self.raw_data = {}


    def read(self):
        """
        Unified dispatcher: calls the corresponding method named `read_<method>` based on `read_method`.
        All concrete implementations must be defined with the name pattern `read_<method>`.
        """
        method_name = f"read_{self.read_method}"
        if not hasattr(self, method_name):
            raise ValueError(f"Unsupported read method: {self.read_method}")
        return getattr(self, method_name)()