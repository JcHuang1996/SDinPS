# -*- coding: utf-8 -*-
# @Time     : 2025/09/05
# @Author   : J. Huang
# @Email    : jiachenghuang0601@gmail.com


from dao.data_reader.reader_base import DataReaderBase
from dao.data_reader.mixin_local_csv import LocalCSVMixin

import logging

from util.project_logger import init_logger
from util.names import InputMethodName


logger = logging.getLogger(__name__)


class DataReader(LocalCSVMixin, DataReaderBase):
    """
    Attach the separately developed read_xxx methods to a single class via multiple inheritance.
    To add new methods later, simply create a new mixin_xxx.py and include the new Mixin in the inheritance list.
    """
    pass


if __name__ == "__main__":
    init_logger()

    # Simple smoke test for this module
    print("Running DataReader smoke test...")

    test_read_method = InputMethodName.LOCAL_CSV
    test_file_path = '/Users/huangjiacheng/OR591/unit test/test_local_csv_file'
    data_set_name = 'function test'

    # Example: test CSV reading
    try:
        r = DataReader(
            read_method=test_read_method,
            local_file_path=test_file_path,
            data_set_name=data_set_name
        )

        r.read()
        print("CSV read success:")
    except Exception as e:
        print("CSV read failed:", e)
