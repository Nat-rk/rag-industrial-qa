# -*- coding: utf-8 -*-
"""日志层的离线测试：幂等、trace_id 注入、级别、非法级别兜底。

这些断言看着琐碎，但对应两个真实坑：
  · `setup_logging()` 在测试/多次启动里会被调用多次，重复加 handler 会让每行日志打印两遍；
  · trace_id 必须能自动出现在格式里，否则排查时还得手工把它传进每个函数。
"""
from __future__ import annotations

import io
import logging
import unittest

import logging_setup


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        logging_setup.reset_logging()
        self.buf = io.StringIO()
        self.addCleanup(logging_setup.reset_logging)

    def test_idempotent_handlers(self):
        logging_setup.setup_logging("INFO", stream=self.buf)
        logging_setup.setup_logging("INFO", stream=self.buf)
        root = logging.getLogger()
        ours = [h for h in root.handlers if getattr(h, "name", "") == "rag-console"]
        self.assertEqual(len(ours), 1)          # 只应有一个，不然日志会重复

    def test_trace_id_appears_in_output(self):
        logging_setup.setup_logging("INFO", stream=self.buf)
        tid = logging_setup.new_trace_id()
        logging.getLogger("tests.demo").info("hello")
        out = self.buf.getvalue()
        self.assertIn(f"trace={tid}", out)
        self.assertIn("hello", out)

    def test_default_trace_is_dash(self):
        logging_setup.setup_logging("INFO", stream=self.buf)
        logging_setup.set_trace_id("-")
        logging.getLogger("tests.demo").info("no trace")
        self.assertIn("trace=-", self.buf.getvalue())

    def test_level_filters_debug(self):
        logging_setup.setup_logging("INFO", stream=self.buf)
        logging.getLogger("tests.demo").debug("should not appear")
        self.assertNotIn("should not appear", self.buf.getvalue())

    def test_invalid_level_falls_back_to_info(self):
        # 日志配置写错不该让服务起不来
        logging_setup.setup_logging("NOT-A-LEVEL", stream=self.buf)
        logging.getLogger("tests.demo").info("still works")
        self.assertIn("still works", self.buf.getvalue())

    def test_new_trace_id_is_short_hex(self):
        tid = logging_setup.new_trace_id()
        self.assertEqual(len(tid), 8)
        int(tid, 16)                            # 不是合法十六进制就会抛 ValueError


if __name__ == "__main__":
    unittest.main()
