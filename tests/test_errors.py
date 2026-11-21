# -*- coding: utf-8 -*-
"""异常层次与 api_key 的离线测试。

重点验证两件行为变化：
  1. 缺 key 不再抛 `SystemExit`（它继承 BaseException，`except Exception` 接不住），
     而是抛 `ConfigError`——CLI 可以打日志后退出 2，将来 FastAPI 可以映射 503。
  2. `RerankError` / `LLMError` 都挂在 `RagError` 下，且**仍然是 RuntimeError 子类**，
     这样旧代码里 `except RuntimeError` 的行为不变（S1 不改变已有语义）。
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from errors import ConfigError, DataError, LLMError, RagError, RerankError
from retrieval import store
from retrieval.reranker import RerankError as RerankerRerankError


class TestExceptionTree(unittest.TestCase):
    def test_all_inherit_rag_error(self):
        for cls in (ConfigError, DataError, LLMError, RerankError):
            self.assertTrue(issubclass(cls, RagError))
            self.assertTrue(issubclass(cls, RuntimeError))   # 兼容旧的 except RuntimeError

    def test_rerank_error_is_shared(self):
        # reranker 里的名字必须是 errors 里的同一个类，不能再有第二份定义
        self.assertIs(RerankerRerankError, RerankError)
        self.assertIsInstance(RerankError("x"), RagError)


class TestApiKey(unittest.TestCase):
    def test_missing_key_raises_config_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError) as ctx:
                store.api_key()
        msg = str(ctx.exception)
        self.assertIn("ZHIPU_API_KEY", msg)
        self.assertNotIn("sk-", msg)          # 报错里不能出现疑似 key 的片段

    def test_missing_key_is_not_system_exit(self):
        """关键回归：以前抛 SystemExit，try/except Exception 根本接不住。"""
        with mock.patch.dict(os.environ, {}, clear=True):
            try:
                store.api_key()
            except SystemExit:                # pragma: no cover - 出现即失败
                self.fail("api_key() 不应再抛 SystemExit")
            except Exception as e:
                self.assertIsInstance(e, ConfigError)

    def test_returns_key(self):
        with mock.patch.dict(os.environ, {"ZHIPU_API_KEY": "test-key-123"}, clear=True):
            self.assertEqual(store.api_key(), "test-key-123")

    def test_strips_whitespace(self):
        with mock.patch.dict(os.environ, {"ZHIPU_API_KEY": "  k  "}, clear=True):
            self.assertEqual(store.api_key(), "k")


if __name__ == "__main__":
    unittest.main()
