# -*- coding: utf-8 -*-
"""Settings 的离线测试：环境变量解析、校验、不可变性、历史默认值。

这里验证的是「配置层本身」：
  · 默认值必须和 S1 之前的 config.py 完全一致（否则 95 项测试与历史评测都会漂移）；
  · 非法输入必须在构造时就抛 ConfigError，而不是等到请求时才炸；
  · Settings 不可变，换模式/换目录要返回新对象。
S3 起 config.py 已删除，默认值只写在 settings.py 一处，所以这里直接断言数字。
"""
from __future__ import annotations

import unittest
from pathlib import Path

from errors import ConfigError
from settings import Settings


class TestSettingsDefaults(unittest.TestCase):
    def test_defaults_match_legacy_config(self):
        """这些值就是历史评测跑出来的那套配置，改任何一个都必须重跑基线。"""
        s = Settings.from_env({})
        self.assertEqual(s.collection, "dense_sentences_v1")
        self.assertEqual(s.retrieval_mode, "bm25")
        self.assertEqual(s.steps, (30, 60, 120, 240))
        self.assertEqual(s.backfill_target, 30)
        self.assertEqual(s.deliver_k, 10)
        self.assertEqual(s.rrf_k, 60)
        self.assertEqual(s.rrf_side, 50)
        self.assertEqual(s.llm_model, "glm-4.5-flash")
        self.assertEqual(s.llm_timeout, 120)
        self.assertEqual(len(s.extra_stopwords), 12)
        self.assertIn("what", s.extra_stopwords)
        self.assertEqual(s.context_parent_max_chars, 2600)
        self.assertEqual(s.context_total_max_chars, 24000)

    def test_default_numbers(self):
        s = Settings.from_env({})
        self.assertEqual(s.retrieval_mode, "bm25")
        self.assertEqual(s.deliver_k, 10)
        self.assertEqual(s.backfill_target, 30)
        self.assertEqual(s.steps, (30, 60, 120, 240))

    def test_paths_derived_from_data_dir(self):
        s = Settings(data_dir=Path("/tmp/rag_data"))
        self.assertEqual(s.parents_path, Path("/tmp/rag_data/out/parents.json"))
        self.assertEqual(s.units_path, Path("/tmp/rag_data/out/units.json"))
        self.assertEqual(s.docstore_path, Path("/tmp/rag_data/storage/docstore.json"))
        self.assertEqual(s.chroma_dir, Path("/tmp/rag_data/chroma_db"))


class TestSettingsFromEnv(unittest.TestCase):
    def test_env_overrides(self):
        s = Settings.from_env({
            "RAG_DATA_DIR": "/tmp/somewhere",
            "RAG_RETRIEVAL_MODE": "hybrid",
            "RAG_DELIVER_K": "5",
            "RAG_BACKFILL_TARGET": "20",
            "RAG_LLM_MODEL": "glm-4.6",
            "RAG_LOG_LEVEL": "debug",
        })
        self.assertEqual(s.data_dir, Path("/tmp/somewhere").resolve())
        self.assertEqual(s.retrieval_mode, "hybrid")
        self.assertEqual(s.deliver_k, 5)
        self.assertEqual(s.backfill_target, 20)
        self.assertEqual(s.llm_model, "glm-4.6")
        self.assertEqual(s.log_level, "DEBUG")

    def test_blank_env_falls_back_to_default(self):
        s = Settings.from_env({"RAG_RETRIEVAL_MODE": "", "RAG_DELIVER_K": "  "})
        self.assertEqual(s.retrieval_mode, "bm25")
        self.assertEqual(s.deliver_k, 10)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env({"RAG_RETRIEVAL_MODE": "dense"})
        self.assertIn("RAG_RETRIEVAL_MODE", str(ctx.exception))

    def test_non_integer_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env({"RAG_DELIVER_K": "ten"})
        self.assertIn("RAG_DELIVER_K", str(ctx.exception))

    def test_deliver_k_cannot_exceed_backfill_target(self):
        # 交付数大于回填数会被静默截断，必须在启动时就失败
        with self.assertRaises(ConfigError):
            Settings.from_env({"RAG_DELIVER_K": "40", "RAG_BACKFILL_TARGET": "30"})
        with self.assertRaises(ConfigError):
            Settings(deliver_k=31, backfill_target=30)

    def test_deliver_k_must_be_positive(self):
        with self.assertRaises(ConfigError):
            Settings.from_env({"RAG_DELIVER_K": "0"})


class TestSettingsValidation(unittest.TestCase):
    def test_steps_must_ascend(self):
        with self.assertRaises(ConfigError):
            Settings(steps=(30, 20, 10))

    def test_steps_must_be_positive(self):
        with self.assertRaises(ConfigError):
            Settings(steps=(0, 30))

    def test_context_caps(self):
        with self.assertRaises(ConfigError):
            Settings(context_parent_max_chars=30000, context_total_max_chars=24000)

    def test_rrf_params_positive(self):
        with self.assertRaises(ConfigError):
            Settings(rrf_k=0)


class TestSettingsImmutable(unittest.TestCase):
    def test_frozen(self):
        s = Settings()
        with self.assertRaises(Exception):
            s.deliver_k = 5          # frozen dataclass 应拒绝赋值

    def test_with_mode_returns_new_object(self):
        s = Settings(retrieval_mode="bm25")
        h = s.with_mode("hybrid")
        self.assertEqual(h.retrieval_mode, "hybrid")
        self.assertEqual(s.retrieval_mode, "bm25")   # 原对象不变
        self.assertIsNot(s, h)

    def test_with_mode_validates(self):
        with self.assertRaises(ConfigError):
            Settings().with_mode("dense")


if __name__ == "__main__":
    unittest.main()
