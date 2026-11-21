# -*- coding: utf-8 -*-
"""S2 新增测试：pipeline 作为"无状态编排函数"的边界与契约。

这些测试关心的是：
    · 同一个 run() 在显式 Settings 下行为可预测（交付数量、策略、路径）；
    · trace_id 在正常/异常/嵌套下都能恢复；
    · pipeline 不偷偷读环境变量、不 import config；
    · main.ask 的 patch 仍然替换 pipeline 真正调用的函数（不是假通过）。

不使用真实数据、不联网：检索器、LLM、rerank 都由桩替代。
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import logging_setup
import main
from qa import pipeline
from settings import Settings
from tests import make_parents, make_units
from tests.test_pipeline import StubLLM, StubRetriever, hits_for

PIPELINE_SRC = Path(pipeline.__file__).read_text(encoding="utf-8")
CONFIG_SRC = (Path(main.__file__).parent / "config.py").read_text(encoding="utf-8")


def _imported_roots(src: str) -> set[str]:
    """解析源码里的 import，返回顶层模块名集合（注释/文档字符串不算）。"""
    import ast
    tree = ast.parse(src)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _calls_mkdir(src: str) -> bool:
    """源码里是否真的调用了 xxx.mkdir(...)（文档字符串里提到不算）。"""
    import ast
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "mkdir":
            return True
    return False


def record_rerank(question, candidates, top_k, **kwargs):
    """记录被传进来的 top_k 与关键字参数，返回原顺序。"""
    record_rerank.calls.append((top_k, kwargs))
    return candidates[:top_k], {"degraded": False, "elapsed": 0.0}


record_rerank.calls = []


def trace_recording_llm(seen: list):
    def _llm(question, context, sources, **kwargs):
        seen.append(logging_setup.current_trace_id())
        return "见 [Source 1]。", {}, {}
    return _llm


class TestPipelineContract(unittest.TestCase):
    def setUp(self):
        record_rerank.calls.clear()
        logging_setup.set_trace_id("-")

    def _run(self, settings=None, **kwargs):
        s = settings or Settings(data_dir=Path("."))
        kwargs.setdefault("retriever", StubRetriever(hits_for("pa03", 7)))
        kwargs.setdefault("rerank_fn", record_rerank)
        kwargs.setdefault("llm_fn", StubLLM())
        return pipeline.run(
            "报警怎么复位？", settings=s,
            parents=make_parents(), units=make_units(), **kwargs,
        )

    def test_result_has_same_14_keys_as_cli(self):
        with mock.patch.object(main, "rerank_candidates", record_rerank), \
                mock.patch.object(main, "llm_answer", StubLLM()):
            expected = main.ask("报警怎么复位？", mode="bm25",
                                retriever=StubRetriever(hits_for("pa03", 7)),
                                parents=make_parents(), units=make_units())
        got = self._run()
        self.assertEqual(set(got), set(expected))
        self.assertEqual(len(got), 14)
        self.assertEqual(got["mode"], "stub")

    def test_deliver_k_comes_from_settings(self):
        s = Settings(data_dir=Path("."), deliver_k=3)
        self._run(settings=s)
        top_k, kwargs = record_rerank.calls[-1]
        self.assertEqual(top_k, 3)
        self.assertIn("model", kwargs)
        self.assertEqual(kwargs["model"], s.rerank_model)

    def test_mode_comes_from_settings_when_not_given(self):
        s = Settings(data_dir=Path("."), retrieval_mode="hybrid")
        with mock.patch.object(pipeline, "build_retriever",
                               return_value=StubRetriever(hits_for("pa03", 7))) as builder, \
                mock.patch.object(pipeline, "load_parents", return_value={}), \
                mock.patch.object(pipeline, "load_units", return_value=[]):
            pipeline.run("q", settings=s, rerank_fn=record_rerank, llm_fn=StubLLM())
        builder.assert_called_once()
        self.assertEqual(builder.call_args.args[0], "hybrid")

    def test_paths_come_from_settings(self):
        s = Settings(data_dir=Path("/tmp/rag_data"))
        with mock.patch.object(pipeline, "load_parents", return_value={}) as lp, \
                mock.patch.object(pipeline, "load_units", return_value=[]) as lu:
            pipeline.run("q", settings=s, retriever=StubRetriever([]),
                         rerank_fn=record_rerank, llm_fn=StubLLM())
        self.assertEqual(lp.call_args.kwargs["parents_path"], s.parents_path)
        self.assertEqual(lp.call_args.kwargs["docstore_path"], s.docstore_path)
        self.assertEqual(lu.call_args.kwargs["units_path"], s.units_path)

    def test_context_caps_come_from_settings(self):
        s = Settings(data_dir=Path("."), context_parent_max_chars=12)
        captured = {}

        def llm(question, context, sources, **kwargs):
            captured["context"] = context
            return "ok", {}, {}

        pipeline.run("q", settings=s, retriever=StubRetriever(hits_for("pa03", 7)),
                     parents=make_parents(), units=make_units(),
                     rerank_fn=record_rerank, llm_fn=llm)
        self.assertIn("本块过长，已截断", captured["context"])
        self.assertLessEqual(len(captured["context"]), 200)

    def test_trace_set_during_run_and_restored_after(self):
        seen = []
        self._run(llm_fn=trace_recording_llm(seen))
        self.assertEqual(len(seen), 1)
        self.assertNotEqual(seen[0], "-")
        self.assertEqual(logging_setup.current_trace_id(), "-")

    def test_trace_restored_even_on_config_error(self):
        def boom(question, context, sources, **kwargs):
            raise main.ConfigError("no key")

        with self.assertRaises(main.ConfigError):
            self._run(llm_fn=boom)
        self.assertEqual(logging_setup.current_trace_id(), "-")

    def test_nested_run_restores_outer_trace(self):
        logging_setup.set_trace_id("outer")
        self._run()
        self.assertEqual(logging_setup.current_trace_id(), "outer")

    def test_two_runs_get_different_trace_ids(self):
        seen = []
        self._run(llm_fn=trace_recording_llm(seen))
        self._run(llm_fn=trace_recording_llm(seen))
        self.assertNotEqual(seen[0], seen[1])

    def test_pipeline_does_not_import_config(self):
        mods = _imported_roots(PIPELINE_SRC)
        self.assertNotIn("config", mods)        # 不 import config
        self.assertNotIn("os", mods)            # 也就没法偷偷读环境变量
        self.assertNotIn("dotenv", mods)

    def test_config_has_no_mkdir_side_effect(self):
        self.assertFalse(_calls_mkdir(CONFIG_SRC))


class TestMainStillPatchable(unittest.TestCase):
    """验收第 3 条：main 上的 patch 必须替换 pipeline 真正调用的函数。"""

    def test_patched_rerank_and_llm_are_actually_used(self):
        seen = {}

        def fake_rerank(question, candidates, top_k, **kwargs):
            seen["rerank"] = top_k
            return candidates[:top_k], {"degraded": False, "elapsed": 0.0}

        llm = StubLLM("见 [Source 1]。")
        with mock.patch.object(main, "rerank_candidates", fake_rerank), \
                mock.patch.object(main, "llm_answer", llm):
            result = main.ask("报警怎么复位？", mode="bm25",
                              retriever=StubRetriever(hits_for("pa03", 7)),
                              parents=make_parents(), units=make_units())

        self.assertEqual(seen["rerank"], main.settings().deliver_k)   # 真的被调用
        self.assertEqual(len(llm.calls), 1)                            # 真的替换了 LLM
        self.assertEqual(result["citations"][0]["page"], 7)


if __name__ == "__main__":
    unittest.main()
