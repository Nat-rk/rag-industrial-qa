# -*- coding: utf-8 -*-
"""端到端链路的离线测试（main.ask，用假检索器 + Mock 掉 LLM 与 rerank）。

这条测试把「检索 → 回填 → rerank → 上下文 → 生成 → 引用」串起来跑一遍，
验证的是**编排与引用映射**，不是答案质量。
"""
from __future__ import annotations

import unittest
from unittest import mock

import main
from retrieval.backfill import Hit
from tests import make_parents, make_units


class StubRetriever:
    """按固定顺序返回命中，不做真实检索。"""
    name = "stub"

    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, qtext: str, n: int):
        return self._hits[:n]


class StubLLM:
    """记录被调用时收到的上下文，并返回固定答案。"""

    def __init__(self, answer: str = "见 [Source 1]。", error: Exception | None = None):
        self.answer_text = answer
        self.error = error
        self.calls: list[tuple[str, list]] = []

    def __call__(self, question, context, sources):
        self.calls.append((context, sources))
        if self.error:
            raise self.error
        return self.answer_text, {"total_tokens": 1}, {}


def hits_for(parent_id: str, page: int, n: int = 3) -> list[Hit]:
    return [Hit(unit_id=f"u{i}", parent_id=parent_id, page=page,
                text=f"句子 {i}", score=1.0 / (i + 1)) for i in range(n)]


def identity_rerank(question, candidates, top_k):
    return candidates[:top_k], {"degraded": False, "elapsed": 0.0}


class TestAskPipeline(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch.object(main, "rerank_candidates", identity_rerank)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_citations_point_to_delivered_parent(self):
        llm = StubLLM("见 [Source 1]。")
        with mock.patch.object(main, "llm_answer", llm):
            result = main.ask("报警怎么复位？", mode="bm25",
                              retriever=StubRetriever(hits_for("pa03", 7)),
                              parents=make_parents(), units=make_units())

        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["page"], 7)
        self.assertEqual(result["citations"][0]["parent_id"], "pa03")
        self.assertEqual(result["invalid_citations"], [])
        self.assertEqual(len(result["sources"]), 1)

    def test_context_sent_to_llm_has_source_headers(self):
        llm = StubLLM()
        with mock.patch.object(main, "llm_answer", llm):
            main.ask("报警怎么复位？", mode="bm25",
                     retriever=StubRetriever(hits_for("pa03", 7)),
                     parents=make_parents(), units=make_units())

        context, sources = llm.calls[0]
        self.assertIn("[Source 1]", context)
        self.assertIn("File: sample_drive_manual.pdf", context)
        self.assertEqual(sources[0]["parent_id"], "pa03")

    def test_llm_failure_is_reported_not_raised(self):
        llm = StubLLM(error=RuntimeError("network down"))
        with mock.patch.object(main, "llm_answer", llm):
            result = main.ask("报警怎么复位？", mode="bm25",
                              retriever=StubRetriever(hits_for("pa03", 7)),
                              parents=make_parents(), units=make_units())

        self.assertTrue(result["answer"].startswith("（生成失败"))
        self.assertIn("network down", result["answer"])
        self.assertEqual(result["citations"], [])        # 失败答案里没有引用
        self.assertTrue(result["sources"])               # 但检索结果仍然交回

    def test_out_of_range_citation_is_flagged(self):
        llm = StubLLM("见 [Source 9]。")
        with mock.patch.object(main, "llm_answer", llm):
            result = main.ask("报警怎么复位？", mode="bm25",
                              retriever=StubRetriever(hits_for("pa03", 7)),
                              parents=make_parents(), units=make_units())

        self.assertEqual(result["invalid_citations"], [9])
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
