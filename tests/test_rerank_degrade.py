# -*- coding: utf-8 -*-
"""rerank 降级路径的离线测试（retrieval/reranker.py）。

**这里验证的是「降级逻辑正确」**：
接口失败时按原顺序交付、不中断问答、错误信息如实写进 rerank 元信息。
它**不验证**降级后答案内容是否正确——那需要真实数据与人工判断。
"""
from __future__ import annotations

import unittest
from unittest import mock

import retrieval.reranker as reranker
from retrieval.reranker import RerankError, rerank_candidates
from tests import make_candidates


class TestRerankDegrade(unittest.TestCase):
    def test_error_falls_back_to_original_order(self):
        candidates = make_candidates()          # 4 个父块，顺序 pa01..pa04

        def boom(*args, **kwargs):
            raise RerankError("重试 3 次仍失败：Timeout")

        with mock.patch.object(reranker, "rerank_documents", boom):
            picked, info = rerank_candidates("q", candidates, top_k=3)

        self.assertEqual([c["parent"].id for c in picked],
                         [c["parent"].id for c in candidates[:3]])   # 原顺序前 3 个
        self.assertTrue(info["degraded"])
        self.assertIn("RerankError", info["error"])

    def test_any_exception_is_caught(self):
        # 不只 RerankError：接口返回怪格式时抛 ValueError 也不能让问答中断
        with mock.patch.object(reranker, "rerank_documents", side_effect=ValueError("bad json")):
            picked, info = rerank_candidates("q", make_candidates(), top_k=2)
        self.assertTrue(info["degraded"])
        self.assertEqual(len(picked), 2)

    def test_normal_path_reorders_and_records_score(self):
        candidates = make_candidates()

        def fake(query, documents, top_n=None):
            return [2, 0, 1], [0.9, 0.5, 0.1], {"total_tokens": 10}, 0.12

        with mock.patch.object(reranker, "rerank_documents", fake):
            picked, info = rerank_candidates("q", candidates, top_k=2)

        self.assertFalse(info["degraded"])
        self.assertEqual([c["parent"].id for c in picked],
                         [candidates[2]["parent"].id, candidates[0]["parent"].id])
        self.assertAlmostEqual(picked[0]["rerank_score"], 0.9)

    def test_empty_candidates(self):
        picked, info = rerank_candidates("q", [], top_k=10)
        self.assertEqual(picked, [])
        self.assertFalse(info["degraded"])
        self.assertEqual(info["error"], "无候选")


if __name__ == "__main__":
    unittest.main()
