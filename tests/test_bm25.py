# -*- coding: utf-8 -*-
"""BM25 检索器的离线测试（retrieval/bm25.py）。

bm25s + llama-index 是纯本地依赖，不需要网络，所以这个测试可以在
「没数据、没 key」的环境里跑——只要装了 requirements.txt。
"""
from __future__ import annotations

import unittest

try:
    from retrieval.bm25 import BM25UnitRetriever

    # 只 import 到 BM25UnitRetriever 不够：它把 bm25s / llama-index 的导入放在
    # __init__ 里（延迟导入）。干净环境里模块能导入、构造时才炸，导致
    # skipUnless 判断失效、变成 setUpClass ERROR。所以这里把真正用到的两个
    # 三方包也探一遍，缺任何一个就整体 skip。
    import bm25s  # noqa: F401
    from llama_index.core.schema import TextNode  # noqa: F401

    HAVE_DEPS = True
except Exception as exc:          # 缺依赖时跳过，而不是让整个测试套件红掉
    HAVE_DEPS = False
    SKIP_REASON = f"缺少 bm25s / llama-index-retrievers-bm25：{exc}"

from tests import make_units


@unittest.skipUnless(HAVE_DEPS, "缺少 BM25 依赖（见 SKIP_REASON）")
class TestBM25UnitRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = make_units()
        cls.retriever = BM25UnitRetriever(cls.units)

    def test_unique_term_hits_expected_unit(self):
        hits = self.retriever.retrieve("overcurrent alarm A13", 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].parent_id, "pa03")

    def test_parameter_number_is_retrievable(self):
        hits = self.retriever.retrieve("parameter 1-00 positioning synchronization", 5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].parent_id, "pa01")

    def test_hit_fields_are_consistent(self):
        hits = self.retriever.retrieve("control word bit", 3)
        self.assertTrue(hits)
        for hit in hits:
            self.assertTrue(hit.unit_id)
            self.assertIn(hit.unit_id, {u["id"] for u in self.units})
            self.assertIsInstance(hit.page, int)

    def test_requesting_more_than_corpus_does_not_crash(self):
        hits = self.retriever.retrieve("drive", 100)
        self.assertLessEqual(len(hits), len(self.units))

    def test_nonsense_query_returns_list(self):
        self.assertIsInstance(self.retriever.retrieve("zzz qqq", 3), list)


if __name__ == "__main__":
    unittest.main()
