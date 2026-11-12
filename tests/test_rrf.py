# -*- coding: utf-8 -*-
"""RRF 融合的离线测试（retrieval/hybrid.py 的 HybridRetriever）。

用两个假检索器代替 Dense 与 BM25：这个测试只验证融合公式与排序，
不验证召回质量（召回质量需要真实向量库，属于集成测试）。
"""
from __future__ import annotations

import unittest

from retrieval.backfill import Hit
from retrieval.hybrid import HybridRetriever


class FakeRetriever:
    """按预设顺序返回命中，并记录被请求的候选数。"""

    def __init__(self, name: str, hits: list[Hit]):
        self.name = name
        self._hits = hits
        self.requested: list[int] = []

    def retrieve(self, qtext: str, n: int) -> list[Hit]:
        self.requested.append(n)
        return self._hits[:n]


def h(unit_id: str, parent_id: str = "p", page: int = 1) -> Hit:
    return Hit(unit_id=unit_id, parent_id=parent_id, page=page, text=f"t-{unit_id}", score=0.0)


class TestRRF(unittest.TestCase):
    def test_score_is_sum_of_reciprocal_ranks(self):
        dense = FakeRetriever("dense", [h("uA"), h("uB")])
        bm25 = FakeRetriever("bm25", [h("uB"), h("uC")])
        fusion = HybridRetriever(dense, bm25, k=60, side=10)

        out = fusion.retrieve("q", n=10)

        by_id = {x.unit_id: x.score for x in out}
        self.assertAlmostEqual(by_id["uA"], 1 / 61)
        self.assertAlmostEqual(by_id["uB"], 1 / 62 + 1 / 61)   # 两路都命中，分数相加
        self.assertAlmostEqual(by_id["uC"], 1 / 62)

    def test_overlapping_unit_ranks_first(self):
        dense = FakeRetriever("dense", [h("uA"), h("uB")])
        bm25 = FakeRetriever("bm25", [h("uB"), h("uC")])
        out = HybridRetriever(dense, bm25, k=60, side=10).retrieve("q", n=10)
        self.assertEqual([x.unit_id for x in out], ["uB", "uA", "uC"])

    def test_truncates_to_requested_n(self):
        dense = FakeRetriever("dense", [h("u1"), h("u2"), h("u3")])
        bm25 = FakeRetriever("bm25", [])
        out = HybridRetriever(dense, bm25, k=60, side=10).retrieve("q", n=2)
        self.assertEqual(len(out), 2)

    def test_side_expands_with_n(self):
        dense = FakeRetriever("dense", [h("u1")])
        bm25 = FakeRetriever("bm25", [h("u2")])
        fusion = HybridRetriever(dense, bm25, k=60, side=10)
        fusion.retrieve("q", n=3)      # n <= side，用 side
        fusion.retrieve("q", n=25)     # n > side，用 n
        self.assertEqual(dense.requested, [10, 25])

    def test_empty_both_sides(self):
        out = HybridRetriever(FakeRetriever("dense", []), FakeRetriever("bm25", []),
                              k=60, side=10).retrieve("q", n=10)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
