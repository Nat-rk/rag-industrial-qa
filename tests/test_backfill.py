# -*- coding: utf-8 -*-
"""父块回填的离线测试（retrieval/backfill.py）。

这两个函数是纯函数，不需要网络也不需要向量库，最适合拿来验证
「去重按名次」「自适应扩展在够数或候选耗尽时停下」这两条行为。
"""
from __future__ import annotations

import unittest

from retrieval.backfill import Hit, adaptive_backfill, dedupe_by_parent
from tests import make_parents

PARENTS = None


def setUpModule():
    global PARENTS
    PARENTS = make_parents()


def hit(unit_id: str, parent_id: str, page: int, score: float) -> Hit:
    return Hit(unit_id=unit_id, parent_id=parent_id, page=page, text=f"text of {unit_id}", score=score)


class TestDedupeByParent(unittest.TestCase):
    def test_keeps_first_hit_and_order(self):
        hits = [
            hit("u1", "pa01", 3, 0.90),
            hit("u2", "pa01", 3, 0.80),   # 同一父块，应被去掉
            hit("u3", "pa02", 4, 0.70),
        ]
        out = dedupe_by_parent(hits, PARENTS)
        self.assertEqual([d["parent"].id for d in out], ["pa01", "pa02"])
        self.assertEqual(out[0]["unit_id"], "u1")
        self.assertAlmostEqual(out[0]["score"], 0.90)

    def test_skips_unknown_parent(self):
        hits = [hit("u1", "no-such-parent", 3, 0.9), hit("u2", "pa03", 7, 0.8)]
        out = dedupe_by_parent(hits, PARENTS)
        self.assertEqual([d["parent"].id for d in out], ["pa03"])

    def test_skips_empty_parent_id(self):
        hits = [hit("u1", "", 3, 0.9)]
        self.assertEqual(dedupe_by_parent(hits, PARENTS), [])


class TestAdaptiveBackfill(unittest.TestCase):
    def test_stops_when_target_reached(self):
        calls = []

        def fetch(n: int):
            calls.append(n)
            return [hit("u1", "pa01", 3, 0.9), hit("u2", "pa02", 4, 0.8), hit("u3", "pa03", 7, 0.7)]

        delivered, trace = adaptive_backfill(fetch, PARENTS, target=2, steps=(3, 6, 12))
        self.assertEqual(len(delivered), 2)
        self.assertEqual(calls, [3])                 # 第一次就够数，不该继续扩
        self.assertEqual(trace[0]["candidates"], 3)
        self.assertEqual(trace[0]["parents"], 2)

    def test_expands_when_not_enough_parents(self):
        def fetch(n: int):
            # 无论 n 多大，都只返回同一个父块：父块数永远不够
            return [hit(f"u{i}", "pa01", 3, 0.9) for i in range(min(n, 10))]

        delivered, trace = adaptive_backfill(fetch, PARENTS, target=3, steps=(2, 4, 8))
        self.assertEqual(len(delivered), 1)
        self.assertEqual(len(trace), 3)
        self.assertEqual([t["candidates"] for t in trace], [2, 4, 8])

    def test_stops_when_candidates_exhausted(self):
        def fetch(n: int):
            # 候选只有 2 条，扩大 n 也拿不到更多：应在第二次发现"没有增长"后退出
            return [hit("u1", "pa01", 3, 0.9), hit("u2", "pa01", 3, 0.8)]

        delivered, trace = adaptive_backfill(fetch, PARENTS, target=10, steps=(2, 4, 8))
        self.assertEqual(len(delivered), 1)
        self.assertEqual([t["candidates"] for t in trace], [2, 4])   # 8 那一轮不该发生

    def test_trace_reports_returned_and_parents(self):
        def fetch(n: int):
            return [hit("u1", "pa01", 3, 0.9), hit("u2", "pa02", 4, 0.8)]

        _, trace = adaptive_backfill(fetch, PARENTS, target=5, steps=(2,))
        self.assertEqual(trace, [{"candidates": 2, "returned": 2, "parents": 2}])


if __name__ == "__main__":
    unittest.main()
