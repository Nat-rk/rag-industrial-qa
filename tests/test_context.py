# -*- coding: utf-8 -*-
"""上下文构建的离线测试（qa/context_builder.py）。

重点验证三件事：编号与来源一一对应、单块超长会截断、总量到顶会丢后面的块。
"""
from __future__ import annotations

import unittest
from unittest import mock

import config
from qa.context_builder import build_context
from tests import make_candidates, make_parents


class TestBuildContext(unittest.TestCase):
    def test_source_numbering_matches_blocks(self):
        candidates = make_candidates()          # fixture 里 4 个父块
        context, sources = build_context(candidates)

        self.assertEqual(len(sources), len(candidates))
        for i, s in enumerate(sources, start=1):
            self.assertEqual(s["n"], i)
            self.assertIn(f"[Source {i}]", context)

    def test_block_carries_file_page_and_id(self):
        context, sources = build_context(make_candidates())
        self.assertIn("File: sample_drive_manual.pdf", context)
        self.assertIn("Page: 3", context)
        self.assertIn("Parent ID: pa01", context)
        self.assertIn("Section: 1 Setup > 1.2 Selecting the Control Mode", context)

    def test_long_parent_is_truncated(self):
        parents = make_parents()
        long_parent = parents["pa01"]
        long_parent.text = "长" * (config.CONTEXT_PARENT_MAX_CHARS + 200)

        with mock.patch.object(config, "CONTEXT_PARENT_MAX_CHARS", 200):
            context, sources = build_context([{"parent": long_parent, "score": 1.0}])

        self.assertIn("（本块过长，已截断）", context)
        # sources 记录的是截断后（含省略标记）的长度，只要求它远小于原文
        self.assertLess(sources[0]["chars"], 260)
        self.assertLess(sources[0]["chars"], len(long_parent.text))

    def test_total_budget_drops_later_blocks(self):
        candidates = make_candidates()
        # 上限设得放不下第二块：只应该留下第一个块
        with mock.patch.object(config, "CONTEXT_TOTAL_MAX_CHARS", 600):
            context, sources = build_context(candidates)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["n"], 1)
        self.assertIn("[Source 1]", context)
        self.assertNotIn("[Source 2]", context)

    def test_empty_candidates(self):
        context, sources = build_context([])
        self.assertEqual(sources, [])
        self.assertIn("没有检索到任何相关文档片段", context)


if __name__ == "__main__":
    unittest.main()
