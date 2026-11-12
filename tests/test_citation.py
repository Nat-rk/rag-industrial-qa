# -*- coding: utf-8 -*-
"""引用模块的离线测试（qa/citation.py，纯标准库，最容易被改坏）。"""
from __future__ import annotations

import unittest

from qa.citation import (build_citations, cited_numbers, format_citations,
                         invalid_numbers, strip_reference_section)

SOURCES = [
    {"n": 1, "file": "sample_drive_manual.pdf", "page": 3, "parent_id": "pa01",
     "short_id": "pa01shor", "section": "1 Setup > 1.2 Selecting the Control Mode", "chars": 100},
    {"n": 2, "file": "sample_drive_manual.pdf", "page": 7, "parent_id": "pa03",
     "short_id": "pa03shor", "section": "2 Diagnostics > 2.1 Alarm Handling", "chars": 80},
]


class TestCitedNumbers(unittest.TestCase):
    def test_dedup_and_keep_order(self):
        answer = "先看 [Source 2]，再看 [Source 1]，最后又回到 [Source 2]。"
        self.assertEqual(cited_numbers(answer), [2, 1])

    def test_accepts_other_spellings(self):
        self.assertEqual(cited_numbers("Source 1 与 source3 与 [source 1]"), [1, 3])

    def test_trailing_reference_list_is_ignored(self):
        answer = ("参数 1-00 的取值见 [Source 2]。\n\n"
                  "参考来源：\n- sample_drive_manual.pdf，第 3 页（Source 1）")
        self.assertEqual(cited_numbers(answer), [2])
        self.assertNotIn("参考来源", strip_reference_section(answer))

    def test_english_references_heading(self):
        answer = "Answer body [Source 1]\n\nReferences:\n[Source 2]"
        self.assertEqual(cited_numbers(answer), [1])

    def test_no_citation(self):
        self.assertEqual(cited_numbers("这是一段没有标注来源的回答。"), [])
        self.assertEqual(cited_numbers(""), [])


class TestInvalidNumbers(unittest.TestCase):
    def test_out_of_range_number_is_reported(self):
        self.assertEqual(invalid_numbers("见 [Source 9]。", SOURCES), [9])

    def test_all_numbers_valid(self):
        self.assertEqual(invalid_numbers("[Source 1] 和 [Source 2]", SOURCES), [])

    def test_out_of_range_inside_stripped_section_is_ignored(self):
        # 末尾清单里的编号不算正文引用，也不应报"越界"
        answer = "正文 [Source 1]\n\n参考来源：\n- [Source 9]"
        self.assertEqual(invalid_numbers(answer, SOURCES), [])


class TestBuildCitations(unittest.TestCase):
    def test_maps_source_to_page(self):
        used, marked = build_citations("见 [Source 2]。", SOURCES)
        self.assertTrue(marked)
        self.assertEqual(len(used), 1)
        self.assertEqual(used[0]["page"], 7)
        self.assertEqual(used[0]["parent_id"], "pa03")

    def test_no_marks_returns_empty(self):
        used, marked = build_citations("没有标注的回答", SOURCES)
        self.assertEqual(used, [])
        self.assertFalse(marked)

    def test_format_without_marks_does_not_claim_citation(self):
        text = format_citations([], SOURCES)
        self.assertIn("模型未在回答中标注", text)
        self.assertIn("可人工核对", text)
        self.assertNotIn("参考来源：\n-", text)

    def test_format_with_marks_lists_only_used(self):
        used, _ = build_citations("[Source 2]", SOURCES)
        text = format_citations(used, SOURCES)
        self.assertIn("第 7 页", text)
        self.assertNotIn("第 3 页", text)


if __name__ == "__main__":
    unittest.main()
