# -*- coding: utf-8 -*-
"""评测脚本自身的打分逻辑（eval/run_cases.py）。

为什么值得测：公开报告里的「检索命中 / 引用命中」就是这几行算出来的。
如果它们被改坏，JSON 里的布尔值会悄悄变错，而指标看起来仍然"正常"。

这里 Mock 掉 ask()，让 run() 走真实的打分分支；输出文件写在 tests/_tmp/，
不污染仓库，也不依赖系统临时目录的权限。
"""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from eval import run_cases
from tests import make_parents

TMP = Path(__file__).resolve().parent / "_tmp"


def fake_ask(answer: str, cited_pages: list[int], picked_pages: list[int]):
    """构造一个假的 ask：picked/citations 的页码由参数指定。"""

    def _ask(question, mode="bm25", retriever=None, parents=None, units=None):
        picked = []
        for page in picked_pages:
            parent = next(p for p in make_parents().values() if p.page == page)
            picked.append({"parent": parent, "score": 1.0})
        return {
            "mode": mode,
            "answer": answer,
            "picked": picked,
            "citations": [{"page": p, "n": 1, "file": "f.pdf", "parent_id": "x",
                           "short_id": "x", "section": "s"} for p in cited_pages],
            "rerank": {"degraded": False},
            "trace": [{"candidates": 30, "returned": 30, "parents": len(picked)}],
            "timing": {"build": 0.0, "recall": 0.0, "rerank": 0.0, "llm": 0.5},
        }

    return _ask


class TestRunScoring(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(exist_ok=True)
        self.out = TMP / f"{self._testMethodName}.json"

    def tearDown(self):
        self.out.unlink(missing_ok=True)

    def run_cases_with(self, cases, answer, cited_pages, picked_pages):
        with mock.patch.object(run_cases, "load_parents", lambda: make_parents()), \
             mock.patch.object(run_cases, "load_units", lambda: []), \
             mock.patch.object(run_cases, "build_retriever", lambda m, u: object()), \
             mock.patch.object(run_cases, "ask", fake_ask(answer, cited_pages, picked_pages)):
            with redirect_stdout(io.StringIO()):        # run() 会打印逐题行，测试里静音
                rows = run_cases.run(cases, ["bm25"], out_path=self.out)
        saved = json.loads(self.out.read_text(encoding="utf-8"))
        return rows, saved

    def test_hit_and_citation_true(self):
        cases = [{"id": "T1", "type": "x", "q": "q", "pages": [7]}]
        rows, saved = self.run_cases_with(cases, "见 [Source 1]。", [7], [7])

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["evidence_ok"])
        self.assertTrue(rows[0]["citations_ok"])
        self.assertFalse(rows[0]["refused"])
        self.assertEqual(len(saved), 1)

    def test_missing_authority_page_is_a_miss(self):
        cases = [{"id": "T2", "type": "x", "q": "q", "pages": [7]}]
        rows, _ = self.run_cases_with(cases, "见 [Source 1]。", [3], [3])
        self.assertFalse(rows[0]["evidence_ok"])
        self.assertFalse(rows[0]["citations_ok"])

    def test_refusal_case_has_no_expected_pages(self):
        cases = [{"id": "N01", "type": "资料不足", "q": "q"}]
        rows, _ = self.run_cases_with(cases, "文档中没有相关信息，无法回答。", [], [3])
        self.assertIsNone(rows[0]["evidence_ok"])       # 没有期望页 → 不进命中率分母
        self.assertTrue(rows[0]["refused"])

    def test_citation_ok_is_intersection_not_equality(self):
        # 期望只有 p7，引用了 p7 和 p3：当前口径判 True（宽松），这是已知设计
        cases = [{"id": "T3", "type": "x", "q": "q", "pages": [7]}]
        rows, _ = self.run_cases_with(cases, "见 [Source 1] [Source 2]。", [7, 3], [7, 3])
        self.assertTrue(rows[0]["citations_ok"])
        self.assertEqual(rows[0]["cited_pages"], [3, 7])

    def test_exception_is_recorded_and_loop_continues(self):
        cases = [{"id": "T4", "type": "x", "q": "q", "pages": [7]},
                 {"id": "T5", "type": "x", "q": "q", "pages": [7]}]
        calls = {"n": 0}
        good = fake_ask("见 [Source 1]。", [7], [7])

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return good(*a, **k)

        with mock.patch.object(run_cases, "load_parents", lambda: make_parents()), \
             mock.patch.object(run_cases, "load_units", lambda: []), \
             mock.patch.object(run_cases, "build_retriever", lambda m, u: object()), \
             mock.patch.object(run_cases, "ask", flaky):
            with redirect_stdout(io.StringIO()):        # run() 会打印逐题行，测试里静音
                rows = run_cases.run(cases, ["bm25"], out_path=self.out)

        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["llm_failed"])
        self.assertIn("RuntimeError", rows[0]["error"])
        self.assertTrue(rows[1]["evidence_ok"])         # 第二题照常完成


if __name__ == "__main__":
    unittest.main()
