# -*- coding: utf-8 -*-
"""数据装载的错误处理（retrieval/store.py）。

S2 收尾新增：数据文件缺失 / 权限不足 / JSON 截断，原本会冒出裸的
`FileNotFoundError`、`PermissionError`、`JSONDecodeError`，命令行上是一屏 traceback。
现在统一包成 `DataError`，消息里带**具体路径与操作**，并用 `from e` 保留原始异常。

这个测试要守住四件事：
    1. 缺文件 → DataError（不是 FileNotFoundError），消息里有路径；
    2. 原始异常还在（`__cause__`），不然排查时看不到 errno；
    3. 正常文件照常读出，file_name 从 docstore 补；
    4. 字段缺失这类"结构错误"**不被吞成 DataError**——那是数据内容问题，
       应该照常抛 KeyError/TypeError，免得把编程错误伪装成配置错误。
"""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from errors import DataError, RagError
from retrieval import store

TMP = Path(__file__).resolve().parent / "_tmp" / "store"


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestStoreDataErrors(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True, exist_ok=True)
        self.parents = write_json(TMP / "parents.json", [
            {"id": "p1", "page": 8, "kind": "section",
             "section": "2 Introduction to IMC > 2.1 An Overview of IMC",
             "text": "body", "chars": 4},
        ])
        self.docstore = write_json(TMP / "docstore.json", {
            "docstore/data": {"p1": {"__data__": {"metadata": {"file_name": "apply.pdf"}}}},
        })
        self.units = write_json(TMP / "units.json", [
            {"id": "u1", "parent_id": "p1", "page": 8, "junk": False,
             "chars": 4, "text": "2.1 > body"},
        ])

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    # ---- 正常路径 ----------------------------------------------------------
    def test_load_parents_ok_and_file_name_from_docstore(self):
        parents = store.load_parents(parents_path=self.parents, docstore_path=self.docstore)
        self.assertEqual(list(parents), ["p1"])
        p = parents["p1"]
        self.assertEqual((p.page, p.kind, p.chars), (8, "section", 4))
        self.assertEqual(p.file_name, "apply.pdf")     # 只有 docstore 里有这个字段
        self.assertEqual(p.short_id, "p1")

    def test_load_units_ok(self):
        units = store.load_units(units_path=self.units)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["parent_id"], "p1")

    # ---- 缺文件 ------------------------------------------------------------
    def test_missing_parents_raises_data_error_with_path(self):
        missing = TMP / "nope.json"
        with self.assertRaises(DataError) as ctx:
            store.load_parents(parents_path=missing, docstore_path=self.docstore)
        msg = str(ctx.exception)
        self.assertIn(str(missing), msg)
        self.assertIn("读取数据文件失败", msg)
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)

    def test_missing_docstore_raises_data_error_with_path(self):
        missing = TMP / "no_docstore.json"
        with self.assertRaises(DataError) as ctx:
            store.load_parents(parents_path=self.parents, docstore_path=missing)
        self.assertIn(str(missing), str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, FileNotFoundError)

    def test_missing_units_raises_data_error(self):
        with self.assertRaises(DataError) as ctx:
            store.load_units(units_path=TMP / "no_units.json")
        self.assertIn("no_units.json", str(ctx.exception))

    # ---- 坏内容 ------------------------------------------------------------
    def test_truncated_json_raises_data_error_with_position(self):
        bad = TMP / "broken.json"
        bad.write_text('{"a": 1,', encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            store.load_units(units_path=bad)
        msg = str(ctx.exception)
        self.assertIn(str(bad), msg)
        self.assertIn("解析数据文件失败", msg)
        self.assertIsInstance(ctx.exception.__cause__, json.JSONDecodeError)

    def test_structure_error_is_not_swallowed(self):
        # 能读、能解析，但缺 id 字段 —— 这是数据内容问题，不该被包装成 DataError
        weird = write_json(TMP / "no_id.json", [{"page": 1}])
        with self.assertRaises(KeyError):
            store.load_parents(parents_path=weird, docstore_path=self.docstore)

    # ---- 异常层级 -----------------------------------------------------------
    def test_data_error_is_rag_and_runtime_error(self):
        # 旧代码里的 `except RuntimeError` 仍能接住，方便迁移
        self.assertTrue(issubclass(DataError, RagError))
        self.assertTrue(issubclass(DataError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
