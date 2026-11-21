# -*- coding: utf-8 -*-
"""数据装载：把磁盘上的父块、句子、docstore 读成 Python 对象。

为什么单独一个文件：
    * parents.json 只有 id/page/kind/section/text/chars，没有文件名；
    * docstore.json 里才有 file_name，所以引用信息要从这里补；
    * 上层（检索、上下文、引用）都只跟 Parent / Unit 这两个数据类打交道。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import config
from errors import ConfigError, DataError


@dataclass
class Parent:
    """一个父块（交付给 LLM 的单位）。"""
    id: str
    page: int
    kind: str          # section / table / toc
    section: str       # 章节路径，如 "2 Introduction to IMC > 2.1 An Overview of IMC"
    text: str          # 正文（表格块是整张表）
    chars: int
    file_name: str     # 来自 docstore，引用要用

    @property
    def short_id(self) -> str:
        return self.id[:8]


def _load_json(path: Path):
    """读一个 JSON 文件，把底层读取/解析错误统一成 DataError。

    为什么要包：
        * 数据资产不在仓库里（要单独下载、或设 RAG_DATA_DIR），最常见的错误就是
          路径不对 / 文件没放 → 原来是裸 `FileNotFoundError`，命令行上是一屏 traceback；
        * 权限不足是 `PermissionError`，同属 OSError；
        * 文件被截断是 `json.JSONDecodeError`。
    只捕获这三类具体异常，不写 `except Exception`——编程错误要原样冒出来。
    `from e` 保留原始异常，调试时 `e.__cause__` 还能看到真正的 errno。
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise DataError(f"读取数据文件失败：{path}（操作：打开并读取文本）") from e
    except json.JSONDecodeError as e:
        raise DataError(
            f"解析数据文件失败：{path}（第 {e.lineno} 行第 {e.colno} 列：{e.msg}）"
        ) from e


def _docstore_metadata(path: Path | None = None) -> dict[str, dict]:
    """docstore.json 的结构是 {"docstore/data": {node_id: {"__data__": {...}}}}。"""
    src = path or config.DOCSTORE
    raw = _load_json(src)
    data = raw.get("docstore/data") or raw.get("docstore", {}).get("data") or {}
    out = {}
    for node_id, entry in data.items():
        body = entry.get("__data__", entry)
        if isinstance(body, str):          # 有些版本会把它存成 JSON 字符串
            try:
                body = json.loads(body)
            except json.JSONDecodeError as e:
                raise DataError(
                    f"解析 docstore 内嵌节点失败：{src}（节点 {node_id}，{e.msg}）"
                ) from e
        out[node_id] = body.get("metadata", {}) or {}
    return out


def load_parents(*, parents_path: Path | None = None,
                 docstore_path: Path | None = None) -> dict[str, Parent]:
    """读 278 个父块；file_name 从 docstore 补。

    参数都是可选的：不传就用 config 里的旧默认值（迁移期兼容），
    pipeline 则显式传 Settings 里的路径。
    """
    meta = _docstore_metadata(docstore_path)
    parents: dict[str, Parent] = {}
    for item in _load_json(parents_path or config.PARENTS):
        m = meta.get(item["id"], {})
        parents[item["id"]] = Parent(
            id=item["id"],
            page=int(item["page"]),
            kind=item["kind"],
            section=item["section"],
            text=item["text"],
            chars=int(item["chars"]),
            file_name=m.get("file_name", "apply.pdf"),
        )
    return parents


def load_units(*, units_path: Path | None = None) -> list[dict]:
    """句子级检索单元：id / parent_id / page / junk / chars / text。"""
    return _load_json(units_path or config.UNITS)


def api_key() -> str:
    """智谱 key：只从环境变量读。

    以前这里抛 `SystemExit`——它继承自 `BaseException`，调用方写
    `except Exception` 根本接不住（`eval/run_cases.py` 就这样漏过一次）。
    改成 `ConfigError`：库层只报"配置不对"，退出码由入口决定。
    """
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not key:
        raise ConfigError(
            "缺少 ZHIPU_API_KEY。\n"
            "  · PyCharm：Run → Edit Configurations → Environment variables 里加一行\n"
            "  · 命令行：$env:ZHIPU_API_KEY = \"你的key\"\n"
            "  · 或参考 README §5 / .env.example"
        )
    return key
