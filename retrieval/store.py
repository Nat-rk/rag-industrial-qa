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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _docstore_metadata() -> dict[str, dict]:
    """docstore.json 的结构是 {"docstore/data": {node_id: {"__data__": {...}}}}。"""
    raw = _load_json(config.DOCSTORE)
    data = raw.get("docstore/data") or raw.get("docstore", {}).get("data") or {}
    out = {}
    for node_id, entry in data.items():
        body = entry.get("__data__", entry)
        if isinstance(body, str):          # 有些版本会把它存成 JSON 字符串
            body = json.loads(body)
        out[node_id] = body.get("metadata", {}) or {}
    return out


def load_parents() -> dict[str, Parent]:
    meta = _docstore_metadata()
    parents: dict[str, Parent] = {}
    for item in _load_json(config.PARENTS):
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


def load_units() -> list[dict]:
    """句子级检索单元：id / parent_id / page / junk / chars / text。"""
    return _load_json(config.UNITS)


def api_key() -> str:
    """智谱 key：优先环境变量，其次 Windows 用户环境变量里读出来的那个。"""
    key = os.environ.get("ZHIPU_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "缺少 ZHIPU_API_KEY。\n"
            "  · PyCharm：Run → Edit Configurations → Environment variables 里加一行\n"
            "  · 命令行：$env:ZHIPU_API_KEY = \"你的key\""
        )
    return key
