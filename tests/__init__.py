# -*- coding: utf-8 -*-
"""离线测试包的共用工具。

这里只做两件事：把仓库根目录放进 sys.path，以及把 fixtures 里的虚构数据
转成生产代码认识的对象（retrieval.store.Parent）。

测试本身不重新实现任何业务逻辑——全部直接调用 retrieval/ 与 qa/ 里的真实函数，
只把「外部依赖」（网络、向量库、数据目录）替换成 fixture 或 Mock。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_FILE = FIXTURES / "sample_device_docs.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_FILE.read_text(encoding="utf-8"))


def make_parents() -> dict:
    """把 fixture 的 parents 变成 {id: Parent}。"""
    from retrieval.store import Parent

    out = {}
    for p in load_fixture()["parents"]:
        out[p["id"]] = Parent(
            id=p["id"], page=p["page"], kind=p["kind"], section=p["section"],
            text=p["text"], chars=p.get("chars", len(p["text"])),
            file_name=load_fixture()["file_name"],
        )
    return out


def make_units() -> list:
    """fixture 的 units 本身就是生产代码使用的那种 dict。"""
    return load_fixture()["units"]


def make_candidates(pages: tuple[int, ...] | None = None) -> list:
    """按页码挑几个父块，包成 build_context / rerank_candidates 需要的形式。"""
    parents = make_parents()
    chosen = list(parents.values())
    if pages is not None:
        chosen = [p for p in chosen if p.page in pages]
    return [{"parent": p, "score": 1.0 / (i + 1), "unit_id": f"u{i}"}
            for i, p in enumerate(chosen)]
