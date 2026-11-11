# -*- coding: utf-8 -*-
"""父块回填：句子级检索结果 → 去重后的父块列表。

为什么需要这一步：
    检索单位是句子（细，能精准命中一句话），但交给 LLM 时必须给整块
    （表格要整张、步骤要整段）。所以按名次扫句子，同一父块只交付一次。

自适应扩展：
    30 个句子里可能挤着好几句话都来自同一张大表，去重后不够 10 个父块。
    这时把候选句子扩到 60/120/240，直到够数或候选耗尽。
    只扩大候选范围，不改变原有检索名次。
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from retrieval.store import Parent


@dataclass
class Hit:
    """一条句子级检索结果。"""
    unit_id: str
    parent_id: str
    page: int
    text: str
    score: float


def dedupe_by_parent(hits: list[Hit], parents: dict[str, Parent]) -> list[dict]:
    """按名次扫句子，按 parent_id 去重；保留触发它的句子与分数。"""
    delivered, seen = [], set()
    for h in hits:
        if not h.parent_id or h.parent_id in seen:
            continue
        parent = parents.get(h.parent_id)
        if parent is None:
            continue
        seen.add(h.parent_id)
        delivered.append({
            "parent": parent,
            "unit_id": h.unit_id,
            "unit_text": h.text,
            "score": h.score,
        })
    return delivered


def adaptive_backfill(fetch, parents: dict[str, Parent],
                      target: int = config.BACKFILL_TARGET,
                      steps: tuple[int, ...] = config.STEPS):
    """fetch(n) 返回前 n 个句子的命中；返回 (父块列表, 每轮轨迹)。"""
    trace, delivered, prev_n = [], [], 0
    for n in steps:
        hits = fetch(n)
        delivered = dedupe_by_parent(hits, parents)[:target]
        trace.append({"candidates": n, "returned": len(hits), "parents": len(delivered)})
        if len(delivered) >= target:
            break
        if len(hits) <= prev_n:      # 候选已经取尽，再扩也没用
            break
        prev_n = len(hits)
    return delivered, trace
