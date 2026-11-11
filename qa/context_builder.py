# -*- coding: utf-8 -*-
"""上下文构建：把 rerank 后的父块整理成 LLM 能读、且可追溯的文本。

设计要点：
    * 每块前面加 [Source n] + 文件名 + 页码 + 父块 ID，后面接正文；
      这样模型可以写 "…… [Source 2]"，我们就能知道它到底用了哪一块。
    * 单块上限 2600 字（最长父块 2607，基本等于不截），总量上限 24000 字。
      截断按 rerank 名次从后往前砍，砍掉的是最不相关的块，不会砍掉答案块。
"""
from __future__ import annotations

import config


def build_context(candidates: list[dict]):
    """返回 (上下文文本, 来源列表)。来源列表顺序与 [Source n] 一一对应。"""
    sources, blocks, total = [], [], 0

    for i, c in enumerate(candidates, start=1):
        p = c["parent"]
        text = p.text.strip()
        if len(text) > config.CONTEXT_PARENT_MAX_CHARS:
            text = text[: config.CONTEXT_PARENT_MAX_CHARS].rstrip() + "\n…（本块过长，已截断）"

        block = (
            f"[Source {i}]\n"
            f"File: {p.file_name}\n"
            f"Page: {p.page}\n"
            f"Parent ID: {p.short_id}\n"
            f"Section: {p.section}\n"
            f"\n{text}\n"
        )
        if total + len(block) > config.CONTEXT_TOTAL_MAX_CHARS and blocks:
            break                      # 总量到顶：丢掉排在后面的（最不相关的）块

        blocks.append(block)
        total += len(block)
        sources.append({
            "n": i,
            "file": p.file_name,
            "page": p.page,
            "parent_id": p.id,
            "short_id": p.short_id,
            "section": p.section,
            "chars": len(text),
        })

    context = "\n".join(blocks) if blocks else "（没有检索到任何相关文档片段）"
    return context, sources
