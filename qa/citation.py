# -*- coding: utf-8 -*-
"""引用整理：把答案里的 [Source n] 映射回 文件名 + 页码 + 父块 ID。

原则：
    * 只列答案里**真正引用过**的来源，不把 10 个候选全列上；
    * 答案没标任何 [Source n] 时，如实显示"模型未标注来源"，并把
      检索到的来源列成"可查阅"而不是"参考来源"——避免让人误以为被引用了。
"""
from __future__ import annotations

import re

SOURCE_RE = re.compile(r"\[?Source\s*(\d+)\]?", re.IGNORECASE)

# 模型有时会自作主张在末尾补一段"参考来源"清单。那一段里的编号并不代表
# 正文真的引用了它——曾经因此出现"正文一句话、末尾列 10 个 Source"的假引用。
# 解析前先把它截掉，引用表由 format_citations() 根据正文 [Source n] 机械生成。
_REF_HEAD_RE = re.compile(r"^[ \t]*(?:参考来源|参考资料|引用来源|References?)[ \t]*[:：]?[ \t]*$",
                          re.IGNORECASE | re.MULTILINE)


def strip_reference_section(answer: str) -> str:
    """去掉答案末尾模型自带的“参考来源”清单，只留正文。"""
    text = answer or ""
    m = _REF_HEAD_RE.search(text)
    return text[:m.start()].rstrip() if m else text


def cited_numbers(answer: str) -> list[int]:
    """按出现顺序去重，拿到答案正文引用过的 Source 编号。"""
    seen, out = set(), []
    for m in SOURCE_RE.finditer(strip_reference_section(answer)):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def invalid_numbers(answer: str, sources: list[dict]) -> list[int]:
    """答案里引用了、但上下文里不存在的编号（模型编造或越界）。

    单独暴露出来，是为了让入口能如实提示"模型引用了不存在的来源"，
    而不是像以前那样静默丢掉——静默会让人以为引用链是完整的。
    """
    valid = {s["n"] for s in sources}
    return [n for n in cited_numbers(answer) if n not in valid]


def build_citations(answer: str, sources: list[dict]):
    """返回 (引用列表, 是否标注了来源)。"""
    by_n = {s["n"]: s for s in sources}
    used = []
    for n in cited_numbers(answer):
        s = by_n.get(n)
        if s:
            used.append({
                "n": n,
                "file": s["file"],
                "page": s["page"],
                "parent_id": s["parent_id"],
                "short_id": s["short_id"],
                "section": s["section"],
            })
    return used, bool(used)


def format_citations(used: list[dict], sources: list[dict]) -> str:
    """打印用：有引用就列引用，没引用就提示未标注。"""
    if used:
        lines = ["参考来源："]
        for c in used:
            lines.append(f"- {c['file']}，第 {c['page']} 页（Source {c['n']}，父块 {c['short_id']}）")
        return "\n".join(lines)

    lines = ["参考来源：模型未在回答中标注 [Source n]，无法确认它实际用了哪几块。",
             "本次检索提供给模型的片段如下，可人工核对："]
    for s in sources:
        lines.append(f"- {s['file']}，第 {s['page']} 页（Source {s['n']}，父块 {s['short_id']}）")
    return "\n".join(lines)
