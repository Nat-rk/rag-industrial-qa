# -*- coding: utf-8 -*-
"""RAG QA 统一入口：一次问答从这里能看清整条链。

运行：
    python main.py "Which parameter selects Positioning or Synchronization mode?"
    python main.py "..." hybrid        # 换 Hybrid 策略
    python main.py                     # 交互模式，输入问题回车

流程（8 步，和下面代码一一对应）：
    1 选策略  2 召回句子  3 回填父块  4 rerank
    5 上下文  6 生成答案  7 整理引用  8 打印
"""
from __future__ import annotations

import sys
import time

import config
from retrieval.store import load_parents, load_units
from retrieval.backfill import adaptive_backfill
from retrieval.bm25 import BM25UnitRetriever
from retrieval.hybrid import DenseRetriever, HybridRetriever
from retrieval.reranker import rerank_candidates
from qa.context_builder import build_context
from qa.generator import answer as llm_answer
from qa.citation import build_citations, format_citations, invalid_numbers


def build_retriever(mode: str, units: list[dict]):
    """第 1 步：按配置组装检索器。两种策略只差这一处。"""
    if mode == "bm25":
        return BM25UnitRetriever(units)
    if mode == "hybrid":
        return HybridRetriever(DenseRetriever(units), BM25UnitRetriever(units))
    raise SystemExit(f"未知 RETRIEVAL_MODE：{mode}（只能是 bm25 或 hybrid）")


def ask(question: str, mode: str = config.RETRIEVAL_MODE, retriever=None,
        parents=None, units=None):
    """跑完整条链，返回结果字典（方便测试脚本复用）。

    retriever 传进来的话就直接用（测试脚本里一次建好、反复问，省掉重建索引的时间）。
    """
    parents = parents or load_parents()
    units = units or load_units()

    t0 = time.time()
    if retriever is None:
        retriever = build_retriever(mode, units)       # 1 选策略
    t_build = time.time() - t0

    t0 = time.time()
    candidates, trace = adaptive_backfill(               # 2 召回 + 3 回填
        lambda n: retriever.retrieve(question, n), parents
    )
    t_recall = time.time() - t0

    t0 = time.time()
    picked, rerank_info = rerank_candidates(question, candidates, config.DELIVER_K)  # 4 rerank
    t_rerank = time.time() - t0

    context, sources = build_context(picked)          # 5 上下文

    t0 = time.time()
    try:
        answer_text, usage, _ = llm_answer(question, context, sources)   # 6 生成
        llm_error = None
    except RuntimeError as e:                         # LLM 失败也要给出可用信息
        answer_text, usage, llm_error = f"（生成失败：{e}）", {}, str(e)
    t_llm = time.time() - t0

    used, marked = build_citations(answer_text, sources)   # 7 引用
    bad = invalid_numbers(answer_text, sources)            # 模型引用越界的编号（应为空）

    return {
        "question": question,
        "mode": retriever.name,
        "candidates": candidates,
        "trace": trace,
        "picked": picked,
        "rerank": rerank_info,
        "sources": sources,
        "context": context,
        "answer": answer_text,
        "usage": usage,
        "citations": used,
        "citations_marked": marked,
        "invalid_citations": bad,
        "timing": {"build": t_build, "recall": t_recall,
                   "rerank": t_rerank, "llm": t_llm},
    }


def show(result: dict) -> None:
    """第 8 步：把过程打印出来（人看的，不是机器解析的）。"""
    print("=" * 78)
    print(f"问题：{result['question']}")
    print(f"策略：{result['mode']}    候选句子：{result['trace'][-1]['candidates']}    "
          f"去重父块：{len(result['candidates'])}   交付：{len(result['picked'])}")
    if result["rerank"].get("degraded"):
        print(f"⚠ rerank 已降级（{result['rerank'].get('error')}），当前顺序 = 原始检索顺序")
    else:
        print(f"rerank 正常，耗时 {result['rerank'].get('elapsed', 0):.2f}s")

    print("\n交付的父块（按最终顺序）：")
    for i, c in enumerate(result["picked"], start=1):
        p = c["parent"]
        score = c.get("rerank_score")
        extra = f"rerank={score:.4f}" if score is not None else f"retrieval={c['score']:.4f}"
        print(f"  {i:>2}. [Source {i}] p{p.page:<3} {p.kind:<7} {p.short_id} "
              f"{p.chars:>4}字 {extra}  {p.section[:52]}")

    print("\n" + "-" * 78)
    print(result["answer"])
    print("-" * 78)
    print(format_citations(result["citations"], result["sources"]))
    if result.get("invalid_citations"):
        print(f"\n⚠ 模型引用了上下文中不存在的编号：{result['invalid_citations']}（已从来源列表剔除）")
    if result["usage"]:
        print(f"\n[token] prompt={result['usage'].get('prompt_tokens')} "
              f"completion={result['usage'].get('completion_tokens')}")
    t = result["timing"]
    print(f"[耗时] 检索 {t['recall']:.2f}s  rerank {t['rerank']:.2f}s  LLM {t['llm']:.2f}s")
    print("=" * 78)


def main(argv: list[str]) -> None:
    mode = config.RETRIEVAL_MODE
    args = [a for a in argv[1:] if not a.startswith("-")]
    if args and args[-1] in ("bm25", "hybrid"):
        mode = args.pop()

    if args:
        show(ask(" ".join(args), mode=mode))
        return

    print(f"交互模式（当前策略 {mode}），直接输入问题，回车提问；空行退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(ask(q, mode=mode))


if __name__ == "__main__":
    # Windows 控制台默认 GBK，打印英文引号和特殊符号会报错，这里统一成 utf-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main(sys.argv)   #argv：完整的命令行参数列表。例如：['main.py', '--debug', '什么是RAG', 'hybrid']
