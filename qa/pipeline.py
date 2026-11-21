# -*- coding: utf-8 -*-
"""一次问答的完整编排（S2 从 main.py 抽出来的）。

为什么单独一个文件：
    S1 之后配置已经是对象（Settings），但"编排"还住在 `main.py` 里，
    而 `main.py` 同时是命令行入口——服务化时（S3）不能 import 一个带
    `sys.argv` 解析的模块。把 8 步流程搬到 qa/pipeline.py，CLI 只留
    "解析参数 + 调用 + 打印"，HTTP 层将来也只需要调用同一个函数。

定位：**无状态的业务流程编排函数**（不是纯函数）。
    * 输入：question、Settings、可选依赖（retriever / parents / units）。
    * 输出：和 CLI 完全一致的 14 键字典。
    * 不读环境变量（Settings 由调用方造好传进来）；
    * 不持有长期资源（检索器、父块都可以由调用方复用后传进来）；
    * 不管进程生命周期与退出码——那是入口的事。

依赖注入的两个口子（rerank_fn / llm_fn）：
    `tests/test_pipeline.py` 用 `mock.patch.object(main, "rerank_candidates")`
    和 `main.llm_answer` 来避免联网。如果本模块直接调用自己 import 进来的函数，
    patch 会失效（测试就变成"假通过"）。所以默认值是 None，由 main.ask 把
    模块属性传进来——patch 替换的是 main 上的名字，而 main 每次调用时再取一次。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

import logging_setup
from errors import ConfigError
from qa.citation import build_citations, invalid_numbers
from qa.context_builder import build_context
from qa.generator import answer as llm_answer
from retrieval.backfill import adaptive_backfill
from retrieval.bm25 import BM25UnitRetriever
from retrieval.hybrid import DenseRetriever, HybridRetriever
from retrieval.reranker import rerank_candidates
from retrieval.store import load_parents, load_units
from settings import Settings

log = logging.getLogger(__name__)


def build_retriever(mode: str, units: list[dict], settings: Settings):
    """第 1 步：按 mode 组装检索器。两种策略只差这一处。

    settings 里的检索参数（停用词、RRF 的 k/side、向量库路径与模型）显式传入，
    不再由各组件自己去 import config。
    """
    bm25 = BM25UnitRetriever(
        units,
        extra_stopwords=settings.extra_stopwords,
        index_size=max(settings.steps),
    )
    if mode == "bm25":
        return bm25
    if mode == "hybrid":
        dense = DenseRetriever(
            units,
            chroma_dir=settings.chroma_dir,
            collection=settings.collection,
            embed_model=settings.embed_model,
            api_base=settings.api_base,
        )
        return HybridRetriever(dense, bm25, k=settings.rrf_k, side=settings.rrf_side)
    raise ConfigError(f"未知 RETRIEVAL_MODE：{mode}（只能是 bm25 或 hybrid）")


def run(question: str, *, settings: Settings, mode: str | None = None,
        retriever=None, parents=None, units=None,
        rerank_fn: Callable[..., Any] | None = None,
        llm_fn: Callable[..., Any] | None = None) -> dict:
    """跑完整条链，返回 14 键结果字典。

    settings 必填：本函数不读环境变量，也不回退到 config。调用方（CLI / 评测 /
    测试）各自决定配置从哪来。
    """
    s = settings

    # 一次问答一个 trace_id；finally 里恢复，避免影响同一进程里的下一次调用。
    # 放在 load 之前：连"数据加载慢"这种问题也能带着同一个 id 一起看。
    token = logging_setup.begin_trace()
    try:
        if parents is None:
            parents = load_parents(parents_path=s.parents_path, docstore_path=s.docstore_path)
        if units is None:
            units = load_units(units_path=s.units_path)

        t0 = time.time()
        if retriever is None:
            retriever = build_retriever(mode or s.retrieval_mode, units, s)   # 1 选策略
        t_build = time.time() - t0

        t0 = time.time()
        candidates, trace = adaptive_backfill(                # 2 召回句子 + 3 回填父块
            lambda n: retriever.retrieve(question, n), parents,
            target=s.backfill_target, steps=s.steps,
        )
        t_recall = time.time() - t0

        t0 = time.time()
        rerank = rerank_fn or rerank_candidates               # 4 rerank
        picked, rerank_info = rerank(
            question, candidates, s.deliver_k,
            model=s.rerank_model, api_base=s.api_base,
        )
        t_rerank = time.time() - t0

        context, sources = build_context(                      # 5 拼上下文
            picked,
            parent_max_chars=s.context_parent_max_chars,
            total_max_chars=s.context_total_max_chars,
        )

        t0 = time.time()
        llm = llm_fn or llm_answer                             # 6 生成答案
        try:
            answer_text, usage, _ = llm(
                question, context, sources,
                model=s.llm_model, temperature=s.llm_temperature,
                api_base=s.api_base, timeout=s.llm_timeout,
            )
        except ConfigError:
            # 配置错误向上抛，交给入口统一处理；不要混进"生成失败"的文案
            raise
        except RuntimeError as e:
            answer_text, usage = f"（生成失败：{e}）", {}
        t_llm = time.time() - t0

        used, marked = build_citations(answer_text, sources)   # 7 整理引用
        bad = invalid_numbers(answer_text, sources)

        log.info(
            "ask.done mode=%s candidates=%d delivered=%d cited=%d degraded=%s "
            "build=%.2fs recall=%.2fs rerank=%.2fs llm=%.2fs",
            retriever.name, len(candidates), len(picked), len(used),
            bool(rerank_info.get("degraded")),
            t_build, t_recall, t_rerank, t_llm,
        )
        if rerank_info.get("degraded"):
            log.warning("rerank 降级：%s", rerank_info.get("error"))
        if bad:
            log.warning("模型引用了不存在的编号：%s", bad)

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
    finally:
        logging_setup.end_trace(token)
