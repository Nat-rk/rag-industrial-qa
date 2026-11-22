# -*- coding: utf-8 -*-
"""RAG QA 统一入口：一次问答从这里能看清整条链。

运行：
    python main.py "Which parameter selects Positioning or Synchronization mode?"
    python main.py "..." hybrid        # 换 Hybrid 策略
    python main.py                     # 交互模式，输入问题回车

流程（8 步，S2 起实际代码在 qa/pipeline.py，本文件只负责入口）：
    1 选策略  2 召回句子  3 回填父块  4 rerank
    5 上下文  6 生成答案  7 整理引用  8 打印

分工：
    main.py         解析 argv、建 Settings、配日志、打印、决定退出码（进程边界）
    qa/pipeline.py  真正的 8 步编排，接收显式 Settings，不读环境变量
    settings.py     配置对象（唯一真源，S3 起 config.py 已删除）
    api/            另一条入口：FastAPI 的 /ask 也走 qa/pipeline.py，不经过本文件
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

import logging_setup
from errors import ConfigError, DataError
from qa import pipeline
from qa.citation import format_citations
from qa.generator import answer as llm_answer
from retrieval.reranker import rerank_candidates
from retrieval.store import api_key
from settings import Settings

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def settings() -> Settings:
    """CLI 用的配置单例：第一次真正用到时才读环境变量。

    为什么惰性：`import main` 曾经会连带 `config` 读一遍环境变量；测试里改
    `os.environ` 再 import 就会互相污染。现在 import 本身不读环境，测试可以先
    改环境再调用 `settings()`（用完 `settings.cache_clear()` 清掉）。

    注意：这个单例只属于 CLI。将来的 Engine/HTTP 服务不得依赖它——它们应自己
    持有 Settings，否则多实例部署时会被这个全局缓存串在一起。
    """
    return Settings.from_env()


def build_retriever(mode: str, units: list[dict]):
    """第 1 步：按配置组装检索器。真实实现已移到 qa/pipeline.py。

    保留这个薄封装是因为 `eval/run_cases.py` 直接 import 它（一次建好、
    反复问，省掉重建索引的时间）。
    """
    return pipeline.build_retriever(mode, units, settings())


def _resolve_settings(explicit: Settings | None) -> Settings:
    """把"可选的 Settings"解析成实例。

    单独一个函数是为了避开名字遮蔽：`ask()` 的参数就叫 `settings`，
    函数体里不能再调同名的全局函数 `settings()`，只能经由这里转一手。
    """
    return explicit if explicit is not None else settings()


def ask(question: str, mode: str | None = None, retriever=None,
        parents=None, units=None, settings: Settings | None = None):
    """跑完整条链，返回结果字典（方便测试脚本复用）。

    mode 不传时用 Settings 里的检索策略；retriever/parents/units 传进来的话
    就直接复用（测试脚本里一次建好、反复问，省掉重建索引的时间）。
    settings 不传时用 CLI 的那个惰性单例；评测脚本应自己 `Settings.from_env()`
    再显式传进来，免得和命令行共用一份全局缓存。

    这里把 `rerank_candidates` 和 `llm_answer` 作为参数传下去，是为了让
    `mock.patch.object(main, "rerank_candidates")` 仍然有效——pipeline 不自己
    持有这两个函数，否则测试的 patch 会静默失效。
    """
    return pipeline.run(
        question,
        settings=_resolve_settings(settings),
        mode=mode,
        retriever=retriever,
        parents=parents,
        units=units,
        rerank_fn=rerank_candidates,
        llm_fn=llm_answer,
    )


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
    s = settings()
    # 输出目录原先由 config 在 import 时创建（副作用）；现在由入口负责，
    # 保证"import 配置"不再碰文件系统。
    s.out_dir.mkdir(parents=True, exist_ok=True)

    # 日志只在这里配一次；库层只 getLogger，不自己加 handler
    logging_setup.setup_logging(s.log_level)
    try:
        api_key()          # 预检：缺 key 立刻失败，而不是先花几秒建索引再失败
        _run(argv)
    except ConfigError as e:
        # 配置错误是"用户能修"的问题：打一行清楚的日志，用退出码 2 结束，
        # 不再像以前那样抛 SystemExit 把调用方一起带走（SystemExit 不属于 Exception）。
        log.error("配置错误：%s", e)
        raise SystemExit(2) from None
    except DataError as e:
        # 数据文件缺失/损坏同样是"用户能修"的问题（下载数据包、设 RAG_DATA_DIR、
        # 检查文件完整性），给一行清楚的日志和退出码 3，而不是一屏 traceback。
        log.error("数据错误：%s", e)
        raise SystemExit(3) from None


def _run(argv: list[str]) -> None:
    mode = settings().retrieval_mode
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
        sys.stderr.reconfigure(encoding="utf-8")   # 日志走 stderr，同样需要
    except Exception:
        pass
    main(sys.argv)   #argv：完整的命令行参数列表。例如：['main.py', '--debug', '什么是RAG', 'hybrid']
