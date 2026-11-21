# -*- coding: utf-8 -*-
"""日志初始化：统一格式 + trace_id，进程入口调用一次。

为什么需要：
    现在全项目只有 `print`（`show()` 打印给人看），没有任何 `logging`。
    服务化之后，一个请求会横跨 recall → rerank → LLM 三段，出问题时既没有
    级别、也没法把同一请求的几行输出串起来。

    S1 先解决两件事：
      1. `setup_logging()`：一次配置 root handler（格式里带 trace_id）；
         重复调用不会叠加 handler（测试里会反复调）。
      2. `new_trace_id()`：每次请求生成一个短 id 放进 ContextVar，格式过滤器
         自动把它塞进每条日志；LLM 库的日志也会带上，不用手动传参。

约定：
    · 库代码里只 `log = logging.getLogger(__name__)`，**不要自己配 handler**；
    · 只有入口（`main()` / 将来 `create_app()`）调用 `setup_logging()`；
    · trace_id 用 ContextVar 而不是全局变量：以后开线程池/async 时不会串号。
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar, Token

# 默认 "-"：没走请求链路（比如启动阶段）时也有一条合法日志
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

_CONFIGURED = False
_HANDLER_NAME = "rag-console"


def new_trace_id() -> str:
    """生成一个 8 位十六进制 trace_id 并写入当前上下文，返回它。"""
    tid = uuid.uuid4().hex[:8]
    _trace_id.set(tid)
    return tid


def begin_trace(tid: str | None = None) -> Token:
    """开启一次请求的 trace，返回"撤销令牌"。

    与 `new_trace_id()` 的区别：这个函数返回 ContextVar 的 Token，
    调用方必须在 finally 里把它交回 `end_trace()`。这样请求结束后上下文里的
    trace_id 会恢复成原来的值（通常是 "-"），不会把上一个请求的 id 留给下一个。
    服务化以后一个进程会连续处理很多请求，这一点比 CLI 重要得多。
    """
    return _trace_id.set(tid or uuid.uuid4().hex[:8])


def end_trace(token: Token) -> None:
    """请求结束（包括抛异常）时恢复到 begin_trace 之前的状态。"""
    _trace_id.reset(token)


def current_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    """测试或上层网关（从请求头透传）用。"""
    _trace_id.set(tid)


class _TraceFilter(logging.Filter):
    """把当前上下文里的 trace_id 贴到 LogRecord 上，供 Formatter 使用。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


def setup_logging(level: str = "INFO", stream=None) -> None:
    """配置 root logger（幂等）。

    level 非法时退回 INFO 而不是抛异常——日志配置失败不应该让服务起不来。
    """
    global _CONFIGURED
    lvl = getattr(logging, str(level).upper(), None)
    if not isinstance(lvl, int):
        lvl = logging.INFO

    root = logging.getLogger()
    root.setLevel(lvl)

    if _CONFIGURED:
        # 已经配过：只更新级别，不再加 handler（否则日志会重复打印）
        return

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.name = _HANDLER_NAME
    handler.setLevel(lvl)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s trace=%(trace_id)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_TraceFilter())
    root.addHandler(handler)
    _CONFIGURED = True


def reset_logging() -> None:
    """测试专用：把 setup_logging 的状态清空，允许重新配置。"""
    global _CONFIGURED
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "name", "") == _HANDLER_NAME:
            root.removeHandler(h)
    _CONFIGURED = False
