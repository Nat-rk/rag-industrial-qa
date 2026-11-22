# -*- coding: utf-8 -*-
"""应用工厂与生命周期：把 RagEngine 的"初始化一次"接到 FastAPI 的 lifespan 上。

为什么要有工厂：
    `uvicorn api.main:app` 需要一个模块级 `app` 对象，但测试需要换成 Fake Engine、
    需要不同的 Settings。`create_app(settings, engine_factory=...)` 让这两件事都不
    需要全局变量：生产入口传 `Settings.from_env()` 与默认的 `RagEngine`，测试传
    临时 Settings 与桩对象。

为什么创建 app 时不初始化 Engine：
    建 app 应该"便宜且无副作用"——测试里频繁 `create_app()` 不应该读 278 个父块、
    不应该连 Chroma。真正的资源准备放在 lifespan 启动阶段，一次做完。

生命周期约定（与 S3 的 RagEngine 对齐）：
    启动：engine_factory(settings) → api_key() 预检 → engine.initialize()
          → 存进 app.state.engine → 建 app.state.ask_lock
    关闭：engine.close() → app.state.engine = None

    初始化失败时，Engine 可能还没存进 app.state，所以必须用**局部变量**清理，
    再把原异常抛出去（uvicorn 因此启动失败，不会带着未就绪的服务接请求）。
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI

from api.errors import register_error_handlers
from api.routes import router
from retrieval.store import api_key
from service.engine import RagEngine
from settings import Settings

log = logging.getLogger(__name__)

#: 测试可以传自己的工厂；生产默认就是 S3 的 RagEngine。
EngineFactory = Callable[[Settings], Any]


def create_app(settings: Settings, *,
               engine_factory: EngineFactory = RagEngine) -> FastAPI:
    """建一个 FastAPI 应用；此时不加载数据、不构建检索器、不读环境变量。"""
    if not isinstance(settings, Settings):
        raise TypeError("create_app(settings=...) 需要 Settings 实例")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1) 只构造（RagEngine 的构造函数不碰磁盘）
        engine = engine_factory(settings)
        try:
            # 2) 配置预检：没有 key 就没必要读几百 MB 数据、建几秒索引
            api_key()
            # 3) 初始化：数据 + 检索器（hybrid 还会预热 Chroma 与 embedding）
            engine.initialize()
        except BaseException:
            # 失败路径用局部引用清理：engine 还没进 app.state
            _safe_close(engine)
            raise

        app.state.engine = engine
        app.state.ask_lock = threading.Lock()
        log.info("engine ready mode=%s parents=%s units=%s",
                 settings.retrieval_mode,
                 _len(getattr(engine, "parents", None)),
                 _len(getattr(engine, "units", None)))

        try:
            yield
        finally:
            # 关闭阶段：幂等；即使 Engine 自己在 shutdown 前出错也要尽力释放
            _safe_close(getattr(app.state, "engine", None))
            app.state.engine = None

    app = FastAPI(
        title="Industrial RAG QA",
        version="0.4.0",
        description=(
            "单文档工业设备问答服务（S4 HTTP 接入层）。\n\n"
            "运行约定：单进程、单 worker、单 Engine，问答请求串行执行；"
            "不面向公网暴露。"
        ),
        lifespan=lifespan,
    )
    # Settings 放在 app.state 上，路由与测试都能读到，但不做全局单例
    app.state.settings = settings
    app.include_router(router)
    register_error_handlers(app)
    return app


def _safe_close(engine: Any) -> None:
    """尽力关闭 Engine；清理失败不覆盖原始异常。"""
    if engine is None:
        return
    close = getattr(engine, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        log.warning("关闭 Engine 时出错（已忽略）", exc_info=True)


def _len(obj: Any) -> Any:
    try:
        return len(obj)
    except Exception:
        return "?"
