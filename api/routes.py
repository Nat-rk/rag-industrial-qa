# -*- coding: utf-8 -*-
"""HTTP 路由：只有三个接口，不增加 /search、/chat、/stream。

线程模型（阶段 4 核实过）：

* `/ask` 是同步 `def`。FastAPI（starlette `run_in_threadpool`）把它丢进 anyio
  工作线程执行，`engine.ask()` 跑 20–60 秒也只占一个线程，不阻塞事件循环。
* `/healthz`、`/readyz` 写成 `async def`，直接在事件循环里返回一次性字典，
  **不占工作线程**。这样即使 40 个工作线程都被排队中的问答占满，健康检查
  仍然能立刻回答（k8s/负载均衡的就绪探针不会被问答拖住）。
* 串行化用 `app.state.ask_lock`（`threading.Lock`），在路由里 `with lock:`
  包住 `engine.ask()`。`with` 保证异常路径也会释放；锁只覆盖单进程内的
  问答入口，healthz/readyz 不拿锁。**不宣称多进程/多 worker 全局串行**——
  那种场景下每个进程各有一把锁、各有一个 Engine，本来就不该开。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from api.errors import EngineNotReady, error_response
from api.schemas import AskRequest, AskResponse

log = logging.getLogger(__name__)

router = APIRouter()

#: 服务名，只用于健康检查的响应体，不影响业务
SERVICE_NAME = "rag-qa"


def _ready_engine(request: Request):
    """取出处于 ready 的 Engine；不存在或状态不对就抛 `EngineNotReady`（→ 503）。"""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise EngineNotReady("Engine 不存在：服务尚未完成初始化")
    state = getattr(engine, "state", None)
    if state != "ready":
        raise EngineNotReady(f"Engine 当前状态为 {state or 'unknown'}，不能接受问答")
    return engine


@router.get("/healthz", summary="存活检查", tags=["ops"])
async def healthz() -> dict:
    """只证明"进程还能响应 HTTP"；不碰 Engine、不碰锁、不碰 LLM。"""
    return {"status": "ok", "service": SERVICE_NAME}


@router.get("/readyz", summary="就绪检查", tags=["ops"])
async def readyz(request: Request):
    """Engine 是否已初始化完成：ready → 200，其余 → 503（不触发初始化、不拿锁）。"""
    engine = getattr(request.app.state, "engine", None)
    state = getattr(engine, "state", None)
    if engine is not None and state == "ready":
        settings = getattr(engine, "settings", None)
        return {
            "status": "ready",
            "service": SERVICE_NAME,
            "mode": getattr(settings, "retrieval_mode", None),
        }
    return error_response(
        503, "not_ready", f"服务尚未就绪（engine state={state or 'none'}）")


@router.post("/ask", response_model=AskResponse, summary="问答", tags=["qa"])
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """一次完整问答：复用启动时初始化好的 Engine，单进程内串行执行。

    请求体只有 `question`；检索模式来自启动配置，不接受请求级覆盖。
    """
    engine = _ready_engine(request)

    lock = getattr(request.app.state, "ask_lock", None)
    if lock is None:
        # 生命周期没跑完（或测试里忘了装锁）时宁可 503，也不要在无串行化保证下执行
        raise EngineNotReady("服务缺少问答锁，拒绝在未串行化的状态下执行")

    with lock:                      # 异常也会释放；锁在 API 层，不在 Engine 里
        result = engine.ask(payload.question)

    return AskResponse.from_result(result)
