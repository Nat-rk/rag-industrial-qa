# -*- coding: utf-8 -*-
"""异常 → HTTP 响应。所有状态码只出现在这一层，底层 `errors.py` 保持纯粹。

映射表（与设计报告一致）：

    ConfigError      503  config_error        服务端配置问题，重启前不会自愈
    DataError        500  data_error          数据资产缺失/损坏，服务端故障
    LLMError         502  llm_error           上游模型失败（网关语义）
    RerankError      502  rerank_error        实际会被 pipeline 降级，防御性映射
    EngineNotReady   503  engine_not_ready    生命周期状态不符（未初始化/已关闭）
    HTTPException    原状态码                 路由自己抛的（如 404），套同一外壳
    校验失败          422  validation_error    请求体不符合 schema
    其它 Exception    500  internal_error      只回稳定文案，traceback 进日志

错误外壳：

    {"error": {"code": "llm_error", "message": "…", "trace_id": "3f9a1c2d"}}
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

import logging_setup
from errors import ConfigError, DataError, LLMError, RagError, RerankError

log = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """Engine 还没初始化好，或在关闭之后又被调用。

    它不是 `RagError` 的子类：这是"服务生命周期"的问题，不是底层 RAG 的错误，
    所以放在 API 层，并且不带 HTTP 语义（状态码在下面的映射表里）。
    """


_STATUS_BY_ERROR: tuple[tuple[type[BaseException], int, str, str], ...] = (
    (ConfigError, 503, "config_error", "服务配置错误，请检查环境变量与数据目录"),
    (DataError, 500, "data_error", "服务端数据资产缺失或损坏"),
    (LLMError, 502, "llm_error", "上游模型服务调用失败"),
    (RerankError, 502, "rerank_error", "重排序服务调用失败"),
    (EngineNotReady, 503, "engine_not_ready", "服务尚未就绪"),
)


def current_trace_id() -> str | None:
    """取当前请求的 trace_id；还没进入 pipeline 时返回 None（不重复建追踪系统）。"""
    try:
        tid = logging_setup.current_trace_id()
    except Exception:            # pragma: no cover - 极端情况下也不该崩
        return None
    return tid if tid and tid != "-" else None


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message,
                           "trace_id": current_trace_id()}},
    )


def register_error_handlers(app: FastAPI) -> None:
    """把异常处理器挂到应用上；调用一次即可。"""

    @app.exception_handler(EngineNotReady)
    async def _engine_not_ready(_request: Request, exc: EngineNotReady) -> JSONResponse:
        log.warning("engine 未就绪：%s", exc)
        return error_response(503, "engine_not_ready", str(exc))

    @app.exception_handler(RagError)
    async def _rag_error(_request: Request, exc: RagError) -> JSONResponse:
        for cls, status, code, message in _STATUS_BY_ERROR:
            if isinstance(exc, cls):
                # 原始异常文本进日志（含路径等细节），客户端只拿固定文案。
                # 已知类型的失败是"预期内故障"，不打 traceback；未预期异常才打。
                log.error("请求失败 %s：%s", code, exc)
                return error_response(status, code, message)
        log.error("未映射的 RagError：%s", exc, exc_info=True)
        return error_response(500, "internal_error", "服务内部错误")

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request,
                                exc: RequestValidationError) -> JSONResponse:
        parts: list[str] = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
            parts.append(f"{loc or 'body'}: {err.get('msg', '不合法')}")
        log.info("请求校验失败：%s", "; ".join(parts))
        return error_response(422, "validation_error", "；".join(parts) or "请求不合法")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request,
                          exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "请求失败"
        return error_response(exc.status_code, "http_error", detail)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # 只回稳定文案；traceback 只进服务端日志
        log.exception("未预期异常：%s", type(exc).__name__)
        return error_response(500, "internal_error", "服务内部错误")
