# -*- coding: utf-8 -*-
"""RagEngine：把"加载数据 + 构建检索器"从每次问答里拿出来，只做一次。

为什么要有它（S3 的目标）：
    S2 的 `pipeline.run()` 已经支持传入 parents/units/retriever 来复用资源，
    但没有任何对象"持有"它们——命令行每次启动都要重新读 312 ms 数据、重建
    4.9 s 的 BM25 索引。Engine 就是那个持有者：初始化一次，之后每次 ask 都
    复用同一批对象，仍然走同一条 `pipeline.run()`。

三个状态（closed 是终态）：
    new    —— 只拿到 Settings，没有碰磁盘；
    ready  —— parents/units 已加载、检索器已构建（hybrid 还完成了 Chroma 与
              embedding 客户端的预热）；
    closed —— 资源已释放，不能再 initialize() 或 ask()。

一个约定（重要）：
    一个数据目录只允许一个同时存活的 Engine。chromadb 按"路径 + 配置"缓存
    System，两个 Engine 指向同一目录时，其中一个 close() 会把另一个的向量库
    一起停掉。隔离测试必须用不同的临时目录。
"""
from __future__ import annotations

import logging

from qa import pipeline
from retrieval.store import load_parents, load_units
from settings import Settings

log = logging.getLogger(__name__)

NEW, READY, CLOSED = "new", "ready", "closed"


class RagEngine:
    """持有一次初始化产生的全部资源，并把每次问答交给 pipeline.run()。

    它不读环境变量、不解析命令行、不依赖 `main.settings()` 那个 CLI 单例——
    Settings 由调用方显式传进来。
    """

    def __init__(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("RagEngine(settings=...) 需要 Settings 实例")
        self._settings = settings
        # 初始化后才有值；new / closed 状态一律是 None
        self._parents = None
        self._units = None
        self._retriever = None
        self._state = NEW

    # -- 只读状态 ----------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def state(self) -> str:
        return self._state

    @property
    def initialized(self) -> bool:
        return self._state == READY

    @property
    def parents(self):
        return self._parents

    @property
    def units(self):
        return self._units

    @property
    def retriever(self):
        return self._retriever

    # -- 生命周期 ----------------------------------------------------------
    def initialize(self) -> None:
        """加载数据、构建检索器；已就绪时重复调用是空操作。

        任何一步失败都会把已经建好的资源清掉、状态退回 new，再原样抛出异常
        （DataError / ConfigError 不包装），所以不会留下"看起来已初始化"的
        半成品。
        """
        if self._state == CLOSED:
            raise RuntimeError("Engine 已关闭（closed 是终态），需要新的 Engine")
        if self._state == READY:
            return

        s = self._settings
        try:
            self._parents = load_parents(
                parents_path=s.parents_path, docstore_path=s.docstore_path)
            self._units = load_units(units_path=s.units_path)
            # 先让 Engine 持有检索器，再预热：预热失败时才能找到它并清理
            self._retriever = pipeline.build_retriever(
                s.retrieval_mode, self._units, s)

            prewarm = getattr(self._retriever, "prewarm", None)
            if prewarm is not None:
                prewarm()          # hybrid：连 Chroma + 建 embedding 客户端
        except BaseException:
            self._cleanup()
            self._state = NEW
            raise

        self._state = READY

    def ask(self, question: str) -> dict:
        """跑一次完整问答；复用初始化时的全部资源，不重新加载/构建。"""
        if self._state == NEW:
            raise RuntimeError("Engine 尚未初始化：先调用 initialize()")
        if self._state == CLOSED:
            raise RuntimeError("Engine 已关闭，不能再提问")
        return pipeline.run(
            question,
            settings=self._settings,
            retriever=self._retriever,
            parents=self._parents,
            units=self._units,
        )

    def close(self) -> None:
        """释放本 Engine 拥有的资源；重复调用安全，未初始化时调用也安全。"""
        if self._state == CLOSED:
            return
        self._cleanup()
        self._state = CLOSED

    # -- 内部 --------------------------------------------------------------
    def _cleanup(self) -> None:
        """丢掉引用并关闭检索器持有的外部客户端。

        清理本身的异常只记日志、不向上抛：它可能正发生在"初始化失败"的
        清理路径上，不能让清理错误盖掉真正的初始化异常。
        """
        retriever, self._retriever = self._retriever, None
        self._parents = None
        self._units = None

        close = getattr(retriever, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:
            log.warning("清理检索器资源时出错（已忽略，避免覆盖原始异常）", exc_info=True)
