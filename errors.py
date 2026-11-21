# -*- coding: utf-8 -*-
"""项目里所有"可预期的失败"都从这里抛。

为什么单独一个文件：
    以前 `store.api_key()` 用 `SystemExit` 报缺 key，`generator` 用裸
    `RuntimeError`，`reranker` 有自己的 `RerankError`。三种错误在调用方看来
    都是"字符串"，没法分类处理——CLI 只能整体崩，将来 FastAPI 也没法区分
    "配置错了（503）"和"上游模型挂了（502）"。

    S1 先把异常类型定下来：库代码只抛 `RagError` 的子类，**怎么退出/怎么转
    HTTP 状态码由入口决定**，库层不替调用方做决定。

层次（都继承 RuntimeError，保证旧代码 `except RuntimeError` 仍然兜得住）：
    RagError
     ├─ ConfigError  配置/环境问题：缺 key、环境变量非法、数据目录不可用
     ├─ DataError    数据资产问题：parents.json / units.json / chroma 读不到
     ├─ LLMError     生成阶段失败：LLM 请求失败、响应格式异常
     └─ RerankError  重排序失败：网络、额度、响应格式——调用方会降级而不是中断
"""
from __future__ import annotations


class RagError(RuntimeError):
    """本项目所有可预期异常的基类。"""


class ConfigError(RagError):
    """配置或运行环境不满足要求（缺 key、环境变量非法、目录不可写）。"""


class DataError(RagError):
    """数据资产缺失或损坏（parents.json / units.json / chroma_db）。"""


class LLMError(RagError):
    """调用大模型失败，或返回内容无法解析。"""


class RerankError(RagError):
    """调用 rerank 接口失败。调用方应降级（保持原顺序）而不是中断问答。"""
