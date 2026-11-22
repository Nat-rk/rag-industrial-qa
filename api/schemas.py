# -*- coding: utf-8 -*-
"""请求/响应模型。

原则：**不改业务结果**。`pipeline.run()` 返回什么，`AskResponse` 就长什么样，
只是把不能直接 JSON 化的 `Parent` 数据类换成同名字段的 `ParentOut`。

14 个键的对应关系（顺序与 `qa/pipeline.py:145-161` 一致）：
    question / mode / candidates / trace / picked / rerank /
    sources / context / answer / usage / citations /
    citations_marked / invalid_citations / timing

`candidates[i]["parent"]`、`picked[i]["parent"]` 是 `retrieval.store.Parent`
（dataclass），直接交给 JSON 会抛 TypeError，所以这里做一次等价转换：
`ParentOut` 的 7 个字段与 `Parent` 完全一致，另外补上它的 `short_id` 属性。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from retrieval.store import Parent

#: 单次提问的字符上限（strip 之后计算）。2000 字对“一句工业设备问题”绰绰有余，
#: 同时避免把整页手册粘进请求。
MAX_QUESTION_CHARS = 2000


class ParentOut(BaseModel):
    """`Parent` 的 JSON 等价物（字段一一对应，不删不减）。"""

    model_config = ConfigDict(extra="allow")

    id: str
    page: int
    kind: str
    section: str
    text: str
    chars: int
    file_name: str
    short_id: str

    @classmethod
    def from_parent(cls, parent: Parent) -> "ParentOut":
        return cls(
            id=parent.id,
            page=parent.page,
            kind=parent.kind,
            section=parent.section,
            text=parent.text,
            chars=parent.chars,
            file_name=parent.file_name,
            short_id=parent.short_id,
        )


class CandidateOut(BaseModel):
    """一条召回候选（触发父块的那一句 + 分数）。"""

    model_config = ConfigDict(extra="allow")

    parent: ParentOut
    unit_id: str
    unit_text: str
    score: float


class PickedOut(BaseModel):
    """交付父块：候选 + rerank 分数（未重排或降级时可能没有该字段）。"""

    model_config = ConfigDict(extra="allow")

    parent: ParentOut
    unit_id: str
    unit_text: str
    score: float
    rerank_score: float | None = None


class TraceStep(BaseModel):
    """回填的自适应过程：这一轮取了多少候选、命中多少句子、去重后几个父块。"""

    candidates: int
    returned: int
    parents: int


class RerankInfo(BaseModel):
    """rerank 元信息；失败时 `degraded=True` 且带 `error`，成功时可能带 `usage`。"""

    model_config = ConfigDict(extra="allow")

    degraded: bool
    elapsed: float = 0.0
    error: str | None = None
    usage: dict[str, Any] | None = None


class SourceOut(BaseModel):
    """上下文里的一段证据；`n` 就是正文里 `[Source n]` 的编号。"""

    model_config = ConfigDict(extra="allow")

    n: int
    file: str
    page: int
    parent_id: str
    short_id: str
    section: str
    chars: int


class CitationOut(BaseModel):
    """答案里真正引用到的来源，由 `qa.citation.build_citations()` 生成。

    注意它**不是** `SourceOut`：`build_citations()` 只带 6 个字段，没有 `chars`。
    S4 阶段六真机验收时就是因为把 citations 声明成 `list[SourceOut]`，
    导致 `POST /ask` 对真实结果抛 `ValidationError: citations.0.chars Field required`
    （FakeEngine 的测试数据随手带了 chars，所以单元测试没暴露）。
    """

    model_config = ConfigDict(extra="allow")

    n: int
    file: str
    page: int
    parent_id: str
    short_id: str
    section: str


class Timing(BaseModel):
    """分段耗时（秒）。"""

    build: float
    recall: float
    rerank: float
    llm: float


class AskRequest(BaseModel):
    """`POST /ask` 的请求体：只有一个问题，不开放检索参数。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description=f"问题文本，1–{MAX_QUESTION_CHARS} 个字符（首尾空白会被去掉）。",
        examples=["FC 302 出现过压故障时应该如何检查？"],
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question 不能为空或只有空白字符")
        return text


class AskResponse(BaseModel):
    """`POST /ask` 的响应：与 `pipeline.run()` 的 14 键完全对应。"""

    model_config = ConfigDict(extra="allow")

    question: str
    mode: str
    candidates: list[CandidateOut]
    trace: list[TraceStep]
    picked: list[PickedOut]
    rerank: RerankInfo
    sources: list[SourceOut]
    context: str
    answer: str
    usage: dict[str, Any]
    citations: list[CitationOut]
    citations_marked: bool
    invalid_citations: list[int]
    timing: Timing

    @classmethod
    def from_result(cls, result: dict) -> "AskResponse":
        """把 `pipeline.run()` 的原始字典转成响应模型（只换 Parent，不改语义）。"""
        return cls.model_validate({
            "question": result["question"],
            "mode": result["mode"],
            "candidates": [
                {**c, "parent": ParentOut.from_parent(c["parent"])}
                for c in result["candidates"]
            ],
            "trace": result["trace"],
            "picked": [
                {**p, "parent": ParentOut.from_parent(p["parent"])}
                for p in result["picked"]
            ],
            "rerank": result["rerank"],
            "sources": result["sources"],
            "context": result["context"],
            "answer": result["answer"],
            "usage": result.get("usage") or {},
            "citations": result["citations"],
            "citations_marked": result["citations_marked"],
            "invalid_citations": result["invalid_citations"],
            "timing": result["timing"],
        })
