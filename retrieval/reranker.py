# -*- coding: utf-8 -*-
"""Rerank：调智谱 rerank-pro 对父块重排，失败时自动降级。

三条规矩（都是上一轮实验踩出来的）：
  1. 输入是**父块**，不是句子；每题最多 30 块，只改顺序、不加候选。
  2. rerank 只负责相关性排序，不负责判断答案对错、也不负责决定是否拒答。
  3. 接口失败（超时/429/额度不足）时**保持原检索顺序**并标 degraded，
     不能让整条问答崩掉，也不能假装成功。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests

import config
from retrieval.store import api_key

MAX_DOCS = 60
TIMEOUT = 120
RETRIES = 3


class RerankError(RuntimeError):
    """rerank 调用失败（网络、额度、响应格式都归到这里）。"""


@dataclass
class RerankOutcome:
    order: list[int] = field(default_factory=list)   # 重排后候选的下标
    scores: list[float] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    elapsed: float = 0.0
    attempts: int = 0
    error: str | None = None


def _post(payload: dict, headers: dict) -> dict:
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(f"{config.API_BASE}/rerank", json=payload,
                              headers=headers, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(0.8 * attempt)
                continue
            if r.status_code != 200:
                raise RerankError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except requests.RequestException as e:      # 网络层错误，重试
            last = f"{type(e).__name__}: {e}"
            time.sleep(0.8 * attempt)
    raise RerankError(f"重试 {RETRIES} 次仍失败：{last}")


def rerank_documents(query: str, documents: list[str], top_n: int | None = None):
    """调用 rerank-pro，返回 (按相关性排序的下标, 分数, usage, 耗时)。"""
    payload = {
        "model": config.RERANK_MODEL,
        "query": query,
        "documents": documents[:MAX_DOCS],
    }
    if top_n:
        payload["top_n"] = min(top_n, len(payload["documents"]))
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}

    t0 = time.time()
    data = _post(payload, headers)
    elapsed = time.time() - t0

    results = data.get("results") or []
    if not results:
        raise RerankError(f"响应里没有 results：{json.dumps(data)[:300]}")

    # 接口只回 document 原文、不回下标，所以按文本反查；同一文本可能出现多次，用队列
    buckets: dict[str, list[int]] = {}
    for i, doc in enumerate(documents):
        buckets.setdefault(doc, []).append(i)

    order, scores = [], []
    for item in results:
        text = item.get("document", "")
        pool = buckets.get(text)
        if not pool:
            continue
        order.append(pool.pop(0))
        scores.append(float(item.get("relevance_score", 0.0)))

    if not order:
        raise RerankError("返回的 document 无法与输入候选对应")
    return order, scores, data.get("usage", {}), elapsed


def rerank_candidates(question: str, candidates: list[dict], top_k: int = config.DELIVER_K):
    """对候选父块重排；返回 (前 top_k 个候选, 元信息)。

    元信息里 degraded=True 表示接口失败、用的是原顺序——调用方必须如实展示。
    """
    if not candidates:
        return [], {"degraded": False, "elapsed": 0.0, "error": "无候选"}

    docs = [c["parent"].text for c in candidates[:MAX_DOCS]]
    try:
        order, scores, usage, elapsed = rerank_documents(question, docs, top_n=top_k)
        picked = [candidates[i] for i in order[:top_k]]
        for c, s in zip(picked, scores):
            c["rerank_score"] = s
        return picked, {"degraded": False, "elapsed": elapsed, "usage": usage}
    except Exception as e:                      # 关键：任何异常都不能让问答中断
        return candidates[:top_k], {
            "degraded": True,
            "elapsed": 0.0,
            "error": f"{type(e).__name__}: {e}",
        }
