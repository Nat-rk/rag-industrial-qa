# -*- coding: utf-8 -*-
"""方案 B 的召回：Dense（向量）+ BM25（词面）→ RRF 融合。

为什么用 RRF 而不是把分数相加：
    dense 的分数是 exp(-L2²)，范围 0~1；BM25 是无上界实数（实测有 26.97 的）。
    两者加起来没有物理意义，只能比“名次”。
"""
from __future__ import annotations

import hashlib

import config
from retrieval.backfill import Hit
from retrieval.bm25 import BM25UnitRetriever
from retrieval.store import api_key


class DenseRetriever:
    """直接查 Chroma 里已有的 1679 条向量，只算查询向量（进程内缓存，不落盘）。"""

    name = "dense"

    def __init__(self, units: list[dict], *, chroma_dir=None, collection: str | None = None,
                 embed_model: str | None = None, api_base: str | None = None):
        # 四个参数都由 pipeline 从 Settings 传进来；不传则退回 config（过渡期兼容）。
        self.by_id = {u["id"]: u for u in units}
        self._chroma_dir = chroma_dir if chroma_dir is not None else config.CHROMA
        self._collection = collection if collection is not None else config.COLLECTION
        self._embed_model = embed_model if embed_model is not None else config.EMBED_MODEL
        self._api_base = api_base if api_base is not None else config.API_BASE
        self._qvec_cache: dict[str, list[float]] = {}
        self._coll = None
        self._embed = None

    def _init(self):
        if self._coll is not None:
            return
        import chromadb
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

        self._embed = ZhipuAIEmbedding(
            model=self._embed_model,
            api_key=api_key(),
            api_base=self._api_base,
        )
        client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._coll = client.get_collection(self._collection)

    def _vector(self, qtext: str) -> list[float]:
        key = hashlib.sha1(qtext.encode("utf-8")).hexdigest()
        if key not in self._qvec_cache:
            self._init()
            self._qvec_cache[key] = list(self._embed.get_query_embedding(qtext))
        return self._qvec_cache[key]

    def retrieve(self, qtext: str, n: int) -> list[Hit]:
        self._init()
        res = self._coll.query(
            query_embeddings=[self._vector(qtext)], n_results=n,
            include=["distances", "metadatas"],
        )
        hits = []
        for nid, dist in zip(res["ids"][0], res["distances"][0]):
            u = self.by_id.get(nid)
            if not u:
                continue
            hits.append(Hit(nid, u["parent_id"], int(u["page"]), u["text"],
                            pow(2.718281828459045, -float(dist))))
        return hits


class HybridRetriever:
    """两路各取 side 个句子 → RRF → 取前 n 个句子。

    RRF 公式：score(unit) = Σ 1 / (k + rank)，k 默认 60。
    同一句话被两路都召到时，两个名次分相加，所以会排得更靠前。
    """

    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25UnitRetriever,
                 *, k: int | None = None, side: int | None = None):
        self.dense, self.bm25 = dense, bm25
        self.k = config.RRF_K if k is None else k
        self.base_side = config.RRF_SIDE if side is None else side

    def retrieve(self, qtext: str, n: int) -> list[Hit]:
        side = self.base_side if n <= self.base_side else n
        left = self.dense.retrieve(qtext, side)
        right = self.bm25.retrieve(qtext, side)

        fused: dict[str, dict] = {}
        for rank, h in enumerate(left, start=1):
            fused[h.unit_id] = {"hit": h, "score": 1.0 / (self.k + rank)}
        for rank, h in enumerate(right, start=1):
            slot = fused.setdefault(h.unit_id, {"hit": h, "score": 0.0})
            slot["score"] += 1.0 / (self.k + rank)

        ordered = sorted(fused.values(), key=lambda x: -x["score"])[:n]
        return [Hit(v["hit"].unit_id, v["hit"].parent_id, v["hit"].page,
                    v["hit"].text, v["score"]) for v in ordered]
