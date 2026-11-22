# -*- coding: utf-8 -*-
"""方案 B 的召回：Dense（向量）+ BM25（词面）→ RRF 融合。

为什么用 RRF 而不是把分数相加：
    dense 的分数是 exp(-L2²)，范围 0~1；BM25 是无上界实数（实测有 26.97 的）。
    两者加起来没有物理意义，只能比“名次”。
"""
from __future__ import annotations

import hashlib

from retrieval.backfill import Hit
from retrieval.bm25 import BM25UnitRetriever
from retrieval.store import api_key


class DenseRetriever:
    """直接查 Chroma 里已有的 1679 条向量，只算查询向量（进程内缓存，不落盘）。"""

    name = "dense"

    def __init__(self, units: list[dict], *, chroma_dir, collection: str,
                 embed_model: str, api_base: str):
        # 四个参数都必填：由 pipeline 从 Settings 传入，组件不再 import config。
        self.by_id = {u["id"]: u for u in units}
        self._chroma_dir = chroma_dir
        self._collection = collection
        self._embed_model = embed_model
        self._api_base = api_base
        self._qvec_cache: dict[str, list[float]] = {}
        self._coll = None
        self._embed = None
        # S3：把 client 也存下来，Engine 才能显式关闭它（以前是 _init 的局部变量）
        self._client = None
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DenseRetriever 已关闭，不能再使用")

    def prewarm(self) -> None:
        """提前连上 Chroma、建好 embedding 客户端。

        由 Engine.initialize() 调用：把"第一次提问才连库"的延迟提前到启动阶段，
        这样"检索器只构建一次"才是真的；缺 key / 缺库也会在启动时就暴露。
        """
        self._check_open()
        self._init()

    def close(self) -> None:
        """关闭本对象打开的 Chroma client；重复调用安全。

        注意只关"自己创建的" client：chromadb 按路径缓存 System，同一路径的
        另一个 Engine 若还活着，会被一起停掉——所以约定"一个数据目录一个 Engine"。
        """
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        self._coll = None
        self._embed = None
        self._qvec_cache.clear()
        if client is not None:
            client.close()

    def _init(self) -> None:
        self._check_open()
        if self._coll is not None:
            return
        import chromadb
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding

        embed = ZhipuAIEmbedding(
            model=self._embed_model,
            api_key=api_key(),
            api_base=self._api_base,
        )
        client = None
        try:
            client = chromadb.PersistentClient(path=str(self._chroma_dir))
            coll = client.get_collection(self._collection)
        except Exception:
            # 打开失败：把刚建好的半成品关掉，再原样抛出原异常
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            raise
        self._embed, self._client, self._coll = embed, client, coll

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
                 *, k: int, side: int):
        self.dense, self.bm25 = dense, bm25
        self.k = k
        self.base_side = side

    def prewarm(self) -> None:
        """转发给 Dense 一侧：BM25 在构造时已经建好索引，只有 Dense 需要预热。"""
        self.dense.prewarm()

    def close(self) -> None:
        """转发给 Dense 一侧：BM25 没有需要释放的外部资源。"""
        self.dense.close()

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
