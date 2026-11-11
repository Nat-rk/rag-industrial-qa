# -*- coding: utf-8 -*-
"""方案 A 的召回：BM25（纯词面，不调任何 API）。

索引对象是 1679 个句子，不是父块——这样和 Dense 用同一批检索单元，
Hybrid 融合时才能按 unit_id 对齐。
"""
from __future__ import annotations

import config
from retrieval.backfill import Hit


def _unit_nodes(units: list[dict]):
    """还原成 llama-index 的 TextNode，供 BM25Retriever 建倒排。

    关键点：把 metadata 全部排除在送检文本外（excluded_embed_metadata_keys），
    否则 BM25 会在 "page_number: 8" 这类元数据上打分，和 Dense 看到的正文不一致。
    """
    from llama_index.core.schema import TextNode

    nodes = []
    for u in units:
        meta = {
            "page_number": int(u["page"]),
            "file_name": "apply.pdf",
            "is_toc": "false",
            "kind": "sentence",
            "section": u["text"].split("\n", 1)[0],
            "parent_id": u["parent_id"],
            "junk": u["junk"],
        }
        node = TextNode(id_=u["id"], text=u["text"], metadata=meta)
        node.excluded_embed_metadata_keys = list(meta.keys())
        node.excluded_llm_metadata_keys = ["parent_id", "junk"]
        nodes.append(node)
    return nodes


class BM25UnitRetriever:
    """索引 1679 个句子；停用词 = bm25s 自带 33 词 + 12 个语法词。"""

    name = "bm25"

    def __init__(self, units: list[dict]):
        from bm25s.tokenization import STOPWORDS_EN
        from llama_index.retrievers.bm25 import BM25Retriever

        self.by_id = {u["id"]: u for u in units}
        self.language = sorted(set(STOPWORDS_EN) | set(config.EXTRA_STOPWORDS))
        self.retriever = BM25Retriever.from_defaults(
            nodes=_unit_nodes(units),
            similarity_top_k=max(config.STEPS),   # 一次取够，扩展时不用重建索引
            language=self.language,
        )

    def retrieve(self, qtext: str, n: int) -> list[Hit]:
        hits = []
        for nws in self.retriever.retrieve(qtext)[:n]:
            u = self.by_id.get(nws.node.node_id)
            if not u:
                continue
            hits.append(Hit(nws.node.node_id, u["parent_id"], int(u["page"]),
                            u["text"], float(nws.score)))
        return hits
