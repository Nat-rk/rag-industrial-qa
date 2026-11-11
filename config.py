# -*- coding: utf-8 -*-
"""全局配置：路径、检索策略、模型、候选数量。

改这里一个常量，就能切换检索策略 / 换模型 / 调候选数，不需要动其他文件。
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
#   BASE   : 本文件所在目录（项目根）
#   DESIGN : 数据资产目录，默认是项目同级的 ../rag_design
#            可以用环境变量 RAG_DATA_DIR 覆盖，避免把数据目录写死在代码里
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # 项目根目录
DESIGN = Path(os.environ.get("RAG_DATA_DIR") or (BASE.parent / "rag_design")).expanduser().resolve()
OUT = BASE / "out"                              # 本目录输出（日志、测试结果）
OUT.mkdir(exist_ok=True)

PARENTS = DESIGN / "out" / "parents.json"       # 278 个父块
UNITS = DESIGN / "out" / "units.json"           # 1679 个句子
DOCSTORE = DESIGN / "storage" / "docstore.json" # 父块元数据（含 file_name）
CHROMA = DESIGN / "chroma_db"                   # 向量库（集合 dense_sentences_v1）

COLLECTION = "dense_sentences_v1"

# ---------------------------------------------------------------------------
# 检索策略：改这一行就能切换
#   "bm25"   = 方案 A：BM25 → 回填 → Rerank → Top10
#   "hybrid" = 方案 B：Dense+BM25(RRF) → 回填 → Rerank → Top10
# ---------------------------------------------------------------------------
RETRIEVAL_MODE = "bm25"

# 两种策略共用的候选数量（沿用实验中验证过的设置，不要随意改）
STEPS = (30, 60, 120, 240)   # 自适应扩展：候选句子数
TOP_K_UNITS = 30             # 首轮召回的句子数
BACKFILL_TARGET = 30         # 回填后交给 rerank 的父块数
DELIVER_K = 10               # rerank 后最终交付的父块数
RRF_K = 60                   # Hybrid 融合参数
RRF_SIDE = 50                # Hybrid 每一路先取多少个句子

# BM25 停用词：bm25s 自带 33 个 + 12 个语法词（上一轮实测 MRR 0.707→0.796）
EXTRA_STOPWORDS = [
    "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "do", "does", "did",
]

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
EMBED_MODEL = "embedding-3"                       # 1024 维，与库里向量一致
RERANK_MODEL = "rerank-pro"
LLM_MODEL = "glm-4.5-flash"                       # 免费；想更强可换 "glm-4.6"
API_BASE = "https://open.bigmodel.cn/api/paas/v4"
LLM_TIMEOUT = 120
LLM_TEMPERATURE = 0.1

# ---------------------------------------------------------------------------
# 上下文长度控制
#   最长父块 2607 字（Table 9），单块上限要盖住它，否则大表会被截断
# ---------------------------------------------------------------------------
CONTEXT_PARENT_MAX_CHARS = 2600
CONTEXT_TOTAL_MAX_CHARS = 24000
