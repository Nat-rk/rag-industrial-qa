# -*- coding: utf-8 -*-
"""全局配置的**别名层**（过渡期）：所有数字的唯一真源是 `settings.Settings`。

为什么保留这个文件：
    47 项测试、`eval/` 的 4 个脚本、`docs/` 里的示例都在 `import config`。
    S1 的目标是先把"配置"变成可注入的对象，而不是一次性改掉所有调用方。
    所以这里不再自己定义任何数值，只做两件事：
      1. `Settings.from_env()` 造出本次运行唯一的配置实例；
      2. 把实例字段映射回旧名字（`DELIVER_K = _S.deliver_k` ……）。

    S2 完成迁移（`create_app(settings)` / `ask(settings)`）后，本文件删除。

注意：`OUT.mkdir()` 已从本文件移走（S2）。现在"import 配置"不写磁盘，
需要输出目录的入口（`main.py`、`eval/` 三个脚本）自己建目录。

**但 import 仍会读一次环境变量**（第 22 行 `Settings.from_env()`），这是本文件
剩余的副作用，也是它必须被删除的原因：
    * `import config` 之后再改 `os.environ`，`config.DELIVER_K` 等不会跟着变；
    * 同一个进程里先后 import 两次，拿到的仍是第一次的快照；
    * 测试若想在不同环境变量下反复验证，只能 `importlib.reload(config)`，
      而 reload 又会重新读一遍环境——不如直接用 `Settings.from_env()` 构造。
新代码请走 `Settings`，把实例传下去；`config` 只服务于尚未迁移的旧调用方。
"""
import os
from pathlib import Path

from settings import Settings

# 本次进程唯一的配置对象；环境变量在这里被读一次
_S = Settings.from_env()

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # 项目根目录
DESIGN = _S.data_dir                            # 数据资产目录（RAG_DATA_DIR 可覆盖）
OUT = BASE / "out"                              # 本目录输出（日志、测试结果）

PARENTS = _S.parents_path                       # 278 个父块
UNITS = _S.units_path                           # 1679 个句子
DOCSTORE = _S.docstore_path                     # 父块元数据（含 file_name）
CHROMA = _S.chroma_dir                          # 向量库（集合 dense_sentences_v1）

COLLECTION = _S.collection

# ---------------------------------------------------------------------------
# 检索策略与候选数量（改环境变量即可，不必改代码）
#   RAG_RETRIEVAL_MODE=bm25 | hybrid
#   RAG_DELIVER_K / RAG_BACKFILL_TARGET
# ---------------------------------------------------------------------------
RETRIEVAL_MODE = _S.retrieval_mode
STEPS = _S.steps                                # 自适应扩展：候选句子数
BACKFILL_TARGET = _S.backfill_target            # 回填后交给 rerank 的父块数
DELIVER_K = _S.deliver_k                        # rerank 后最终交付的父块数
RRF_K = _S.rrf_k                                # Hybrid 融合参数
RRF_SIDE = _S.rrf_side                          # Hybrid 每一路先取多少个句子

# BM25 停用词：bm25s 自带 33 个 + 12 个语法词（实测 MRR 0.707→0.796）
EXTRA_STOPWORDS = list(_S.extra_stopwords)

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
EMBED_MODEL = _S.embed_model                    # 1024 维，与库里向量一致
RERANK_MODEL = _S.rerank_model
LLM_MODEL = _S.llm_model
API_BASE = _S.api_base
LLM_TIMEOUT = _S.llm_timeout
LLM_TEMPERATURE = _S.llm_temperature

# ---------------------------------------------------------------------------
# 上下文长度控制（最长父块 2607 字，单块上限要盖住它）
# ---------------------------------------------------------------------------
CONTEXT_PARENT_MAX_CHARS = _S.context_parent_max_chars
CONTEXT_TOTAL_MAX_CHARS = _S.context_total_max_chars

# 日志级别（RAG_LOG_LEVEL，默认 INFO）；入口调用 setup_logging 时使用
LOG_LEVEL = _S.log_level
