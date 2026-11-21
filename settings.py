# -*- coding: utf-8 -*-
"""运行配置：一个不可变的 Settings 实例，替代"import 即定型"的模块常量。

为什么要有它（和 config.py 的区别）：
    旧 `config.py` 是模块级常量：import 时就把 `OUT.mkdir()` 执行掉，全进程
    只有一份，测试想指向别的数据目录只能改环境变量再 reload；值写错（比如
    `DELIVER_K = "abc"`）也要等到用它的那一刻才炸。

    Settings 是一个显式的对象：
      · 构造时校验，错就立刻抛 ConfigError（启动即失败，而不是请求时才失败）；
      · 可以造多份（测试用临时目录、评测分别建 bm25 / hybrid 两个引擎）；
      · 不碰磁盘、不读全局状态，`from_env()` 是唯一的"环境变量入口"。

    S1 阶段 `config.py` 会变成它的"别名层"（值全部来自这里），这样旧代码
    一行不用改，而数字只写一份。S2 迁移完成后 config.py 即可删除。

新增环境变量（都有默认值，不设也能跑）：
    RAG_DATA_DIR        数据资产目录（默认 ../rag_design）
    RAG_RETRIEVAL_MODE  bm25 | hybrid（默认 bm25）
    RAG_DELIVER_K       交付父块数（默认 10，必须 <= RAG_BACKFILL_TARGET）
    RAG_BACKFILL_TARGET 回填父块数（默认 30）
    RAG_LLM_MODEL       模型名（默认 glm-4.5-flash）
    RAG_LOG_LEVEL       DEBUG/INFO/WARNING/ERROR（默认 INFO）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from errors import ConfigError

# settings.py 自己就在项目根目录下，所以默认数据目录是"上一级的 rag_design"
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT.parent / "rag_design"

VALID_MODES = ("bm25", "hybrid")


def _env_str(env: dict, name: str, default: str) -> str:
    value = (env.get(name) or "").strip()
    return value or default


def _env_int(env: dict, name: str, default: int) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name} 必须是整数，收到 {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    """一次运行需要的全部参数。frozen=True：建好之后不允许被悄悄改动。"""

    # 路径
    data_dir: Path = _DEFAULT_DATA_DIR
    # 检索
    collection: str = "dense_sentences_v1"
    retrieval_mode: str = "bm25"
    steps: tuple[int, ...] = (30, 60, 120, 240)
    backfill_target: int = 30
    deliver_k: int = 10
    rrf_k: int = 60
    rrf_side: int = 50
    extra_stopwords: tuple[str, ...] = (
        "what", "which", "who", "whom", "whose", "when",
        "where", "why", "how", "do", "does", "did",
    )
    # 模型
    embed_model: str = "embedding-3"
    rerank_model: str = "rerank-pro"
    llm_model: str = "glm-4.5-flash"
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_timeout: int = 120
    llm_temperature: float = 0.1
    # 上下文长度
    context_parent_max_chars: int = 2600
    context_total_max_chars: int = 24000
    # 日志
    log_level: str = "INFO"

    # -- 校验 --------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.retrieval_mode not in VALID_MODES:
            raise ConfigError(
                f"RAG_RETRIEVAL_MODE 只能是 {VALID_MODES}，收到 {self.retrieval_mode!r}"
            )
        if not self.steps or any(not isinstance(s, int) or s <= 0 for s in self.steps):
            raise ConfigError(f"steps 必须是正整数序列，收到 {self.steps!r}")
        if tuple(sorted(self.steps)) != tuple(self.steps):
            raise ConfigError(f"steps 必须递增，收到 {self.steps!r}")
        if self.backfill_target <= 0:
            raise ConfigError(f"backfill_target 必须 > 0，收到 {self.backfill_target}")
        if not 1 <= self.deliver_k <= self.backfill_target:
            # 超过回填数时，交付会被候选数静默截断——不如启动就失败
            raise ConfigError(
                f"deliver_k 必须在 1..{self.backfill_target} 之间，收到 {self.deliver_k}"
            )
        if self.rrf_k <= 0 or self.rrf_side <= 0:
            raise ConfigError("rrf_k / rrf_side 必须 > 0")
        if self.llm_timeout <= 0:
            raise ConfigError("llm_timeout 必须 > 0")
        if self.context_parent_max_chars <= 0 or self.context_total_max_chars <= 0:
            raise ConfigError("context 长度上限必须 > 0")
        if self.context_parent_max_chars > self.context_total_max_chars:
            raise ConfigError("context_parent_max_chars 不能大于 context_total_max_chars")

    # -- 派生路径 ----------------------------------------------------------
    @property
    def out_dir(self) -> Path:
        """数据目录下的 out/（父块、句子、结果文件都在这里）。"""
        return self.data_dir / "out"

    @property
    def parents_path(self) -> Path:
        return self.out_dir / "parents.json"

    @property
    def units_path(self) -> Path:
        return self.out_dir / "units.json"

    @property
    def docstore_path(self) -> Path:
        return self.data_dir / "storage" / "docstore.json"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    # -- 构造 --------------------------------------------------------------
    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        """从环境变量构造；不传 env 时读 os.environ（测试可传 dict 注入）。"""
        env = dict(os.environ if env is None else env)
        data_dir = _env_str(env, "RAG_DATA_DIR", str(_DEFAULT_DATA_DIR))
        return cls(
            data_dir=Path(data_dir).expanduser().resolve(),
            retrieval_mode=_env_str(env, "RAG_RETRIEVAL_MODE", "bm25"),
            backfill_target=_env_int(env, "RAG_BACKFILL_TARGET", 30),
            deliver_k=_env_int(env, "RAG_DELIVER_K", 10),
            llm_model=_env_str(env, "RAG_LLM_MODEL", "glm-4.5-flash"),
            log_level=_env_str(env, "RAG_LOG_LEVEL", "INFO").upper(),
        )

    def with_mode(self, mode: str) -> "Settings":
        """换检索策略，其余参数不变（CLI 的位置参数 / 评测分模式跑题时用）。"""
        return replace(self, retrieval_mode=mode)

    def with_data_dir(self, path: str | Path) -> "Settings":
        return replace(self, data_dir=Path(path).expanduser().resolve())
