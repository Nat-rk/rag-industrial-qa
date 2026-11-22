# 基于 RAG 的工业设备知识助手

项目状态：核心 RAG 问答、评测与 HTTP 服务入口均已完成；当前为**单文档**版本，提供 CLI 与 FastAPI 两种入口（HTTP 层已在 S4 阶段六完成真实验收）。
面向 Danfoss VLT® AutomationDrive FC 302 技术文档，构建基于 RAG 的工业设备知识问答系统，支持设备参数查询、功能解释与故障排查，并提供文件名和页码引用，方便用户回溯原始资料。对于资料不足的问题，系统按照拒答约束处理。

回答里每个结论后面跟 `[Source n]`，n 对应到具体文件名与页码；资料里查不到时明确拒答，不靠模型自由发挥。

> 这是一个**单文档、单轮问答**项目，附带一套基于固定测试集的批量评测能力；模型推理与重排序依赖智谱 API，**不属于可独立运行的离线推理系统**。
> HTTP 入口只做“把 `pipeline.run()` 包成 3 个接口”，固定**单进程、单 worker、单 Engine**，不面向公网、没有认证与限流。
> 能力边界写在「已实现 / 未实现」一节，请不要按多用户 Web 服务或 Agent 的预期来读它。

项目演示地址：https://www.bilibili.com/video/BV1MgWo6KEAX/?vd_source=326b9c922ec409b3f17224151543f18e
---

## 1. 解决什么问题

工业设备手册通常 100 页以上，参数表、选项值、报警码散落在正文与表格中。工程师现场排查时的典型问题是：

- “定位模式和同步模式是由哪个参数选的？可选值是什么？”
- “参数 8-10 设成 [3] 时，控制字每一位是什么含义？”
- “正向找原点时先碰到限位开关会怎样？报警码多少？”

这类问题的难点不在“语义相似”，而在**精确匹配 + 完整表格 + 可核对的出处**：参数号不能错、整张表不能只剩一半、答案必须能翻到那一页去看。

---

## 2. 已实现 / 未实现

**已实现（有代码、有测试）**

| 能力 | 位置 |
|---|---|
| 句子级检索 + 父块回填 | `retrieval/backfill.py` |
| BM25 关键词检索（含 12 个语法停用词） | `retrieval/bm25.py` |
| Dense 向量检索 | `retrieval/hybrid.py` |
| Hybrid 混合检索 + RRF 融合 | `retrieval/hybrid.py` |
| Reranker 重排序，失败自动降级 | `retrieval/reranker.py` |
| 上下文构建（带 `[Source n]`、文件名、页码、父块 ID） | `qa/context_builder.py` |
| LLM 答案生成 + 拒答约束 | `qa/generator.py` |
| 引用映射与越界编号检查 | `qa/citation.py` |
| 启动初始化一次、多请求复用（`RagEngine`） | `service/engine.py` |
| HTTP 服务：`/ask`、`/healthz`、`/readyz` | `api/`（`app.py` / `routes.py` / `schemas.py` / `errors.py`） |
| 单进程内问答串行（应用级锁） | `api/routes.py` |
| 批量问答评测（15 题 × 2 策略） | `eval/run_cases.py` |
| 公开评测报告生成（脱敏 + 严格引用口径） | `eval/reports/build_public_report.py` |
| 离线单元测试（156 项，不需要数据与 key，需先装依赖） | `tests/` |
| Reranker 故障注入与降级验证 | `eval/test_rerank_failure.py`（需真实数据与 API） |

**未实现（不要当成已有能力）**

- HTTP 层只有 3 个接口：没有认证、没有 CORS、没有限流、没有前端、没有 SSE/流式
- 不支持多进程 / 多 worker（同一 `chroma_db` 目录只允许一个进程使用）
- 没有多轮对话、没有会话记忆
- 没有多文档管理，只挂 1 份手册
- 没有 Agent、没有自动检索策略路由（模式由启动配置决定）

---

## 3. 架构

两条入口，共用同一段业务代码：

```text
CLI：python main.py "问题" [bm25|hybrid]
        │
        ▼
HTTP Client ──▶ FastAPI（api/app.py 的 create_app + lifespan）
        │            ├─ 请求校验（api/schemas.py，失败 → 422）
        │            ├─ 路由（api/routes.py：/ask、/healthz、/readyz）
        │            └─ 异常映射（api/errors.py：503 / 500 / 422；502 映射保留）
        ▼
   RagEngine（仅 HTTP 路径；启动时 initialize 一次，之后复用）
        ▼
   qa/pipeline.py 的 run(question, settings=…)
```

`pipeline.run()` 内部固定 8 步：

```
1 选策略 ──▶ 2 句子召回（两种策略只差这一步）
                A: BM25 召回 30 句
                B: Dense 50 句 + BM25 50 句 → RRF(k=60) → 取前 30 句
          ──▶ 3 句子去重成父块，不足 30 个就扩到 60/120/240
          ──▶ 4 rerank-pro 对 30 个父块重排 → 取前 10
          ──▶ 5 拼上下文（[Source n] + 文件 + 页码 + 父块 ID）
          ──▶ 6 LLM 生成（只依据上下文，查不到就拒答）
          ──▶ 7 解析 [Source n] → 文件名 + 页码 + 父块 ID
```

第 3–7 步两条链路**完全共用同一份代码**。区别只在“谁来编排”：CLI 由 `main.py` 的 `ask()`
直接调 `pipeline.run()`（每次启动重新加载数据与检索器）；HTTP 由 `api/routes.py` 调
`RagEngine.ask()`（数据与检索器在启动时初始化一次，之后每个请求复用）。

---

## 4. 目录结构

```text
.
├── main.py                  # CLI 入口：参数 / 交互 / 打印；只解析参数、配日志
├── qa/pipeline.py           # 8 步流程的真正实现：run(question, settings=...)
├── settings.py              # 配置对象 Settings：默认值、环境变量、构造即校验
├── errors.py                # 统一异常：ConfigError / DataError / LLMError / RerankError
├── logging_setup.py         # 日志一次配置 + trace_id（每次问答一个短 id）
├── api/                     # HTTP 入口（S4）：只做协议转换，不碰检索与生成
│   ├── app.py               # create_app(settings) + lifespan：建 Engine、关闭释放
│   ├── main.py              # uvicorn 入口：api.main:app（--workers 1）
│   ├── routes.py            # /ask /healthz /readyz；/ask 用应用级锁串行
│   ├── schemas.py           # 请求/响应模型：AskRequest、AskResponse（14 键）
│   └── errors.py            # 异常 → HTTP 状态码 + 统一错误外壳，不含业务逻辑
├── retrieval/
│   ├── store.py             # 读 parents / units / docstore，补出 file_name
│   ├── bm25.py              # 方案 A 的词面召回（本地 BM25，不调 API）
│   ├── hybrid.py            # Dense 检索 + RRF 融合
│   ├── backfill.py          # 句子 → 父块去重、自适应扩展
│   └── reranker.py          # rerank-pro 封装；失败保序并标 degraded
├── qa/
│   ├── context_builder.py   # 父块 → 带编号的上下文，超长按名次截断
│   ├── generator.py         # Prompt 约束 + LLM 调用
│   └── citation.py          # 引用解析、越界检查、来源表格式化
├── service/
│   └── engine.py            # RagEngine：启动时加载一次数据与检索器，多次问答复用
├── eval/
│   ├── qa_cases.json        # 15 道验收题：12 道有答案 + 3 道资料不足
│   ├── run_cases.py         # 批量问答，逐题落盘，可断点续跑
│   ├── test_rerank_failure.py  # 5 类故障注入 + 正常对照（需真实数据与 API）
│   ├── compare_runs.py      # 两次结果逐题对比
│   └── reports/             # 可公开的评测证据（脱敏，可直接看）
│       ├── build_public_report.py   # 从历史 JSON 生成下面三份
│       ├── evaluation.md            # 人读报告：口径、汇总、逐题、失败案例、局限
│       ├── evaluation_details.csv   # 逐题明细（可自行重算指标）
│       └── evaluation_summary.json  # 机器可读汇总
├── tests/                   # 离线单元测试：不需要数据、不需要 key
│   ├── fixtures/sample_device_docs.json   # 虚构手册片段（非真实文档）
│   ├── test_citation.py / test_context.py / test_backfill.py
│   ├── test_rrf.py / test_bm25.py / test_rerank_degrade.py
│   ├── test_pipeline.py / test_run_cases_scoring.py
│   ├── test_settings.py / test_errors.py / test_logging_setup.py
│   ├── test_pipeline_module.py   # S2：Settings 注入、trace 恢复、patch 有效性
│   ├── test_engine.py            # S3：初始化一次、资源复用、三状态生命周期
│   ├── test_dense_retriever.py   # S3：prewarm / close 的真实 Chroma 生命周期
│   ├── test_api.py               # S4：应用生命周期（FakeEngine + mock api_key）
│   └── test_api_routes.py        # S4：校验/响应/错误映射/串行化/约束断言
├── docs/
│   ├── 架构与模块说明.md      # 每个模块解决什么问题、改哪里
│   └── 验收报告.md           # 评测方法、指标口径、结果、限制
├── out/                     # 运行产物（已 gitignore，仅保留 .gitkeep）
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 5. 快速开始

前提：**开发及实际验证环境为 Python 3.13.9**；其他版本尚未逐一验证。

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 准备数据资产（见下一节），放到 ../rag_design/，或用 RAG_DATA_DIR 指定

# 3. 配置 key（不要写进代码）
#    Windows PowerShell:
$env:ZHIPU_API_KEY = "你的key"
#    macOS / Linux:
export ZHIPU_API_KEY="你的key"

# 4. 提问
python main.py "Which parameter selects Positioning or Synchronization mode?"
python main.py "..." hybrid        # 换 Hybrid 策略
python main.py                     # 交互模式

# 5. 批量评测（需要数据 + key，会产生费用）
python eval/run_cases.py                       # 15 题 × 2 策略
python eval/run_cases.py --ids Q04 N03 --modes bm25
python eval/test_rerank_failure.py             # 故障注入（12 次）

# 6. 离线自检（不需要数据、不需要 key、不联网——但依赖仍要先装好）
python -m unittest discover -s tests -t .      # 156 项，约 14 秒（含真实 Chroma 用例）
python eval/reports/build_public_report.py     # 从已有结果重新生成公开报告
```

> `.env.example` 只是变量清单，本项目**不加载** `.env`。请用系统环境变量或 IDE 的运行配置。
>
> 关于“离线”：指的是**不需要数据资产、不需要 API key、不联网**，不等于“不需要安装依赖”——
> 先执行 `pip install -r requirements.txt`。没装依赖时 `unittest discover` 会出现
> `ModuleNotFoundError`（例如缺 `requests` / `llama-index-*`），那是环境没准备好，不是测试失败。

### 5.1 HTTP 服务（可选入口）

前提同上：依赖已装、数据已放、`ZHIPU_API_KEY` 已配。

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

- 默认只绑本机回环地址，`/docs` 本地可用，不面向公网。
- **必须 `--workers 1`**：每个 worker 会各建一个 `RagEngine`、各开一次 `chroma_db`；同一数据目录不支持多进程共享。
- 启动时 `initialize()` 一次（读 278 父块 + 1679 句 + 建检索器；实测 bm25 约 2 s、hybrid 约 4 s），之后请求复用同一个 Engine；关闭时 `engine.close()` 释放 Chroma 客户端。

| 方法 | 路径 | 作用 | 就绪条件 |
|---|---|---|---|
| POST | `/ask` | 工业设备知识问答 | 仅 `ready` 时可用，否则 503 |
| GET | `/healthz` | 存活检查（进程能响应就是 200） | 无 |
| GET | `/readyz` | 就绪检查 | Engine `ready` 才 200，否则 503 `not_ready` |

`/ask` 请求与响应（响应只展示关键字段，实际返回 14 键，与 `api/schemas.py` 一致）：

```bash
curl -s http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Which parameter selects Positioning or Synchronization mode?"}'
```

```json
{
  "question": "Which parameter selects Positioning or Synchronization mode?",
  "mode": "bm25",
  "answer": "参数 1-00 Configuration Mode 用于选择定位或同步模式，对应选项为 [9] Positioning 与 [10] Synchronization [Source 2]。",
  "citations": [
    {"n": 2, "file": "apply.pdf", "page": 8, "parent_id": "…", "short_id": "…", "section": "2.1 An Overview of IMC"}
  ],
  "citations_marked": true,
  "invalid_citations": [],
  "timing": {"build": 1.9, "recall": 0.01, "rerank": 1.3, "llm": 15.6}
}
```

完整响应固定 14 个键（与 `api/schemas.py` 的 `AskResponse` 一一对应）：

| 字段 | 含义 |
|---|---|
| `question` | 去掉首尾空白后的原始问题 |
| `mode` | 本次实际使用的策略（`bm25` / `hybrid`，来自启动配置） |
| `candidates` | 回填得到的候选父块（`parent` 为展开对象，含 `unit_text` / `score` / `rerank_score`） |
| `trace` | 自适应回填的每一步：候选句子数、去重后的父块数 |
| `picked` | 最终交付的父块（rerank 之后、按顺序取前 10） |
| `rerank` | 是否降级、模型名、耗时、错误信息（正常时 `degraded=false`） |
| `sources` | 上下文里每个 `[Source n]` 的元信息（n / 文件 / 页 / 父块 ID / 章节 / 字数） |
| `context` | 实际拼给 LLM 的完整上下文文本（最长 24000 字） |
| `answer` | 模型答案，行内带 `[Source n]` |
| `usage` | LLM 的 token 用量（`prompt` / `completion`，缺失时为 `null`） |
| `citations` | 解析后的引用列表（文件 / 页 / 父块 ID / 章节；**没有 `chars`**，与 `sources` 不同） |
| `citations_marked` | 答案里是否出现了引用标记（布尔值，不是字符串） |
| `invalid_citations` | 模型引用、但上下文里不存在的编号 |
| `timing` | 各阶段耗时（`build` / `recall` / `rerank` / `llm`，秒） |

错误响应（所有错误共用一个外壳）：
```json
{"error": {"code": "data_error", "message": "读取数据文件失败", "trace_id": "d22945d9"}}
```

```json
{"error": {"code": "validation_error", "message": "question 不能为空", "trace_id": null}}
```

要点：

- 问题长度 1–2000 字，空白会被拒绝（422）。
- 检索模式由**启动时的** `RAG_RETRIEVAL_MODE` 决定；请求里传 `mode`/`top_k` 会被 422 拒绝（不按请求切换）。
- 请求在单进程内**排队串行**：同一时刻只跑一个 `engine.ask()`；排队没有超时，慢请求会占住后面的请求。
- 错误码：422 参数不合法、503 未就绪、500 数据或未预期错误（不返回堆栈）。**当前 `/ask` 不会返回 502**：LLM 失败被 `pipeline.run()` 捕获成 200 + 答案“（生成失败：…）”，rerank 失败降级成 200 + `rerank.degraded=true`；`api/errors.py` 仍保留 `LLMError`/`RerankError` → 502 的映射，供以后把错误上抛的路径使用。
- 接口文档：启动后访问 http://127.0.0.1:8000/docs （Swagger UI）与 http://127.0.0.1:8000/redoc 。

---

## 6. 数据准备（重要）

本仓库**不包含**技术手册 PDF、解析文本、切片和向量库——它们体积大，且属于第三方版权内容。运行前需要自己准备 4 份资产，放在同一个目录下：

```text
<数据目录>/
├── out/
│   ├── parents.json        # 父块：交付给 LLM 的单位
│   └── units.json          # 句子：真正被检索的单位
├── storage/
│   └── docstore.json       # 父块元数据（引用要用的 file_name 在这里）
└── chroma_db/              # Chroma 持久化目录，集合名 dense_sentences_v1
```

字段约定（与 `retrieval/store.py` 一一对应）：

| 文件 | 结构 |
|---|---|
| `parents.json` | `[{"id","page","kind","section","text","chars"}]`，`kind ∈ {section, table, toc}` |
| `units.json` | `[{"id","parent_id","page","junk","chars","text"}]`，`junk` 为脏句标记 |
| `docstore.json` | llama-index 文档存储格式：`{"docstore/data": {node_id: {"__data__": {...,"metadata":{...}}}}}` |
| `chroma_db` | 集合 `dense_sentences_v1`，每条向量 1024 维（`embedding-3`），metadata 含 `page_number / file_name / is_toc / kind / section / parent_id` |

参考规模：278 个父块（207 段落 + 57 表格 + 14 目录）、1679 个句子（其中 186 条标记为脏句）。

**这些文件怎么来？** 由上游离线流水线生成（PDF → 按章节切块 → 切句 → 向量化），该流水线不在本仓库内。如果你只想要“跑起来看效果”，最小路径是：

1. 用自己的 PDF 解析 + 章节切块产出 `parents.json`；
2. 按父块切句产出 `units.json`，并用相同规则构建 `docstore.json` 的 metadata；
3. 用同一个 embedding 模型（`embedding-3`，1024 维）写入 Chroma 集合 `dense_sentences_v1`。

数据目录的解析逻辑只有 `settings.py` 一处（`Settings.from_env()`），可用环境变量覆盖：

```bash
export RAG_DATA_DIR=/path/to/your/data
```

---

## 7. 评测

15 道题：12 道手册里有明确答案，3 道手册里确实没有（用来测“会不会编”）。两种策略各跑一遍，共 30 次。

| 指标 | bm25 | hybrid |
|---|---:|---:|
| 有答案题：检索命中权威页 | 12/12 | 12/12 |
| 有答案题：引用命中权威页（宽松，有交集） | 11/12 | 11/12 |
| 有答案题：严格引用匹配（引用页集合完全相同） | 8/12 | 8/12 |
| 资料不足题：正确拒答 | 3/3 | 3/3 |
| 平均交付父块数 | 10.0 | 10.0 |
| 平均耗时 | 24.3 s | 37.3 s |
| 降级次数 / 生成失败 | 0 / 0 | 0 / 0 |

- “检索命中”= 权威父块进了交付的前 10 块；“引用命中”= 答案里引用的 `[Source n]` 落到了权威页。**两者不是一回事**，不要混用。
- “严格引用匹配”是更严的口径：引用页集合必须和权威页集合**相同**，两种策略都是 8/12。按策略看，BM25 没通过的是 Q04（多引 p16）、Q06（漏引 p32）、Q10（多引 p20/p36/p40）、Q20（引到同值但不同场景的 p49）；Hybrid 没通过的是 Q06、Q10（多引 p48/p56）、Q19（多引 p48）、Q20。逐题“多引/漏引了哪页”见公开报告第 5 节。
- Reranker 故障注入：连接异常 / 超时 / 500 / 429 / 空响应 5 类 × 2 策略 + 2 次正常对照，12/12 通过。**它证明的是“降级逻辑可用”（保序、页码映射不丢、问答不中断、引用不越界），不证明降级后答案内容正确。**
- 唯一未命中的引用是 Q20：两张不同应用场景的表数值相同，答案取到了另一张。详见 `eval/reports/evaluation.md` 第 5 节与 `docs/验收报告.md`。

### 7.1 三类证据，不要混为一谈

| 证据 | 位置 | 能证明 | 不能证明 |
|---|---|---|---|
| 历史评测（2026-11-06，15 题 × 2 策略） | `eval/reports/evaluation.md` / `.csv` / `.json` | 那一轮的检索、引用、拒答与耗时 | 当前代码在所有数据上的表现 |
| 当前代码抽检（Q04 + Q20 × 2 策略，4 次） | 同上第 6 节 | 这 4 次的检索可复现（权威页 4/4 在交付里）；引用有运行间波动 | 30 行历史结果都来自当前 HEAD |
| 离线单元测试（156 项，无网络/无 key，需先装依赖） | `tests/` | 引用解析、上下文编号与截断、去重与自适应扩展、RRF 公式、降级路径、评测打分逻辑、Engine 生命周期、HTTP 接口行为（Fake Engine） | 真实召回质量、答案质量、真实 LLM/Chroma 的端到端行为 |
- 本轮完整记录（30 行原始 JSON，含模型原文与手册章节标题）按约定**不随仓库发布**；可公开的脱敏版本见 [`eval/reports/evaluation.md`](eval/reports/evaluation.md)（同目录还有逐题明细 CSV 与生成脚本，指标可逐项重算）。

指标口径、逐题结果与已知失败案例见 [`eval/reports/evaluation.md`](eval/reports/evaluation.md) 与 [`docs/验收报告.md`](docs/验收报告.md)。

---

## 8. 几个刻意的设计

| 设计 | 为什么 |
|---|---|
| 检索用句子、交付用父块 | 句子定位准，但表格/步骤必须整块给模型，否则答案缺一半 |
| 两者用 RRF 而非分数相加 | dense 是 `exp(-L2²)`（0–1），BM25 无上界（实测 26.97），相加没有物理意义 |
| rerank 失败保持原序并标 `degraded` | 一次 429 不该让问答崩掉，但也不能假装没发生 |
| 拒答靠 Prompt，不设分数阈值 | rerank 分数跨题不可比（0.998 与 0.651 都出现过），阈值只会误杀 |
| 单块上下文上限 2600 字 | 最长父块 2607 字，再小会把大表截断 |
| 查询向量只放内存 | 每题算一次，落盘反而会写脏实验目录 |

---

## 9. 已知限制

1. **只验证过一份文档**（76 页手册），多文档/多领域未验证。
2. **评测集 15 题，规模小**；其中 12 题来自更早的 27 题实验（9 dev + 3 test），可视为回归集，不能当作泛化能力的证据。
3. **引用到“父块/页”粒度**，不是句子级；同一页多块时页码相同、块不同。
4. **答案质量受 LLM 影响**：`glm-4.5-flash` 免费够用，复杂参数表偶有串行（Q04 案例）。
5. **相似表竞争**未根治：数值相同的不同场景表，rerank 分不开（Q20）。
6. **无自动路由**：BM25 / Hybrid 靠启动时的 `RAG_RETRIEVAL_MODE` 手选，不按请求切换。
7. **中文提问**会让 BM25 近乎失效（倒排是英文 token），只能用 Hybrid；这一点未做多语言查询改写。
8. **HTTP 层是单进程的**：`--workers 1` + 进程内锁，排队没有超时；没有认证、CORS、限流，不适合直接暴露到公网。
9. **真实 HTTP 端到端已在阶段六跑通**（2026-11-22）：真实 uvicorn + 真实 Engine + 真实 LLM，`POST /ask` 的 Q04 返回 200/14 键/36 s，拒答 N01 如实回答“没找到足够信息”；接口测试里的 Engine 仍是 Fake，不代表真实 RAG 质量；高并发与大响应耗时未验证。

---

## 10. 版权与数据声明

本项目原创代码采用 MIT License。项目不包含 Danfoss 原始技术手册及其解析数据。Danfoss、VLT 等相关商标及第三方资料权利归其各自权利人所有。

- 仓库里的评测题只记录**问题**（英文提问）、权威页码和短语级答案要点，`eval/qa_cases.json` 不含手册正文段落。
- 运行所需的手册解析数据（`parents.json`、`units.json`、`docstore.json`、`chroma_db/`）需自行准备，不在本仓库分发。
- 使用本项目时，请自行确保对所用文档拥有合法使用权。
