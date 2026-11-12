# 基于 RAG 的工业设备知识助手

项目状态：核心 RAG 问答与评测流程已完成，当前为单文档命令行版本。
面向 Danfoss VLT® AutomationDrive FC 302 技术文档，构建基于 RAG 的工业设备知识问答系统，支持设备参数查询、功能解释与故障排查，并提供文件名和页码引用，方便用户回溯原始资料。对于资料不足的问题，系统按照拒答约束处理。

回答里每个结论后面跟 `[Source n]`，n 对应到具体文件名与页码；资料里查不到时明确拒答，不靠模型自由发挥。

> 这是一个**单文档、单轮问答**项目，附带一套基于固定测试集的批量评测能力；模型推理与重排序依赖智谱 API，**不属于可独立运行的离线推理系统**。能力边界写在「已实现 / 未实现」一节，请不要按 Web 服务或 Agent 的预期来读它。

项目演示地址：https://www.bilibili.com/video/BV1MgWo6KEAX/?vd_source=326b9c922ec409b3f17224151543f18e
<img width="1252" height="649" alt="image" src="https://github.com/user-attachments/assets/afe7203a-5f28-4969-877c-ba01393e3669" />

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
| 批量问答评测（15 题 × 2 策略） | `eval/run_cases.py` |
| 公开评测报告生成（脱敏 + 严格引用口径） | `eval/reports/build_public_report.py` |
| 离线单元测试（47 项，不需要数据与 key） | `tests/` |
| Reranker 故障注入与降级验证 | `eval/test_rerank_failure.py`（需真实数据与 API） |

**未实现（不要当成已有能力）**

- 没有 FastAPI / Web 服务，只有命令行与交互模式
- 没有多轮对话、没有会话记忆
- 没有多文档管理，只挂 1 份手册
- 没有 Agent、没有自动检索策略路由
- 没有前端

---

## 3. 架构

```
问题 ──▶ 1 选策略 ──▶ 2 句子召回（两种策略只差这一步）
                        A: BM25 召回 30 句
                        B: Dense 50 句 + BM25 50 句 → RRF(k=60) → 取前 30 句
                    ──▶ 3 句子去重成父块，不足 30 个就扩到 60/120/240
                    ──▶ 4 rerank-pro 对 30 个父块重排 → 取前 10
                    ──▶ 5 拼上下文（[Source n] + 文件 + 页码 + 父块 ID）
                    ──▶ 6 LLM 生成（只依据上下文，查不到就拒答）
                    ──▶ 7 解析 [Source n] → 文件名 + 页码 + 父块 ID
```

第 3–7 步两条链路**完全共用同一份代码**：`main.py` 的 `ask()` 是唯一的编排处。

---

## 4. 目录结构

```text
.
├── main.py                  # 入口：命令行 / 交互模式；ask() 是整条链的骨架
├── config.py                # 路径、策略、模型、候选数、上下文长度
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
│   └── test_pipeline.py / test_run_cases_scoring.py
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

# 6. 离线自检（不需要数据、不需要 key、不联网）
python -m unittest discover -s tests -t .      # 47 项，约 2 秒
python eval/reports/build_public_report.py     # 从已有结果重新生成公开报告
```

> `.env.example` 只是变量清单，本项目**不加载** `.env`。请用系统环境变量或 IDE 的运行配置。

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

数据目录的解析逻辑只有 `config.py` 一处，可用环境变量覆盖：

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
| 离线单元测试（47 项，无网络/无 key） | `tests/` | 引用解析、上下文编号与截断、去重与自适应扩展、RRF 公式、降级路径、评测打分逻辑 | 真实召回质量与答案质量 |
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
6. **无自动路由**：BM25 / Hybrid 靠 `config.RETRIEVAL_MODE` 手选。
7. **中文提问**会让 BM25 近乎失效（倒排是英文 token），只能用 Hybrid；这一点未做多语言查询改写。

---

## 10. 版权与数据声明

本项目原创代码采用 MIT License。项目不包含 Danfoss 原始技术手册及其解析数据。Danfoss、VLT 等相关商标及第三方资料权利归其各自权利人所有。

- 仓库里的评测题只记录**问题**（英文提问）、权威页码和短语级答案要点，`eval/qa_cases.json` 不含手册正文段落。
- 运行所需的手册解析数据（`parents.json`、`units.json`、`docstore.json`、`chroma_db/`）需自行准备，不在本仓库分发。
- 使用本项目时，请自行确保对所用文档拥有合法使用权。
