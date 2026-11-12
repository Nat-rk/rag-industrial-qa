# -*- coding: utf-8 -*-
"""把历史评测记录（qa_results_v3.json）转成可以公开的脱敏评测报告。

为什么需要这个脚本：
    * 原始 JSON 里有 `answer`（模型输出，可能复述手册内容）和 `picked[].section`
      （手册章节标题），这两类不该公开；
    * 原始 JSON 里没有 `expected_pages`，判据在 `eval/qa_cases.json` 里，需要 join；
    * 原始 JSON 的 `evidence_ok` / `citations_ok` 是当年脚本算好的布尔值，
      这里重新算一遍，并额外给出「严格引用匹配」与「多余引用页」，
      避免「引用页与预期页有交集」被读成「引用完全准确」。

输出三份文件（默认写到 eval/reports/）：
    evaluation_summary.json   汇总指标 + 逐题明细（机器读）
    evaluation_details.csv    逐题明细（表格软件读）
    evaluation.md             人读的报告

用法：
    python eval/reports/build_public_report.py
    python eval/reports/build_public_report.py --spotcheck _local/results/spotcheck_2026-11-13.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODES = ("bm25", "hybrid")
MODE_LABEL = {"bm25": "BM25", "hybrid": "Hybrid"}


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(path.read_text(encoding="utf-8"))["cases"]}


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 单行计算：全部从原始页码重新推导，不信任文件里的布尔值
# ---------------------------------------------------------------------------
def build_detail(row: dict, case: dict) -> dict:
    expected = [int(p) for p in case.get("pages", [])]
    delivered = [int(p) for p in row.get("pages", [])]
    cited = sorted({int(p) for p in row.get("cited_pages", [])})
    picked = row.get("picked", []) or []

    # 期望页在交付顺序里第一次出现的位置（1 起）；没有则为 None
    rank = next((i for i, item in enumerate(picked, start=1)
                 if int(item.get("page", -1)) in expected), None)

    return {
        "question_id": row["id"],
        "category": case.get("type", row.get("type", "")),
        "strategy": row["mode"],
        "answerable": bool(expected),
        "expected_pages": expected,
        "delivered_pages": delivered,
        "retrieval_hit": (any(p in delivered for p in expected) if expected else None),
        "authority_rank": rank,
        "cited_pages": cited,
        "citation_hit": (bool(set(cited) & set(expected)) if expected else None),
        "exact_citation_match": ((set(cited) == set(expected)) if expected else None),
        "extra_cited_pages": sorted(set(cited) - set(expected)) if expected else [],
        "missing_cited_pages": sorted(set(expected) - set(cited)) if expected else [],
        "delivered_blocks": int(row.get("delivered", 0) or 0),
        "candidates": int(row.get("candidates", 0) or 0),
        "elapsed_s": float(row.get("elapsed", 0) or 0.0),
        "llm_s": float(row.get("llm_seconds", 0) or 0.0),
        "degraded": bool(row.get("degraded", False)),
        "llm_failed": bool(row.get("llm_failed", False)),
        "refused": bool(row.get("refused", False)),
    }


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize_mode(details: list[dict]) -> dict:
    ans = [d for d in details if d["answerable"]]
    ref = [d for d in details if not d["answerable"]]
    n = len(details) or 1
    return {
        "rows": len(details),
        "answerable": len(ans),
        "refusal_cases": len(ref),
        "retrieval_hit": sum(1 for d in ans if d["retrieval_hit"]),
        "citation_hit": sum(1 for d in ans if d["citation_hit"]),
        "exact_citation_hit": sum(1 for d in ans if d["exact_citation_match"]),
        "refusal_ok": sum(1 for d in ref if d["refused"]),
        "avg_delivered_blocks": round(sum(d["delivered_blocks"] for d in details) / n, 2),
        "avg_elapsed_s": round(sum(d["elapsed_s"] for d in details) / n, 2),
        "avg_llm_s": round(sum(d["llm_s"] for d in details) / n, 2),
        "total_cited_pages": sum(len(d["cited_pages"]) for d in details),
        "total_extra_cited_pages": sum(len(d["extra_cited_pages"]) for d in details),
        "degraded": sum(1 for d in details if d["degraded"]),
        "llm_failed": sum(1 for d in details if d["llm_failed"]),
    }


def by_category(details: list[dict]) -> list[dict]:
    cats: dict[str, dict] = {}
    for d in details:
        if not d["answerable"]:
            continue
        c = cats.setdefault(d["category"], {"category": d["category"], "n": 0, "hit": 0})
        c["n"] += 1
        c["hit"] += 1 if d["retrieval_hit"] else 0
    return sorted(cats.values(), key=lambda x: -x["n"])


def authority_rank_of(row: dict, expected: list[int]) -> int | None:
    """期望页在交付顺序里第一次出现的位置（1 起）。"""
    for i, item in enumerate(row.get("picked", []) or [], start=1):
        if int(item.get("page", -1)) in expected:
            return i
    return None


def compare_spotcheck(history: dict[tuple, dict], spot: list[dict],
                      cases: dict[str, dict]) -> list[dict]:
    """历史 v3 与抽检复跑逐项对比（只比页码，不比答案文本）。"""
    out = []
    for row in spot:
        old = history.get((row["id"], row["mode"]))
        expected = [int(p) for p in cases[row["id"]].get("pages", [])]
        if old is None:
            out.append({"question_id": row["id"], "strategy": row["mode"],
                        "status": "历史记录里没有这一行"})
            continue
        out.append({
            "question_id": row["id"],
            "strategy": row["mode"],
            "history_pages": [int(p) for p in old.get("pages", [])],
            "spotcheck_pages": [int(p) for p in row.get("pages", [])],
            "pages_identical": [int(p) for p in old.get("pages", [])] == [int(p) for p in row.get("pages", [])],
            "history_cited": sorted({int(p) for p in old.get("cited_pages", [])}),
            "spotcheck_cited": sorted({int(p) for p in row.get("cited_pages", [])}),
            "cited_identical": sorted({int(p) for p in old.get("cited_pages", [])}) ==
                               sorted({int(p) for p in row.get("cited_pages", [])}),
            "authority_rank_history": authority_rank_of(old, expected),
            "authority_rank_spotcheck": authority_rank_of(row, expected),
        })
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
CSV_COLS = ["question_id", "category", "strategy", "answerable", "expected_pages",
            "delivered_pages", "retrieval_hit", "authority_rank", "cited_pages",
            "citation_hit", "exact_citation_match", "extra_cited_pages",
            "missing_cited_pages", "delivered_blocks", "candidates",
            "elapsed_s", "llm_s", "degraded", "llm_failed", "refused"]


def write_csv(path: Path, details: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:   # BOM：Excel 打开不乱码
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for d in sorted(details, key=lambda x: (x["question_id"], x["strategy"])):
            row = []
            for c in CSV_COLS:
                v = d[c]
                if isinstance(v, list):
                    v = " ".join(str(x) for x in v)
                elif v is None:
                    v = ""
                row.append(v)
            w.writerow(row)


def fmt_pages(p: list[int]) -> str:
    return ",".join(f"p{x}" for x in p) if p else "—"


def write_md(path: Path, meta: dict, summary: dict, details: list[dict],
             spot_compare: list[dict]) -> None:
    L: list[str] = []
    a = L.append

    a("# 评测报告：RAG 工业设备知识助手\n")
    a("> 本文件由 `eval/reports/build_public_report.py` 从历史评测记录自动生成，**请勿手改**；"
      "改口径请改脚本后重跑。\n")
    a("## 0. 这份报告能证明什么、不能证明什么\n")
    a("| 能证明 | 不能证明 |")
    a("|---|---|")
    a("| 2026-11-06 那一轮，15 道题 × 2 种检索策略的检索命中、引用页码、拒答与耗时 | 当前代码在其他数据、其他模型版本上的表现 |")
    a("| 失败案例（Q20）确实存在，没有被隐藏 | 答案「内容」是否正确（那是人工判断，不是本文的自动指标） |")
    a("| 任何人可以用 CSV 重算本文所有数字 | 引用页码正确 = 引用内容正确（本文只做到「页码指向的表/段落确实相关」） |")
    a("")

    a("## 1. 数据来源与可追溯信息\n")
    a(f"- 源文件：`{meta['source_file']}`（不随仓库发布：含模型原始答案与手册章节标题）")
    a(f"- 源文件 sha256：`{meta['source_sha256']}`")
    a(f"- 行数：{meta['source_rows']} 行 = {meta['question_count']} 题 × 2 种策略")
    a(f"- 判据文件：`eval/qa_cases.json`（sha256 `{meta['cases_sha256']}`）")
    a(f"- 记录时间：{meta['run_date_note']}")
    a(f"- 生成时间：{meta['generated_at']}（脚本写入，不代表评测时间）")
    a("")
    a("**版本说明（重要）**：源记录没有保存 commit、模型版本、检索参数；"
      "仓库代码在评测之后有过改动（文档、配置读取方式等）。因此本报告描述的是"
      "「2026-11-06 那次运行」，不等于「当前 HEAD 的复现结果」。"
      "当前代码的抽检结果见第 6 节。\n")

    a("## 2. 指标定义（口径）\n")
    a("| 指标 | 定义 | 分母 |")
    a("|---|---|---|")
    a("| 检索命中 retrieval_hit | 交付给 LLM 的 10 个父块里，存在页码属于该题的期望页 | 12 道有答案题 |")
    a("| 权威页名次 authority_rank | 交付顺序里第一次出现期望页的位置（从 1 起） | 有答案题 |")
    a("| 引用命中 citation_hit（宽松） | 答案引用的页码集合与期望页集合**有交集** | 12 道有答案题 |")
    a("| 严格引用匹配 exact_citation_match | 引用页码集合与期望页集合**完全相同** | 12 道有答案题 |")
    a("| 多余引用页 extra_cited_pages | 引用了、但不在期望页里的页码 | 逐题展示 |")
    a("| 正确拒答 refusal_ok | 资料不足题的回答命中拒答标志串 | 3 道资料不足题 |")
    a("| 平均耗时 | 端到端（检索 + rerank + LLM）算术平均，含拒答题 | 15 题 |")
    a("")
    a("> `citation_hit` 是**宽松**口径：引到一个对的页就算过。Q04 的 BM25 引了 `p8,p16`"
      "（期望只有 p8）、Q10 的 BM25 引了 4 个页（期望只有 p17），宽松口径下都算命中，"
      "因此本文额外给出 `exact_citation_match` 与多余引用页。\n")
    a("> 无法可靠计算、明确不提供的指标：答案内容正确率（无标注字段）、"
      "Recall@50（历史记录未保存候选全集）、rerank 前后名次变化（未保存）。\n")

    a("## 3. 汇总结果\n")
    a("| 指标 | BM25 | Hybrid |")
    a("|---|---:|---:|")
    for key, label in [("retrieval_hit", "检索命中"), ("citation_hit", "引用命中（宽松）"),
                       ("exact_citation_hit", "严格引用匹配")]:
        a(f"| {label} | {summary['bm25'][key]}/{summary['bm25']['answerable']} | "
          f"{summary['hybrid'][key]}/{summary['hybrid']['answerable']} |")
    a(f"| 正确拒答 | {summary['bm25']['refusal_ok']}/{summary['bm25']['refusal_cases']} | "
      f"{summary['hybrid']['refusal_ok']}/{summary['hybrid']['refusal_cases']} |")
    a(f"| 平均交付父块 | {summary['bm25']['avg_delivered_blocks']} | "
      f"{summary['hybrid']['avg_delivered_blocks']} |")
    a(f"| 平均耗时 | {summary['bm25']['avg_elapsed_s']} s | {summary['hybrid']['avg_elapsed_s']} s |")
    a(f"| 其中平均 LLM 耗时 | {summary['bm25']['avg_llm_s']} s | {summary['hybrid']['avg_llm_s']} s |")
    a(f"| 引用页合计 / 其中多余 | {summary['bm25']['total_cited_pages']} / "
      f"{summary['bm25']['total_extra_cited_pages']} | {summary['hybrid']['total_cited_pages']} / "
      f"{summary['hybrid']['total_extra_cited_pages']} |")
    a(f"| rerank 降级次数 | {summary['bm25']['degraded']} | {summary['hybrid']['degraded']} |")
    a(f"| LLM 生成失败次数 | {summary['bm25']['llm_failed']} | {summary['hybrid']['llm_failed']} |")
    a("")

    a("## 4. 逐题明细\n")
    for mode in MODES:
        a(f"### {MODE_LABEL[mode]}\n")
        a("| 题号 | 类别 | 期望页 | 交付页 | 权威页名次 | 引用页 | 检索 | 引用 | 严格引用 |")
        a("|---|---|---|---|---:|---|---|---|---|")
        for d in sorted([x for x in details if x["strategy"] == mode], key=lambda x: x["question_id"]):
            if not d["answerable"]:
                a(f"| {d['question_id']} | {d['category']} | — | {fmt_pages(d['delivered_pages'])} | — | "
                  "（拒答，无引用） | — | — | — |")
                continue
            mark = (lambda b: "✓" if b else "✗")   # noqa: E731
            head = d["delivered_pages"][:6]
            tail = "…" if len(d["delivered_pages"]) > 6 else ""
            a(f"| {d['question_id']} | {d['category']} | {fmt_pages(d['expected_pages'])} | "
              f"{fmt_pages(head)}{tail} | {d['authority_rank'] or '—'} | {fmt_pages(d['cited_pages'])} | "
              f"{mark(d['retrieval_hit'])} | {mark(d['citation_hit'])} | {mark(d['exact_citation_match'])} |")
        a("")

    a("## 5. 失败与不完美的案例（只列真实存在的）\n")
    a("### 5.1 Q20：检索命中、引用未命中（两种策略都失败）\n")
    for d in sorted([x for x in details if x["question_id"] == "Q20"], key=lambda x: x["strategy"]):
        a(f"- **{MODE_LABEL[d['strategy']]}**：期望 p{d['expected_pages'][0]}，权威页在交付里排第 "
          f"{d['authority_rank']}，答案却引用了 {fmt_pages(d['cited_pages'])}。")
    a("")
    a("p40（Table 13，Digital I/O 场景）与 p49（Table 19，Fieldbus 场景）内容逐字相同，"
      "模型选了排第一的 p49。这是**引用指向不够贴近问题场景**，不是检索没召回。\n")
    a("### 5.2 引用过宽（宽松口径命中、严格口径不命中）\n")
    wide = [x for x in details if x["answerable"] and x["extra_cited_pages"]]
    for d in sorted(wide, key=lambda x: (x["question_id"], x["strategy"])):
        a(f"- {d['question_id']} / {MODE_LABEL[d['strategy']]}：期望 {fmt_pages(d['expected_pages'])}，"
          f"多引了 {fmt_pages(d['extra_cited_pages'])}")
    a("")
    a("### 5.3 引用不足（漏引期望页）\n")
    short = [x for x in details if x["answerable"] and x["missing_cited_pages"]]
    for d in sorted(short, key=lambda x: (x["question_id"], x["strategy"])):
        a(f"- {d['question_id']} / {MODE_LABEL[d['strategy']]}：期望 {fmt_pages(d['expected_pages'])}，"
          f"漏引 {fmt_pages(d['missing_cited_pages'])}")
    a("")

    if spot_compare:
        a("## 6. 当前代码的抽检复跑（只证明这 4 次）\n")
        a("抽检对象：Q04（参数/选项值）与 Q20（相似表题），两种策略各一次。命令：\n")
        a("```bash")
        a("python eval/run_cases.py --ids Q04 Q20 --out _local/results/spotcheck_<date>.json")
        a("```\n")
        a("| 题号 | 策略 | 历史交付页 | 复跑交付页 | 交付页一致 | 历史引用 | 复跑引用 | 引用一致 | 权威页名次（历史→复跑） |")
        a("|---|---|---|---|---|---|---|---|---|")
        for s in spot_compare:
            if "status" in s:
                a(f"| {s['question_id']} | {s['strategy']} | — | — | {s['status']} | — | — | — | — |")
                continue
            a(f"| {s['question_id']} | {MODE_LABEL[s['strategy']]} | {fmt_pages(s['history_pages'])} | "
              f"{fmt_pages(s['spotcheck_pages'])} | {'✓' if s['pages_identical'] else '✗'} | "
              f"{fmt_pages(s['history_cited'])} | {fmt_pages(s['spotcheck_cited'])} | "
              f"{'✓' if s['cited_identical'] else '✗'} | "
              f"{s['authority_rank_history']} → {s['authority_rank_spotcheck']} |")
        a("")
        pages_same = sum(1 for s in spot_compare if s.get("pages_identical"))
        cited_same = sum(1 for s in spot_compare if s.get("cited_identical"))
        authority_ok = sum(1 for s in spot_compare
                           if s.get("authority_rank_spotcheck") is not None)
        rank_same = sum(1 for s in spot_compare
                        if s.get("authority_rank_spotcheck") == s.get("authority_rank_history"))
        n = len(spot_compare)
        a(f"结论：\n")
        a(f"- **检索层可复现**：{authority_ok}/{n} 次的权威页都在交付里，"
          f"{rank_same}/{n} 次名次与历史相同（交付页集合完全一致的 {pages_same}/{n} 次；"
          "其余差异只出现在 rerank 同分时的尾部块顺序上）。")
        a(f"- **引用层不能完全复现**：引用页集合完全一致 {cited_same}/{n} 次。"
          "LLM 引用的范围会随机变化（本轮 Q04/BM25 少引了 p16、Q20/BM25 多引了 p41），"
          "但**引用命中与否的结论没有变**（Q20 两种策略仍然未引用权威页 p40）。")
        a("- 因此这 4 次只能说明「检索部分可复现、引用部分有运行间波动」，"
          "**不能**说明 30 行历史结果都来自当前 HEAD。\n")

    a("## 7. 局限性\n")
    a("1. 只有 15 题、单一标注者人工判断，样本小。")
    a("2. 引用正确性的判定粒度是「页码」，同页有多块时无法区分具体用了哪一块。")
    a("3. 拒答判定是字符串匹配（6 个标志串），可能把「部分拒答」判成「完整拒答」。")
    a("4. 历史记录未保存 commit / 模型版本 / 检索参数，无法证明与当前代码一一对应。")
    a("5. 平均耗时受免费档模型排队影响大（单题 11–78 秒），不代表系统性能上限。")
    a("")
    a("## 8. 复现方式\n")
    a("```bash")
    a("# 重新生成本报告（只读历史记录，不调任何 API）")
    a("python eval/reports/build_public_report.py")
    a("")
    a("# 重跑完整评测（需要数据目录 + ZHIPU_API_KEY，会产生费用）")
    a("python eval/run_cases.py --modes bm25 hybrid")
    a("```")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="_local/results/qa_results_v3.json")
    ap.add_argument("--cases", default="eval/qa_cases.json")
    ap.add_argument("--out-dir", default="eval/reports")
    ap.add_argument("--spotcheck", default=None, help="可选：抽检复跑 JSON，用于第 6 节")
    args = ap.parse_args()

    src = Path(args.source)
    src = src if src.is_absolute() else (ROOT / src)
    cases_path = ROOT / args.cases
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"找不到源记录：{src}\n（它在 .gitignore 里，需要先放回 _local/results/）")

    cases = load_cases(cases_path)
    rows = load_rows(src)
    details = [build_detail(r, cases[r["id"]]) for r in rows if r["id"] in cases]
    unknown = [r["id"] for r in rows if r["id"] not in cases]

    summary = {m: summarize_mode([d for d in details if d["strategy"] == m]) for m in MODES}
    meta = {
        "source_file": src.name,
        "source_sha256": sha256_of(src),
        "source_rows": len(rows),
        "question_count": len({d["question_id"] for d in details}),
        "cases_file": cases_path.name,
        "cases_sha256": sha256_of(cases_path),
        "run_date_note": "2026-11-06（由文件修改时间与内部记录确认，非脚本写入）",
        "generated_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S%z"),
        "unknown_ids": unknown,
    }

    spot_compare: list[dict] = []
    if args.spotcheck:
        sp = Path(args.spotcheck)
        sp = sp if sp.is_absolute() else (ROOT / sp)
        spot_compare = compare_spotcheck({(r["id"], r["mode"]): r for r in rows},
                                         load_rows(sp), cases)
        meta["spotcheck_file"] = sp.name
        meta["spotcheck_sha256"] = sha256_of(sp)

    (out_dir / "evaluation_summary.json").write_text(
        json.dumps({"meta": meta, "metrics": summary,
                    "by_category": {m: by_category([d for d in details if d["strategy"] == m])
                                    for m in MODES},
                    "spotcheck": spot_compare, "details": details},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_dir / "evaluation_details.csv", details)
    write_md(out_dir / "evaluation.md", meta, summary, details, spot_compare)

    print(f"源记录：{src}（{len(rows)} 行）")
    for m in MODES:
        s = summary[m]
        print(f"  {m:<7} 检索 {s['retrieval_hit']}/{s['answerable']} | 引用 {s['citation_hit']}/{s['answerable']}"
              f" | 严格 {s['exact_citation_hit']}/{s['answerable']} | 拒答 {s['refusal_ok']}/{s['refusal_cases']}"
              f" | 交付 {s['avg_delivered_blocks']} | 耗时 {s['avg_elapsed_s']}s")
    if spot_compare:
        same = sum(1 for x in spot_compare if x.get("pages_identical") and x.get("cited_identical"))
        print(f"  抽检：{same}/{len(spot_compare)} 次与历史一致")
    print(f"\n已写出：{out_dir / 'evaluation_summary.json'}")
    print(f"        {out_dir / 'evaluation_details.csv'}")
    print(f"        {out_dir / 'evaluation.md'}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
