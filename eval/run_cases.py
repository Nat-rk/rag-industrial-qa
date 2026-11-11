# -*- coding: utf-8 -*-
"""批量跑 QA 验收题：两种检索策略 × 15 道题，落盘 + 打印对照表。

用法：
    python eval/run_cases.py                 # 两种策略都跑
    python eval/run_cases.py --modes bm25    # 只跑 BM25
    python eval/run_cases.py --ids Q01 Q04   # 只跑指定几道
    python eval/run_cases.py --limit 3       # 先跑前 3 道（省钱省时间）

说明：
    * 自动判定的只有两件事——「检索到的父块里有没有正确页」和「资料不足的题有没有拒答」；
      答案对不对、引用准不准，脚本只把证据摆出来，仍需人工对照原文。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                    # noqa: E402
from main import ask, build_retriever            # noqa: E402
from retrieval.store import load_parents, load_units  # noqa: E402

REFUSAL_MARKS = ["没有找到足够", "没有足够的信息", "信息不足", "无法回答", "未提及", "文档中没有"]
CASES = Path(__file__).resolve().parent / "qa_cases.json"


def load_cases() -> list[dict]:
    with open(CASES, encoding="utf-8") as f:
        return json.load(f)["cases"]


def run(cases: list[dict], modes: list[str],
        out_path: Path | None = None, merge: bool = False) -> list[dict]:
    parents, units = load_parents(), load_units()
    retrievers = {m: build_retriever(m, units) for m in modes}   # 每种策略只建一次

    rows: list[dict] = []
    kept: list[dict] = []          # --merge 时保留的旧行
    if merge and out_path and out_path.exists():
        kept = json.loads(out_path.read_text(encoding="utf-8"))

    def save() -> None:
        """每跑完一次就落盘：网络抖动/中途崩溃不至于把前面几十次白跑。"""
        if not out_path:
            return
        keys = {(r["id"], r["mode"]) for r in rows}
        merged = [r for r in kept if (r["id"], r["mode"]) not in keys] + rows
        merged.sort(key=lambda r: (r["id"], r["mode"]))
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    for case in cases:
        for mode in modes:
            t0 = time.time()
            try:
                result = ask(case["q"], mode=mode, retriever=retrievers[mode],
                             parents=parents, units=units)
            except Exception as e:                  # 网络/接口异常：记一行失败，继续下一题
                rows.append({"id": case["id"], "type": case["type"], "mode": mode,
                             "question": case["q"], "error": f"{type(e).__name__}: {e}",
                             "answer": "", "pages": [], "evidence_pages": [],
                             "evidence_ok": None, "refused": False, "cited_pages": [],
                             "citations_ok": False, "degraded": False,
                             "llm_failed": True, "delivered": 0, "candidates": 0,
                             "llm_seconds": 0, "elapsed": round(time.time() - t0, 1),
                             "picked": []})
                save()
                print(f"{case['id']:<5}{mode:<8}接口异常：{type(e).__name__}")
                continue
            elapsed = time.time() - t0

            pages = [c["parent"].page for c in result["picked"]]
            evidence = [p for p in case.get("pages", []) if p in pages]
            refused = any(mark in result["answer"] for mark in REFUSAL_MARKS)
            cited_pages = sorted({c["page"] for c in result["citations"]})

            row = {
                "id": case["id"], "type": case["type"], "mode": result["mode"],
                "question": case["q"],
                "answer": result["answer"],
                "pages": pages,
                "evidence_pages": evidence,
                "evidence_ok": bool(evidence) if case.get("pages") else None,
                "refused": refused,
                "cited_pages": cited_pages,
                "citations_ok": bool(set(cited_pages) & set(case.get("pages", []))) if cited_pages else False,
                "degraded": bool(result["rerank"].get("degraded")),
                "llm_failed": result["answer"].startswith("（生成失败"),
                "delivered": len(result["picked"]),
                "candidates": result["trace"][-1]["candidates"],
                "llm_seconds": round(result["timing"]["llm"], 1),
                "elapsed": round(elapsed, 1),
                "picked": [{"page": c["parent"].page, "kind": c["parent"].kind,
                            "short_id": c["parent"].short_id,
                            "chars": c["parent"].chars,
                            "rerank": round(c.get("rerank_score", 0.0), 4),
                            "section": c["parent"].section} for c in result["picked"]],
            }
            rows.append(row)
            save()

            flag = "检索✓" if row["evidence_ok"] else ("检索✗" if case.get("pages") else "—")
            print(f"{row['id']:<5}{row['mode']:<8}{flag:<7}"
                  f"页={str(pages)[:34]:<36}引用={str(cited_pages)[:14]:<16}"
                  f"{'拒答' if refused else '    '} {row['elapsed']:>5.1f}s")
    return rows


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 92)
    print(f"{'策略':<9}{'有答案题 检索命中':<20}{'资料不足 拒答':<16}{'平均交付':<10}{'引用命中':<10}{'降级':<6}{'平均耗时'}")
    for mode in sorted({r["mode"] for r in rows}):
        sub = [r for r in rows if r["mode"] == mode]
        ans = [r for r in sub if r["evidence_ok"] is not None]
        non = [r for r in sub if r["evidence_ok"] is None]
        hit = sum(1 for r in ans if r["evidence_ok"])
        ref = sum(1 for r in non if r["refused"])
        cited = sum(1 for r in ans if r["citations_ok"])
        avg_del = sum(r["delivered"] for r in sub) / max(len(sub), 1)
        deg = sum(1 for r in sub if r["degraded"])
        avg_t = sum(r["elapsed"] for r in sub) / max(len(sub), 1)
        print(f"{mode:<11}{hit}/{len(ans):<18}{ref}/{len(non):<14}"
              f"{avg_del:<11.1f}{cited}/{len(ans):<8}{deg:<7}{avg_t:.1f}s")
    print("=" * 92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["bm25", "hybrid"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(config.OUT / "qa_results.json"))
    ap.add_argument("--merge", action="store_true",
                    help="把结果合并进已有 --out 文件（按 题号+策略 覆盖），用于补跑个别题")
    args = ap.parse_args()

    cases = load_cases()
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    if args.limit:
        cases = cases[: args.limit]

    print(f"共 {len(cases)} 道题 × {len(args.modes)} 种策略 = {len(cases) * len(args.modes)} 次问答\n")
    out_path = Path(args.out)
    rows = run(cases, args.modes, out_path=out_path, merge=args.merge)
    summarize(rows)

    if args.merge and out_path.exists():
        old = json.loads(out_path.read_text(encoding="utf-8"))
        keys = {(r["id"], r["mode"]) for r in rows}
        rows = [r for r in old if (r["id"], r["mode"]) not in keys] + rows
        rows.sort(key=lambda r: (r["id"], r["mode"]))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
