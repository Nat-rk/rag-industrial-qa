# -*- coding: utf-8 -*-
"""对比两次验收结果（修改前 vs 修改后），只为报告服务，不参与问答。

用法：
    python eval/compare_runs.py --old out/qa_results.json --new out/qa_results_v2.json

关注三件事：
    1. 引用条数（Q26 是重点）：是变窄了还是没变；
    2. 检索/拒答/引用命中等硬指标有没有退化；
    3. 答案长度（防止"为了少引用把答案写长了"或"写短了漏步骤"）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def load(path: str) -> dict[tuple[str, str], dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(r["id"], r["mode"]): r for r in rows}


def flag(v) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(config.OUT / "qa_results.json"))
    ap.add_argument("--new", default=str(config.OUT / "qa_results_v2.json"))
    ap.add_argument("--out", default=str(config.OUT / "compare_v1_v2.json"))
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    keys = sorted(new, key=lambda k: (k[0], k[1]))

    print(f"{'题':<5}{'策略':<9}{'引用(旧)':<22}{'引用(新)':<22}{'条数':<7}{'检索':<9}{'拒答':<7}{'字数(旧→新)':<14}")
    print("-" * 96)
    diffs = []
    for k in keys:
        o, n = old.get(k), new.get(k)
        if not o:
            continue
        oc, nc = o["cited_pages"], n["cited_pages"]
        delta = len(nc) - len(oc)
        row = {
            "id": k[0], "mode": k[1],
            "old_cited": oc, "new_cited": nc, "delta": delta,
            "old_evidence_ok": o["evidence_ok"], "new_evidence_ok": n["evidence_ok"],
            "old_refused": o["refused"], "new_refused": n["refused"],
            "old_len": len(o["answer"]), "new_len": len(n["answer"]),
            "old_degraded": o["degraded"], "new_degraded": n["degraded"],
        }
        diffs.append(row)
        print(f"{k[0]:<5}{k[1]:<9}{str(oc)[:21]:<22}{str(nc)[:21]:<22}"
              f"{delta:+d}     {flag(o['evidence_ok'])}→{flag(n['evidence_ok']):<5}"
              f"{'是' if n['refused'] else '否':<7}{len(o['answer'])}→{len(n['answer'])}")

    print("-" * 96)
    total_old = sum(len(r["old_cited"]) for r in diffs)
    total_new = sum(len(r["new_cited"]) for r in diffs)
    fewer = [r for r in diffs if r["delta"] < 0]
    more = [r for r in diffs if r["delta"] > 0]
    same = [r for r in diffs if r["delta"] == 0]
    ev_bad = [r for r in diffs if r["old_evidence_ok"] and not r["new_evidence_ok"]]
    ref_bad = [r for r in diffs if r["old_refused"] and not r["new_refused"]]
    print(f"引用总条数：{total_old} → {total_new}（{total_new - total_old:+d}）")
    print(f"变窄 {len(fewer)} 处、不变 {len(same)} 处、变宽 {len(more)} 处")
    print(f"检索命中退化：{ev_bad or '无'}    拒答退化：{ref_bad or '无'}")
    if more:
        print("变宽明细：" + "；".join(f"{r['id']}/{r['mode']} {r['old_cited']}→{r['new_cited']}" for r in more))

    Path(args.out).write_text(json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {args.out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
