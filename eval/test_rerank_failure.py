# -*- coding: utf-8 -*-
"""S3：真实触发一次 Rerank 失败，验证降级行为（不花 rerank 的钱）。

做法：把 `retrieval.reranker` 里的 `requests` 临时换成一个"假 shim"，
让它按预设方式失败（连接异常 / 超时 / 500 / 429 额度 / 200 但响应为空），
从而真实走一遍 `_post` 的重试与 `rerank_candidates` 的兜底分支。
LLM 走的仍是真实接口（glm-4.5-flash，免费），所以能验证"降级后仍能出答案"。

检查五件事：
    1. 是否退回原始候选顺序；
    2. 是否仍能构建上下文并调用 LLM（答案非空、不是"生成失败"）；
    3. 父块 ID / 文件名 / 页码是否与 picked 一一对应；
    4. 引用是否仍能映射到真实来源、且没有越界编号；
    5. 结果里是否明确标了 degraded。

用法：
    python eval/test_rerank_failure.py            # 默认 5 种故障 + 1 次正常对照
    python eval/test_rerank_failure.py --modes bm25
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests                                    # noqa: E402

import config                                      # noqa: E402
import retrieval.reranker as rr                    # noqa: E402
from main import ask                               # noqa: E402
from retrieval.store import load_parents, load_units  # noqa: E402

QUESTION = "Which parameter selects Positioning or Synchronization mode, and what are the corresponding option values?"


class FakeResp:
    """够用的假响应：只有 status_code / text / json。"""

    def __init__(self, status: int, body: str = "", payload: dict | None = None):
        self.status_code = status
        self.text = body
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _raise(exc):
    def _post(payload, headers, *a, **kw):
        raise exc
    return _post


def _status(code: int, body: str = ""):
    def _post(payload, headers, *a, **kw):
        return FakeResp(code, body)
    return _post


def _empty_200():
    def _post(payload, headers, *a, **kw):
        return FakeResp(200, "{}", {})            # 没有 results
    return _post


@contextmanager
def failing_rerank(post_fn):
    """只替换 reranker 看到的 requests，不影响 generator 调 LLM。"""
    shim = types.SimpleNamespace(
        post=post_fn,
        RequestException=requests.RequestException,
        Timeout=requests.Timeout,
        ConnectionError=requests.ConnectionError,
    )
    old_requests, old_retries = rr.requests, rr.RETRIES
    rr.requests = shim
    rr.RETRIES = 1                                 # 只跑 1 次重试，省时间
    try:
        yield
    finally:
        rr.requests, rr.RETRIES = old_requests, old_retries


CASES = [
    ("network", "连接异常（RequestException）", _raise(requests.ConnectionError("mock: 连接被重置"))),
    ("timeout", "读超时（Timeout）", _raise(requests.Timeout("mock: 读超时"))),
    ("http_500", "服务端错误 HTTP 500", _status(500, '{"error":"upstream failed"}')),
    ("quota_429", "额度不足 HTTP 429", _status(429, '{"error":{"code":"1113","message":"余额不足或无可用资源包"}}')),
    ("empty_200", "HTTP 200 但 results 为空", _empty_200()),
]


def check(result: dict, candidates: list, parents: dict) -> dict:
    """对一次降级问答做五项核对。"""
    picked = result["picked"]
    sources = result["sources"]
    expect = candidates[: len(picked)]

    order_ok = [(c["parent"].id, c["parent"].page) for c in picked] == \
               [(c["parent"].id, c["parent"].page) for c in expect]
    mapping_ok = all(
        s["parent_id"] == c["parent"].id and s["page"] == c["parent"].page
        and s["file"] == c["parent"].file_name
        for s, c in zip(sources, picked)
    )
    cited_pages = {c["page"] for c in result["citations"]}
    picked_pages = {c["parent"].page for c in picked}
    cite_ok = cited_pages <= picked_pages
    answer_ok = bool(result["answer"]) and not result["answer"].startswith("（生成失败")

    return {
        "degraded": bool(result["rerank"].get("degraded")),
        "error": result["rerank"].get("error"),
        "n_candidates": len(candidates),
        "n_picked": len(picked),
        "order_preserved": order_ok,
        "source_mapping_ok": mapping_ok,
        "answer_ok": answer_ok,
        "cited_pages": sorted(cited_pages),
        "citations_subset_of_picked": cite_ok,
        "invalid_citations": result.get("invalid_citations", []),
        "answer_head": result["answer"][:120].replace("\n", " "),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["bm25", "hybrid"])
    ap.add_argument("--cases", nargs="*", default=None,
                    help="只跑指定故障（normal/network/timeout/http_500/quota_429/empty_200）")
    ap.add_argument("--out", default=str(config.OUT / "rerank_failure_test.json"))
    args = ap.parse_args()

    cases = [c for c in CASES if args.cases is None or c[0] in args.cases]
    run_normal = args.cases is None or "normal" in args.cases

    parents, units = load_parents(), load_units()
    rows, all_ok = [], True

    # 先跑一次正常对照，确认降级测试没有污染正常流程
    for mode in (args.modes if run_normal else []):
        result = ask(QUESTION, mode=mode, parents=parents, units=units)
        info = check(result, result["candidates"], parents)
        normal_ok = (not info["degraded"]) and info["source_mapping_ok"] and info["answer_ok"]
        rows.append({"case": "normal", "mode": mode, "expect_degraded": False,
                     "ok": normal_ok, **info})
        all_ok &= normal_ok
        print(f"{'normal':<10}{mode:<8}degraded={info['degraded']!s:<6}"
              f"映射={info['source_mapping_ok']!s:<6}答案={info['answer_ok']!s:<6}"
              f"引用={info['cited_pages']}  {'✓' if normal_ok else '✗'}")

    # 再逐个故障跑
    for name, label, fn in cases:
        for mode in args.modes:
            with failing_rerank(fn):
                result = ask(QUESTION, mode=mode, parents=parents, units=units)
            info = check(result, result["candidates"], parents)
            ok = (info["degraded"] and info["order_preserved"] and info["source_mapping_ok"]
                  and info["answer_ok"] and info["citations_subset_of_picked"]
                  and not info["invalid_citations"])
            all_ok &= ok
            rows.append({"case": name, "label": label, "mode": mode,
                         "expect_degraded": True, "ok": ok, **info})
            print(f"{name:<10}{mode:<8}degraded={info['degraded']!s:<6}"
                  f"原序={info['order_preserved']!s:<6}映射={info['source_mapping_ok']!s:<6}"
                  f"答案={info['answer_ok']!s:<6}引用={info['cited_pages']}  {'✓' if ok else '✗'}")

    out_path = Path(args.out)
    if out_path.exists():                     # 只重跑部分故障时合并旧结果，不丢历史
        old = json.loads(out_path.read_text(encoding="utf-8"))
        keys = {(r["case"], r["mode"]) for r in rows}
        rows = [r for r in old if (r["case"], r["mode"]) not in keys] + rows
        rows.sort(key=lambda r: (r["case"] != "normal", r["case"], r["mode"]))
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    passed = sum(1 for r in rows if r["ok"])
    print(f"总计 {passed}/{len(rows)} 通过（含历史记录）"
          f"    {'全部通过' if all_ok else '本次仍有失败项，见上表'}")
    print("=" * 88)
    print(f"\n明细已写入 {out_path}")
    if not all_ok:
        raise SystemExit(1)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
