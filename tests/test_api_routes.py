# -*- coding: utf-8 -*-
"""阶段 4：HTTP 接口的行为测试（路由层，不启动 lifespan）。

三种证据要分清：

* **真实组件**：`FastAPI` + `TestClient`/`httpx.ASGITransport`、真正的线程池调度、
  真正的 `threading.Lock`、真正的 Pydantic 校验与异常处理器。
* **FakeEngine**：只有 Engine 是假的（返回固定 14 键、或按需抛异常），
  用来把 HTTP 语义与 LLM/Chroma 隔离。
* **Mock**：本文件只在"约束扫描"里不涉及 mock；`tests/test_api.py` 用
  `mock.patch` 替换 engine_factory 与 api_key，属于生命周期测试。

真正跑通"HTTP → Engine → 数据 → Chroma → LLM"的端到端冒烟已于 2026-11-22 阶段六手工执行
（Q04 有答案、N01 拒答），不在本文件内——避免每次跑测试都花钱。
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import threading
import time
import unittest

try:                                      # 干净环境（未 pip install）时整模块跳过
    import httpx
    from fastapi.testclient import TestClient

    from api.app import create_app
    from errors import ConfigError, DataError, LLMError, RerankError
    from retrieval.store import Parent
    from settings import Settings
except ModuleNotFoundError as exc:        # pragma: no cover - 依赖缺失时走这里
    raise unittest.SkipTest(f"接口测试需要 fastapi/httpx：{exc}")

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: 生产代码目录/文件（写死比 rglob 更稳：不会把 tests/_tmp 或 __pycache__ 算进来）
PRODUCTION_PATHS = [
    ROOT / "main.py",
    ROOT / "settings.py",
    ROOT / "errors.py",
    ROOT / "logging_setup.py",
    *sorted((ROOT / "api").glob("*.py")),
    *sorted((ROOT / "qa").glob("*.py")),
    *sorted((ROOT / "retrieval").glob("*.py")),
    *sorted((ROOT / "service").glob("*.py")),
]

EXPECTED_KEYS = {
    "question", "mode", "candidates", "trace", "picked", "rerank",
    "sources", "context", "answer", "usage", "citations",
    "citations_marked", "invalid_citations", "timing",
}


def fake_parent() -> Parent:
    return Parent(
        id="abcdef1234567890", page=8, kind="section",
        section="2 Introduction to IMC > 2.1 An Overview of IMC",
        text="To set the motion control mode, select [9] Positioning or [10] …",
        chars=614, file_name="apply.pdf",
    )


def fake_result(question: str = "q") -> dict:
    p = fake_parent()
    src = {"n": 1, "file": "apply.pdf", "page": 8,
           "parent_id": p.id, "short_id": p.short_id,
           "section": "2.1", "chars": 614}
    return {
        "question": question,
        "mode": "bm25",
        "candidates": [{"parent": p, "unit_id": "u1",
                        "unit_text": "select [9] Positioning", "score": 0.77}],
        "trace": [{"candidates": 30, "returned": 12, "parents": 3}],
        "picked": [{"parent": p, "unit_id": "u1",
                    "unit_text": "select [9] Positioning", "score": 0.77,
                    "rerank_score": 0.91}],
        "rerank": {"degraded": False, "elapsed": 1.3,
                   "usage": {"total_tokens": 10}},
        "sources": [src],
        "context": "[Source 1]\nFile: apply.pdf\nPage: 8\n\nTo set…",
        "answer": "参数 1-00 用于选择定位或同步模式 [Source 1]。",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        "citations": [{"n": 1, "file": "apply.pdf", "page": 8,
                       "parent_id": p.id, "short_id": p.short_id,
                       "section": "2.1"}],   # 注意：真实 build_citations() 没有 chars
        "citations_marked": True,
        "invalid_citations": [],
        "timing": {"build": 0.0, "recall": 0.01, "rerank": 1.3, "llm": 15.6},
    }


class FakeEngine:
    """假 Engine：只实现路由需要的三个成员。"""

    def __init__(self, state: str = "ready"):
        self.state = state
        self.settings = Settings()
        self.error: Exception | None = None
        self.calls: list[str] = []

    def initialize(self) -> None:            # 本文件不跑 lifespan，但保持接口完整
        self.state = "ready"

    def close(self) -> None:
        self.state = "closed"

    def ask(self, question: str) -> dict:
        self.calls.append(question)
        if self.error is not None:
            raise self.error
        return fake_result(question)


def make_app(engine, *, with_lock: bool = True, settings: Settings | None = None):
    app = create_app(settings or Settings(),
                     engine_factory=lambda s: engine)
    # 不走 lifespan：直接把 ready 的 Engine 与锁装进 state
    app.state.engine = engine
    app.state.ask_lock = threading.Lock() if with_lock else None
    return app


def make_client(engine, **kwargs) -> TestClient:
    # 不用 `with TestClient(...)`，避免触发 lifespan 把 FakeEngine 换成真的 RagEngine
    return TestClient(make_app(engine, **kwargs), raise_server_exceptions=False)


class AskValidationTests(unittest.TestCase):
    """/ask 的请求校验（Pydantic 层，422 外壳由 api/errors.py 统一生成）。"""

    def setUp(self) -> None:
        self.client = make_client(FakeEngine())

    def tearDown(self) -> None:
        self.client.close()

    def _assert_422(self, payload) -> dict:
        r = self.client.post("/ask", json=payload)
        self.assertEqual(r.status_code, 422, r.text)
        body = r.json()
        self.assertEqual(body["error"]["code"], "validation_error")
        self.assertIn("message", body["error"])
        return body

    def test_empty_question(self) -> None:
        self._assert_422({"question": ""})

    def test_blank_question(self) -> None:
        self._assert_422({"question": "   \n\t "})

    def test_missing_question(self) -> None:
        self._assert_422({})

    def test_too_long_question(self) -> None:
        self._assert_422({"question": "x" * 2001})

    def test_extra_field_rejected(self) -> None:
        # 只开放 question：mode/top_k 这类"请求级检索参数"必须被拒绝
        self._assert_422({"question": "ok", "mode": "hybrid"})

    def test_wrong_type(self) -> None:
        self._assert_422({"question": 123})

    def test_validation_error_has_trace_id_key(self) -> None:
        body = self._assert_422({"question": ""})
        # 还没进 pipeline，所以允许为 null，但字段必须在
        self.assertIn("trace_id", body["error"])


class AskSuccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.client = make_client(self.engine)

    def tearDown(self) -> None:
        self.client.close()

    def test_returns_all_14_keys(self) -> None:
        body = self.client.post("/ask", json={"question": "问题"}).json()
        self.assertEqual(set(body), EXPECTED_KEYS)

    def test_question_is_stripped(self) -> None:
        body = self.client.post("/ask", json={"question": "  哪个参数？  "}).json()
        self.assertEqual(body["question"], "哪个参数？")
        # Engine 收到的也是 strip 之后的文本
        self.assertEqual(self.engine.calls, ["哪个参数？"])

    def test_nested_parent_serialized(self) -> None:
        body = self.client.post("/ask", json={"question": "q"}).json()
        parent = body["picked"][0]["parent"]
        self.assertEqual(
            set(parent),
            {"id", "page", "kind", "section", "text", "chars", "file_name",
             "short_id"})
        self.assertEqual(parent["id"], "abcdef1234567890")
        self.assertEqual(parent["short_id"], "abcdef12")
        self.assertEqual(parent["page"], 8)

    def test_candidates_also_serialized(self) -> None:
        body = self.client.post("/ask", json={"question": "q"}).json()
        self.assertEqual(body["candidates"][0]["parent"]["page"], 8)
        self.assertEqual(body["candidates"][0]["score"], 0.77)

    def test_rerank_score_passthrough(self) -> None:
        body = self.client.post("/ask", json={"question": "q"}).json()
        self.assertEqual(body["picked"][0]["rerank_score"], 0.91)
        self.assertFalse(body["rerank"]["degraded"])

    def test_types_and_empty_lists(self) -> None:
        body = self.client.post("/ask", json={"question": "q"}).json()
        self.assertIsInstance(body["citations_marked"], bool)
        self.assertIsInstance(body["invalid_citations"], list)
        self.assertIsInstance(body["context"], str)
        self.assertEqual(body["trace"][0]["candidates"], 30)
        self.assertEqual(body["timing"]["llm"], 15.6)


class HealthTests(unittest.TestCase):
    def test_healthz_without_engine(self) -> None:
        client = make_client(FakeEngine(state="new"))
        try:
            r = client.get("/healthz")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json(), {"status": "ok", "service": "rag-qa"})
        finally:
            client.close()

    def test_readyz_ready(self) -> None:
        client = make_client(FakeEngine())
        try:
            r = client.get("/readyz")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "ready")
            self.assertIn("mode", r.json())
        finally:
            client.close()

    def test_readyz_not_ready(self) -> None:
        engine = FakeEngine(state="new")
        client = make_client(engine)
        try:
            r = client.get("/readyz")
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["code"], "not_ready")
        finally:
            client.close()

    def test_readyz_without_engine(self) -> None:
        app = create_app(Settings(), engine_factory=lambda s: FakeEngine())
        client = TestClient(app, raise_server_exceptions=False)
        try:
            r = client.get("/readyz")
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["code"], "not_ready")
        finally:
            client.close()


class ErrorMappingTests(unittest.TestCase):
    def _post_error(self, exc: Exception):
        engine = FakeEngine()
        engine.error = exc
        client = make_client(engine)
        try:
            return client.post("/ask", json={"question": "q"})
        finally:
            client.close()

    def test_config_error_503(self) -> None:
        r = self._post_error(ConfigError("缺少 ZHIPU_API_KEY"))
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "config_error")

    def test_data_error_500(self) -> None:
        r = self._post_error(DataError("parents.json 缺失"))
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["code"], "data_error")

    def test_llm_error_502(self) -> None:
        r = self._post_error(LLMError("上游 500"))
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.json()["error"]["code"], "llm_error")

    def test_rerank_error_502(self) -> None:
        r = self._post_error(RerankError("重排失败"))
        self.assertEqual(r.status_code, 502)
        self.assertEqual(r.json()["error"]["code"], "rerank_error")

    def test_unexpected_error_hides_message(self) -> None:
        r = self._post_error(RuntimeError("boom-secret"))
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["code"], "internal_error")
        self.assertNotIn("boom-secret", r.text)
        self.assertEqual(r.json()["error"]["message"], "服务内部错误")

    def test_unknown_route_uses_envelope(self) -> None:
        client = make_client(FakeEngine())
        try:
            r = client.get("/nope")
            self.assertEqual(r.status_code, 404)
            self.assertEqual(r.json()["error"]["code"], "http_error")
        finally:
            client.close()

    def test_not_ready_when_closed(self) -> None:
        engine = FakeEngine(state="closed")
        client = make_client(engine)
        try:
            r = client.post("/ask", json={"question": "q"})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["code"], "engine_not_ready")
            self.assertEqual(engine.calls, [])
        finally:
            client.close()

    def test_missing_lock_refuses(self) -> None:
        client = make_client(FakeEngine(), with_lock=False)
        try:
            r = client.post("/ask", json={"question": "q"})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["code"], "engine_not_ready")
        finally:
            client.close()


class BlockingEngine(FakeEngine):
    """ask() 会卡住的 Engine：用来观察"同时进来了几个请求"。"""

    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def ask(self, question: str) -> dict:
        self.calls.append(question)
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            self.gate.wait(timeout=10)
        finally:
            with self._guard:
                self.active -= 1
        return fake_result(question)


class ConcurrencyTests(unittest.TestCase):
    """真实 ASGI + 真实线程池 + 真实 threading.Lock。"""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_two_asks_are_serialized(self) -> None:
        engine = BlockingEngine()
        app = make_app(engine)

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                    transport=transport, base_url="http://test") as ac:
                first = asyncio.create_task(
                    ac.post("/ask", json={"question": "第一个"}))
                # 等第一个真正进入 engine.ask()，再发第二个
                await asyncio.to_thread(engine.entered.wait, 5)
                second = asyncio.create_task(
                    ac.post("/ask", json={"question": "第二个"}))
                # 给第二个足够时间尝试进锁；如果锁没生效，它会同时进入 ask()
                await asyncio.sleep(0.3)
                engine.gate.set()
                return await first, await second

        r1, r2 = self._run_async(scenario())
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertEqual(engine.max_active, 1, "同一时刻只能有一个 ask 进入 Engine")
        self.assertEqual(sorted(engine.calls), sorted(["第一个", "第二个"]))

    def test_health_endpoints_not_blocked_by_running_ask(self) -> None:
        engine = BlockingEngine()
        app = make_app(engine)

        async def scenario():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                    transport=transport, base_url="http://test") as ac:
                ask = asyncio.create_task(ac.post("/ask", json={"question": "慢"}))
                await asyncio.to_thread(engine.entered.wait, 5)
                t0 = time.perf_counter()
                h = await ac.get("/healthz")
                r = await ac.get("/readyz")
                cost = time.perf_counter() - t0
                engine.gate.set()
                await ask
                return h, r, cost

        h, r, cost = self._run_async(scenario())
        self.assertEqual(h.status_code, 200)
        self.assertEqual(r.status_code, 200)
        # 事件循环里的 async 路由不会排队等锁；实测远小于 1 秒
        self.assertLess(cost, 1.0, f"健康检查被阻塞了 {cost:.3f}s")

    def test_lock_released_after_exception(self) -> None:
        engine = FakeEngine()
        engine.error = LLMError("第一次失败")
        client = make_client(engine)
        try:
            first = client.post("/ask", json={"question": "a"})
            self.assertEqual(first.status_code, 502)
            engine.error = None
            second = client.post("/ask", json={"question": "b"})
            self.assertEqual(second.status_code, 200,
                             "异常后锁必须释放，否则第二次会一直等锁")
            self.assertEqual(second.json()["question"], "b")
        finally:
            client.close()


class ConstraintTests(unittest.TestCase):
    """把阶段 4 的约定变成断言，防止以后被无意破坏。"""

    def test_only_store_reads_the_api_key(self) -> None:
        offenders = []
        for path in PRODUCTION_PATHS:
            text = path.read_text(encoding="utf-8")
            if "ZHIPU_API_KEY" in text and path.name != "store.py":
                offenders.append(path.name)
        self.assertEqual(
            offenders, [],
            "只允许 retrieval/store.py 读取 ZHIPU_API_KEY；"
            f"这些文件也读了：{offenders}")

    def test_production_modules_do_not_import_config(self) -> None:
        offenders = []
        for path in PRODUCTION_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "").split(".")[0]]
                else:
                    continue
                if "config" in names:
                    offenders.append(path.name)
        self.assertEqual(offenders, [], f"这些文件仍在 import config：{offenders}")

    def test_routes_do_not_touch_engine_internal_state(self) -> None:
        # 路由只允许调用 engine.ask / engine.state / engine.settings
        text = (ROOT / "api" / "routes.py").read_text(encoding="utf-8")
        for forbidden in ("engine._", "engine.parents", "engine.retriever"):
            self.assertNotIn(forbidden, text)


class SchemaShapeTests(unittest.TestCase):
    """阶段 6 回归：响应模型必须能吃下 `pipeline.run()` 的真实返回形状。

    背景：S4 阶段六真机验收时 `POST /ask` 返回 500，
    `pydantic_core.ValidationError: citations.0.chars Field required`——
    原因是 `citations` 被声明成了 `list[SourceOut]`，而
    `qa.citation.build_citations()` 产出的引用只有 6 个字段、没有 `chars`。
    当时 FakeEngine 的测试数据正好带上了 chars，所以单元测试没暴露。
    这个用例直接调用真实的 `build_citations()`，不手写引用形状。
    """

    def test_citations_from_build_citations_validate(self) -> None:
        from api.schemas import AskResponse
        from qa.citation import build_citations

        result = fake_result()
        used, marked = build_citations(result["answer"], result["sources"])
        self.assertEqual(len(used), 1)
        self.assertNotIn("chars", used[0], "build_citations 不应该凭空加 chars")

        result["citations"] = used
        result["citations_marked"] = marked
        resp = AskResponse.from_result(result)

        self.assertEqual(resp.citations[0].n, 1)
        self.assertEqual(resp.citations[0].page, 8)
        self.assertEqual(resp.citations_marked, True)
        # 14 键一个不少，且 citations 是独立类型、不是 SourceOut
        dumped = resp.model_dump()
        self.assertEqual(set(dumped), EXPECTED_KEYS)
        self.assertEqual(type(resp.citations[0]).__name__, "CitationOut")


if __name__ == "__main__":
    unittest.main()
