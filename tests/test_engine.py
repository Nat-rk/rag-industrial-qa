# -*- coding: utf-8 -*-
"""RagEngine 的离线测试：资源只加载一次、只构建一次、多次问答复用同一批对象。

这里用桩检索器 + patch 掉数据加载与 pipeline.run，验证的是**所有权与调用次数**，
不是检索质量；真实链路在 test_engine_smoke.py（需 key 与数据，手工跑）。
"""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from errors import DataError
from service import engine as engine_mod
from service.engine import CLOSED, NEW, READY, RagEngine
from settings import Settings

TMP = Path(__file__).resolve().parent / "_tmp" / "engine"


class StubRetriever:
    """只记录调用次数；不带 prewarm 的版本用来验证 bm25 分支。"""

    name = "stub"

    def __init__(self):
        self.retrieve_calls = 0
        self.prewarm_calls = 0
        self.close_calls = 0
        self.prewarm_error: Exception | None = None
        self.close_error: Exception | None = None

    def retrieve(self, qtext: str, n: int):
        self.retrieve_calls += 1
        return []

    def prewarm(self) -> None:
        self.prewarm_calls += 1
        if self.prewarm_error is not None:
            raise self.prewarm_error

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class PlainRetriever(StubRetriever):
    """没有 prewarm/close 的检索器（模拟 bm25：纯内存索引，无需释放）。"""
    prewarm = None
    close = None


def make_settings(name: str = "a", mode: str = "hybrid", **kw) -> Settings:
    TMP.mkdir(parents=True, exist_ok=True)
    return Settings(data_dir=TMP / name, retrieval_mode=mode, **kw)


class EngineTestCase(unittest.TestCase):
    """统一把数据加载与检索器构建换成桩，避免碰真实数据资产。"""

    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)
        self.parents = {"p1": object()}
        self.units = [{"id": "u1"}]
        self.patch_parents = mock.patch.object(
            engine_mod, "load_parents", return_value=self.parents)
        self.patch_units = mock.patch.object(
            engine_mod, "load_units", return_value=self.units)
        self.patch_build = mock.patch.object(
            engine_mod.pipeline, "build_retriever", side_effect=lambda *a, **k: StubRetriever())
        self.load_parents = self.patch_parents.start()
        self.load_units = self.patch_units.start()
        self.build_retriever = self.patch_build.start()
        self.addCleanup(self.patch_parents.stop)
        self.addCleanup(self.patch_units.stop)
        self.addCleanup(self.patch_build.stop)

    def make_engine(self, **kw) -> RagEngine:
        return RagEngine(make_settings(**kw))


class TestInitialize(EngineTestCase):
    def test_constructor_does_not_touch_disk(self):
        eng = self.make_engine()
        self.assertEqual(eng.state, NEW)
        self.assertFalse(eng.initialized)
        self.load_parents.assert_not_called()
        self.load_units.assert_not_called()
        self.build_retriever.assert_not_called()

    def test_initialize_loads_and_builds_once(self):
        eng = self.make_engine()
        eng.initialize()

        self.assertEqual(eng.state, READY)
        self.assertEqual(self.load_parents.call_count, 1)
        self.assertEqual(self.load_units.call_count, 1)
        self.assertEqual(self.build_retriever.call_count, 1)
        # 传给 build_retriever 的就是 Engine 自己那份 Settings 与刚加载的 units
        args = self.build_retriever.call_args.args
        self.assertIs(args[2], eng.settings)
        self.assertIs(args[1], eng.units)
        self.assertEqual(eng.parents, self.parents)

    def test_initialize_is_idempotent(self):
        eng = self.make_engine()
        eng.initialize()
        retriever = eng.retriever
        eng.initialize()

        self.assertEqual(self.load_parents.call_count, 1)
        self.assertEqual(self.load_units.call_count, 1)
        self.assertEqual(self.build_retriever.call_count, 1)
        self.assertIs(eng.retriever, retriever)

    def test_initialize_prewarms_hybrid(self):
        eng = self.make_engine(mode="hybrid")
        eng.initialize()
        self.assertEqual(eng.retriever.prewarm_calls, 1)

    def test_initialize_skips_prewarm_when_absent(self):
        self.build_retriever.side_effect = lambda *a, **k: PlainRetriever()
        eng = self.make_engine(mode="bm25")
        eng.initialize()                       # 不应因为"没有 prewarm"而报错
        self.assertEqual(eng.state, READY)

    def test_initialize_failure_returns_to_new(self):
        self.load_parents.side_effect = DataError("parents.json 不存在")
        eng = self.make_engine()
        with self.assertRaises(DataError):
            eng.initialize()

        self.assertEqual(eng.state, NEW)
        self.assertIsNone(eng.parents)
        self.assertIsNone(eng.units)
        self.assertIsNone(eng.retriever)
        # 修好数据后可以重新初始化
        self.load_parents.side_effect = None
        eng.initialize()
        self.assertEqual(eng.state, READY)

    def test_prewarm_failure_cleans_up_and_raises_original(self):
        boom = RuntimeError("chroma 打不开")
        stub = StubRetriever()
        stub.prewarm_error = boom
        self.build_retriever.side_effect = lambda *a, **k: stub
        eng = self.make_engine(mode="hybrid")

        with self.assertRaises(RuntimeError) as ctx:
            eng.initialize()

        self.assertIs(ctx.exception, boom)
        self.assertEqual(eng.state, NEW)
        self.assertEqual(stub.close_calls, 1)
        self.assertIsNone(eng.retriever)

    def test_cleanup_error_does_not_mask_initialize_error(self):
        original = RuntimeError("预热失败")
        stub = StubRetriever()
        stub.prewarm_error = original
        stub.close_error = RuntimeError("close 也炸了")
        self.build_retriever.side_effect = lambda *a, **k: stub
        eng = self.make_engine()

        with self.assertRaises(RuntimeError) as ctx:
            eng.initialize()

        self.assertIs(ctx.exception, original)


class TestResourceReuse(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.run_patch = mock.patch.object(
            engine_mod.pipeline, "run",
            side_effect=lambda q, **kw: {"question": q, "kw": kw},
        )
        self.run = self.run_patch.start()
        self.addCleanup(self.run_patch.stop)

    def test_three_asks_reuse_same_objects(self):
        eng = self.make_engine()
        eng.initialize()
        parents, units, retriever = eng.parents, eng.units, eng.retriever

        results = [eng.ask(f"问题{i}") for i in range(3)]

        self.assertEqual(self.run.call_count, 3)
        for i, res in enumerate(results):
            self.assertEqual(res["question"], f"问题{i}")
            kw = res["kw"]
            self.assertIs(kw["parents"], parents)
            self.assertIs(kw["units"], units)
            self.assertIs(kw["retriever"], retriever)
            self.assertIs(kw["settings"], eng.settings)

    def test_three_asks_do_not_reload_or_rebuild(self):
        eng = self.make_engine()
        eng.initialize()
        for i in range(3):
            eng.ask(f"问题{i}")

        self.assertEqual(self.load_parents.call_count, 1)
        self.assertEqual(self.load_units.call_count, 1)
        self.assertEqual(self.build_retriever.call_count, 1)
        self.assertEqual(eng.retriever.prewarm_calls, 1)


class TestIsolation(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)
        retriever_a, retriever_b = StubRetriever(), StubRetriever()
        self.stubs = [retriever_a, retriever_b]
        pending = [retriever_a, retriever_b]

        def build(mode, units, settings):
            return pending.pop(0)      # 按初始化顺序各给一个独立检索器

        self.patches = [
            mock.patch.object(engine_mod, "load_parents", return_value={"p": object()}),
            mock.patch.object(engine_mod, "load_units", return_value=[{"id": "u"}]),
            mock.patch.object(engine_mod.pipeline, "build_retriever", side_effect=build),
            mock.patch.object(engine_mod.pipeline, "run",
                              side_effect=lambda q, **kw: {"question": q, "mode": kw["retriever"].name}),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_two_engines_do_not_share_state(self):
        a = RagEngine(make_settings("a", mode="hybrid"))
        b = RagEngine(make_settings("b", mode="bm25"))
        a.initialize()
        b.initialize()

        self.assertIsNot(a.settings, b.settings)
        self.assertIsNot(a.retriever, b.retriever)
        self.assertIs(a.retriever, self.stubs[0])
        self.assertIs(b.retriever, self.stubs[1])

    def test_closing_one_engine_keeps_the_other_alive(self):
        a = RagEngine(make_settings("a", mode="hybrid"))
        b = RagEngine(make_settings("b", mode="hybrid"))
        a.initialize()
        b.initialize()

        a.close()
        self.assertEqual(a.state, CLOSED)
        self.assertEqual(b.state, READY)
        self.assertEqual(b.ask("还在吗")["question"], "还在吗")
        self.assertEqual(b.retriever.close_calls, 0)

    def test_engines_do_not_touch_cli_settings_singleton(self):
        import main

        with mock.patch.object(main, "settings",
                               side_effect=AssertionError("Engine 不许用 CLI 单例")):
            a = RagEngine(make_settings("a"))
            a.initialize()
            a.ask("问题")


class TestLifecycle(EngineTestCase):
    def test_ask_before_initialize_raises(self):
        eng = self.make_engine()
        with self.assertRaises(RuntimeError):
            eng.ask("问题")

    def test_close_before_initialize_is_safe(self):
        eng = self.make_engine()
        eng.close()
        eng.close()
        self.assertEqual(eng.state, CLOSED)

    def test_close_is_idempotent(self):
        eng = self.make_engine()
        eng.initialize()
        stub = eng.retriever
        eng.close()
        eng.close()

        self.assertEqual(eng.state, CLOSED)
        self.assertEqual(stub.close_calls, 1)
        self.assertIsNone(eng.retriever)
        self.assertIsNone(eng.parents)

    def test_ask_after_close_raises(self):
        eng = self.make_engine()
        eng.initialize()
        eng.close()
        with self.assertRaises(RuntimeError):
            eng.ask("问题")

    def test_initialize_after_close_raises(self):
        eng = self.make_engine()
        eng.initialize()
        eng.close()
        with self.assertRaises(RuntimeError):
            eng.initialize()

    def test_ask_after_failed_init_then_close(self):
        self.load_parents.side_effect = DataError("坏了")
        eng = self.make_engine()
        with self.assertRaises(DataError):
            eng.initialize()
        eng.close()
        self.assertEqual(eng.state, CLOSED)


if __name__ == "__main__":
    unittest.main()
