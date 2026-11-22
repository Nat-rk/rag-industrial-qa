# -*- coding: utf-8 -*-
"""S4 接口测试。

当前只覆盖阶段 2 的应用生命周期（A 组）；请求/响应、异常映射、并发测试会在
后续阶段追加到本文件。

Mock 边界（重要）：
    * 这里的 Engine 是 `FakeEngine`，**不是**真实 RagEngine；
    * `api_key()` 被 patch 成返回假 key，所以**不需要** ZHIPU_API_KEY；
    * 不调用真实 LLM、不读真实数据、不连 Chroma；
    * 真实链路由阶段 6 的 HTTP 冒烟单独验证（会明确标注）。
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

try:                                      # 干净环境（未 pip install）时整模块跳过
    from fastapi.testclient import TestClient

    from api.app import create_app
    from errors import ConfigError, DataError
    from settings import Settings
except ModuleNotFoundError as exc:        # pragma: no cover - 依赖缺失时走这里
    raise unittest.SkipTest(f"接口测试需要 fastapi：{exc}")

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    """最小的 Engine 替身：只记录调用次数与状态。"""

    def __init__(self) -> None:
        self.initialize_calls = 0
        self.close_calls = 0
        self.ask_calls: list[str] = []
        self.state = "new"
        self.parents = {"p1": object()}
        self.units = [{"id": "u1"}]
        self.init_error: Exception | None = None

    def initialize(self) -> None:
        self.initialize_calls += 1
        if self.init_error is not None:
            raise self.init_error
        self.state = "ready"

    def close(self) -> None:
        self.close_calls += 1
        self.state = "closed"

    def ask(self, question: str) -> dict:
        self.ask_calls.append(question)
        return {"question": question, "answer": "fake"}


def make_app(testcase: unittest.TestCase, engine: FakeEngine):
    """建一个用 FakeEngine 的应用；api_key 预检在整个测试期间放行。

    patch 必须活到 lifespan 执行完（也就是 `with TestClient(...)` 期间），
    所以挂在 testcase.addCleanup 上，而不是函数返回前就停掉。
    """
    patcher = mock.patch("api.app.api_key", return_value="fake-key")
    patcher.start()
    testcase.addCleanup(patcher.stop)
    return create_app(Settings(), engine_factory=lambda s: engine)


class LifecycleTest(unittest.TestCase):
    """A 组：应用工厂与 lifespan。"""

    def test_create_app_does_not_touch_engine(self) -> None:
        created: list[object] = []

        def factory(settings):
            created.append(settings)
            return FakeEngine()

        app = create_app(Settings(), engine_factory=factory)
        self.assertEqual(created, [])          # 建 app 时不调用工厂
        self.assertFalse(hasattr(app.state, "engine"))
        self.assertEqual(app.state.settings.__class__, Settings)

    def test_lifespan_initializes_exactly_once(self) -> None:
        engine = FakeEngine()
        app = make_app(self, engine)
        with TestClient(app) as client:
            self.assertEqual(client.get("/openapi.json").status_code, 200)
            self.assertEqual(engine.initialize_calls, 1)
            self.assertEqual(engine.state, "ready")
            self.assertIs(app.state.engine, engine)
            self.assertIsNotNone(app.state.ask_lock)
        # 退出后：关闭一次，状态清空
        self.assertEqual(engine.close_calls, 1)
        self.assertIsNone(app.state.engine)
        self.assertEqual(engine.state, "closed")

    def test_startup_failure_propagates_and_closes(self) -> None:
        engine = FakeEngine()
        engine.init_error = DataError("parents.json 不见了")
        app = make_app(self, engine)
        with self.assertRaises(DataError):
            with TestClient(app):
                pass
        # 局部引用清理：即使没进 app.state 也要 close
        self.assertEqual(engine.close_calls, 1)
        self.assertFalse(hasattr(app.state, "engine"))

    def test_missing_key_fails_before_initialize(self) -> None:
        engine = FakeEngine()
        with mock.patch("api.app.api_key",
                        side_effect=ConfigError("缺少 ZHIPU_API_KEY")):
            app = create_app(Settings(), engine_factory=lambda s: engine)
            with self.assertRaises(ConfigError):
                with TestClient(app):
                    pass
        # 预检失败就不该读数据、建索引
        self.assertEqual(engine.initialize_calls, 0)
        self.assertEqual(engine.close_calls, 1)

    def test_close_failure_does_not_mask_startup_error(self) -> None:
        engine = FakeEngine()
        engine.init_error = DataError("坏数据")
        engine.close = mock.Mock(side_effect=RuntimeError("关不掉"))  # type: ignore[method-assign]
        app = make_app(self, engine)
        with self.assertRaises(DataError):     # 原异常胜出
            with TestClient(app):
                pass
        engine.close.assert_called_once()

    def test_api_layer_does_not_import_cli_or_config(self) -> None:
        for name in ("api/app.py", "api/main.py"):
            tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    mods = [node.module or ""]
                else:
                    continue
                for m in mods:
                    self.assertNotIn(m, {"config", "main"},
                                     f"{name} 不应 import {m}")


if __name__ == "__main__":   # pragma: no cover
    unittest.main()
