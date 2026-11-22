# -*- coding: utf-8 -*-
"""DenseRetriever 的 prewarm/close 生命周期测试（真实 Chroma，不联网）。

用临时目录建一个只有 4 维向量的小集合：只验证"客户端打开一次、关闭一次、
关闭后不能再用"，不做任何检索质量断言。缺依赖时整类跳过。
"""
from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from retrieval.hybrid import DenseRetriever

TMP = Path(__file__).resolve().parent / "_tmp" / "dense"


def _deps_available() -> bool:
    try:
        import chromadb  # noqa: F401
        from llama_index.embeddings.zhipuai import ZhipuAIEmbedding  # noqa: F401
    except Exception:
        return False
    return True


@unittest.skipUnless(_deps_available(), "缺少 chromadb / llama-index")
class TestDenseRetrieverLifecycle(unittest.TestCase):
    def setUp(self):
        # chromadb 按"路径"缓存 System，同一路径重复建集合会撞 already exists，
        # 所以每个测试用独立目录（也顺带验证"不同目录互不影响"）。
        self.tmp = TMP / uuid.uuid4().hex[:8]
        self.tmp.mkdir(parents=True, exist_ok=True)

        import chromadb

        client = chromadb.PersistentClient(path=str(self.tmp))
        coll = client.create_collection("dense_test")
        coll.add(ids=["u1"], embeddings=[[0.1, 0.2, 0.3, 0.4]], documents=["虚构句子"])

        self.dense = DenseRetriever(
            [],
            chroma_dir=self.tmp,
            collection="dense_test",
            embed_model="embedding-3",
            api_base="https://example.invalid/api",
        )

    def tearDown(self):
        self.dense.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _prewarm(self):
        # 只为绕过"必须有 ZHIPU_API_KEY"的检查；ZhipuAIEmbedding 构造不会发请求
        with mock.patch("retrieval.hybrid.api_key", return_value="fake-key"):
            self.dense.prewarm()

    def test_prewarm_opens_client_once(self):
        self._prewarm()
        client = self.dense._client
        self.assertIsNotNone(client)
        self._prewarm()                     # 幂等：复用同一个 client
        self.assertIs(self.dense._client, client)

    def test_close_releases_client_and_guards_use(self):
        self._prewarm()
        self.dense.close()
        self.dense.close()                  # 重复关闭安全
        self.assertIsNone(self.dense._client)
        self.assertIsNone(self.dense._coll)

        with self.assertRaises(RuntimeError):
            self.dense.prewarm()
        with self.assertRaises(RuntimeError):
            self.dense.retrieve("问题", 5)

    def test_close_without_prewarm_is_safe(self):
        self.dense.close()
        with self.assertRaises(RuntimeError):
            self.dense.prewarm()


if __name__ == "__main__":
    unittest.main()
