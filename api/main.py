# -*- coding: utf-8 -*-
"""uvicorn 的加载入口：`uvicorn api.main:app --workers 1`。

为什么 Settings 在这里创建：
    进程入口是唯一适合读环境变量的地方（`Settings.from_env()` 是项目里唯一的
    RAG_* 配置读取入口）。`create_app()` 本身不碰磁盘、不连 Chroma，
    那些都发生在 lifespan 里。

为什么必须 `--workers 1`：
    每个 worker 都会各建一个 Engine、各打开一次 Chroma。两个进程同时指向
    同一个数据目录没有做过验证（chromadb 的按路径缓存在进程内有效，跨进程
    会变成两个客户端）。所以启动命令固定单 worker，问答由应用内锁串行化。
"""
from __future__ import annotations

from api.app import create_app
from settings import Settings

#: ASGI 应用对象；`uvicorn api.main:app` 加载的就是它
app = create_app(Settings.from_env())
