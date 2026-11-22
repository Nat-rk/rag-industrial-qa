# -*- coding: utf-8 -*-
"""服务层：S3 的 RagEngine 在这里；S4 的 FastAPI 应用也放这一层。

这一层只做"把长期资源管起来、把请求转给 qa/pipeline.py"，
不实现检索算法，也不定义 HTTP 状态码。
"""
