# -*- coding: utf-8 -*-
"""LLM 生成：把问题和上下文拼成 Prompt，调智谱 chat 接口拿回答。

Prompt 的九条约束不是装饰，每一条都对应一种真实失败：
    1. 只能依据上下文 —— 防止模型用自己训练时的知识冒充手册内容；
    2. 资料不足要说 —— 手册里没有的参数，编一个比说"没有"危害大得多；
    3. 同参数多取值要分场景/保留不确定性 —— 本手册有数字量、现场总线、编码器等多种应用，
       同一个参数在不同应用里取值不同，合并成一个数会直接误导现场调试；
    4. 句末标 [Source n] —— 引用环节要靠它判断"这句到底来自哪一块"；
    5. 引用指向最直接的那块 —— Q26 曾出现"5 张不同表都写了 1024 PPR，模型全列上"，
       引用过宽会让读者无法判断主证据；但允许"多块拼起来"时分别引用，不搞一刀切限条数；
    6. 不许引用不存在的编号 —— 模型曾写出 [Source 11]，而上下文只有 10 块；
    7. 参数值原样抄 —— 3-22 写成 3.22 或 322 都会误导现场调试；
    8. 不要复述整段 —— 现场要的是结论，不是把手册抄一遍；
    9. 不要把一张表的选项套到另一个参数上。
"""
from __future__ import annotations

import time

import requests

import config
from retrieval.store import api_key

SYSTEM_PROMPT = """你是工业变频器/伺服设备的技术支持工程师，正在根据《VLT Integrated Motion Controller Application Guide》回答现场工程师的问题。

必须遵守：
1. 只依据【文档片段】回答；片段里没有的内容，一律不要补充，也不要用你的通用知识代替。
2. 如果片段不足以回答，直接说"文档中没有找到足够的信息来回答这个问题"，并说明缺什么。
3. 同一参数在片段中出现多个不同取值时，先判断差异是否与问题相关：
   - 问题问的是"哪个参数 / 定义 / 一般做法"：给出最直接适用的取值即可，其他场景一句话带过
     （如"其他应用场景下另有取值"），不必逐条罗列每个例子；
   - 问题本身就在问具体数值，且片段显示不同场景取值不同：分场景列出，
     每类只引用最直接的一条来源；
   - 片段看不出差异原因：只如实并列，例如"文档中存在不同记录：… [Source 2]；… [Source 5]；
     当前片段不足以判断两者差异的原因"，不要自行编造原因；
   - 任何情况下都不要把不同场景下的取值合并成一个通用结论。
4. 每条关键结论后面标注来源编号，格式如 [Source 2]；没有来源支撑的话不要写。
5. 引用要指向直接支持该结论的那一块：同一个事实若多块都提到，优先引用与问题场景最贴近、内容最直接的那一块（通常排在最前面）；确实需要多块拼起来才能回答时，才分别引用。不要因为多个片段"都写了同一个数值"就把它们全部列出。
6. 只能引用【文档片段】里真实存在的编号（1 到片段总数），不要编造不存在的 [Source n]。
7. 参数编号、参数值、选项值、页码必须逐字照抄，不要改写、不要四舍五入、不要翻译成中文数字。
8. 回答用中文，先给结论，再给必要细节（参数号、取值、步骤），不要整段复述原文；
   普通问题保持简洁，一般不超过 5 行，不要为了"全面"把片段里所有同类例子都列一遍。
9. 不要把一张表的选项套到另一个参数上：如果问题问的是 1-00，就只引用 1-00 的选项行；
   控制字（8-10）的 bit 选项、其他参数（如 1-09）的模式列表都不能当成 1-00 的选项。

输出格式：
回答：
<正文，关键结论后标 [Source n]>

不要在末尾另列"参考来源"清单——系统会根据你写的 [Source n] 自动生成来源表，
你重复列一遍只会造成"引用了但正文没标"的错觉。
"""


def answer(question: str, context: str, sources: list[dict]):
    """调 LLM；返回 (回答文本, 用量, 耗时)。失败抛 RuntimeError，由 main 决定怎么处理。"""
    user_prompt = (
        f"【文档片段】\n{context}\n\n"
        f"【问题】\n{question}\n\n"
        "请按系统提示的格式回答。"
    )
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.LLM_TEMPERATURE,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}

    t0 = time.time()
    try:
        r = requests.post(f"{config.API_BASE}/chat/completions", json=payload,
                          headers=headers, timeout=config.LLM_TIMEOUT)
    except requests.RequestException as e:
        raise RuntimeError(f"LLM 请求失败：{type(e).__name__}: {e}") from e

    elapsed = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"LLM 返回 HTTP {r.status_code}: {r.text[:300]}")

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM 响应异常：{str(data)[:300]}")

    text = (choices[0].get("message") or {}).get("content", "").strip()
    return text, data.get("usage", {}), elapsed
