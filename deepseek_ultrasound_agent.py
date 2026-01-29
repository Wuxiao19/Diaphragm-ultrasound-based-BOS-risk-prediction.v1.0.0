"""
使用 DeepSeek-R1-Distill-Qwen-7B + FastMCP + 你的 MCP 工具（agent_et_mcp.py）的 Agent 示例。

设计思路（尽量简单稳定）：
1. 还是用你现有的 MCP 工具负责“算概率”（detect_single_pair / detect_batch_folders）；
2. DeepSeek 模型只负责“看 JSON 结果 + 用中文解释和总结”，不亲自调工具，这样不依赖函数调用特性，也更稳。

你可以先用这个脚本跑通一轮“单组 B/M 图片 → DetectionPipeline → DeepSeek 解释”的闭环。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI


# ============================================================
# 环境变量 & DeepSeek 客户端初始化
# ============================================================

DEFAULT_DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def get_deepseek_client(api_key: str, base_url: str = DEFAULT_DEEPSEEK_BASE_URL) -> OpenAI:
    """
    获取一个 OpenAI-兼容客户端，用于调用 DeepSeek-R1-Distill-Qwen-7B（SiliconFlow/OpenAI兼容接口）。

    注意：
    - 不要把 API key 写死在代码里；建议用环境变量或 Streamlit secrets/输入框传入。
    """
    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# MCP 工具调用封装（沿用你已有的 DetectionPipeline）
# ============================================================


async def mcp_detect_single_pair(
    b_image_path: str,
    m_image_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    通过 FastMCP Client 调用 MCP 工具 detect_single_pair，获取单组 B/M 检测结果。
    """
    async with Client(mcp_entry) as client:
        result = await client.call_tool(
            "detect_single_pair",
            {
                "b_image_path": b_image_path,
                "m_image_path": m_image_path,
            },
        )
        # FastMCP 一般返回 Content 列表，这里尽量解析成 dict
        if isinstance(result, list) and result:
            first = result[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}


async def mcp_detect_batch_folders(
    b_folder_path: str,
    m_folder_path: str,
    mcp_entry: str = "agent_et_mcp.py",
) -> Dict[str, Any]:
    """
    通过 FastMCP Client 调用 MCP 工具 detect_batch_folders，获取批量 B/M 检测结果。
    """
    async with Client(mcp_entry) as client:
        result = await client.call_tool(
            "detect_batch_folders",
            {
                "b_folder_path": b_folder_path,
                "m_folder_path": m_folder_path,
            },
        )
        if isinstance(result, list) and result:
            first = result[0]
            text = getattr(first, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except Exception:
                    return {"raw": text}
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}


# ============================================================
# DeepSeek 负责“看结果 + 说人话”的部分
# ============================================================


DEEPSEEK_SYSTEM_PROMPT = """
你是一名资深的重症医学和呼吸康复专家，熟悉基于膈肌 B 模式和 M 模式超声图像的风险评估模型。

你会收到一份 JSON 结构的“自动检测结果”，其中包含：
- 对于单个患者：merged_key、risk_probability、prediction、prediction_label 等字段；
- 对于多个患者：上述字段的列表、总样本数、平均风险等信息。

你的任务：
1. 用清晰、专业但尽量易懂的中文解释这些检测结果的含义；
2. 根据 risk_probability 对风险进行分级（如很低、较低、中等、较高、很高）；
3. 如果是批量结果，请指出其中明显高风险的患者（例如风险概率 > 0.7），并列出他们的 ID 和概率；
4. 给出 1~3 条临床或管理上的建议（例如是否需要进一步检查、随访、注意点等）；
5. 明确提醒：这是基于图像的机器学习模型结果，不能替代医生的最终诊断。
"""


def deepseek_explain_detection_sync(
    detection_json: Dict[str, Any],
    user_intent: str,
    api_key: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 0.4,
) -> str:
    """
    同步调用 DeepSeek-R1-Distill-Qwen-7B，对检测 JSON 做中文解释。

    参数：
      - detection_json: 从 MCP 工具拿到的检测结果（单个或批量）
      - user_intent: 用户想知道的内容（例如“请解释这名患者的风险并给建议”）
    """
    # 把 JSON 格式化成字符串给模型看
    json_str = json.dumps(detection_json, ensure_ascii=False, indent=2)

    user_prompt = f"""
下面是自动检测模型给出的 JSON 结果：

{json_str}

用户问题/需求如下：
{user_intent}

请按照系统提示中的要求，给出一段完整的中文解读。
"""

    client = get_deepseek_client(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    # DeepSeek-OpenAI 接口风格：choices[0].message.content
    return resp.choices[0].message.content or ""


async def deepseek_explain_detection(
    detection_json: Dict[str, Any],
    user_intent: str,
    api_key: str,
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    temperature: float = 0.4,
) -> str:
    """
    异步包装：在需要 async 的场景下使用（内部仍是同步 HTTP 请求）。
    """
    return deepseek_explain_detection_sync(
        detection_json=detection_json,
        user_intent=user_intent,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


# ============================================================
# 示例：单组 B/M + DeepSeek 解释
# ============================================================


async def run_single_example() -> None:
    """
    示例：一组 B/M 图片 → MCP 检测 → DeepSeek 解释。

    你可以先用你之前已经跑通的那组图片路径，确认整个链路没问题。
    """
    # 你可以把这里换成你自己的图片路径
    b_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\B_model_one\25-08-11-B013_25-08-11-B013-L-Tdi-exp1.jpg"
    m_image_path = r"D:\A_SJTU\yolo\Miafex_RF\detect_pic\M_model_one\25-08-11-B013_25-08-11-B013-R-DE-DB1.jpg"

    if not Path(b_image_path).exists() or not Path(m_image_path).exists():
        print("❌ 示例 B/M 图片路径不存在，请先修改 deepseek_ultrasound_agent.py 中的路径。")
        return

    print("==============================================")
    print("  步骤1：通过 MCP 工具进行单组 B/M 检测")
    print("==============================================")
    detection = await mcp_detect_single_pair(b_image_path, m_image_path)
    print("原始检测结果（截断展示）：")
    print(json.dumps(detection, ensure_ascii=False, indent=2)[:600] + "...\n")

    print("==============================================")
    print("  步骤2：调用 DeepSeek 模型做中文解释")
    print("==============================================\n")

    user_intent = (
        "请根据这名患者的检测结果，说明大致患病风险概率和风险级别，并给出 1~3 条临床建议。"
    )
    # 说明：示例里为了方便，你可以直接在这里临时填 key；
    # 更推荐的做法：用环境变量/`.env`/Streamlit 输入框传入。
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY（环境变量或 .env），无法调用 DeepSeek。")
        return

    explanation = await deepseek_explain_detection(
        detection_json=detection,
        user_intent=user_intent,
        api_key=api_key,
    )

    print("******** DeepSeek 模型的中文解读 ********\n")
    print(explanation)
    print("\n****************************************\n")


async def main() -> None:
    """
    命令行入口：
      默认先跑一个“单组 B/M + 解释”的示例。
      之后你可以仿照写一个批量版本。
    """
    await run_single_example()


if __name__ == "__main__":
    asyncio.run(main())


