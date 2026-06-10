"""辩论导演 — 将原始辩词提炼为戏剧性、易读的竞技场展示文本

核心功能：
- 对每条发言生成「竞技场摘要」（arena_summary）：精简保留核心论点与冲突，
  使用 markdown 格式（**观点**/**论述** 标签突出，换行分隔）
- 对每条发言生成「气泡咒语」（arena_bubble）：一句话精简表达，
  像 RPG 法术咒语般简练有力

优化：
- 论述部分更详尽，保留2-4个关键论据并引用证据
- 注入已处理消息的上下文，让导演了解辩论进程
- 气泡放宽至15字，允许更丰富的表达
"""

import json
import re
from pathlib import Path

from openai import OpenAI


DIRECTOR_SYSTEM = """\
你是一个RPG辩论竞技场的戏剧导演。你的任务是将辩手的原始发言提炼为两种精简表达。
你将看到辩论的上下文（此前发言的摘要），请综合考虑整个辩论进程，避免重复、突出新论点和冲突升级。

## 输出格式

对每条发言，你必须严格按照以下格式输出（不要输出其他内容）：

【竞技场】
**观点**：[核心立场/预测，一句话]
**论述**：[2-4个关键论据，保留证据编号引用（如E001），精简但完整]
【气泡】[一句法术咒语式的精简攻击/防守语，15字以内]

## 规则

1. **竞技场摘要**：
   - 必须保留核心立场和预测方向
   - 论述保留2-4个最具冲击力的论据，关键数据不可省略
   - 如果原文引用了证据（E001等格式），必须在论述中保留引用
   - 使用 **观点** 和 **论述** 作为标签（必须用**粗体**包裹）
   - 观点和论述必须换行分隔
   - 总长度控制在150字以内
   - 综合考虑辩论上下文：如果前面的发言已提出某论点，此处应突出新的论据或反驳，而非重复

2. **气泡咒语**：
   - 像RPG法术名称一样精练有力
   - 质疑类：攻击性，如"数据不攻自破！"、"此论纯属虚妄！"
   - 回应/防守类：防守性，如"铁证如山！"、"事实胜于雄辩！"
   - 立场陈述类：宣告性，如"胜局已定！"、"数据铁证在此！"
   - 反质疑类：反击性，如"倒打一耙！"、"以彼之道还施彼身！"
   - 必须在15字以内
   - 不要加引号或标点结尾

3. **冲突优先**：
   - 突出与对手观点的对立和冲突升级
   - 强调证据和数据的冲击力
   - 保留最具说服力的关键信息
   - 注意辩论进程：随着辩论深入，冲突应逐步升级而非原地踏步
"""

DIRECTOR_USER_TEMPLATE = """\
辩题：{topic}
当前发言者：{speaker}（{perspective}）
行为：{action}{target_info}
当前预测：{prediction}
{prior_context}
原始发言：
{content}

请按格式输出精简表达："""


def _parse_director_output(text: str) -> dict:
    """解析导演输出，提取竞技场摘要和气泡咒语"""
    result = {"arena_summary": "", "arena_bubble": ""}

    # 提取竞技场摘要
    arena_match = re.search(r"【竞技场】\s*(.+?)(?=【气泡】|$)", text, re.DOTALL)
    if arena_match:
        result["arena_summary"] = arena_match.group(1).strip()

    # 提取气泡咒语
    bubble_match = re.search(r"【气泡】(.+?)(?:\n|$)", text)
    if bubble_match:
        result["arena_bubble"] = bubble_match.group(1).strip()

    return result


def process_debate_json(
    data: dict,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    batch_size: int = 5,
    verbose: bool = False,
) -> dict:
    """对辩论 JSON 数据进行导演提炼，添加 arena_summary 和 arena_bubble 字段

    Args:
        data: 原始辩论 JSON 数据（dict）
        client: OpenAI 客户端
        model: 使用的 LLM 模型
        temperature: 生成温度
        batch_size: 每批处理的消息数（减少API调用次数）
        verbose: 是否输出处理进度

    Returns:
        添加了浓缩字段的数据（原地修改并返回）
    """
    topic = data.get("topic", "")
    agents_map = {a["name"]: a for a in data.get("agents", [])}

    # 收集所有需要处理的消息
    messages_to_process = []
    for round_msgs in data.get("rounds", []):
        for msg in round_msgs:
            if msg.get("role") == "assistant" and msg.get("speaker"):
                messages_to_process.append(msg)

    # 也处理仲裁/总结等特殊消息
    for key in ("arbitrator_verdict", "verdict", "summary"):
        special = data.get(key)
        if special and special.get("content"):
            messages_to_process.append(special)

    # 也处理每阶段仲裁消息（phase_verdicts）
    for pv in data.get("phase_verdicts", []):
        if pv.get("content"):
            # 构造与普通消息一致的结构
            messages_to_process.append({
                "speaker": pv.get("speaker", "仲裁者"),
                "action": "仲裁",
                "target": "",
                "content": pv["content"],
                "role": "assistant",
            })

    if not messages_to_process:
        return data

    # 维护辩论上下文：最近4条已处理消息的摘要
    prior_summaries: list[str] = []
    MAX_PRIOR = 4

    # 批量处理
    total = len(messages_to_process)
    for i in range(0, total, batch_size):
        batch = messages_to_process[i:i + batch_size]

        for msg in batch:
            speaker = msg.get("speaker", "")
            action = msg.get("action", "立场")
            target = msg.get("target", "")
            content = msg.get("content", "")
            perspective = agents_map.get(speaker, {}).get("perspective_short", "")
            prediction = msg.get("speaker_prediction", "")

            # 如果消息太短（<20字），直接用原文
            if len(content) < 20:
                msg["arena_summary"] = content
                msg["arena_bubble"] = content[:15]
                # 也加入上下文
                prior_summaries.append(f"{speaker}：{content[:30]}")
                if len(prior_summaries) > MAX_PRIOR:
                    prior_summaries.pop(0)
                continue

            target_info = f"→{target}" if target else ""

            # 构建辩论上下文
            prior_context = ""
            if prior_summaries:
                ctx_lines = "\n".join(prior_summaries)
                prior_context = f"\n此前发言摘要：\n{ctx_lines}"

            user_prompt = DIRECTOR_USER_TEMPLATE.format(
                topic=topic,
                speaker=speaker,
                perspective=perspective,
                action=action,
                target_info=target_info,
                prediction=prediction,
                prior_context=prior_context,
                content=content[:1500],  # 放宽截断以保留证据引用
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": DIRECTOR_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=400,
                )
                output = response.choices[0].message.content.strip()
                parsed = _parse_director_output(output)
                msg["arena_summary"] = parsed["arena_summary"]
                msg["arena_bubble"] = parsed["arena_bubble"]
            except Exception as e:
                if verbose:
                    print(f"  ⚠️ 导演处理失败（{speaker}）: {e}")
                # 降级：用原文的前120字作为摘要，前15字作为气泡
                msg["arena_summary"] = content[:120] + ("..." if len(content) > 120 else "")
                msg["arena_bubble"] = content[:15].replace("\n", " ")

            # 更新辩论上下文
            summary_brief = f"{speaker}（{action}）：{msg['arena_summary'][:50]}"
            prior_summaries.append(summary_brief)
            if len(prior_summaries) > MAX_PRIOR:
                prior_summaries.pop(0)

        if verbose:
            print(f"  🎬 导演处理进度：{min(i + batch_size, total)}/{total}")

    # ── 生成辩手立场档案 ──────────────────────────────────
    generate_agent_profiles(data, client, model, verbose=verbose)

    return data


def generate_agent_profiles(
    data: dict,
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.5,
    verbose: bool = False,
) -> None:
    """为每位辩手生成立场档案，写入 data["agent_profiles"]

    在所有消息处理完毕后调用。批量提交所有辩手信息，
    一次 LLM 调用生成所有档案。
    """
    agents_data = data.get("agents", [])
    if not agents_data:
        return

    # 收集每位辩手的信息
    agents_info: list[str] = []
    for agent in agents_data:
        name = agent.get("name", "")
        perspective = agent.get("perspective_short", "")
        history = agent.get("prediction_history", [])
        stance = agent.get("stance", "")
        assigned_side = agent.get("assigned_side", "")

        # 初始预测
        initial = history[0] if history else {}
        initial_pred = f"{initial.get('winner', '?')}{initial.get('score', '?')}准{initial.get('confidence', '?')}"

        # 终局预测
        final = history[-1] if history else {}
        final_pred = f"{final.get('winner', '?')}{final.get('score', '?')}准{final.get('confidence', '?')}"

        # 观点变化节点
        changes: list[str] = []
        for entry in history:
            if entry.get("changed") and entry.get("count", 0) > 1:
                changes.append(
                    f"第{entry['count']}次论述：{entry.get('winner', '')}{entry.get('score', '')}准{entry.get('confidence', '')}"
                )

        # 收集该辩手的 arena_summary
        summaries: list[str] = []
        for round_msgs in data.get("rounds", []):
            for msg in round_msgs:
                if msg.get("speaker") == name and msg.get("arena_summary"):
                    summaries.append(msg["arena_summary"][:80])

        side_label = f"（{stance}）" if stance else ""
        side_label += f" [支持{assigned_side}]" if assigned_side else ""

        info_parts = [
            f"【辩手】{name}{side_label}",
            f"视角：{perspective}",
            f"初始预测：{initial_pred}",
            f"终局预测：{final_pred}",
        ]
        if changes:
            info_parts.append(f"观点变化：{'；'.join(changes)}")
        else:
            info_parts.append("观点变化：无")
        if summaries:
            # 最多取3条摘要
            for i, s in enumerate(summaries[:3]):
                info_parts.append(f"摘要{i+1}：{s}")

        agents_info.append("\n".join(info_parts))

    if not agents_info:
        return

    PROFILE_SYSTEM = """\
你是一个辩论竞技场的叙事导演。请为每位辩手生成一份简洁的立场档案。

## 输出格式

对每位辩手，严格按照以下格式输出（每位之间空一行）：

【辩手】{name}
【初始】{winner} {score}（准{confidence}）— {一句话概括初始理由，30字以内}
【演变】{立场坚定 / 描述转变原因，一句话40字以内}
【终局】{winner} {score}（准{confidence}）

## 规则
- 初始理由突出该辩手最独特的论点，不要重复其他辩手已说过的
- 演变指出转折原因（如"被X质疑后转变立场"）；立场未变则写"立场坚定"
- 每位辩手总长度控制在80字以内
- 不要输出其他内容"""

    PROFILE_USER = f"""\
辩题：{data.get('topic', '')}

各位辩手信息：
{chr(10).join(agents_info)}

请按格式输出每位辩手的立场档案："""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROFILE_SYSTEM},
                {"role": "user", "content": PROFILE_USER},
            ],
            temperature=temperature,
            max_tokens=600,
        )
        output = response.choices[0].message.content.strip()
        profiles = _parse_agent_profiles(output, agents_data)
        data["agent_profiles"] = profiles
    except Exception as e:
        if verbose:
            print(f"  ⚠️ 辩手档案生成失败：{e}")
        # 降级：从现有数据拼凑
        data["agent_profiles"] = _fallback_agent_profiles(agents_data)


def _parse_agent_profiles(text: str, agents_data: list[dict]) -> dict:
    """解析导演输出的辩手档案"""
    profiles: dict = {}

    # 按【辩手】分割
    blocks = re.split(r"【辩手】", text)
    for block in blocks[1:]:  # 跳过第一个空块
        block = block.strip()
        if not block:
            continue

        # 提取辩手名
        name_match = re.match(r"(.+?)(?:\n|【)", block)
        name = name_match.group(1).strip() if name_match else ""
        if not name:
            continue

        # 清理名字中的附加信息
        name = re.sub(r"[（(].+?[）)]", "", name).strip()
        name = re.sub(r"\[.+?\]", "", name).strip()

        profile: dict = {}

        # 提取初始
        initial_match = re.search(r"【初始】(.+?)(?:\n|【|$)", block)
        if initial_match:
            profile["initial"] = initial_match.group(1).strip()

        # 提取演变
        evolution_match = re.search(r"【演变】(.+?)(?:\n|【|$)", block)
        if evolution_match:
            profile["evolution"] = evolution_match.group(1).strip()

        # 提取终局
        final_match = re.search(r"【终局】(.+?)(?:\n|$)", block)
        if final_match:
            profile["final"] = final_match.group(1).strip()

        if profile:
            profiles[name] = profile

    return profiles


def _fallback_agent_profiles(agents_data: list[dict]) -> dict:
    """降级：从现有数据拼凑辩手档案"""
    profiles: dict = {}
    for agent in agents_data:
        name = agent.get("name", "")
        history = agent.get("prediction_history", [])
        if not history:
            continue

        initial = history[0]
        final = history[-1]
        changed = any(e.get("changed") and e.get("count", 0) > 1 for e in history)

        initial_str = f"{initial.get('winner', '')}{initial.get('score', '')}（准{initial.get('confidence', '')}）"
        final_str = f"{final.get('winner', '')}{final.get('score', '')}（准{final.get('confidence', '')}）"

        profiles[name] = {
            "initial": initial_str,
            "evolution": "观点有变化" if changed else "立场坚定",
            "final": final_str,
        }

    return profiles


def process_json_file(
    input_path: str,
    output_path: str = "",
    client: OpenAI | None = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> str:
    """对现有 JSON 文件进行导演提炼

    Args:
        input_path: 输入 JSON 文件路径
        output_path: 输出路径（默认覆盖原文件）
        client: OpenAI 客户端
        model: LLM 模型
        verbose: 是否输出进度

    Returns:
        输出文件路径
    """
    if client is None:
        client = OpenAI()

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    process_debate_json(data, client, model, verbose=verbose)

    out = output_path or input_path
    Path(out).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="辩论导演 — 提炼竞技场展示文本")
    parser.add_argument("input", help="输入 JSON 文件路径")
    parser.add_argument("-o", "--output", default="", help="输出路径（默认覆盖）")
    parser.add_argument("-m", "--model", default="", help="LLM 模型（留空则从 .env 配置读取）")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示进度")
    args = parser.parse_args()

    # 自动加载 .env 中的 LLM 配置（与系统其他 agent 一致）
    if not args.model:
        try:
            from .config import load_llm_config
            llm = load_llm_config()
            client = OpenAI(api_key=llm.api_key, base_url=llm.base_url)
            model = llm.model
            print(f"  使用配置：{model} @ {llm.base_url}")
        except Exception:
            # fallback：尝试从环境变量直接创建
            import os
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            if not api_key:
                print("❌ 未找到 API 配置，请设置 .env 或使用 -m 指定模型")
                exit(1)
            client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI()
        model = args.model

    result = process_json_file(args.input, args.output, client=client, model=model, verbose=args.verbose)
    print(f"✅ 导演处理完成：{result}")
