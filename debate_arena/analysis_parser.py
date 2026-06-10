"""百度体育分析数据解析器

解析 analysis_data.json（百度体育比赛分析页数据），提取：
- 有利/不利情报 → 注入提示词 + 归入证据池
- 近期战绩/交锋记录 → 注入提示词（替代 YAML 中重复数据）

排除项（不读取、不注入）：
- result.percentage — 胜负预测百分比
- result.team[].winrate — 球队胜率
- guess.* — 球迷投票数据
"""

import json
from pathlib import Path

from pydantic import BaseModel


# ── 数据模型 ──────────────────────────────────────────────


class IntelItem(BaseModel):
    """一条情报"""

    team: str       # "韩国" / "捷克"
    content: str    # 情报文本
    category: str   # "有利情报" / "不利情报"


class MatchRecordSummary(BaseModel):
    """一组战绩摘要"""

    title: str              # "历史战绩" / "韩国近期战绩"
    subtitle: str           # "以下数据均为主队(韩国)视角"
    result_summary: str     # "4胜0平2负"
    matches: list[str]      # ["2026-06-04 韩国 1-0 萨尔瓦多 (国际友谊)", ...]


class AnalysisData(BaseModel):
    """解析后的分析数据（不含任何预测信息）"""

    match_number: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    intel_items: list[IntelItem] = []
    record_summaries: list[MatchRecordSummary] = []


# ── 解析 ──────────────────────────────────────────────────


def parse_analysis_json(filepath: str) -> AnalysisData:
    """解析百度体育分析 JSON，提取情报+战绩，排除所有预测数据

    排除的预测数据：
    - result.percentage（胜负百分比）
    - result.team[].winrate（球队胜率）
    - guess.*（球迷投票）

    仅保留 result.num（赛次编号，用作标识符）。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"分析数据文件不存在：{filepath}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # 赛次编号（仅用作标识符，不含预测信息）
    match_number = data.get("result", {}).get("num")

    # 解析情报
    intel_items = _extract_intel(data.get("igence", []))

    # 推断主客队名
    home_team = None
    away_team = None
    if intel_items:
        teams = {item.team for item in intel_items}
        if len(teams) >= 2:
            home_team = intel_items[0].team
            away_team = intel_items[-1].team
            if home_team == away_team and len(teams) == 2:
                away_team = list(teams - {home_team})[0]

    # 从 igence 结构直接取队名（更可靠）
    igence = data.get("igence", [])
    if igence:
        first_intel = igence[0].get("intelligence", {})
        team_info = first_intel.get("intelligenceTeamInfo", {})
        leater_info = first_intel.get("intelligenceteamLeaterInfo", {})
        if team_info.get("name"):
            home_team = team_info["name"]
        if leater_info.get("name"):
            away_team = leater_info["name"]

    # 解析战绩
    record_summaries = _extract_records(data.get("homeRecord", []))

    return AnalysisData(
        match_number=match_number,
        home_team=home_team,
        away_team=away_team,
        intel_items=intel_items,
        record_summaries=record_summaries,
    )


def _extract_intel(igence_list: list) -> list[IntelItem]:
    """从 igence 数组提取情报条目"""
    items: list[IntelItem] = []
    for igence_entry in igence_list:
        intel = igence_entry.get("intelligence", {})
        category = igence_entry.get("intelligencetitle", "")

        # 主队情报
        home_name = intel.get("intelligenceTeamInfo", {}).get("name", "主队")
        for entry in intel.get("intelligenceteam", []):
            content = entry.get("content", "").strip()
            if content:
                items.append(IntelItem(team=home_name, content=content, category=category))

        # 客队情报
        away_name = intel.get("intelligenceteamLeaterInfo", {}).get("name", "客队")
        for entry in intel.get("intelligenceteamleater", []):
            content = entry.get("content", "").strip()
            if content:
                items.append(IntelItem(team=away_name, content=content, category=category))

    return items


def _extract_records(home_record_list: list) -> list[MatchRecordSummary]:
    """从 homeRecord 数组提取战绩摘要"""
    summaries: list[MatchRecordSummary] = []
    for record_group in home_record_list:
        history = record_group.get("history", {})
        if not history:
            continue

        title = history.get("title", "战绩")
        subtitle = history.get("subTitle", "")

        # 胜率摘要
        result_summary = _extract_result_summary(history.get("probability", []))

        # 比赛列表（最多8场）
        match_list = history.get("list", [])
        matches = []
        for entry in match_list[:8]:
            formatted = _format_match_entry(entry)
            if formatted:
                matches.append(formatted)

        if result_summary or matches:
            summaries.append(MatchRecordSummary(
                title=title,
                subtitle=subtitle,
                result_summary=result_summary,
                matches=matches,
            ))

    return summaries


def _extract_result_summary(probability: list) -> str:
    """从 probability 数组提取胜率摘要"""
    if not probability:
        return ""

    # 第一个 probability 是胜率
    first = probability[0]
    win = first.get("win", {}).get("value", 0)
    draw = first.get("draw", {}).get("value", 0)
    loss = first.get("loss", {}).get("value", 0)

    parts = []
    if win:
        parts.append(f"{win}胜")
    if draw:
        parts.append(f"{draw}平")
    if loss:
        parts.append(f"{loss}负")

    return "".join(parts) if parts else ""


def _format_match_entry(entry: dict) -> str:
    """格式化单场比赛条目

    输出：'2026-06-04 韩国 1-0 萨尔瓦多 (国际友谊)'
    省略赔率数据。
    """
    date = entry.get("date", "")
    left = entry.get("left", {})
    right = entry.get("right", {})
    match_name = entry.get("match", "")
    vs = entry.get("vs", "")

    left_name = left.get("name", "")
    right_name = right.get("name", "")

    if not left_name or not right_name:
        return ""

    parts = [date, f"{left_name} {vs} {right_name}"]
    if match_name:
        parts.append(f"({match_name})")

    return " ".join(parts)


# ── 格式化 ────────────────────────────────────────────────


def format_analysis_for_prompt(data: AnalysisData) -> str:
    """格式化分析数据为提示词文本

    输出段：## 战绩记录（有利/不利情报已归入证据池，不在此处注入 match_context）
    战绩每类最多8场，省略赔率数据。
    """
    if not data.record_summaries:
        return ""

    sections: list[str] = []

    # 战绩记录
    if data.record_summaries:
        lines = ["## 战绩记录", ""]
        for rec in data.record_summaries:
            header = rec.title
            if rec.subtitle:
                header += f"（{rec.subtitle}）"
            lines.append(f"### {header}")
            if rec.result_summary:
                lines.append(f"总计：{rec.result_summary}")
            for match_str in rec.matches:
                lines.append(f"- {match_str}")
            lines.append("")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def get_intel_as_evidence_items(data: AnalysisData) -> list[str]:
    """返回情报条目列表，可直接传入 evidence_pool.add_local()

    每条格式：[有利情报·韩国] 韩国6场比赛4胜0平，状态出色。
    """
    items: list[str] = []
    for intel in data.intel_items:
        items.append(f"[{intel.category}·{intel.team}] {intel.content}")
    return items
