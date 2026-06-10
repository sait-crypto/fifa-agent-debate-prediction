"""输出渲染 — 终端彩色 + Markdown 文件

终端输出增强：
- 扩展颜色调色板（bright 系列 + dim/italic/underline）
- 正方/反方 agent 分色显示
- 证据编号高亮，置信度色彩编码
- 置信度可视化条
- 更精致的分隔线

Markdown 输出增强：
- 证据展示：标题 + 摘要 + 链接格式（有 URL 时）
"""

import datetime
import re
from pathlib import Path

from .evidence import EvidencePool
from .models import DebateResult, Message, PredictionResult

# ═══════════════════════════════════════════════════════════
# ANSI 颜色码
# ═══════════════════════════════════════════════════════════

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

# Agent 配色方案
_PRO_COLORS = ["bright_green", "green", "cyan"]
_CON_COLORS = ["bright_red", "red", "magenta"]
_ROUNDTABLE_COLORS = ["bright_blue", "bright_magenta", "bright_cyan", "yellow", "green", "blue"]


def _color(text: str, color: str) -> str:
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"


def _bold(text: str) -> str:
    return f"{_COLORS['bold']}{text}{_COLORS['reset']}"


def _dim(text: str) -> str:
    return f"{_COLORS['dim']}{text}{_COLORS['reset']}"


def _confidence_color(confidence: float) -> str:
    """根据置信度返回颜色名"""
    if confidence >= 0.7:
        return "bright_green"
    elif confidence >= 0.4:
        return "yellow"
    else:
        return "bright_red"


def _confidence_bar(confidence: float, width: int = 8) -> str:
    """置信度可视化条：[██████░░] 0.70"""
    filled = round(confidence * width)
    empty = width - filled
    color = _confidence_color(confidence)
    bar = "█" * filled + "░" * empty
    return f"{_color(bar, color)} {confidence:.2f}"


def _separator(char: str = "─", length: int = 60) -> str:
    """分隔线"""
    return _dim(char * length)


class Renderer:
    """辩论结果渲染器"""

    def __init__(
        self,
        result: DebateResult,
        evidence_pool: EvidencePool | None = None,
        pro_names: list[str] | None = None,
        con_names: list[str] | None = None,
        agents: list | None = None,  # Agent 对象列表，用于获取论述计数和预测历史
        match_context: str = "",              # 整体上下文（赛事信息、阵容、战绩等）
        team_info: dict | None = None,        # 两队信息（含世界排名），来自 MatchConfig.get_team_info()
        match_meta: dict | None = None,       # 比赛元数据（stage/match_time等），来自 MatchConfig.get_match_meta()
        cli_truncate: int = 300,           # 终端显示截断字数
        filename_topic_length: int = 40,   # 文件名话题最大长度
    ):
        self.result = result
        self.evidence_pool = evidence_pool
        self.pro_names = set(pro_names or [])
        self.con_names = set(con_names or [])
        self.agents = agents or []
        self.match_context = match_context
        self.team_info = team_info or {}
        self.match_meta = match_meta or {}
        self._cli_truncate = cli_truncate
        self._filename_topic_length = filename_topic_length
        self._color_map: dict[str, str] = {}
        self._agent_counts: dict[str, int] = {}  # 追踪当前渲染到的论述序号
        self._build_color_map()

    def _build_color_map(self) -> None:
        """为每个 agent 分配颜色（正方绿系、反方红系、其他蓝系）"""
        # 正方 agents
        pro_seen = [n for n in self.pro_names if n]
        for i, name in enumerate(pro_seen):
            self._color_map[name] = _PRO_COLORS[i % len(_PRO_COLORS)]

        # 反方 agents
        con_seen = [n for n in self.con_names if n]
        for i, name in enumerate(con_seen):
            self._color_map[name] = _CON_COLORS[i % len(_CON_COLORS)]

        # 其他 agents（圆桌、仲裁者、审核员等）
        other_seen: list[str] = []
        for round_msgs in self.result.rounds:
            for msg in round_msgs:
                if msg.speaker and msg.speaker not in self._color_map and msg.speaker not in other_seen:
                    other_seen.append(msg.speaker)
        if self.result.verdict and self.result.verdict.speaker:
            if self.result.verdict.speaker not in self._color_map and self.result.verdict.speaker not in other_seen:
                other_seen.append(self.result.verdict.speaker)

        for i, name in enumerate(other_seen):
            self._color_map[name] = _ROUNDTABLE_COLORS[i % len(_ROUNDTABLE_COLORS)]

        # 固定角色颜色
        self._color_map["仲裁者"] = "bright_yellow"
        self._color_map["审核员"] = "bright_cyan"
        self._color_map["总结分析师"] = "bright_white"
        self._color_map["场记"] = "bright_magenta"

    def _get_stance_tag(self, speaker: str) -> str:
        """获取 agent 阵营标签"""
        if speaker in self.pro_names:
            return _color("正", "bright_green")
        elif speaker in self.con_names:
            return _color("反", "bright_red")
        return ""

    def _is_prediction_changed(self, speaker: str, statement_count: int) -> bool:
        """检查 agent 在第N次论述时预测是否变化"""
        for agent in self.agents:
            if agent.name == speaker:
                for entry in agent.prediction_history:
                    if entry["count"] == statement_count:
                        return entry.get("changed", False)
        return False

    def _get_prediction_label(self, speaker: str, statement_count: int) -> str:
        """获取 agent 在第N次论述时的预测标签（如 '韩国2-1信6'）"""
        for agent in self.agents:
            if agent.name == speaker:
                for entry in agent.prediction_history:
                    if entry["count"] == statement_count:
                        w = entry.get("winner", "")
                        s = entry.get("score", "")
                        c = entry.get("confidence", "")
                        return f"{w}{s}准{c}" if w else ""
        return ""

    def _get_latest_prediction_label(self, speaker: str) -> str:
        """获取 agent 当前（最新）的预测标签"""
        for agent in self.agents:
            if agent.name == speaker and agent.prediction_history:
                latest = agent.prediction_history[-1]
                w = latest.get("winner", "")
                s = latest.get("score", "")
                c = latest.get("confidence", "")
                return f"{w}{s}准{c}" if w else ""
        return ""

    def print_live(self) -> None:
        """终端彩色输出"""
        print()
        print(_bold(_color(f"  ⚽  {self.result.topic}", "bright_yellow")))
        mode_label = "正反方辩论" if self.result.mode == "pro_con" else "圆桌辩论"
        print(f"  模式：{_color(mode_label, 'bright_cyan')}")
        print(_separator("═"))

        # 证据池摘要
        if self.evidence_pool:
            print(f"  📚 {self.evidence_pool.format_summary()}")

        # 每轮输出
        phase_names = {0: "陈述", 1: "质疑/回应", 2: "分配质疑"}
        for round_num, round_msgs in enumerate(self.result.rounds):
            phase = phase_names.get(round_num, f"阶段{round_num+1}")
            print()
            print(f"  {_bold(_color(f'── {phase} ──', 'bright_blue'))}")
            for msg in round_msgs:
                self._print_message(msg)

        # 共识状态
        print()
        if self.result.consensus:
            print(_bold(_color("  ✅ 共识达成（预测+论述一致）！", "bright_green")))
        else:
            print(_bold(_color("  ❌ 未达成共识", "bright_yellow")))

        # 仲裁者共识评判
        if self.result.arbitrator_verdict:
            print()
            print(_bold(_color("  ⚖️  仲裁者共识评判", "bright_yellow")))
            print(_separator())
            self._print_message(self.result.arbitrator_verdict)
            print(_separator())

        # 仲裁者最终裁定（未共识时）
        if self.result.verdict and not self.result.consensus:
            print()
            print(_bold(_color("  ⚖️  仲裁者最终裁定", "bright_yellow")))
            print(_separator())
            self._print_message(self.result.verdict)
            print(_separator())

        # 最终预测
        if self.result.final_prediction:
            pred = self.result.final_prediction
            print()
            print(_separator("═"))
            print(_bold(_color("  🏆  最终预测", "bright_yellow")))
            print(_separator())
            winner_color = "bright_green" if pred.confidence >= 7 else "yellow"
            print(f"  胜方：{_bold(_color(pred.winner, winner_color))}")
            print(f"  比分：{_color(pred.score, 'bright_white')}")
            conf_bar = _confidence_bar(pred.confidence / 10)
            print(f"  准确度：{conf_bar}")
            if pred.key_factors:
                print(f"  依据：{', '.join(pred.key_factors)}")
            print(_separator("═"))

        # 总结分析师
        if self.result.summary:
            print()
            print(_bold(_color("  📝  总结分析师", "bright_white")))
            print(_separator())
            self._print_message(self.result.summary)
            print(_separator())

        # 预测汇总表
        self._print_prediction_summary()
        print()

    def _print_message(self, msg: Message) -> None:
        color = self._color_map.get(msg.speaker, "white")
        stance_tag = self._get_stance_tag(msg.speaker)

        # 论述计数：所有发言都计数，质疑也计数但标记为🔍
        if msg.speaker and msg.role == "assistant":
            self._agent_counts[msg.speaker] = self._agent_counts.get(msg.speaker, 0) + 1
            count = self._agent_counts[msg.speaker]
        else:
            count = 0

        # 构建标签：【发言人；第N次🔍/行为；→目标】
        tag = _color(f"【{msg.speaker}】", color)
        if stance_tag:
            tag = f"{stance_tag} {tag}"

        # 论述序号 + 行为标签
        is_challenge = (msg.action == "质疑")
        parts = []
        if count > 0:
            # 检查预测是否变化
            changed = self._is_prediction_changed(msg.speaker, count)
            change_marker = _color("⚡", "bright_yellow") if changed else ""
            # 质疑标记 🔍 表示不是完整观点表达
            challenge_tag = _color("🔍", "bright_magenta") if is_challenge else ""
            # 反质疑标记 ↩️
            cc_tag = _color("↩️反质疑", "bright_yellow") if getattr(msg, 'counter_challenge', False) else ""
            parts.append(f"第{count}次{challenge_tag}{cc_tag}{change_marker}")
        if msg.action:
            parts.append(msg.action)
        if msg.target:
            # 有明确对象时标注双方当前预测
            speaker_pred = self._get_prediction_label(msg.speaker, count) if count > 0 else self._get_latest_prediction_label(msg.speaker)
            target_pred = self._get_latest_prediction_label(msg.target)
            target_color = self._color_map.get(msg.target, "white")
            target_label = _color(msg.target, target_color)
            if target_pred:
                target_label += f"({_dim(target_pred)})"
            # 说话者预测加在发言人标签后（仅质疑/回应时显示）
            if speaker_pred and msg.action in ("质疑", "回应"):
                tag = f"{tag} ({_dim(speaker_pred)})"
            parts.append(f"→【{target_label}】")

        if parts:
            tag = f"{tag} {_dim('; '.join(parts))}"

        # 只显示前 N 字的摘要
        content = msg.content
        if len(content) > self._cli_truncate:
            content = content[:self._cli_truncate] + _dim("...")
        lines = content.split("\n")
        indented = "\n".join("    " + line for line in lines)
        print(f"  {tag}")
        print(indented)

    def _print_prediction_summary(self) -> None:
        """打印预测汇总表"""
        if not self.result.predictions:
            return
        print()
        print(_bold(_color("  📊  预测汇总", "bright_cyan")))
        print(_separator("─", 50))
        for agent_name, preds in self.result.predictions.items():
            if preds:
                latest = preds[-1]
                color = self._color_map.get(agent_name, "white")
                stance_tag = self._get_stance_tag(agent_name)
                tag = _color(f"  {agent_name}", color)
                if stance_tag:
                    tag = f"{stance_tag} {tag}"
                conf_color = "bright_green" if latest.confidence >= 7 else ("yellow" if latest.confidence >= 4 else "bright_red")
                conf_str = _color(f"{latest.confidence}/10", conf_color)
                print(f"{tag}: {_bold(latest.winner)} {latest.score} (准确度{conf_str})")
        print(_separator("─", 50))

    def save_markdown(self, path: str = "") -> str:
        """保存为 Markdown 文件到 debate_output/ 目录，返回文件路径

        如果未指定 path，自动生成带时间戳的文件名：
        debate_output/YYYY-MM-DD_HHMMSS_话题.md
        """
        if not path:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H%M%S")
            # 清理话题名中的特殊字符，用作文件名
            safe_topic = re.sub(r'[\\/:*?"<>|\s]', '_', self.result.topic)[:self._filename_topic_length]
            output_dir = Path("debate_output")
            output_dir.mkdir(exist_ok=True)
            path = str(output_dir / f"{timestamp}_{safe_topic}.md")
        lines: list[str] = []
        lines.append(f"# 比赛预测：{self.result.topic}")
        lines.append("")
        lines.append(f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 辩论模式：{'正反方' if self.result.mode == 'pro_con' else '圆桌'}")
        lines.append(f"> 共识状态：{'已达成' if self.result.consensus else '未达成'}")
        lines.append("")

        # 比赛上下文
        if self.match_context:
            lines.append("## 比赛上下文")
            lines.append("")
            lines.append(self.match_context)
            lines.append("")

        # 证据池
        if self.evidence_pool:
            lines.append("## 证据池")
            lines.append("")
            for ev in self.evidence_pool.shared:
                ev_line = self._format_evidence_md(ev)
                lines.append(f"- {ev_line}")
            for agent_name, evs in self.evidence_pool.private.items():
                private_evs = [e for e in evs if not e.shared]
                for ev in private_evs:
                    ev_line = self._format_evidence_md(ev)
                    lines.append(f"- {ev_line}")
            lines.append("")

        # 每轮记录
        md_agent_counts: dict[str, int] = {}  # Markdown 输出的论述计数
        for round_num, round_msgs in enumerate(self.result.rounds, 1):
            lines.append(f"## 第 {round_num} 轮")
            lines.append("")
            for msg in round_msgs:
                # 构建标题：发言人；第N次论述🔍；行为→目标(预测)
                parts = [msg.speaker]
                if msg.speaker and msg.role == "assistant":
                    md_agent_counts[msg.speaker] = md_agent_counts.get(msg.speaker, 0) + 1
                    count = md_agent_counts[msg.speaker]
                    is_challenge = (msg.action == "质疑")
                    changed = self._is_prediction_changed(msg.speaker, count)
                    change_marker = " ⚡观点变化" if changed else ""
                    challenge_tag = " 🔍质疑" if is_challenge else ""
                    cc_tag = " ↩️反质疑" if getattr(msg, 'counter_challenge', False) else ""
                    parts.append(f"第{count}次论述{challenge_tag}{cc_tag}{change_marker}")
                    # 说话者预测标注（仅质疑/回应时）
                    if msg.action in ("质疑", "回应"):
                        sp = self._get_prediction_label(msg.speaker, count)
                        if sp:
                            parts[0] = f"{msg.speaker}({sp})"
                if msg.action:
                    parts.append(msg.action)
                if msg.target:
                    tp = self._get_latest_prediction_label(msg.target)
                    target_str = f"{msg.target}({tp})" if tp else msg.target
                    parts.append(f"→{target_str}")
                title = "；".join(parts)
                lines.append(f"### {title}")
                lines.append("")
                lines.append(msg.content)
                lines.append("")

        # 预测汇总
        if self.result.predictions:
            lines.append("## 预测汇总")
            lines.append("")
            lines.append("| 参与者 | 预测胜方 | 预测比分 | 准确度 |")
            lines.append("|--------|----------|----------|------|")
            for agent_name, preds in self.result.predictions.items():
                if preds:
                    p = preds[-1]
                    lines.append(f"| {agent_name} | {p.winner} | {p.score} | {p.confidence}/10 |")
            lines.append("")

        # 预测变化链
        if self.result.prediction_chains:
            lines.append("## 预测变化链")
            lines.append("")
            lines.append("> 📌 标记表示该次论述中预测发生了变化")
            lines.append("")
            for agent_name, chain in self.result.prediction_chains.items():
                lines.append(f"### {agent_name}")
                lines.append("")
                lines.append(f"```")
                lines.append(chain)
                lines.append(f"```")
                lines.append("")

        # 最终预测
        if self.result.final_prediction:
            pred = self.result.final_prediction
            lines.append("## 最终预测")
            lines.append("")
            lines.append(f"**胜方**：{pred.winner}")
            lines.append(f"**比分**：{pred.score}")
            lines.append(f"**准确度**：{pred.confidence}/10")
            lines.append("")

        # 仲裁者共识评判
        if self.result.arbitrator_verdict:
            lines.append("## 仲裁者共识评判")
            lines.append("")
            lines.append(self.result.arbitrator_verdict.content)
            lines.append("")

        # 仲裁者最终裁定
        if self.result.verdict and not self.result.consensus:
            lines.append("## 仲裁者最终裁定")
            lines.append("")
            lines.append(self.result.verdict.content)
            lines.append("")

        # 总结分析师
        if self.result.summary:
            lines.append("## 总结分析师")
            lines.append("")
            lines.append(self.result.summary.content)
            lines.append("")

        content = "\n".join(lines)
        Path(path).write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _format_evidence_md(ev: "Evidence") -> str:
        """格式化单条证据为 Markdown

        格式：[E001] (70%·本地池) [标题](URL) — 摘要
        或：  [E001] (90%·本地池) 证据内容
        """
        conf_str = f"{ev.confidence:.0%}"
        id_part = f"[{ev.id}]"
        source_part = f"({conf_str}·{ev.discovered_by})"

        if ev.url and ev.title:
            # 有标题和链接
            title_link = f"[{ev.title}]({ev.url})"
            if ev.summary:
                return f"{id_part} {source_part} {title_link} — {ev.summary}"
            else:
                return f"{id_part} {source_part} {title_link}"
        elif ev.url:
            # 有链接无标题
            url_link = f"[链接]({ev.url})"
            if ev.summary:
                return f"{id_part} {source_part} {url_link} — {ev.summary}: {ev.content}"
            else:
                return f"{id_part} {source_part} {url_link} — {ev.content}"
        else:
            # 本地证据，无链接
            return f"{id_part} {source_part} {ev.content}"

    def save_json(self, path: str = "", agents: list | None = None) -> str:
        """输出完整 JSON 记录，供动画等后续展示使用

        JSON 包含完整的辩论过程、agent 配置、预测历史、证据池等，
        所有内容不截断，保留原始完整数据。
        """
        import json

        if not path:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H%M%S")
            safe_topic = re.sub(r'[\\/:*?"<>|\s]', '_', self.result.topic)[:self._filename_topic_length]
            output_dir = Path("debate_output")
            output_dir.mkdir(exist_ok=True)
            path = str(output_dir / f"{timestamp}_{safe_topic}.json")

        data: dict = {
            "topic": self.result.topic,
            "mode": self.result.mode,
            "consensus": self.result.consensus,
            "generated_at": datetime.datetime.now().isoformat(),
            "agents": [],
            "rounds": [],
            "predictions": {},
            "prediction_chains": self.result.prediction_chains,
            "evidence_pool": [],
            "arbitrator_verdict": None,
            "verdict": None,
            "final_prediction": None,
            "summary": None,
            "phase_verdicts": [],
            "match_context": self.match_context,
            "team_info": self.team_info,
            "match_meta": self.match_meta,
        }

        # Agent 配置（含视角短名和预测历史）
        agents = agents or []
        for agent in agents:
            agent_data = {
                "name": agent.name,
                "stance": agent.stance,
                "perspective_short": getattr(agent, "perspective_short", ""),
            }
            # 预测历史
            if hasattr(agent, "prediction_history"):
                agent_data["prediction_history"] = agent.prediction_history
            # 分配立场标记（圆桌模式下被指定支持某队的 agent）
            if getattr(agent, "assigned_side", None):
                agent_data["assigned_side"] = agent.assigned_side
            data["agents"].append(agent_data)

        # 完整辩论历史（不截断）
        json_agent_counts: dict[str, int] = {}  # JSON 输出的论述计数
        for round_msgs in self.result.rounds:
            round_data = []
            for msg in round_msgs:
                msg_data = {
                    "role": msg.role,
                    "content": msg.content,
                    "speaker": msg.speaker,
                    "target": msg.target,
                    "action": msg.action,
                    "counter_challenge": getattr(msg, 'counter_challenge', False),
                    "new_evidence": getattr(msg, 'new_evidence', []),
                }
                # 有明确对象时标注双方当前预测
                if msg.speaker and msg.role == "assistant" and msg.target:
                    json_agent_counts[msg.speaker] = json_agent_counts.get(msg.speaker, 0) + 1
                    count = json_agent_counts[msg.speaker]
                    sp = self._get_prediction_label(msg.speaker, count)
                    tp = self._get_latest_prediction_label(msg.target)
                    if sp:
                        msg_data["speaker_prediction"] = sp
                    if tp:
                        msg_data["target_prediction"] = tp
                elif msg.speaker and msg.role == "assistant":
                    json_agent_counts[msg.speaker] = json_agent_counts.get(msg.speaker, 0) + 1
                round_data.append(msg_data)
            data["rounds"].append(round_data)

        # 预测结果
        for agent_name, preds in self.result.predictions.items():
            data["predictions"][agent_name] = [
                {
                    "winner": p.winner,
                    "score": p.score,
                    "confidence": p.confidence,
                    "key_factors": p.key_factors,
                }
                for p in preds
            ]

        # 证据池
        if self.evidence_pool:
            for ev in self.evidence_pool.shared:
                data["evidence_pool"].append({
                    "id": ev.id,
                    "content": ev.content,
                    "source": ev.source,
                    "confidence": ev.confidence,
                    "discovered_by": ev.discovered_by,
                    "shared": ev.shared,
                    "title": ev.title,
                    "url": ev.url,
                    "summary": ev.summary,
                })
            for agent_name, evs in self.evidence_pool.private.items():
                for ev in evs:
                    if not ev.shared:
                        data["evidence_pool"].append({
                            "id": ev.id,
                            "content": ev.content,
                            "source": ev.source,
                            "confidence": ev.confidence,
                            "discovered_by": ev.discovered_by,
                            "shared": ev.shared,
                            "title": ev.title,
                            "url": ev.url,
                            "summary": ev.summary,
                            "private_to": agent_name,
                        })

        # 仲裁者共识评判
        if self.result.arbitrator_verdict:
            v = self.result.arbitrator_verdict
            data["arbitrator_verdict"] = {
                "role": v.role,
                "content": v.content,
                "speaker": v.speaker,
            }

        # 仲裁者最终裁定（未共识时）
        if self.result.verdict and not self.result.consensus:
            v = self.result.verdict
            data["verdict"] = {
                "role": v.role,
                "content": v.content,
                "speaker": v.speaker,
            }

        # 最终预测
        if self.result.final_prediction:
            fp = self.result.final_prediction
            data["final_prediction"] = {
                "winner": fp.winner,
                "score": fp.score,
                "confidence": fp.confidence,
                "key_factors": fp.key_factors,
            }

        # 总结分析师
        if self.result.summary:
            s = self.result.summary
            data["summary"] = {
                "role": s.role,
                "content": s.content,
                "speaker": s.speaker,
            }

        # 每阶段仲裁检查结果
        if self.result.phase_verdicts:
            data["phase_verdicts"] = self.result.phase_verdicts

        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
