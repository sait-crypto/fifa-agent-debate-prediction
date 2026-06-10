"""增强智能体 — 证据感知 + CoT 思考/检索管道 + 结构化预测

核心设计：
- 每个 Agent 维护自己的状态：论述计数、预测历史、交互伙伴
- speak() 接收完整 history 但通过 build_context() 过滤为个人上下文
- 上下文规则：
  1. 自己的所有发言 → 明确标注为「你的第N次论述」
  2. 交互伙伴的发言 → 保留交互结构（辩论/质疑/回应）
  3. 自由质疑时 → 一次性看到所有人的最新论述
  4. 所有辩论/质疑/回应 → 针对对方最新论述，其余为上下文摘要

发言流程（agent_search=True 时）：
1. 思考（CoT）→ 判断是否需要检索 → 提取检索词
2. 如需检索：执行搜索 → 结果入私人池 → 审核员审核
3. 基于个人上下文 + 可见证据 → 生成结构化论述
"""

import re

from openai import OpenAI

from .evidence import EvidencePool
from .models import Message, PredictionResult
from .prompts import ROUND_SUMMARY_INJECTION, RETHINKING_AFTER_SEARCH, SYSTEM_THINKING
from .researcher import MatchFact, search_web


def _parse_prediction(text: str) -> PredictionResult:
    """从 agent 回复中解析预测结果"""
    winner = ""
    score = ""
    confidence = 0
    key_factors: list[str] = []

    m = re.search(r"【预测】(.+?)(?:\n|$)", text)
    if m:
        parts = m.group(1)
        for part in parts.split("|"):
            part = part.strip()
            if part.startswith("胜方："):
                winner = part[3:].strip()
            elif part.startswith("比分："):
                score = part[3:].strip()
            elif part.startswith("准确度："):
                conf_str = part[4:].strip().replace("/10", "")
                # Handle "5 → 4" style change notation — take the last number
                if "→" in conf_str or "->" in conf_str:
                    conf_str = re.split(r"→|->", conf_str)[-1].strip()
                try:
                    confidence = int(conf_str)
                except ValueError:
                    confidence = 5

    return PredictionResult(
        winner=winner,
        score=score,
        confidence=confidence,
        key_factors=key_factors,
    )


def _parse_counter_challenge(text: str) -> str:
    """从 agent 回复中解析反质疑内容

    格式：【反质疑】反质疑的具体内容
    返回反质疑文本，无则返回空字符串。
    """
    m = re.search(r"【反质疑】(.+?)(?=(?:【|$))", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _needs_search(thought: str) -> bool:
    """判断思考结果中是否需要检索"""
    m = re.search(r"【检索】\s*(是|否|yes|no|true|false)", thought, re.IGNORECASE)
    if m:
        return m.group(1).lower() in ("是", "yes", "true")
    return False


def _extract_search_query(thought: str) -> str:
    """从思考结果中提取检索关键词"""
    m = re.search(r"【检索词】\s*(.+?)(?:\n|$)", thought)
    if m:
        return m.group(1).strip()
    return ""


def _parse_curation(text: str) -> list[tuple[str, bool, float]]:
    """解析审核员输出"""
    results: list[tuple[str, bool, float]] = []
    for m in re.finditer(
        r"证据编号：(E\d{3})\s*\|\s*是否共享：(是|否)\s*\|\s*置信度：([\d.]+)",
        text,
    ):
        eid = m.group(1)
        share = m.group(2) == "是"
        try:
            conf = float(m.group(3))
        except ValueError:
            conf = 0.5
        conf = max(0.15, min(0.95, conf))
        results.append((eid, share, conf))
    return results


class Agent:
    """增强智能体 — 带个人状态、上下文过滤和思考/检索管道"""

    def __init__(
        self,
        name: str,
        stance: str,
        system_prompt: str,
        client: OpenAI,
        model: str,
        temperature: float = 0.7,
        evidence_pool: EvidencePool | None = None,
        agent_search: bool = True,
        perspective_short: str = "",           # 视角短名（用于 JSON 输出和显示）
        other_summary_limit: int = 100,       # 非交互 agent 消息摘要截断长度
        thinking_temperature: float = 0.3,    # CoT 思考步骤温度
        verbose: bool = False,               # verbose 模式下输出思考内容
    ):
        self.name = name
        self.stance = stance
        self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.temperature = temperature
        self.evidence_pool = evidence_pool
        self.agent_search = agent_search
        self.perspective_short = perspective_short
        self.other_summary_limit = other_summary_limit
        self.thinking_temperature = thinking_temperature
        self.verbose = verbose

        # ── 个人状态 ──
        self.statement_count: int = 0          # 论述计数
        self.prediction_history: list[dict] = []  # 预测变化链
        self.interaction_partners: set[str] = set()  # 交互过的 agent
        self.round_summaries: list[str] = []   # 每轮的场记摘要

    def speak(
        self,
        history: list[Message],
        curator: "Agent | None" = None,
        match_context: str = "",
        all_latest_arguments: str = "",  # 自由质疑时使用
        temp_foreign_evidence: list | None = None,  # 临时外部私有证据（仅本次回应可见）
    ) -> tuple[Message, PredictionResult]:
        """发言，返回 (消息, 预测结果)

        all_latest_arguments: 自由质疑环节，所有 agent 最新论述的汇总
        temp_foreign_evidence: 质疑者引用的私有证据，仅本次回应时临时可见
        """
        if self.agent_search and self.evidence_pool:
            return self._speak_with_search(history, curator, match_context, all_latest_arguments, temp_foreign_evidence)
        else:
            return self._speak_direct(history, all_latest_arguments, temp_foreign_evidence)

    # ── 上下文构建（核心改动）────────────────────────────────

    def build_context(
        self,
        history: list[Message],
        all_latest_arguments: str = "",
    ) -> list[dict]:
        """为当前 agent 构建个性化上下文

        规则：
        1. 场记摘要 → 全局形势（客观记录，提示不可从众）
        2. 自己直接参与的所有辩论 → 按时间线排列为「辩论历程」
           包括：自己的陈述/质疑/回应 + 对方对自己的质疑/回应
        3. 回应等同于陈述（需完整表述观点），质疑不等同于陈述（只需达到质疑目的）
        4. 非我参与的其他论述 → 压缩为摘要
        5. 自由质疑时 all_latest_arguments 非空 → 一次性看到所有人最新论述
        """
        messages: list[dict] = []

        # 场记摘要（全局形势供参考）
        if self.round_summaries:
            summary_text = "\n\n".join(self.round_summaries)
            messages.append({
                "role": "user",
                "content": ROUND_SUMMARY_INJECTION.format(summary=summary_text),
            })

        # 自己的预测历史（注入到 system 后）
        if self.prediction_history:
            pred_lines = ["## 你的预测变化历史\n"]
            for entry in self.prediction_history:
                pred_lines.append(
                    f"- 第{entry['count']}次论述({entry['action']})："
                    f"{entry['winner']} {entry['score']} (准确度{entry['confidence']}/10)"
                )
            pred_lines.append("")
            messages.append({"role": "user", "content": "\n".join(pred_lines)})

        # 自由质疑模式：一次性展示所有人的最新论述
        if all_latest_arguments:
            messages.append({
                "role": "user",
                "content": f"## 所有参与者最新论述（供你选择质疑对象）\n\n{all_latest_arguments}",
            })
            return messages

        # 正常模式：按时间线构建自己的完整辩论历程
        my_debate_flow: list[str] = []   # 自己参与的完整交互（按时间线）
        other_summary: list[str] = []    # 非我参与的摘要

        for msg in history:
            if msg.role == "user" and msg.speaker == "系统":
                # 系统提示保留
                messages.append({"role": "user", "content": msg.content})
            elif msg.speaker == self.name:
                # 自己的发言 → 标注行为和目标
                action = msg.action or "立场"
                target_info = f"→{msg.target}" if msg.target else ""
                count_info = self._get_statement_tag(msg)
                my_debate_flow.append(
                    f"【你的{action}{target_info}（{count_info}）】\n{msg.content}"
                )
            elif msg.role == "assistant" and self._is_my_interaction(msg):
                # 对方对我是交互（质疑我 / 回应我）→ 保留完整内容
                action = msg.action or "发言"
                my_debate_flow.append(
                    f"【{msg.speaker}→你·{action}】\n{msg.content}"
                )
            elif msg.speaker and msg.role == "assistant":
                # 非我参与 → 摘要
                other_summary.append(
                    f"[{msg.speaker}：{msg.content[:self.other_summary_limit]}...]"
                )

        # 组装：辩论历程 → 其他摘要
        if my_debate_flow:
            header = (
                "## 你的辩论历程\n\n"
                "⚠️ 回应等同于陈述——回应时应完整表述你的观点和论据。\n"
                "质疑不等同于陈述——质疑只需指出对方论证的问题，不必重述自己的完整观点。\n"
            )
            # 追加当前预测状态，明确"这是你自己的立场"
            if self.prediction_history:
                latest = self.prediction_history[-1]
                header += (
                    f"\n你当前的预测：{latest['winner']} {latest['score']} "
                    f"(准确度{latest['confidence']}/10) — 这是你自己的立场，不是别人的。\n"
                )
            header += "\n"
            debate_flow_text = header + "\n\n".join(my_debate_flow)
            messages.append({"role": "user", "content": debate_flow_text})
        if other_summary:
            messages.append({
                "role": "user",
                "content": "## 其他论述摘要\n\n" + "\n".join(other_summary),
            })

        return messages

    def _get_statement_tag(self, msg: Message) -> str:
        """获取论述标签（第N次·行为）"""
        count = 0
        for entry in self.prediction_history:
            count = max(count, entry["count"])
        action = msg.action or "立场"
        if count > 0:
            return f"第{count}次论述({action})"
        return f"论述({action})"

    def _is_my_interaction(self, msg: Message) -> bool:
        """判断这条消息是否是我参与的交互

        我参与的交互 = 我是这条消息的 target（对方质疑/回应我）
        不是我参与的 = 伙伴和其他人的交互（即使伙伴是 Con1，
        但 Con1 质疑 Pro2 的内容与我无关，我只看我和 Con1 的交互）
        """
        return msg.target == self.name

    def record_statement(self, msg: Message, pred: PredictionResult) -> None:
        """记录一次论述和预测

        所有发言（包括质疑）都递增 statement_count 并记录。
        质疑标记为 is_challenge=True，表示其不是完整观点表达：
        - 不会被别人当作质疑对象（_get_latest_argument 跳过质疑）
        - 回应视作完整陈述，可作为质疑对象
        - 质疑不应改变质疑者自身的观点，只是尝试说服对方
        """
        self.statement_count += 1
        is_challenge = (msg.action == "质疑")
        action = msg.action or "立场"

        # 注册交互伙伴
        if msg.target:
            self.interaction_partners.add(msg.target)

        # 检查预测是否变化
        prev = self.prediction_history[-1] if self.prediction_history else None
        changed = (
            prev is None
            or prev["winner"] != pred.winner
            or prev["score"] != pred.score
        )

        self.prediction_history.append({
            "count": self.statement_count,
            "action": action,
            "target": msg.target,
            "winner": pred.winner,
            "score": pred.score,
            "confidence": pred.confidence,
            "changed": changed,
            "is_challenge": is_challenge,
        })

    # ── 发言流程 ──────────────────────────────────────────

    def _speak_with_search(
        self,
        history: list[Message],
        curator: "Agent | None" = None,
        match_context: str = "",
        all_latest_arguments: str = "",
        temp_foreign_evidence: list | None = None,
    ) -> tuple[Message, PredictionResult]:
        """思考 → 判断检索 → 审核员审核 → 再思考 → 最终论述"""
        thought = self._think(history, all_latest_arguments, temp_foreign_evidence)
        self._print_verbose_thought(thought, "思考")

        discovered_ids: list[str] = []  # 记录本次搜到的证据ID
        if _needs_search(thought):
            query = _extract_search_query(thought)
            if query:
                print(f"    🔍 {self.name} 检索：{query}")
                facts = search_web(query, max_results=3, fetch_content=True)
                if facts:
                    new_evidence = self.evidence_pool.add_agent_search(self.name, facts)
                    discovered_ids = [ev.id for ev in new_evidence]
                    print(f"    📄 {self.name} 获得 {len(new_evidence)} 条私人证据")
                    if curator and match_context:
                        self._curate_evidence(new_evidence, curator, match_context)
                    else:
                        for ev in new_evidence:
                            self.evidence_pool.share_to_pool(self.name, ev.id)
                            print(f"    🔗 {self.name} 共享证据 {ev.id} 到共用池")

                    # 获得新证据后重新思考论述策略
                    thought2 = self._rethink(history, all_latest_arguments, temp_foreign_evidence)
                    self._print_verbose_thought(thought2, "再思考")

        msg, pred = self._speak_direct(history, all_latest_arguments, temp_foreign_evidence)
        if discovered_ids:
            msg.new_evidence = discovered_ids
        return msg, pred

    def _think(self, history: list[Message], all_latest_arguments: str = "", temp_foreign_evidence: list | None = None) -> str:
        """调用 LLM 生成 CoT 思考"""
        evidence_text = self.evidence_pool.format_for_prompt(self.name, temp_foreign_evidence=temp_foreign_evidence) if self.evidence_pool else ""
        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{SYSTEM_THINKING}"},
        ]
        if evidence_text:
            messages.append({"role": "user", "content": evidence_text})

        # 用过滤后的上下文
        context = self.build_context(history, all_latest_arguments)
        messages.extend(context)

        messages.append({"role": "user", "content": "请先思考你的论述策略，判断是否需要检索额外证据，用途包括支持自己的观点和证据，以及质疑对方的观点和证据等。"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.thinking_temperature,
        )
        return response.choices[0].message.content.strip()

    def _rethink(
        self,
        history: list[Message],
        all_latest_arguments: str = "",
        temp_foreign_evidence: list | None = None,
    ) -> str:
        """获得新证据后重新思考论述策略（不再判断是否需要检索）"""
        evidence_text = self.evidence_pool.format_for_prompt(self.name, temp_foreign_evidence=temp_foreign_evidence) if self.evidence_pool else ""
        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{RETHINKING_AFTER_SEARCH}"},
        ]
        if evidence_text:
            messages.append({"role": "user", "content": evidence_text})

        context = self.build_context(history, all_latest_arguments)
        messages.extend(context)

        messages.append({"role": "user", "content": "请基于当前所有证据（含新获得的），重新组织你的论述要点。"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.thinking_temperature,
        )
        return response.choices[0].message.content.strip()

    def _print_verbose_thought(self, thought: str, label: str = "思考") -> None:
        """verbose 模式下输出思考内容"""
        if not self.verbose or not thought:
            return
        print(f"    💭 {label}：")
        for line in thought.split("\n"):
            print(f"      {line}")

    def _curate_evidence(
        self,
        new_evidence: list,
        curator: "Agent",
        match_context: str,
    ) -> None:
        """由审核员审核私人证据"""
        if not new_evidence:
            return

        from .prompts import USER_EVIDENCE_CURATOR_REVIEW

        evidence_text = "\n".join(
            f"[{ev.id}] (来源：{ev.source}，初始置信度：{ev.confidence:.1f}) {ev.content}"
            for ev in new_evidence
        )

        messages = [
            {"role": "system", "content": curator.system_prompt},
            {"role": "user", "content": USER_EVIDENCE_CURATOR_REVIEW(evidence_text, match_context)},
        ]

        try:
            response = self.client.chat.completions.create(
                model=curator.model,
                messages=messages,
                temperature=curator.temperature,
            )
            curation_text = response.choices[0].message.content.strip()
            decisions = _parse_curation(curation_text)

            for eid, should_share, confidence in decisions:
                self.evidence_pool.update_confidence(eid, confidence)
                if should_share:
                    self.evidence_pool.share_to_pool(self.name, eid)
                    print(f"    ✅ 审核员：证据 {eid} 共享（置信度 {confidence:.2f}）")
                else:
                    print(f"    ⏸️  审核员：证据 {eid} 不共享（置信度 {confidence:.2f}）")
        except Exception as e:
            print(f"    ⚠️ 审核员调用失败：{e}，默认全部共享")
            for ev in new_evidence:
                self.evidence_pool.share_to_pool(self.name, ev.id)

    def _speak_direct(
        self,
        history: list[Message],
        all_latest_arguments: str = "",
        temp_foreign_evidence: list | None = None,
    ) -> tuple[Message, PredictionResult]:
        """直接生成论述（含结构化预测）"""
        evidence_text = ""
        if self.evidence_pool:
            evidence_text = self.evidence_pool.format_for_prompt(self.name, temp_foreign_evidence=temp_foreign_evidence)

        # 构建 messages：system + 证据 + 过滤后上下文
        messages = [{"role": "system", "content": self.system_prompt}]
        if evidence_text:
            messages.append({"role": "user", "content": evidence_text})

        context = self.build_context(history, all_latest_arguments)
        messages.extend(context)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content.strip()
        prediction = _parse_prediction(content)

        msg = Message(role="assistant", content=content, speaker=self.name)

        # 注意：record_statement 由编排器在设置 action/target 后调用
        return msg, prediction

    def format_prediction_chain(self) -> str:
        """格式化预测变化链（用于报告）

        质疑用 🔍 标记（不是完整观点表达），变化用 📌 标记。
        """
        if not self.prediction_history:
            return ""
        lines = []
        for entry in self.prediction_history:
            is_ch = entry.get("is_challenge", False)
            changed_marker = "📌" if entry["changed"] else "  "
            target_info = f" →{entry['target']}" if entry.get("target") else ""
            if is_ch:
                lines.append(
                    f"{changed_marker} 🔍第{entry['count']}次({entry['action']}{target_info})："
                    f"{entry['winner']} {entry['score']} (准确度{entry['confidence']}/10)"
                )
            else:
                lines.append(
                    f"{changed_marker} 第{entry['count']}次({entry['action']}{target_info})："
                    f"{entry['winner']} {entry['score']} (准确度{entry['confidence']}/10)"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        search_flag = " +search" if self.agent_search else ""
        return f"Agent({self.name!r}, stance={self.stance!r}{search_flag})"
