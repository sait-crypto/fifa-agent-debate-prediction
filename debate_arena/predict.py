"""统一编排器 — 正反方辩论 + 圆桌辩论

核心流程：
1. 加载本地证据 → 初始检索 → 构建共用证据池
2. 创建 agents（根据 mode）+ 证据审核员 + 场记总结者
3. 辩论循环
   - pro_con: 每轮正反方发言 → 质疑回应 → 场记总结 → 仲裁评判
   - roundtable: 陈述 → 按 phase_order 执行各环节 → 场记总结 → 仲裁评判
4. 共识提前终止 / 仲裁裁定 → 总结分析师

质疑流程保证：
- 质疑不算表述：statement_count 不递增，_get_latest_argument 跳过质疑
- 质疑与回应紧邻：有一个质疑就紧接一个回应
- 自由质疑可拆分为多次：每 agent 有人次计数，达标后统一让被质疑者回应
- 必须指定对象：否则打回重写（失败不记录）
- 分配质疑双向配对：每对 a→b 质疑回应 + b→a 质疑回应
"""

import datetime
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAI

from .agent import Agent, _parse_counter_challenge, _parse_prediction

if TYPE_CHECKING:
    from .config import SystemConfig
from .evidence import EvidencePool
from .match_config import MatchConfig, format_match_context, load_evidence_files
from .models import DebateResult, Message, PredictionResult
from .prompts import (
    SPEECH_CONCISE_HINT,
    SYSTEM_ARBITRATOR,
    SYSTEM_CON_SUPPORTER,
    SYSTEM_EVIDENCE_CURATOR,
    SYSTEM_PRO_SUPPORTER,
    SYSTEM_ROUND_SUMMARIZER,
    SYSTEM_ROUNDTABLE_AGENT,
    SYSTEM_ROUNDTABLE_ASSIGNED_AGENT,
    SYSTEM_SUMMARY_ANALYST,
    USER_ARBITRATE,
    USER_CONSENSUS_CHECK,
    USER_COUNTER_CHALLENGE_RESPOND,
    USER_PREDICT_OPENING,
    USER_PRO_CON_CHALLENGE,
    USER_PRO_CON_RESPOND_CHALLENGE,
    USER_ROUND_SUMMARY,
    USER_ROUNDTABLE_ASSIGNED_CHALLENGE,
    USER_ROUNDTABLE_ASSIGNED_OPENING,
    USER_ROUNDTABLE_FREE_CHALLENGE,
    USER_ROUNDTABLE_OPENING,
    USER_ROUNDTABLE_RESPOND_CHALLENGE,
    USER_SUMMARY,
    get_perspective,
)
from .researcher import format_context, research_match


class MatchPredictor:
    """比赛预测编排器"""

    def __init__(
        self,
        config: MatchConfig,
        client: OpenAI,
        model: str,
        sys_config: "SystemConfig | None" = None,
    ):
        self.config = config
        self.client = client
        self.model = model
        self.evidence_pool = EvidencePool()
        self.sys_config = sys_config

        # ── 从系统配置解析参数（有默认值兜底）──
        self.cli_verbose = _cfg_get(sys_config, "output.cli_verbose", False)
        self._limits = _resolve_limits(sys_config)
        self._temps = _resolve_temperatures(sys_config)
        self._fetch_cfg = _resolve_fetch(sys_config)
        self._speech_hint = _cfg_get(sys_config, "debate.agent_speech_hint", 0)
        self._dump_evidence = _cfg_get(sys_config, "output.dump_evidence_pool", False)
        self._output_dir: Path | None = None      # 由 main.py 设置
        self._run_timestamp: str | None = None     # 由 main.py 设置

        self.pro_agents: list[Agent] = []
        self.con_agents: list[Agent] = []
        self.roundtable_agents: list[Agent] = []
        self.arbitrator: Agent | None = None
        self.summary_analyst: Agent | None = None
        self.curator: Agent | None = None
        self.round_summarizer: Agent | None = None
        self.predictions: dict[str, list[PredictionResult]] = {}
        self.match_context: str = ""
        self.research_context: str = ""
        self._analysis_data = None
        self._phase_verdicts: list[dict] = []  # 每阶段仲裁检查结果

    def run(self) -> DebateResult:
        """执行预测流程"""
        local_evidence = load_evidence_files(self.config.evidence_files)
        self.evidence_pool.add_local(local_evidence)
        if local_evidence:
            print(f"  📋 已加载 {len(local_evidence)} 条本地证据")

        # 加载分析数据（情报归入证据池）
        if self.config.analysis_file:
            from .analysis_parser import parse_analysis_json, get_intel_as_evidence_items
            try:
                self._analysis_data = parse_analysis_json(self.config.analysis_file)
                intel_items = get_intel_as_evidence_items(self._analysis_data)
                if intel_items:
                    self.evidence_pool.add_local(intel_items)
                    print(f"  📊 已从分析数据加载 {len(intel_items)} 条情报证据")
            except Exception as e:
                print(f"  ⚠️  分析数据加载失败：{e}")

        print(f"  🔍 正在检索 {self.config.home} vs {self.config.away} 的相关信息...")
        result = research_match(
            self.config.home, self.config.away, self.client, self.model,
            fetch_config=self._fetch_cfg,
            research_llm_temperature=self._temps["research_llm"],
            search_max_results=self._fetch_cfg.get("search_max_results", 5) if self._fetch_cfg else 5,
        )
        if result.facts:
            self.evidence_pool.add_research(result.facts)
            print(f"  📚 检索到 {len(result.facts)} 条补充证据")
        self.research_context = format_context(result)

        self.match_context = format_match_context(self.config, self._analysis_data)

        if self.config.debate.mode == "pro_con":
            self._create_pro_con_agents()
        else:
            self._create_roundtable_agents()
        self._create_arbitrator()
        self._create_summary_analyst()
        self._create_evidence_curator()
        self._create_round_summarizer()

        if self.config.debate.mode == "pro_con":
            debate_result = self._run_pro_con_debate()
        else:
            debate_result = self._run_roundtable_debate()

        debate_result = self._summarize(debate_result)
        return debate_result

    # ── 创建 Agents ───────────────────────────────────────

    def _create_pro_con_agents(self) -> None:
        debate_cfg = self.config.debate
        base_temp = debate_cfg.temperature or 0.7

        for i in range(debate_cfg.pro_count):
            name = f"{self.config.home}支持方{i+1}" if debate_cfg.pro_count > 1 else f"{self.config.home}支持方"
            sys_prompt = SYSTEM_PRO_SUPPORTER(self.config.home, self.config.away, self.match_context)
            perspective_section, perspective_short = get_perspective(i)
            sys_prompt += perspective_section
            agent_temp = round(base_temp + (i * 0.05 - 0.025), 2)
            self.pro_agents.append(self._make_agent(name, f"支持{self.config.home}", sys_prompt, agent_temp, perspective_short))

        for i in range(debate_cfg.con_count):
            name = f"{self.config.away}支持方{i+1}" if debate_cfg.con_count > 1 else f"{self.config.away}支持方"
            sys_prompt = SYSTEM_CON_SUPPORTER(self.config.home, self.config.away, self.match_context)
            perspective_section, perspective_short = get_perspective(i + debate_cfg.pro_count)
            sys_prompt += perspective_section
            agent_temp = round(base_temp + (i * 0.05 - 0.025), 2)
            self.con_agents.append(self._make_agent(name, f"支持{self.config.away}", sys_prompt, agent_temp, perspective_short))

    def _create_roundtable_agents(self) -> None:
        debate_cfg = self.config.debate
        base_temp = debate_cfg.temperature or 0.7
        total = debate_cfg.agent_count
        assigned_count = min(debate_cfg.assigned_stance_count or 0, total)
        free_count = total - assigned_count

        for i in range(total):
            name = f"分析师{i+1}"
            sys_prompt = SYSTEM_ROUNDTABLE_AGENT(name, self.config.display_title, self.match_context)
            perspective_section, perspective_short = get_perspective(i)
            sys_prompt += perspective_section
            agent_temp = round(base_temp + (i * 0.05 - 0.025), 2)
            agent = self._make_agent(name, "独立分析", sys_prompt, agent_temp, perspective_short)
            # 后 N 个 agent 为"分配立场"（初始不指定具体立场，发言前根据形势分配）
            if i >= free_count:
                agent.assigned = True
            self.roundtable_agents.append(agent)

    def _make_agent(
        self,
        name: str,
        stance: str,
        system_prompt: str,
        temperature: float | None = None,
        perspective_short: str = "",
    ) -> Agent:
        if self._speech_hint > 0:
            system_prompt += SPEECH_CONCISE_HINT(self._speech_hint)
        return Agent(
            name=name, stance=stance, system_prompt=system_prompt,
            client=self.client, model=self.model,
            temperature=temperature or self.config.debate.temperature or 0.7,
            evidence_pool=self.evidence_pool,
            agent_search=self.config.debate.agent_search,
            perspective_short=perspective_short,
            other_summary_limit=self._limits["other_agent_summary"],
            thinking_temperature=self._temps["thinking"],
            verbose=self.cli_verbose,
        )

    def _create_arbitrator(self) -> None:
        self.arbitrator = Agent(
            name="仲裁者", stance="中立",
            system_prompt=SYSTEM_ARBITRATOR(self.config.display_title, self.match_context),
            client=self.client, model=self.model, temperature=self._temps["arbitrator"],
            evidence_pool=self.evidence_pool, agent_search=False,
        )

    def _create_summary_analyst(self) -> None:
        self.summary_analyst = Agent(
            name="总结分析师", stance="中立",
            system_prompt=SYSTEM_SUMMARY_ANALYST(self.config.display_title, self.match_context),
            client=self.client, model=self.model, temperature=self._temps["summary_analyst"],
            evidence_pool=self.evidence_pool, agent_search=False,
        )

    def _create_evidence_curator(self) -> None:
        self.curator = Agent(
            name="审核员", stance="中立",
            system_prompt=SYSTEM_EVIDENCE_CURATOR,
            client=self.client, model=self.model, temperature=self._temps["evidence_curator"],
            evidence_pool=self.evidence_pool, agent_search=False,
        )

    def _create_round_summarizer(self) -> None:
        self.round_summarizer = Agent(
            name="场记", stance="中立",
            system_prompt=SYSTEM_ROUND_SUMMARIZER,
            client=self.client, model=self.model, temperature=self._temps["round_summarizer"],
            evidence_pool=self.evidence_pool, agent_search=False,
        )

    # ── 场记总结 ────────────────────────────────────────────

    def _format_debate_events(self, history: list[Message]) -> str:
        """从历史消息中提取结构化辩论事件"""
        events: list[str] = []
        for msg in history:
            if msg.role != "assistant" or not msg.speaker:
                continue
            if msg.action == "质疑" and msg.target:
                events.append(f"- {msg.speaker} 质疑了 {msg.target}")
            elif msg.action == "回应" and msg.target:
                if getattr(msg, 'counter_challenge', False):
                    events.append(f"- {msg.speaker} 回应了 {msg.target} 的质疑并提出反质疑")
                else:
                    events.append(f"- {msg.speaker} 回应了 {msg.target} 的质疑")
            elif not msg.action or msg.action == "立场":
                events.append(f"- {msg.speaker} 陈述了立场")

        # 预测变化
        for agent in self._all_agents():
            for entry in agent.prediction_history:
                if entry.get("changed") and not entry.get("is_challenge"):
                    prev_idx = agent.prediction_history.index(entry) - 1
                    if prev_idx >= 0:
                        prev = agent.prediction_history[prev_idx]
                        events.append(
                            f"- {agent.name} 改变了预测："
                            f"{prev['winner']} {prev['score']} → {entry['winner']} {entry['score']}"
                        )
                    else:
                        events.append(f"- {agent.name} 初始预测：{entry['winner']} {entry['score']}")

        return "\n".join(events)

    def _summarize_round(self, history: list[Message], round_num: int) -> None:
        """场记总结当前全局形势，分发给所有 agent 和仲裁者"""
        arg_limit = self._limits["round_summary_arg"]
        all_arguments = "\n\n".join(
            f"【{msg.speaker}】：{msg.content[:arg_limit]}"
            for msg in history if msg.role == "assistant"
        )
        debate_events = self._format_debate_events(history)

        summary_msg = Message(
            role="user",
            content=USER_ROUND_SUMMARY(round_num, all_arguments, debate_events),
            speaker="系统",
        )
        summary_result, _ = self.round_summarizer.speak([summary_msg])
        print(f"    📋 场记：第{round_num}轮总结完成")

        # 分发给所有辩论 agent 和仲裁者
        for agent in self._all_agents():
            agent.round_summaries.append(summary_result.content)
        if self.arbitrator:
            self.arbitrator.round_summaries.append(summary_result.content)

    # ── 正反方辩论 ────────────────────────────────────────

    def _run_pro_con_debate(self) -> DebateResult:
        history: list[Message] = []
        rounds: list[list[Message]] = []
        debate_cfg = self.config.debate

        opening = Message(
            role="user",
            content=USER_PREDICT_OPENING(
                self.config.home, self.config.away, self.config.tournament.stage
            ),
            speaker="系统",
        )
        history.append(opening)

        for round_num in range(1, debate_cfg.max_rounds + 1):
            round_msgs: list[Message] = []
            print(f"\n  ── 第 {round_num}/{debate_cfg.max_rounds} 轮 ──")

            # 正方依次发言
            for agent in self.pro_agents:
                msg, pred = agent.speak(history, curator=self.curator, match_context=self.config.display_title)
                for con in self.con_agents:
                    agent.interaction_partners.add(con.name)
                agent.record_statement(msg, pred)  # 编排器记录
                round_msgs.append(msg)
                history.append(msg)
                self._record_prediction(agent.name, pred)
                self._print_prediction(agent.name, pred, full_content=msg.content)

            # 反方依次发言
            for agent in self.con_agents:
                msg, pred = agent.speak(history, curator=self.curator, match_context=self.config.display_title)
                for pro in self.pro_agents:
                    agent.interaction_partners.add(pro.name)
                agent.record_statement(msg, pred)
                round_msgs.append(msg)
                history.append(msg)
                self._record_prediction(agent.name, pred)
                self._print_prediction(agent.name, pred, full_content=msg.content)

            # 质疑回应子阶段
            if debate_cfg.pro_con_challenge_enabled:
                challenge_msgs = self._run_pro_con_challenge(history, round_num)
                round_msgs.extend(challenge_msgs)

            # 场记总结
            self._summarize_round(history, round_num)
            self._maybe_dump_evidence(f"正反方第{round_num}轮")
            rounds.append(round_msgs)

            # 仲裁者评判共识
            consensus, verdict = self._check_consensus_via_arbitrator(history)
            self._record_phase_verdict(round_num, consensus, verdict)
            if consensus:
                print(f"\n  ✅ 仲裁者判定：共识达成！")
                return self._build_result(rounds, consensus=True, arbitrator_verdict=verdict)
            else:
                print(f"  ⚖️ 仲裁者判定：未达成共识")

        print(f"\n  ⚖️ 到达上限轮次，由仲裁者裁定")
        result = self._arbitrate(history, rounds)
        result.arbitrator_verdict = result.verdict
        result.phase_verdicts = self._phase_verdicts
        return result

    def _run_pro_con_challenge(self, history: list[Message], round_num: int) -> list[Message]:
        """正反方质疑回应 — 紧邻配对"""
        debate_cfg = self.config.debate
        per_agent = debate_cfg.pro_con_challenge_per_agent
        result_msgs: list[Message] = []

        pro_targets = self._assign_pro_con_challenges(
            self.pro_agents, self.con_agents, per_agent, round_num
        )
        con_targets = self._assign_pro_con_challenges(
            self.con_agents, self.pro_agents, per_agent, round_num
        )

        # 正方质疑反方 → 立即回应
        print(f"    ⚔️ 正方质疑反方")
        for challenger, target in pro_targets:
            ch_msg, ch_pred = self._do_assigned_challenge(challenger, target, history, "正")
            result_msgs.append(ch_msg)
            history.append(ch_msg)
            self._record_prediction(challenger.name, ch_pred)
            self._print_prediction(challenger.name, ch_pred, f"质疑→{target.name}", full_content=ch_msg.content)

            resp_msg, resp_pred = self._do_respond(target, challenger, ch_msg, history, pro_con_mode=True)
            result_msgs.append(resp_msg)
            history.append(resp_msg)
            self._record_prediction(target.name, resp_pred)
            self._print_prediction(target.name, resp_pred, f"回应→{challenger.name}", full_content=resp_msg.content)

            # 反质疑链
            self._process_counter_challenge_chain(resp_msg, target, challenger, history, result_msgs, pro_con_mode=True)

        # 反方质疑正方 → 立即回应
        print(f"    ⚔️ 反方质疑正方")
        for challenger, target in con_targets:
            ch_msg, ch_pred = self._do_assigned_challenge(challenger, target, history, "反")
            result_msgs.append(ch_msg)
            history.append(ch_msg)
            self._record_prediction(challenger.name, ch_pred)
            self._print_prediction(challenger.name, ch_pred, f"质疑→{target.name}", full_content=ch_msg.content)

            resp_msg, resp_pred = self._do_respond(target, challenger, ch_msg, history, pro_con_mode=True)
            result_msgs.append(resp_msg)
            history.append(resp_msg)
            self._record_prediction(target.name, resp_pred)
            self._print_prediction(target.name, resp_pred, f"回应→{challenger.name}", full_content=resp_msg.content)

            # 反质疑链
            self._process_counter_challenge_chain(resp_msg, target, challenger, history, result_msgs, pro_con_mode=True)

        return result_msgs

    @staticmethod
    def _assign_pro_con_challenges(
        challengers: list[Agent],
        targets: list[Agent],
        per_agent: int,
        round_num: int,
    ) -> list[tuple[Agent, Agent]]:
        """分配质疑目标：跨轮轮换 + 确保每个 target 至少被质疑一次"""
        if not challengers or not targets:
            return []

        n_challengers = len(challengers)
        n_targets = len(targets)
        offset = (round_num - 1) % n_targets

        pairs: list[tuple[Agent, Agent]] = []
        for challenger in challengers:
            for i in range(min(per_agent, n_targets)):
                target_idx = (offset + i) % n_targets
                pairs.append((challenger, targets[target_idx]))

        # 补充：确保每个 target 至少被质疑一次
        challenged_targets = {t.name for _, t in pairs}
        for target in targets:
            if target.name not in challenged_targets:
                ch_idx = (round_num - 1) % n_challengers
                pairs.append((challengers[ch_idx], target))

        return pairs

    # ── 圆桌辩论 ──────────────────────────────────────────

    def _run_roundtable_debate(self) -> DebateResult:
        history: list[Message] = []
        rounds: list[list[Message]] = []
        agents = self.roundtable_agents
        rt_cfg = self.config.debate.roundtable

        phases: list[str] = []
        for p in rt_cfg.phase_order:
            if p == "phase1" and rt_cfg.phase1_enabled:
                phases.append("phase1")
            elif p == "phase2" and rt_cfg.phase2_enabled:
                phases.append("phase2")
        if not phases:
            phases = ["phase1"]

        # 陈述阶段 — 区分自由 agent 和分配立场 agent
        free_agents = [a for a in agents if not getattr(a, 'assigned', False)]
        assigned_agents = [a for a in agents if getattr(a, 'assigned', False)]

        print(f"\n  ── 陈述阶段 ──" + (f"（{len(free_agents)} 自由 + {len(assigned_agents)} 分配）" if assigned_agents else ""))
        opening = Message(
            role="user",
            content=USER_ROUNDTABLE_OPENING(self.config.display_title),
            speaker="系统",
        )
        history.append(opening)

        opening_msgs: list[Message] = []

        # 自由 agent 先发言
        for agent in free_agents:
            msg, pred = agent.speak(history, curator=self.curator, match_context=self.config.display_title)
            msg.action = "立场"
            agent.record_statement(msg, pred)
            opening_msgs.append(msg)
            history.append(msg)
            self._record_prediction(agent.name, pred)
            self._print_prediction(agent.name, pred, full_content=msg.content)

        # 分配立场 agent 逐个指定并发言
        if assigned_agents:
            print(f"\n  ── 分配立场阶段（{len(assigned_agents)} 人）──")
            for agent in assigned_agents:
                assigned_side = self._determine_assigned_side()
                opponent_side = self.config.away if assigned_side == self.config.home else self.config.home

                # 更新 agent 的系统提示词和立场
                idx = agents.index(agent)
                perspective_section, _ = get_perspective(idx)
                new_prompt = SYSTEM_ROUNDTABLE_ASSIGNED_AGENT(
                    agent.name, self.config.display_title, self.match_context,
                    assigned_side, opponent_side,
                )
                new_prompt += perspective_section
                if self._speech_hint > 0:
                    new_prompt += SPEECH_CONCISE_HINT(self._speech_hint)
                agent.system_prompt = new_prompt
                agent.stance = f"支持{assigned_side}"
                agent.assigned_side = assigned_side

                # 以指定立场发言
                opening_assigned = Message(
                    role="user",
                    content=USER_ROUNDTABLE_ASSIGNED_OPENING(self.config.display_title, assigned_side),
                    speaker="系统",
                )
                msg, pred = agent.speak(
                    history + [opening_assigned],
                    curator=self.curator,
                    match_context=self.config.display_title,
                )
                msg.action = "立场"
                agent.record_statement(msg, pred)
                opening_msgs.append(msg)
                history.append(msg)
                self._record_prediction(agent.name, pred)
                self._print_prediction(agent.name, pred, f"指定支持{assigned_side}", full_content=msg.content)
                print(f"    📌 {agent.name} 被指定支持 {assigned_side}")

        rounds.append(opening_msgs)

        self._summarize_round(history, 1)
        self._maybe_dump_evidence("圆桌陈述后")
        consensus, verdict = self._check_consensus_via_arbitrator(history)
        self._record_phase_verdict(1, consensus, verdict)
        if consensus:
            print(f"\n  ✅ 仲裁者判定：共识达成！")
            return self._build_result(rounds, consensus=True, arbitrator_verdict=verdict)
        else:
            print(f"  ⚖️ 仲裁者判定：未达成共识")

        for phase_idx, phase in enumerate(phases, 2):
            if phase == "phase1":
                self._run_phase1(agents, history, rounds)
            elif phase == "phase2":
                self._run_phase2(agents, history, rounds)

            self._summarize_round(history, phase_idx)
            self._maybe_dump_evidence(f"圆桌阶段{phase}后")

            consensus, verdict = self._check_consensus_via_arbitrator(history)
            self._record_phase_verdict(phase_idx, consensus, verdict)
            if consensus:
                print(f"\n  ✅ 仲裁者判定：共识达成！")
                return self._build_result(rounds, consensus=True, arbitrator_verdict=verdict)
            else:
                print(f"  ⚖️ 仲裁者判定：未达成共识")

        print(f"\n  ⚖️ 未达成共识，由仲裁者裁定")
        result = self._arbitrate(history, rounds)
        result.arbitrator_verdict = result.verdict
        result.phase_verdicts = self._phase_verdicts
        return result

    def _record_phase_verdict(self, phase: int, consensus: bool, verdict: Message | None) -> None:
        """记录每阶段仲裁检查结果，供 viewer 展示"""
        entry = {
            "phase": phase,
            "consensus": consensus,
            "content": verdict.content if verdict else "",
            "speaker": "仲裁者",
        }
        self._phase_verdicts.append(entry)

    def _determine_assigned_side(self) -> str:
        """根据当前预测分布，决定下一个分配立场 agent 应支持的队伍

        规则：
        - 统计当前所有 agent 的最新预测（忽略平局预测）
        - 支持人数少的队伍优先分配
        - 双方人数相同时随机选择
        - 逐个分配，每次看全局最新状态，保证真正平衡
        """
        home = self.config.home
        away = self.config.away

        home_count = 0
        away_count = 0

        for agent_name, preds in self.predictions.items():
            if preds:
                last = preds[-1]
                if last.winner:
                    if home in last.winner:
                        home_count += 1
                    elif away in last.winner:
                        away_count += 1
                    # 平局预测不计入任何一方

        if home_count < away_count:
            return home
        elif away_count < home_count:
            return away
        else:
            # 双方相同，随机选择
            return random.choice([home, away])

    def _run_phase1(self, agents: list[Agent], history: list[Message], rounds: list[list[Message]]) -> None:
        """阶段1：自由质疑 — 逐个质疑+立即回应+反质疑链

        每个 agent 每次只质疑一个对象，立刻让被质疑者回应并处理反质疑链，
        然后再质疑下一个（如果配额未满）。
        """
        rt_cfg = self.config.debate.roundtable
        challenge_count = rt_cfg.phase1_challenge_count
        agent_names = {a.name for a in agents}
        max_retries = self._limits["free_challenge_retries"]
        multi_target = getattr(self.config.debate, 'allow_multi_target_challenge', False)
        max_attempts_per_agent = challenge_count + max_retries

        print(f"\n  ── 自由质疑（每人需质疑 {challenge_count} 人次）──")

        phase_msgs: list[Message] = []

        for agent in agents:
            challenges_issued = 0
            attempts = 0

            while challenges_issued < challenge_count and attempts < max_attempts_per_agent:
                # 不允许多目标时每次只质疑1人
                remaining = (challenge_count - challenges_issued) if multi_target else 1

                # 构建最新表述（跳过质疑）
                all_latest = self._build_all_latest_arguments(agents, history)

                single_target = not multi_target
                msg = self._do_free_challenge_attempt(
                    agent, history, agent_names, all_latest, remaining,
                    single_target=single_target,
                )

                if msg is None:
                    # 本次尝试所有重试失败
                    break

                # 不允许多目标但LLM指定了多人，只取第一个
                target_names = [t for t in msg.target.split("、") if t]
                if not multi_target and len(target_names) > 1:
                    msg.target = target_names[0]
                    target_names = [target_names[0]]
                    print(f"    ⚠️ {agent.name} 质疑多人，仅取第一个：{msg.target}")

                challenges_issued += len(target_names)

                phase_msgs.append(msg)
                history.append(msg)
                pred = _parse_prediction(msg.content)
                agent.record_statement(msg, pred)
                self._record_prediction(agent.name, pred)
                self._print_prediction(agent.name, pred, f"质疑→{msg.target}", full_content=msg.content)

                # ★ 立刻让被质疑者回应 + 反质疑链
                for target_name in target_names:
                    target_agent = self._find_agent(target_name)
                    if not target_agent:
                        continue
                    resp_msg, resp_pred = self._do_respond(target_agent, agent, msg, history)
                    phase_msgs.append(resp_msg)
                    history.append(resp_msg)
                    self._record_prediction(target_agent.name, resp_pred)
                    self._print_prediction(target_agent.name, resp_pred, f"回应→{agent.name}", full_content=resp_msg.content)

                    # 反质疑链
                    self._process_counter_challenge_chain(resp_msg, target_agent, agent, history, phase_msgs)

                attempts += 1

        if phase_msgs:
            rounds.append(phase_msgs)

    def _do_free_challenge_attempt(
        self,
        agent: Agent,
        history: list[Message],
        agent_names: set[str],
        all_latest_arguments: str,
        remaining: int,
        single_target: bool = False,
    ) -> Message | None:
        """单次自由质疑尝试（带重试），必须指定对象

        失败的尝试不记录（record_statement 由调用方在成功后调用）。
        """
        max_retries = self._limits["free_challenge_retries"]
        for attempt in range(max_retries + 1):
            challenge_msg = Message(
                role="user",
                content=USER_ROUNDTABLE_FREE_CHALLENGE(
                    agent.name, all_latest_arguments, remaining,
                    single_target=single_target,
                ),
                speaker="系统",
            )
            msg, pred = agent.speak(
                history + [challenge_msg],
                curator=self.curator,
                match_context=self.config.display_title,
                all_latest_arguments=all_latest_arguments,
            )

            targets = self._extract_targets(msg.content, agent_names - {agent.name})

            # 单目标模式下如果匹配到多人，只取第一个
            if single_target and targets:
                first = [t for t in targets.split("、") if t]
                if len(first) > 1:
                    targets = first[0]

            if targets:
                msg.target = targets
                msg.action = "质疑"
                for t in targets.split("、"):
                    if t:
                        agent.interaction_partners.add(t)
                return msg

            # 失败：不记录，直接重试
            if attempt < max_retries:
                print(f"    ⚠️ {agent.name} 未指定质疑对象，重试 ({attempt + 1}/{max_retries})")
            else:
                print(f"    ❌ {agent.name} 多次未指定质疑对象，跳过")

        return None

    def _run_phase2(self, agents: list[Agent], history: list[Message], rounds: list[list[Message]]) -> None:
        """阶段2：分配质疑 — 双向配对，紧邻回应"""
        rt_cfg = self.config.debate.roundtable
        challenge_rounds = rt_cfg.phase2_challenge_count

        print(f"\n  ── 分配质疑（轮换配对，{challenge_rounds} 轮）──")

        for cr in range(challenge_rounds):
            paired = self._pair_agents_round_robin(agents, cr)
            phase_msgs: list[Message] = []
            for a, b in paired:
                print(f"    {a.name} ⚔ {b.name}")

                # a 质疑 b → b 回应
                ch_a, pred_a = self._do_assigned_challenge(a, b, history)
                phase_msgs.append(ch_a)
                history.append(ch_a)
                self._record_prediction(a.name, pred_a)
                self._print_prediction(a.name, pred_a, f"质疑→{b.name}", full_content=ch_a.content)

                resp_b, pred_b = self._do_respond(b, a, ch_a, history)
                phase_msgs.append(resp_b)
                history.append(resp_b)
                self._record_prediction(b.name, pred_b)
                self._print_prediction(b.name, pred_b, f"回应→{a.name}", full_content=resp_b.content)
                # 反质疑链
                self._process_counter_challenge_chain(resp_b, b, a, history, phase_msgs)

                # b 质疑 a → a 回应
                ch_b, pred_b2 = self._do_assigned_challenge(b, a, history)
                phase_msgs.append(ch_b)
                history.append(ch_b)
                self._record_prediction(b.name, pred_b2)
                self._print_prediction(b.name, pred_b2, f"质疑→{a.name}", full_content=ch_b.content)

                resp_a, pred_a2 = self._do_respond(a, b, ch_b, history)
                phase_msgs.append(resp_a)
                history.append(resp_a)
                self._record_prediction(a.name, pred_a2)
                self._print_prediction(a.name, pred_a2, f"回应→{b.name}", full_content=resp_a.content)
                # 反质疑链
                self._process_counter_challenge_chain(resp_a, a, b, history, phase_msgs)

            rounds.append(phase_msgs)

    # ── 质疑/回应原子操作 ──────────────────────────────────

    def _do_assigned_challenge(
        self,
        challenger: Agent,
        target: Agent,
        history: list[Message],
        side_label: str = "",
    ) -> tuple[Message, PredictionResult]:
        """执行分配质疑（目标确定，不会失败）"""
        if side_label:
            prompt = USER_PRO_CON_CHALLENGE(
                side_label, target.name, self._get_latest_argument(target.name, history)
            )
        else:
            prompt = USER_ROUNDTABLE_ASSIGNED_CHALLENGE(
                target.name, self._get_latest_argument(target.name, history)
            )

        challenge_msg = Message(role="user", content=prompt, speaker="系统")
        msg, pred = challenger.speak(
            history + [challenge_msg],
            curator=self.curator,
            match_context=self.config.display_title,
        )
        msg.target = target.name
        msg.action = "质疑"
        challenger.interaction_partners.add(target.name)
        challenger.record_statement(msg, pred)
        return msg, pred

    def _do_respond(
        self,
        responder: Agent,
        challenger: Agent,
        challenge_msg: Message,
        history: list[Message],
        pro_con_mode: bool = False,
    ) -> tuple[Message, PredictionResult]:
        """执行回应（紧接质疑之后）

        回应者可临时看到质疑者引用的私有证据（仅本次回应可见）。
        """
        if pro_con_mode:
            prompt_text = USER_PRO_CON_RESPOND_CHALLENGE(challenger.name, challenge_msg.content)
        else:
            prompt_text = USER_ROUNDTABLE_RESPOND_CHALLENGE(challenger.name, challenge_msg.content)

        # 提取质疑者引用的私有证据，临时注入回应者上下文
        temp_foreign = self._get_temp_foreign_evidence(challenge_msg.content, responder.name)

        respond_prompt = Message(role="user", content=prompt_text, speaker="系统")
        msg, pred = responder.speak(
            history + [respond_prompt],
            curator=self.curator,
            match_context=self.config.display_title,
            temp_foreign_evidence=temp_foreign,
        )
        msg.target = challenger.name
        msg.action = "回应"
        responder.interaction_partners.add(challenger.name)
        responder.record_statement(msg, pred)

        # 检测回应中是否包含反质疑（标志留给 _process_counter_challenge_chain 打印和处理）
        if _parse_counter_challenge(msg.content):
            msg.counter_challenge = True

        return msg, pred

    # ── 反质疑链 ──────────────────────────────────────────

    def _process_counter_challenge_chain(
        self,
        response_msg: Message,
        responder: Agent,
        challenger: Agent,
        history: list[Message],
        result_msgs: list[Message],
        pro_con_mode: bool = False,
    ) -> None:
        """检查回应是否包含反质疑，如果是则执行反质疑链

        反质疑链是嵌套的辩论：回应者→原质疑者→回应者→...
        直到某次回应不包含反质疑，或达到最大深度。
        所有链内消息直接追加到 history 和 result_msgs。
        """
        cc_text = _parse_counter_challenge(response_msg.content)
        if not cc_text:
            return

        response_msg.counter_challenge = True
        max_depth = self._limits.get("counter_challenge_max_depth", 3)
        if max_depth <= 0:
            return

        # 初始反质疑算作 depth=1
        print(f"      ↩️ {responder.name} 对 {challenger.name} 提出反质疑（深度1/{max_depth}）")

        current_responder = challenger   # 原质疑者现在变成回应者
        current_challenger = responder   # 原回应者变成反质疑者
        current_cc_text = cc_text
        current_response_msg = response_msg

        # 如果 max_depth=1，初始反质疑已耗尽深度，不需要进一步处理
        if max_depth <= 1:
            return

        # 从 depth=2 开始，初始反质疑已占 depth=1
        for depth in range(2, max_depth + 1):
            # 构建反质疑回应提示词
            prompt_text = USER_COUNTER_CHALLENGE_RESPOND(
                current_challenger.name, current_response_msg.content, current_cc_text
            )

            # 提取反质疑者引用的私有证据，临时注入回应者上下文
            temp_foreign = self._get_temp_foreign_evidence(current_cc_text, current_responder.name)

            respond_prompt = Message(role="user", content=prompt_text, speaker="系统")
            msg, pred = current_responder.speak(
                history + [respond_prompt],
                curator=self.curator,
                match_context=self.config.display_title,
                temp_foreign_evidence=temp_foreign,
            )
            msg.target = current_challenger.name
            msg.action = "回应"
            current_responder.interaction_partners.add(current_challenger.name)
            current_responder.record_statement(msg, pred)

            result_msgs.append(msg)
            history.append(msg)
            self._record_prediction(current_responder.name, pred)
            self._print_prediction(
                current_responder.name, pred,
                f"回应(反质疑{depth - 1})→{current_challenger.name}",
                full_content=msg.content,
            )

            # 检查这次回应是否也有反质疑
            next_cc_text = _parse_counter_challenge(msg.content)
            if not next_cc_text:
                break

            msg.counter_challenge = True
            print(f"      ↩️ {current_responder.name} 对 {current_challenger.name} 提出反质疑（深度{depth}/{max_depth}）")

            # 交换角色
            prev_responder = current_responder
            current_responder = current_challenger
            current_challenger = prev_responder
            current_cc_text = next_cc_text
            current_response_msg = msg
        else:
            # 循环耗尽 = 达到最大深度
            print(f"      ↩️ 反质疑链达到最大深度({max_depth})，终止")

    # ── 共识评判 ──────────────────────────────────────────

    def _check_consensus_via_arbitrator(self, history: list[Message]) -> tuple[bool, Message | None]:
        """仲裁者评判：使用最新表述（跳过质疑）+ 场记摘要

        即使预测不一致也请仲裁者评判，确保每阶段都有仲裁意见记录。
        """
        all_agents = self._all_agents()
        winners: list[str] = []
        for agent in all_agents:
            preds = self.predictions.get(agent.name, [])
            if preds and preds[-1].winner:
                winners.append(preds[-1].winner)
        predictions_match = bool(winners) and len(set(winners)) == 1

        # 构建最新表述（跳过质疑）
        latest_statements = self._build_all_latest_arguments(all_agents, history)

        check_msg = Message(
            role="user",
            content=USER_CONSENSUS_CHECK(latest_statements),
            speaker="系统",
        )
        verdict_msg, _ = self.arbitrator.speak(history + [check_msg])

        # 共识需要预测一致 + 仲裁者确认
        m = re.search(r"【共识】\s*(是|否|yes|no|true|false)", verdict_msg.content, re.IGNORECASE)
        arb_consensus = m.group(1).lower() in ("是", "yes", "true") if m else False
        consensus = predictions_match and arb_consensus

        return consensus, verdict_msg

    # ── 仲裁 ──────────────────────────────────────────────

    def _arbitrate(self, history: list[Message], rounds: list[list[Message]]) -> DebateResult:
        """仲裁者最终裁定 — 使用最新表述 + 场记摘要"""
        latest_statements = self._build_all_latest_arguments(self._all_agents(), history)

        judge_msg = Message(
            role="user",
            content=USER_ARBITRATE(latest_statements),
            speaker="系统",
        )
        verdict, pred = self.arbitrator.speak(history + [judge_msg])

        result = self._build_result(rounds, consensus=False)
        result.verdict = verdict
        result.final_prediction = pred
        return result

    # ── 总结分析师 ────────────────────────────────────────

    def _summarize(self, result: DebateResult) -> DebateResult:
        arg_limit = self._limits["summary_analyst_arg"]
        all_arguments = "\n\n".join(
            f"【{msg.speaker}】：{msg.content[:arg_limit]}"
            for round_msgs in result.rounds
            for msg in round_msgs if msg.role == "assistant"
        )
        arbitrator_text = ""
        if result.arbitrator_verdict:
            arbitrator_text = result.arbitrator_verdict.content
        elif result.verdict:
            arbitrator_text = result.verdict.content

        summary_msg = Message(
            role="user",
            content=USER_SUMMARY(all_arguments, arbitrator_text),
            speaker="系统",
        )
        summary_result, _ = self.summary_analyst.speak([summary_msg])
        result.summary = summary_result

        if not result.final_prediction:
            result.final_prediction = _parse_prediction(summary_result.content)

        return result

    # ── 辅助方法 ──────────────────────────────────────────

    def _all_agents(self) -> list[Agent]:
        return self.pro_agents + self.con_agents + self.roundtable_agents

    def _find_agent(self, name: str) -> Agent | None:
        return next((a for a in self._all_agents() if a.name == name), None)

    def _build_all_latest_arguments(self, agents: list[Agent], history: list[Message]) -> str:
        """构建所有人的最新表述（跳过质疑消息）"""
        arg_limit = self._limits["latest_arg_for_arbitration"]
        parts = []
        for agent in agents:
            latest = MatchPredictor._get_latest_argument(agent.name, history)
            if latest:
                parts.append(f"【{agent.name}】：{latest[:arg_limit]}")
        return "\n\n".join(parts)

    def _record_prediction(self, agent_name: str, pred: PredictionResult) -> None:
        if agent_name not in self.predictions:
            self.predictions[agent_name] = []
        self.predictions[agent_name].append(pred)

    def _get_temp_foreign_evidence(self, challenge_content: str, responder_name: str) -> list:
        """提取质疑者引用的私有证据，供回应者临时查看

        从质疑内容中提取 E001 格式的证据 ID，
        找到回应者不可见的私有证据，作为临时外部证据注入。
        """
        cited_ids = re.findall(r"E\d{3}", challenge_content)
        if not cited_ids:
            return []

        # 回应者已可见的证据 ID
        existing_ids = {ev.id for ev in self.evidence_pool.get_all_for_agent(responder_name)}

        temp_foreign = []
        for eid in cited_ids:
            ev = self.evidence_pool.get_by_id(eid)
            if ev and not ev.shared and eid not in existing_ids:
                temp_foreign.append(ev)

        return temp_foreign

    def _maybe_dump_evidence(self, round_label: str = "") -> None:
        """如果配置启用，实时导出完整证据池文件"""
        if not self._dump_evidence or not self.evidence_pool:
            return
        output_dir = self._output_dir or Path("debate_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_topic = re.sub(r'[\\/:*?"<>|\s]', '_', self.config.display_title)[:40]
        timestamp = self._run_timestamp or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dump_path = output_dir / f"evidence_pool_{timestamp}.md"
        content = f"# 证据池完整记录\n\n"
        if round_label:
            content += f"当前进度：{round_label}\n\n"
        content += self.evidence_pool.format_full_dump()
        dump_path.write_text(content, encoding="utf-8")

    def _print_prediction(self, agent_name: str, pred: PredictionResult, action: str = "", full_content: str = "") -> None:
        # 构建预测简写标签
        pred_short = f"{pred.winner}{pred.score}准{pred.confidence}"

        # 为质疑/回应标注说话者和目标的预测
        pred_label = ""
        if action and "→" in action:
            pred_label = f"({pred_short})"
            # 提取目标名并标注其预测
            target_match = re.search(r"→(.+)$", action)
            if target_match:
                target_name = target_match.group(1).strip()
                target_preds = self.predictions.get(target_name, [])
                if target_preds:
                    tp = target_preds[-1]
                    pred_label += f" →{target_name}({tp.winner}{tp.score}准{tp.confidence})"

        tag = f"{agent_name}{pred_label} {action}" if action else agent_name
        prev_preds = self.predictions.get(agent_name, [])
        changed = ""
        if len(prev_preds) >= 2:
            prev = prev_preds[-2]
            if prev.winner != pred.winner or prev.score != pred.score:
                changed = " ⚡"
        print(f"    {tag}：预测 {pred.winner} {pred.score} (准确度{pred.confidence}/10){changed}")

        # Verbose 模式：打印完整 LLM 回复
        if self.cli_verbose and full_content:
            print(f"    {'─' * 50}")
            for line in full_content.split("\n"):
                print(f"    {line}")
            print(f"    {'─' * 50}")

    def _build_result(
        self,
        rounds: list[list[Message]],
        consensus: bool,
        arbitrator_verdict: Message | None = None,
    ) -> DebateResult:
        prediction_chains: dict[str, str] = {}
        for agent in self._all_agents():
            chain = agent.format_prediction_chain()
            if chain:
                prediction_chains[agent.name] = chain

        result = DebateResult(
            topic=self.config.display_title,
            mode=self.config.debate.mode,
            rounds=rounds,
            predictions=dict(self.predictions),
            prediction_chains=prediction_chains,
            consensus=consensus,
            arbitrator_verdict=arbitrator_verdict,
            phase_verdicts=self._phase_verdicts,
        )
        if consensus:
            for agent_preds in self.predictions.values():
                for p in reversed(agent_preds):
                    if p.winner:
                        result.final_prediction = p
                        break
                if result.final_prediction:
                    break
        return result

    @staticmethod
    def _pair_agents_round_robin(
        agents: list[Agent],
        round_offset: int = 0,
    ) -> list[tuple[Agent, Agent]]:
        """Round-robin 配对，每轮配对不同。奇数人时最后一人轮空（不参与）。"""
        n = len(agents)
        if n < 2:
            return []

        indices = list(range(n))
        if round_offset > 0 and n > 2:
            rotated = indices[1:]
            shift = round_offset % len(rotated)
            rotated = rotated[-shift:] + rotated[:-shift]
            indices = [indices[0]] + rotated

        pairs: list[tuple[Agent, Agent]] = []
        paired_count = n if n % 2 == 0 else n - 1  # 奇数时最后一人不参与
        for i in range(paired_count // 2):
            j = paired_count - 1 - i
            pairs.append((agents[indices[i]], agents[indices[j]]))

        return pairs

    @staticmethod
    def _get_latest_argument(agent_name: str, history: list[Message]) -> str:
        """获取 agent 最新完整观点表达（跳过质疑，只看立场和回应）

        质疑不是完整观点表达，不应被别人当作质疑对象。
        回应视作完整陈述，可作为质疑对象。
        """
        for msg in reversed(history):
            if msg.speaker == agent_name and msg.action != "质疑":
                return msg.content
        return ""

    @staticmethod
    def _extract_targets(content: str, candidate_names: set[str]) -> str:
        """从 agent 回复内容中提取被质疑的对象名字"""
        found = [name for name in candidate_names if name in content]
        return "、".join(found)


# ── 配置解析辅助函数 ──────────────────────────────────────


def _cfg_get(sys_config, dotted_path: str, default):
    """沿点号路径安全取值，None 或缺失则返回 default"""
    if sys_config is None:
        return default
    parts = dotted_path.split(".")
    obj = sys_config
    for part in parts:
        obj = getattr(obj, part, None)
        if obj is None:
            return default
    return obj


def _resolve_limits(sys_config) -> dict:
    """解析 limits 配置，返回字典（含默认值兜底）"""
    defaults = {
        "other_agent_summary": 100,
        "round_summary_arg": 300,
        "summary_analyst_arg": 300,
        "latest_arg_for_arbitration": 500,
        "free_challenge_retries": 2,
        "counter_challenge_max_depth": 3,
    }
    limits = _cfg_get(sys_config, "limits", None)
    if limits is None:
        return defaults
    return {k: getattr(limits, k, None) or defaults[k] for k in defaults}


def _resolve_temperatures(sys_config) -> dict:
    """解析 temperatures 配置，返回字典（含默认值兜底）"""
    defaults = {
        "arbitrator": 0.3,
        "summary_analyst": 0.3,
        "evidence_curator": 0.1,
        "round_summarizer": 0.1,
        "thinking": 0.3,
        "research_llm": 0.3,
    }
    temps = _cfg_get(sys_config, "temperatures", None)
    if temps is None:
        return defaults
    return {k: getattr(temps, k, None) or defaults[k] for k in defaults}


def _resolve_fetch(sys_config) -> dict | None:
    """解析 fetch 配置，返回字典（含默认值兜底）"""
    defaults = {
        "max_content_length": 3000,
        "fetch_timeout": 10,
        "search_max_results": 5,
    }
    fetch = _cfg_get(sys_config, "fetch", None)
    if fetch is None:
        return defaults
    return {k: getattr(fetch, k, None) or defaults[k] for k in defaults}
