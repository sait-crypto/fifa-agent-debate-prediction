"""比赛配置 — 结构化的比赛上下文 + 辩论配置 + 本地证据

从 YAML 文件加载，所有参数统一管理。
只有 match.home 和 match.away 必填，其他全部可选。

辩论参数的默认值不在代码中硬编码，全部来自 system_default.yaml：
- system_default.yaml → system.yaml 覆盖 → 作为辩论默认值
- matches/*.yaml 的 debate 字段可选择性覆盖系统默认值
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


# ── 数据模型 ──────────────────────────────────────────────


class TournamentInfo(BaseModel):
    """赛事信息"""

    name: str = ""
    stage: str = ""   # group / round-of-16 / quarter-final / semi-final / final
    group: str = ""   # 小组赛时填写，如 "A"


class MatchInfo(BaseModel):
    """比赛基本信息（home/away 必填）"""

    home: str
    away: str
    date: str = ""
    match_time: str = ""  # 开赛时间（如 "21:00"），区别于 date
    venue: str = ""
    match_number: int | None = None


class TeamLineup(BaseModel):
    """球队阵容信息"""

    world_ranking: int | None = None  # FIFA 世界排名
    formation: str = ""
    lineup: list[str] = []
    injuries: list[str] = []
    suspensions: list[str] = []


class HistoryMatch(BaseModel):
    """历史比赛记录"""

    opponent: str
    score: str = ""
    stage: str = ""


class H2H(BaseModel):
    """交锋记录"""

    recent_results: list[str] = []
    notes: str = ""


class RoundtablePhaseConfig(BaseModel):
    """圆桌辩论环节配置 — 默认值来自 YAML，不硬编码"""

    phase1_enabled: bool | None = None
    phase2_enabled: bool | None = None
    phase_order: list[str] | None = None
    phase1_challenge_count: int | None = None
    phase2_challenge_count: int | None = None


class DebateConfig(BaseModel):
    """辩论配置 — 默认值来自 YAML，不硬编码

    所有字段均可选（None），实际默认值由 system_default.yaml 定义。
    加载时通过 merge_debate_config() 合并系统默认值和比赛覆盖值。
    """

    mode: str | None = None                # "pro_con" | "roundtable"
    pro_count: int | None = None           # 正方 agent 数量
    con_count: int | None = None           # 反方 agent 数量
    agent_count: int | None = None         # 圆桌 agent 数量
    assigned_stance_count: int | None = None  # 圆桌中分配立场agent数量
    max_rounds: int | None = None
    agent_search: bool | None = None       # agent 是否可自主检索
    temperature: float | None = None
    agent_speech_hint: int | None = None   # 发言字数软限制提示（0=不限）
    allow_multi_target_challenge: bool | None = None  # 单次质疑能否指定多个对象
    pro_con_challenge_enabled: bool | None = None   # 正反方质疑环节
    pro_con_challenge_per_agent: int | None = None   # 每个 agent 质疑几个对手
    roundtable: RoundtablePhaseConfig | None = None


class MatchConfig(BaseModel):
    """比赛配置 — 完整的比赛上下文 + 辩论参数 + 证据文件引用"""

    tournament: TournamentInfo = TournamentInfo()
    match: MatchInfo
    home_team: TeamLineup = TeamLineup()
    away_team: TeamLineup = TeamLineup()
    home_history: list[HistoryMatch] = []
    away_history: list[HistoryMatch] = []
    h2h: H2H = H2H()
    notes: str = ""
    debate: DebateConfig | None = None     # None 表示使用系统默认值
    evidence_files: list[str] = []         # 引用 evidence/ 目录下的 YAML 文件名
    analysis_file: str = ""                # 百度体育分析数据 JSON 文件路径

    @property
    def home(self) -> str:
        return self.match.home

    @property
    def away(self) -> str:
        return self.match.away

    @property
    def display_title(self) -> str:
        stage = self.tournament.stage
        if stage:
            from .prompts import STAGE_CN
            stage_cn = STAGE_CN.get(stage, stage)
            return f"{self.home} vs {self.away}（{stage_cn}）"
        return f"{self.home} vs {self.away}"

    def get_team_info(self) -> dict:
        """返回两队信息（含世界排名），供 renderer 和前端使用"""
        return {
            self.match.home: {"world_ranking": self.home_team.world_ranking},
            self.match.away: {"world_ranking": self.away_team.world_ranking},
        }

    def get_match_meta(self) -> dict:
        """返回比赛元数据（stage/group/match_time 等），供前端显示"""
        meta: dict = {}
        # Build stage string
        parts = []
        if self.tournament.group:
            parts.append(self.tournament.group)
        if self.tournament.stage:
            from .prompts import STAGE_CN
            stage_cn = STAGE_CN.get(self.tournament.stage, self.tournament.stage)
            parts.append(stage_cn)
        if parts:
            meta["stage"] = " · ".join(parts)
        if self.match.match_time:
            meta["match_time"] = self.match.match_time
        if self.match.date:
            meta["date"] = self.match.date
        return meta


# ── 合并逻辑 ──────────────────────────────────────────────


def merge_debate_config(
    system_debate: DebateConfig,
    match_debate: DebateConfig | None = None,
) -> DebateConfig:
    """合并系统默认辩论配置和比赛覆盖配置

    优先级：match_debate > system_debate
    match_debate 中为 None 的字段使用 system_debate 的值。
    roundtable 子配置也做同样的合并。
    """
    if match_debate is None:
        return system_debate

    # 取 match_debate 非 None 的值，否则用 system_debate
    merged_data: dict = {}
    for field_name in DebateConfig.model_fields:
        match_val = getattr(match_debate, field_name)
        system_val = getattr(system_debate, field_name)

        if field_name == "roundtable":
            # 子配置递归合并
            merged_data[field_name] = _merge_roundtable(
                system_val, match_val
            )
        elif match_val is not None:
            merged_data[field_name] = match_val
        elif system_val is not None:
            merged_data[field_name] = system_val
        # 两者都 None 则不设（Pydantic 会报错，但 YAML 一定会提供完整默认值）

    return DebateConfig(**merged_data)


def _merge_roundtable(
    system_rt: RoundtablePhaseConfig | None,
    match_rt: RoundtablePhaseConfig | None,
) -> RoundtablePhaseConfig:
    """合并圆桌子配置"""
    if system_rt is None and match_rt is None:
        return RoundtablePhaseConfig()
    if system_rt is None:
        return match_rt
    if match_rt is None:
        return system_rt

    merged_data: dict = {}
    for field_name in RoundtablePhaseConfig.model_fields:
        match_val = getattr(match_rt, field_name)
        system_val = getattr(system_rt, field_name)
        merged_data[field_name] = match_val if match_val is not None else system_val

    return RoundtablePhaseConfig(**merged_data)


# ── 加载 ──────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EVIDENCE_DIR = _PROJECT_ROOT / "evidence"


def load_evidence_files(names: list[str]) -> list[str]:
    """从 evidence/ 目录加载指定文件，返回所有证据条目

    每个证据文件是 YAML 格式，内容为字符串列表。
    支持多种选择方式：
    - "bra_vs_arg_brazil.yaml" — 指定文件名
    - "bra_vs_arg_brazil"      — 自动补 .yaml 后缀
    """
    items: list[str] = []
    for name in names:
        # 自动补后缀
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        filepath = _EVIDENCE_DIR / filename
        if not filepath.exists():
            print(f"  ⚠️  证据文件不存在：{filename}")
            continue
        data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
        if isinstance(data, list):
            items.extend(str(item) for item in data if item)
        elif isinstance(data, str):
            items.append(data)
    return items


def load_match_config(path: str) -> MatchConfig:
    """从 YAML 文件加载比赛配置"""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not data or "match" not in data:
        raise ValueError(f"配置文件缺少必填的 'match' 字段：{path}")
    if "home" not in data["match"] or "away" not in data["match"]:
        raise ValueError(f"配置文件的 match 字段需要 home 和 away：{path}")
    return MatchConfig(**data)


def quick_match_config(home: str, away: str) -> MatchConfig:
    """从队名快速创建最小配置（向后兼容）"""
    return MatchConfig(match=MatchInfo(home=home, away=away))


# ── 格式化 ────────────────────────────────────────────────


def format_match_context(config: MatchConfig, analysis_data: "AnalysisData | None" = None) -> str:
    """将比赛配置格式化为可注入 prompt 的上下文

    只输出用户实际填写的字段，空字段不出现。
    当 analysis_data 非空时，插入情报和战绩数据（替代 YAML 中的历史战绩和交锋记录）。
    """
    from .analysis_parser import format_analysis_for_prompt

    sections: list[str] = []

    # ── 赛事信息 ──
    if config.tournament.name or config.tournament.stage or config.tournament.group:
        lines = ["## 赛事信息", ""]
        if config.tournament.name:
            lines.append(f"- 赛事：{config.tournament.name}")
        if config.tournament.stage:
            from .prompts import STAGE_CN
            stage_cn = STAGE_CN.get(config.tournament.stage, config.tournament.stage)
            lines.append(f"- 阶段：{stage_cn}")
        if config.tournament.group:
            lines.append(f"- 小组：{config.tournament.group}")
        lines.append("")
        sections.append("\n".join(lines))

    # ── 比赛基本信息 ──
    match_lines = ["## 比赛信息", ""]
    match_lines.append(f"- 主队：{config.match.home}")
    if config.home_team.world_ranking is not None:
        match_lines.append(f"- {config.match.home}世界排名：{config.home_team.world_ranking}")
    match_lines.append(f"- 客队：{config.match.away}")
    if config.away_team.world_ranking is not None:
        match_lines.append(f"- {config.match.away}世界排名：{config.away_team.world_ranking}")
    if config.match.date:
        match_lines.append(f"- 日期：{config.match.date}")
    if config.match.match_time:
        match_lines.append(f"- 开赛时间：{config.match.match_time}")
    if config.match.venue:
        match_lines.append(f"- 场地：{config.match.venue}")
    if config.match.match_number is not None:
        match_lines.append(f"- 场次：第 {config.match.match_number} 场")
    match_lines.append("")
    sections.append("\n".join(match_lines))

    # ── 阵容信息 ──
    for label, team in [(config.match.home, config.home_team), (config.match.away, config.away_team)]:
        has_info = team.formation or team.lineup or team.injuries or team.suspensions
        if has_info:
            lines = [f"## {label}阵容", ""]
            if team.formation:
                lines.append(f"- 阵型：{team.formation}")
            if team.lineup:
                lines.append(f"- 首发：{', '.join(team.lineup)}")
            if team.injuries:
                lines.append(f"- 伤停：{', '.join(team.injuries)}")
            if team.suspensions:
                lines.append(f"- 停赛：{', '.join(team.suspensions)}")
            lines.append("")
            sections.append("\n".join(lines))

    # ── 分析数据（情报+战绩）──
    if analysis_data:
        analysis_text = format_analysis_for_prompt(analysis_data)
        if analysis_text:
            sections.append(analysis_text)

    # ── 历史战绩（YAML 提供，分析数据存在时跳过）──
    if not analysis_data or not analysis_data.record_summaries:
        for label, history in [(config.match.home, config.home_history), (config.match.away, config.away_history)]:
            if history:
                lines = [f"## {label}本届战绩", ""]
                for h in history:
                    stage_info = f"（{h.stage}）" if h.stage else ""
                    lines.append(f"- vs {h.opponent} {h.score}{stage_info}")
                lines.append("")
                sections.append("\n".join(lines))

    # ── 交锋记录（YAML 提供，分析数据存在时跳过）──
    if not analysis_data or not analysis_data.record_summaries:
        if config.h2h.recent_results or config.h2h.notes:
            lines = ["## 交锋记录", ""]
            if config.h2h.recent_results:
                lines.append(f"- 近期比分：{', '.join(config.h2h.recent_results)}")
            if config.h2h.notes:
                lines.append(f"- 备注：{config.h2h.notes}")
            lines.append("")
            sections.append("\n".join(lines))

    # ── 补充说明 ──
    if config.notes:
        sections.append(f"## 补充说明\n\n{config.notes}\n")

    return "\n".join(sections)
