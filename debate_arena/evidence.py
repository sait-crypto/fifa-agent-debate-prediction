"""证据池 — 共用池 + 私人池，统一编号防篡改

证据来源三层（初始置信度递减）：
1. local:        用户在配置文件中提供的本地证据（0.9）
2. research:     辩论开始时统一检索的证据（0.7）
3. agent_search: agent 自主检索的证据（0.5，初始在私人池，可共享）

私人证据初始置信度为中等（0.5），其目的是防止无关证据涌入共用池，
而非因为证据本身不可靠。是否入共用池由证据审核员判断。

所有证据都有唯一编号（E001, E002 ...），保证在传递中不因幻觉发生变化。
agent 只引用编号，所有人可查阅原始证据验证无编造和篡改。

置信度为 0-1 浮点数，审核员可在基准上浮动，但 hard clamp [0.15, 0.95]。
"""

from pydantic import BaseModel

from .researcher import MatchFact


# 来源中文标签（用于 prompt 和报告显示）
_SOURCE_LABEL = {
    "local": "本地池",
    "research": "统一检索",
    "agent_search": "Agent检索",
}


class Evidence(BaseModel):
    """一条证据"""

    id: str              # 唯一编号 "E001"
    content: str         # 证据正文（抓取的网页文本 / 本地文本）
    source: str          # "local" | "research" | "agent_search"
    confidence: float    # 0.0-1.0 置信度
    discovered_by: str   # "本地池" | "统一检索" | "Agent检索:N名"
    shared: bool         # 是否在共用池中
    title: str = ""      # 网页标题 / 证据标题
    url: str = ""        # 来源链接
    summary: str = ""    # 摘要（搜索 snippet / 简短描述）

    @property
    def source_label(self) -> str:
        """来源中文标签"""
        return _SOURCE_LABEL.get(self.source, self.source)


class EvidencePool:
    """证据池 — 管理共用和私人证据"""

    def __init__(self) -> None:
        self.shared: list[Evidence] = []
        self.private: dict[str, list[Evidence]] = {}
        self._next_id: int = 1

    def _alloc_id(self) -> str:
        eid = f"E{self._next_id:03d}"
        self._next_id += 1
        return eid

    # ── 添加证据 ──────────────────────────────────────────

    def add_local(self, items: list[str]) -> None:
        """添加本地证据（置信度 0.9，直接入共用池）"""
        for item in items:
            eid = self._alloc_id()
            self.shared.append(Evidence(
                id=eid,
                content=item,
                source="local",
                confidence=0.9,
                discovered_by="本地池",
                shared=True,
            ))

    def add_research(self, facts: list[MatchFact]) -> None:
        """添加检索证据（置信度 0.7，直接入共用池）"""
        for fact in facts:
            eid = self._alloc_id()
            self.shared.append(Evidence(
                id=eid,
                content=fact.content,
                source="research",
                confidence=0.7,
                discovered_by="统一检索",
                shared=True,
                title=getattr(fact, "title", ""),
                url=getattr(fact, "url", "") or fact.source,
                summary=getattr(fact, "body", ""),
            ))

    def add_agent_search(self, agent_name: str, facts: list[MatchFact]) -> list[Evidence]:
        """添加 agent 自主检索的证据（初始入私人池，置信度 0.5）
        私人证据初始置信度为中等，其目的是防止无关证据涌入共用池。
        """
        if agent_name not in self.private:
            self.private[agent_name] = []

        new_evidence: list[Evidence] = []
        for fact in facts:
            eid = self._alloc_id()
            ev = Evidence(
                id=eid,
                content=fact.content,
                source="agent_search",
                confidence=0.5,
                discovered_by=f"Agent检索:{agent_name}",
                shared=False,
                title=getattr(fact, "title", ""),
                url=getattr(fact, "url", "") or fact.source,
                summary=getattr(fact, "body", ""),
            )
            self.private[agent_name].append(ev)
            new_evidence.append(ev)
        return new_evidence

    # ── 共享机制 ──────────────────────────────────────────

    def share_to_pool(self, agent_name: str, evidence_id: str) -> None:
        """将 agent 私人证据共享到共用池"""
        if agent_name not in self.private:
            return
        for ev in self.private[agent_name]:
            if ev.id == evidence_id and not ev.shared:
                ev.shared = True
                self.shared.append(ev)
                return

    def update_confidence(self, evidence_id: str, confidence: float) -> None:
        """审核员修改证据置信度（clamp 到 [0.15, 0.95]）"""
        confidence = max(0.15, min(0.95, confidence))
        ev = self.get_by_id(evidence_id)
        if ev:
            ev.confidence = confidence

    def curate_and_share(self, agent_name: str, evidence_id: str, confidence: float) -> None:
        """原子操作：共享到共用池 + 设置审核置信度"""
        self.share_to_pool(agent_name, evidence_id)
        self.update_confidence(evidence_id, confidence)

    # ── 查询 ──────────────────────────────────────────────

    def get_by_id(self, evidence_id: str) -> Evidence | None:
        """按编号查找证据（先查共用池，再查所有私人池）"""
        for ev in self.shared:
            if ev.id == evidence_id:
                return ev
        for agent_evs in self.private.values():
            for ev in agent_evs:
                if ev.id == evidence_id:
                    return ev
        return None

    def get_all_for_agent(self, agent_name: str) -> list[Evidence]:
        """获取 agent 可见的所有证据（共用池 + 自身未共享的私人证据）"""
        result = list(self.shared)
        if agent_name in self.private:
            # 只包含尚未共享的私人证据（已共享的已在 shared 列表中）
            result.extend(ev for ev in self.private[agent_name] if not ev.shared)
        # 按置信度降序，同置信度按编号升序
        result.sort(key=lambda e: (-e.confidence, e.id))
        return result

    def get_shared_count(self) -> int:
        return len(self.shared)

    def get_private_count(self, agent_name: str) -> int:
        return len(self.private.get(agent_name, []))

    # ── 格式化 ────────────────────────────────────────────

    def format_for_prompt(
        self,
        agent_name: str,
        temp_foreign_evidence: list[Evidence] | None = None,
    ) -> str:
        """格式化 agent 可见的所有证据，注入到 prompt

        temp_foreign_evidence: 临时外部证据（质疑者引用的私有证据），
        仅本次调用可见，不永久添加到池中，标记为"临时"。
        """
        evidence = self.get_all_for_agent(agent_name)

        # 添加临时外部证据（去重：避免已在共用池或自己私人池中的）
        temp_ids: set[str] = set()
        if temp_foreign_evidence:
            existing_ids = {ev.id for ev in evidence}
            for fev in temp_foreign_evidence:
                if fev.id not in existing_ids:
                    evidence.append(fev)
                    temp_ids.add(fev.id)

        if not evidence:
            return "（暂无证据）"

        lines = ["## 可用证据", ""]
        for ev in evidence:
            if ev.id in temp_ids:
                visibility = "临时"
            elif ev.shared:
                visibility = "共用"
            else:
                visibility = "私人"
            conf_pct = f"{ev.confidence:.0%}"
            source_tag = ev.source_label
            lines.append(
                f"[{ev.id}] ({conf_pct}·{source_tag}·{visibility}) {ev.content}"
            )
        lines.append("")
        return "\n".join(lines)

    def format_summary(self) -> str:
        """格式化证据池摘要（用于终端输出）"""
        lines = [f"共用证据：{len(self.shared)} 条"]
        for name, evs in self.private.items():
            private_count = len([e for e in evs if not e.shared])
            if private_count > 0:
                lines.append(f"{name} 私人证据：{private_count} 条")
        return " | ".join(lines)

    def format_full_dump(self) -> str:
        """输出所有证据为 YAML 列表格式，可直接作为 match config 的本地证据文件读取

        格式：YAML 字符串列表，每条证据为一个含元数据的字符串，
        可被 load_evidence_files() 直接加载。
        """
        all_evidence = list(self.shared)
        seen_ids = {ev.id for ev in self.shared}
        for agent_name, evs in self.private.items():
            for ev in evs:
                if ev.id not in seen_ids:
                    all_evidence.append(ev)
                    seen_ids.add(ev.id)

        lines = ["# 证据池 — 可作为 match config 的 evidence_files 直接引用"]
        lines.append(f"# 共用证据：{len(self.shared)} 条")
        private_total = sum(len([e for e in evs if not e.shared]) for evs in self.private.values())
        lines.append(f"# 私人证据：{private_total} 条")
        lines.append("")

        for ev in sorted(all_evidence, key=lambda e: e.id):
            if ev.shared:
                visibility = "共用"
            else:
                owner = "未知"
                for agent_name, evs in self.private.items():
                    if any(e.id == ev.id for e in evs):
                        owner = agent_name
                        break
                visibility = f"私人({owner})"
            conf_pct = f"{ev.confidence:.0%}"
            source_tag = ev.source_label
            # Build the evidence string: [ID](meta) content
            meta_parts = [conf_pct, source_tag, visibility]
            if ev.url:
                meta_parts.append(ev.url)
            if ev.title:
                meta_parts.append(ev.title)
            meta_str = "·".join(meta_parts)
            content_clean = ev.content.replace('"', "'").replace("\n", " ")
            lines.append(f'- "[{ev.id}]({meta_str}) {content_clean}"')

        return "\n".join(lines)
