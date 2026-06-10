"""RAG 检索层 — 搜索外部信息并格式化为可注入的上下文

核心流程：Retrieve → Format → Inject
这是 LLM 应用中注入外部信息的最普遍模式（RAG 的简化版）。

检索策略（降级链）：
1. Web Search：DuckDuckGo 搜索真实网页数据 + URL 内容抓取
2. LLM Research：用 LLM 自身知识生成结构化研究报告
"""

from openai import OpenAI

from .models import Message
from .prompts import SYSTEM_RESEARCH_ANALYST


class MatchFact:
    """一条检索到的比赛相关事实"""

    def __init__(
        self,
        source: str,
        content: str,
        title: str = "",
        url: str = "",
        body: str = "",
    ):
        self.source = source      # URL（向后兼容）
        self.content = content    # 正文（抓取的网页文本 / 拼接的 snippet）
        self.title = title        # 页面标题
        self.url = url or source  # 可点击链接
        self.body = body          # DDG snippet 摘要


class ResearchResult:
    """检索结果"""

    def __init__(self, query: str, facts: list[MatchFact], method: str = "web"):
        self.query = query
        self.facts = facts
        self.method = method  # "web" | "llm"


def search_web(
    query: str,
    max_results: int = 5,
    fetch_content: bool = True,
    fetch_config: dict | None = None,
) -> list[MatchFact]:
    """使用 DuckDuckGo 搜索，返回检索到的事实列表

    如果搜索不可用或返回空结果，返回空列表。
    当 fetch_content=True 时，尝试抓取每个搜索结果的 URL 正文。
    fetch_config 可选：{"max_content_length": int, "fetch_timeout": int}
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except ImportError:
            return []

    facts: list[MatchFact] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "web")

                # 尝试抓取 URL 正文
                page_content = None
                if fetch_content and href and href.startswith("http"):
                    from .fetcher import fetch_url
                    fetch_kwargs = {}
                    if fetch_config:
                        if "max_content_length" in fetch_config:
                            fetch_kwargs["max_content_length"] = fetch_config["max_content_length"]
                        if "fetch_timeout" in fetch_config:
                            fetch_kwargs["fetch_timeout"] = fetch_config["fetch_timeout"]
                    page_content = fetch_url(href, **fetch_kwargs)
                    if page_content:
                        print(f"    🌐 抓取：{title[:40]}...")

                content = page_content if page_content else f"{title}\n{body}"

                facts.append(MatchFact(
                    source=href,
                    content=content,
                    title=title,
                    url=href,
                    body=body,
                ))
    except Exception:
        pass  # 网络错误静默降级
    return facts


def research_via_llm(
    team_a: str, team_b: str, client: OpenAI, model: str,
    temperature: float = 0.3,
) -> ResearchResult:
    """使用 LLM 自身知识生成结构化研究报告（Web Search 降级方案）

    当 Web Search 不可用时，用 LLM 的知识作为外部信息源。
    这在生产系统中也很常见——比如处理私有数据或搜索受限的场景。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SYSTEM_RESEARCH_ANALYST(team_a, team_b)}],
        temperature=temperature,
    )
    content = response.choices[0].message.content.strip()

    return ResearchResult(
        query=f"{team_a} vs {team_b}",
        facts=[MatchFact(source="LLM知识库", content=content)],
        method="llm",
    )


def research_match(
    team_a: str,
    team_b: str,
    client: OpenAI | None = None,
    model: str = "",
    fetch_config: dict | None = None,
    research_llm_temperature: float = 0.3,
    search_max_results: int = 3,
) -> ResearchResult:
    """检索两支球队的相关信息

    降级链：Web Search（含 URL 抓取） → LLM Research
    优先使用真实搜索数据，搜索不可用时降级到 LLM 知识库。
    fetch_config 可选：{"max_content_length": int, "fetch_timeout": int}
    """
    # ── 策略 1: Web Search ──
    all_facts: list[MatchFact] = []
    queries = [
        f"{team_a} vs {team_b} match prediction stats",
        f"{team_a} {team_b} head to head recent form",
        f"{team_a} {team_b} key players injuries",
    ]

    for q in queries:
        facts = search_web(q, max_results=search_max_results, fetch_content=True, fetch_config=fetch_config)
        all_facts.extend(facts)

    if all_facts:
        print(f"  📚 Web Search 检索到 {len(all_facts)} 条相关信息")
        return ResearchResult(
            query=f"{team_a} vs {team_b}",
            facts=all_facts,
            method="web",
        )

    # ── 策略 2: LLM Research（降级） ──
    if client and model:
        print("  🤖 Web Search 无结果，使用 LLM 知识库生成研究报告...")
        return research_via_llm(team_a, team_b, client, model, temperature=research_llm_temperature)

    # ── 都不可用 ──
    print("  ⚠️  无法检索外部信息，智能体将基于自身知识辩论")
    return ResearchResult(
        query=f"{team_a} vs {team_b}",
        facts=[],
        method="none",
    )


def format_context(result: ResearchResult) -> str:
    """将检索结果格式化为可注入 system prompt 的上下文文本

    这是 RAG 中 "Augment" 的核心——将外部信息结构化后注入提示词。
    格式化原则：
    - 清晰标注信息来源
    - 标明检索方式（web/llm）
    - 明确分隔上下文和指令
    """
    if not result.facts:
        return ""

    method_label = {
        "web": "Web 搜索",
        "llm": "LLM 知识库",
        "none": "无",
    }.get(result.method, result.method)

    lines = [
        f"## 外部信息（检索方式：{method_label}）",
        "",
    ]

    for i, fact in enumerate(result.facts, 1):
        source_label = fact.title or fact.url or fact.source
        lines.append(f"### 资料 {i}（来源：{source_label}）")
        lines.append(fact.content)
        lines.append("")

    lines.append("请在辩论中引用以上信息来支撑你的论点。如果信息不足，请说明。")
    return "\n".join(lines)
