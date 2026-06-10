"""URL 内容抓取 — 抓取网页并清洗为纯文本

用于将搜索结果的 URL 抓取正文，作为更丰富的证据来源。
失败时静默降级，不影响主流程。

依赖：httpx（HTTP 请求）、lxml（HTML 解析）
两者均为 ddgs 的传递依赖，同时也在 pyproject.toml 中显式声明。
"""

import re

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# 要移除的 HTML 标签（不含正文信息）
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "noscript")


def fetch_url(url: str, max_content_length: int = 3000, fetch_timeout: int = 10) -> str | None:
    """抓取 URL 并返回清洗后的文本内容，失败返回 None

    流程：HTTP GET → 检查状态码 → HTML 转文本 → 截断
    """
    try:
        import httpx
    except ImportError:
        return None

    try:
        with httpx.Client(
            timeout=fetch_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return _html_to_text(resp.text, max_content_length)
    except Exception:
        return None


def _html_to_text(raw_html: str, max_content_length: int = 3000) -> str:
    """将 HTML 转换为清洗后的纯文本

    1. 使用 lxml 解析 HTML
    2. 移除 script/style/nav 等非正文标签
    3. 提取文本内容
    4. 清洗空白和换行
    5. 截断到 max_content_length
    """
    text = _lxml_extract(raw_html)
    if not text:
        # 回退：正则去标签
        text = re.sub(r"<[^>]+>", " ", raw_html)

    # 清洗连续空白
    text = re.sub(r"\s+", " ", text).strip()

    # 截断
    if len(text) > max_content_length:
        text = text[:max_content_length] + "…"

    return text


def _lxml_extract(raw_html: str) -> str:
    """使用 lxml 提取 HTML 正文文本"""
    try:
        from lxml import html as lxml_html
    except ImportError:
        return ""

    try:
        tree = lxml_html.fromstring(raw_html)
        # 移除非正文标签
        for tag in _STRIP_TAGS:
            for el in tree.iter(tag):
                parent = el.getparent()
                if parent is not None:
                    parent.remove(el)
        return tree.text_content()
    except Exception:
        return ""
