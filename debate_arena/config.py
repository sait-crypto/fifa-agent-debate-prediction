"""系统配置

配置分层：
- system_default.yaml / system.yaml  — 辩论参数 + llm_preset（选择使用哪个 LLM）
- .env                                — 所有 LLM 预设的密钥和连接信息

切换 LLM：只需修改 system.yaml 中的 llm_preset 字段。
辩论参数的默认值全部来自 system_default.yaml，不在代码中硬编码。
"""

import os
import shutil
from pathlib import Path

import yaml
from pydantic import BaseModel

from .match_config import DebateConfig


# ── 配置模型 ──────────────────────────────────────────────


class OutputConfig(BaseModel):
    """CLI 输出行为"""

    cli_verbose: bool | None = None         # true 时实时输出 LLM 完整回复
    cli_truncate: int | None = None         # 终端显示截断字数
    filename_topic_length: int | None = None  # 文件名话题最大长度
    dump_evidence_pool: bool | None = None  # 辩论中实时导出完整证据池文件


class LimitsConfig(BaseModel):
    """上下文/摘要截断限制"""

    other_agent_summary: int | None = None       # 非交互 agent 消息摘要长度
    round_summary_arg: int | None = None         # 场记参数截断
    summary_analyst_arg: int | None = None       # 总结分析师参数截断
    latest_arg_for_arbitration: int | None = None  # 仲裁/共识检查参数截断
    free_challenge_retries: int | None = None    # 自由质疑重试次数
    counter_challenge_max_depth: int | None = None  # 反质疑链最大深度（0=禁用）


class FetchConfig(BaseModel):
    """检索/抓取参数"""

    max_content_length: int | None = None  # 网页抓取最大字符数
    fetch_timeout: int | None = None       # HTTP 超时（秒）
    search_max_results: int | None = None  # 搜索结果数上限


class TemperatureConfig(BaseModel):
    """特殊角色温度覆盖（基础 agent 温度仍用 debate.temperature）"""

    arbitrator: float | None = None
    summary_analyst: float | None = None
    evidence_curator: float | None = None
    round_summarizer: float | None = None
    thinking: float | None = None
    research_llm: float | None = None      # researcher.py LLM 降级检索


class SystemConfig(BaseModel):
    """系统配置（来自 YAML）"""

    llm_preset: str = "openai"       # 当前使用的 LLM 预设名（对应 .env 中的前缀）
    debate: DebateConfig | None = None  # 辩论参数（来自 YAML，不硬编码）
    output: OutputConfig | None = None
    limits: LimitsConfig | None = None
    fetch: FetchConfig | None = None
    temperatures: TemperatureConfig | None = None


class LLMConfig(BaseModel):
    """LLM 连接配置（从 .env 按 preset 名解析）"""

    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7


# ── 加载逻辑 ──────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SYSTEM_DEFAULT = _PROJECT_ROOT / "system_default.yaml"
_SYSTEM_USER = _PROJECT_ROOT / "system.yaml"
_ENV_FILE = _PROJECT_ROOT / ".env"


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 覆盖 base"""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_env() -> dict[str, str]:
    """手动解析 .env 文件，返回 key=value 映射（不依赖 pydantic-settings）"""
    env: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return env
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _resolve_preset(preset: str, env: dict[str, str]) -> LLMConfig:
    """根据预设名从 .env 映射中解析 LLM 配置

    .env 中的格式：{PRESET}_API_KEY / {PRESET}_BASE_URL / {PRESET}_MODEL
    例如 llm_preset=deepseek → DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
    """
    prefix = preset.upper().replace("-", "_")
    api_key = env.get(f"{prefix}_API_KEY", "")
    base_url = env.get(f"{prefix}_BASE_URL", "")
    model = env.get(f"{prefix}_MODEL", "")

    if not api_key:
        # 尝试 fallback 到系统环境变量
        api_key = os.environ.get(f"{prefix}_API_KEY", "")
        base_url = base_url or os.environ.get(f"{prefix}_BASE_URL", "")
        model = model or os.environ.get(f"{prefix}_MODEL", "")

    if not api_key:
        print(f"  ⚠️  预设 '{preset}' 未找到 API Key")
        print(f"     请在 .env 中设置 {prefix}_API_KEY")
        print(f"     可用预设前缀：{', '.join(_list_presets(env))}")

    return LLMConfig(
        api_key=api_key or "sk-xxx",
        base_url=base_url or "https://api.openai.com/v1",
        model=model or "gpt-4o-mini",
    )


def _list_presets(env: dict[str, str]) -> list[str]:
    """列出 .env 中所有可用的 LLM 预设名"""
    presets: set[str] = set()
    for key in env:
        if key.endswith("_API_KEY"):
            prefix = key[: -len("_API_KEY")]
            presets.add(prefix.lower().replace("_", "-"))
    return sorted(presets)


def load_system_config() -> SystemConfig:
    """加载系统配置（合并 system_default.yaml + system.yaml）

    默认值全部来自 YAML 文件，不在代码中硬编码。
    """
    default_data: dict = {}
    if _SYSTEM_DEFAULT.exists():
        default_data = yaml.safe_load(_SYSTEM_DEFAULT.read_text(encoding="utf-8")) or {}

    if not _SYSTEM_USER.exists():
        shutil.copy2(_SYSTEM_DEFAULT, _SYSTEM_USER)

    user_data: dict = {}
    if _SYSTEM_USER.exists():
        user_data = yaml.safe_load(_SYSTEM_USER.read_text(encoding="utf-8")) or {}

    merged = _deep_merge(default_data, user_data)
    return SystemConfig(**merged)


def load_llm_config(preset: str | None = None) -> LLMConfig:
    """加载 LLM 配置

    如果指定 preset，使用该预设；否则从 system.yaml 读取 llm_preset。
    """
    if preset is None:
        sys_config = load_system_config()
        preset = sys_config.llm_preset

    env = _load_env()
    return _resolve_preset(preset, env)


def list_available_presets() -> list[str]:
    """列出 .env 中所有可用的 LLM 预设"""
    env = _load_env()
    return _list_presets(env)
