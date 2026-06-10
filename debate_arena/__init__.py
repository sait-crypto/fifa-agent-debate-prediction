"""多智能体辩论预测系统 — 证据驱动"""

from .agent import Agent
from .config import LLMConfig, SystemConfig, list_available_presets, load_llm_config, load_system_config
from .evidence import Evidence, EvidencePool
from .match_config import (
    DebateConfig,
    MatchConfig,
    RoundtablePhaseConfig,
    load_match_config,
    quick_match_config,
)
from .models import DebateResult, Message, PredictionResult
from .predict import MatchPredictor
from .renderer import Renderer

__all__ = [
    "Agent",
    "DebateConfig",
    "DebateResult",
    "Evidence",
    "EvidencePool",
    "LLMConfig",
    "MatchConfig",
    "MatchPredictor",
    "Message",
    "PredictionResult",
    "Renderer",
    "RoundtablePhaseConfig",
    "SystemConfig",
    "list_available_presets",
    "load_llm_config",
    "load_match_config",
    "load_system_config",
    "quick_match_config",
]
