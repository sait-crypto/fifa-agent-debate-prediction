"""数据模型 — 消息、预测结果、辩论结果"""

from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """一条发言记录"""

    role: Literal["system", "user", "assistant"]
    content: str
    speaker: str = ""      # 智能体名字（辅助字段，不发给 LLM）
    target: str = ""       # 辩论/质疑对象（辅助字段，用于报告展示）
    action: str = ""       # 行为标签："" | "辩论" | "质疑" | "回应"
    counter_challenge: bool = False  # 回应中包含【反质疑】时为 True
    new_evidence: list[str] = []     # 本次发言中搜到的新证据ID列表（如 ["E007","E008"]）


class PredictionResult(BaseModel):
    """一个 agent 的预测结果"""

    winner: str          # 预测胜方
    score: str           # 预测比分 "2-1"
    confidence: int      # 预测准确度 1-10
    key_factors: list[str] = []  # 关键依据


class DebateResult(BaseModel):
    """一场辩论的完整结果"""

    topic: str
    mode: str            # "pro_con" | "roundtable"
    rounds: list[list[Message]]
    predictions: dict[str, list[PredictionResult]] = {}  # agent_name -> 历次预测
    prediction_chains: dict[str, str] = {}  # agent_name -> 预测变化链文本
    arbitrator_verdict: Message | None = None  # 仲裁者共识评判
    consensus: bool = False          # 是否达成共识（预测+论述一致）
    verdict: Message | None = None   # 仲裁者最终裁定（未共识时）
    final_prediction: PredictionResult | None = None
    summary: Message | None = None   # 总结分析师的结构化总结
    phase_verdicts: list[dict] = []  # 每阶段仲裁检查结果 [{phase, content, speaker}]
