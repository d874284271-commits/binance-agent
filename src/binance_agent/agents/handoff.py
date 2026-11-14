"""Explicit, protected context passed between research participants."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HandoffContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    run_id: str
    agent_id: str
    parent_agent_id: str | None = None
    task_id: str
    role: str
    root_question: str = ""
    turn_question: str = ""
    user_updates: list[str] = Field(default_factory=list)
    instruction: str
    completion: str = "返回可追溯的结果,并明确失败、冲突和未解决事项"
    allowed_tools: list[str] = Field(default_factory=list)
    snapshot_ref: str
    analysis_at: str | None = None
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    historical_context: list[dict[str, Any]] = Field(default_factory=list)
    input_revision: int = Field(ge=0)
    budget_ref: str
    config_fingerprint: str
