"""Typed repair strategy actions.

Actions are declarative templates. RECAP never executes them. The host agent
remains responsible for authorization and execution.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from recap.domain.enums import ActionType, SideEffectClass


class StrategyAction(BaseModel):
    action_type: ActionType
    tool_name: str = ""
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    max_attempts: int = Field(default=1, ge=1, le=5)
    backoff_ms: int = Field(default=0, ge=0, le=60_000)

    @field_validator("parameters")
    @classmethod
    def _bounded_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 32:
            raise ValueError("too many action parameters")
        for key in value:
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("invalid action parameter name")
        return value

    @field_validator("description")
    @classmethod
    def _bounded_description(cls, value: str) -> str:
        if len(value) > 2000:
            raise ValueError("action description too long")
        return value


class RepairStrategy(BaseModel):
    summary: str
    actions: list[StrategyAction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)

    @field_validator("actions")
    @classmethod
    def _non_empty(cls, value: list[StrategyAction]) -> list[StrategyAction]:
        if not value:
            raise ValueError("a strategy requires at least one action")
        if len(value) > 8:
            raise ValueError("strategy exceeds maximum action count")
        return value

    @property
    def side_effect_class(self) -> SideEffectClass:
        order = {
            SideEffectClass.READ_ONLY: 0,
            SideEffectClass.WRITE: 1,
            SideEffectClass.DESTRUCTIVE: 2,
        }
        return max((action.side_effect_class for action in self.actions), key=order.__getitem__)

    @property
    def action_signature(self) -> str:
        """Stable signature used for diversity caps and deduplication."""
        return "|".join(
            f"{action.action_type}:{action.tool_name}" for action in self.actions
        )
