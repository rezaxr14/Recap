"""Request/response models for grounded Q&A.

The citation contract is deliberately small: a citation is a source id plus a
verbatim quote from that source. Verification is programmatic and does not use
an LLM as a judge.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["memory", "run", "event"]
AnswerStatus = Literal["answered", "insufficient_evidence"]
Answerer = Literal["extractive", "llm"]


class AskRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["extractive", "llm"] = "extractive"
    max_sources: int = Field(default=8, ge=1, le=20)
    max_runs: int = Field(default=40, ge=1, le=200)


class Citation(BaseModel):
    type: SourceType
    id: str
    quote: str


class CitationVerification(BaseModel):
    checked: int = 0
    valid: int = 0
    invalid: int = 0
    rejected: list[dict[str, str]] = Field(default_factory=list)


class RetrievalTrace(BaseModel):
    mode: str = "hybrid_rrf"
    memory_hits: int = 0
    run_hits: int = 0
    event_hits: int = 0
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)


class AskAnswer(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    answerer: Answerer = "extractive"
    provider: str = ""
    model: str = ""
    verification: CitationVerification = Field(default_factory=CitationVerification)
    retrieval: RetrievalTrace = Field(default_factory=RetrievalTrace)
    notes: list[str] = Field(default_factory=list)
