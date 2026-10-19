"""Retrieval endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from recap.domain.retrieval import RetrievalDecision, RetrievalRequest
from recap.errors import NotFoundError, SearchUnavailable

router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/search", response_model=RetrievalDecision)
def search(payload: RetrievalRequest, request: Request) -> RetrievalDecision:
    container = request.app.state.get_container()
    with container.services() as services:
        return services.recall.retrieve(payload)


@router.post("/retrieval/select", response_model=RetrievalDecision)
def select(payload: RetrievalRequest, request: Request) -> RetrievalDecision:
    container = request.app.state.get_container()
    with container.services() as services:
        return services.recall.retrieve(payload)


@router.get("/retrieval/decisions/{decision_id}", response_model=RetrievalDecision)
def get_decision(decision_id: str, request: Request, workspace_id: str) -> RetrievalDecision:
    container = request.app.state.get_container()
    with container.services() as services:
        decision = services.decisions.get(decision_id, workspace_id)
        if decision is None:
            raise NotFoundError(f"decision {decision_id} not found")
        return decision
