"""Grounded Q&A endpoint."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from recap.ask.models import AskAnswer, AskRequest

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskAnswer)
def ask(payload: AskRequest, request: Request) -> AskAnswer:
    container = request.app.state.get_container()
    with container.services() as services:
        return cast(AskAnswer, services.ask.answer(payload))
