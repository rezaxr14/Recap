"""Run and event endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from recap.domain.runs import AgentRun, RunCreate, RunEventCreate, RunFinish, RunSummary
from recap.errors import NotFoundError

router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=AgentRun, status_code=201)
def create_run(payload: RunCreate, request: Request) -> AgentRun:
    container = request.app.state.get_container()
    with container.services() as services:
        run = AgentRun(
            **payload.model_dump(),
        )
        return services.recorder.start_run(run)


@router.post("/runs/{run_id}/events", status_code=201)
def append_event(run_id: str, payload: RunEventCreate, request: Request) -> dict[str, object]:
    container = request.app.state.get_container()
    with container.services() as services:
        run = services.recorder.get_run(run_id)
        event = services.recorder.append_event(run, payload)
        return {"event_id": event.id, "sequence_number": event.sequence_number}


@router.post("/runs/{run_id}/finish", response_model=AgentRun)
def finish_run(run_id: str, payload: RunFinish, request: Request) -> AgentRun:
    container = request.app.state.get_container()
    with container.services() as services:
        run = services.recorder.get_run(run_id)
        fields = payload.model_dump(exclude={"metadata"})
        if payload.metadata:
            fields["config_snapshot"] = {**run.config_snapshot, "finish_metadata": payload.metadata}
        return services.recorder.finish_run(run, **fields)


@router.get("/runs/{run_id}", response_model=RunSummary)
def get_run(run_id: str, request: Request) -> RunSummary:
    container = request.app.state.get_container()
    with container.services() as services:
        run = services.recorder.get_run(run_id)
        events = services.runs.list_events(run_id)
        return RunSummary(
            run=run,
            event_count=len(events),
            last_sequence=events[-1].sequence_number if events else 0,
        )


@router.get("/runs/{run_id}/events")
def list_events(run_id: str, request: Request, limit: int = 100, offset: int = 0) -> dict[str, object]:
    container = request.app.state.get_container()
    with container.services() as services:
        services.recorder.get_run(run_id)
        events = services.runs.list_events(run_id)
        page = events[offset : offset + limit]
        return {
            "events": [event.model_dump(mode="json") for event in page],
            "total": len(events),
            "offset": offset,
            "limit": limit,
        }
