"""RECAP command line interface.

Every command performs real work against the configured database and search
index. Commands never print success for an operation that did not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from recap.config import get_settings, load_settings
from recap.container import Container
from recap.domain.enums import MemoryStatus, RetrievalMode
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.errors import RecapError

app = typer.Typer(help="RECAP memory and recovery engine", no_args_is_help=True)
db_app = typer.Typer(help="Database operations")
search_app = typer.Typer(help="Elasticsearch operations")
memory_app = typer.Typer(help="Memory inspection")
repair_app = typer.Typer(help="Repair compilation and validation")
demo_app = typer.Typer(help="Deterministic local demo")
eval_app = typer.Typer(help="Evaluation harness")
app.add_typer(db_app, name="db")
app.add_typer(search_app, name="search")
app.add_typer(memory_app, name="memory")
app.add_typer(repair_app, name="repair")
app.add_typer(demo_app, name="demo")
app.add_typer(eval_app, name="eval")

console = Console()


def _container(config: str | None = None) -> Container:
    settings = load_settings(config) if config else get_settings()
    return Container.build(settings)


def _fail(exc: Exception) -> None:
    if isinstance(exc, RecapError):
        console.print(f"[red]{exc.code}[/red]: {exc.message}")
        if exc.details:
            console.print(json.dumps(exc.details, indent=2, default=str))
    else:
        console.print(f"[red]{type(exc).__name__}[/red]: {exc}")
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print the installed version."""
    from recap import __version__

    console.print(f"recap {__version__}")


@app.command()
def health(config: str | None = None) -> None:
    """Check database and Elasticsearch readiness."""
    try:
        container = _container(config)
        result = container.health()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, default=str))


# ---------------------------------------------------------------------- db
@db_app.command("migrate")
def db_migrate(config: str | None = None) -> None:
    """Apply database migrations (Alembic)."""
    from alembic import command
    from alembic.config import Config

    settings = load_settings(config) if config else get_settings()
    root = Path(__file__).resolve().parents[2]
    alembic_cfg = Config(str(root / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(root / "migrations"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.storage.postgres_url)
    try:
        command.upgrade(alembic_cfg, "head")
    except Exception as exc:
        _fail(exc)
        return
    console.print("[green]database migrations applied[/green]")


@db_app.command("init")
def db_init(config: str | None = None) -> None:
    """Create the baseline schema directly (development shortcut)."""
    try:
        container = _container(config)
        container.migrate()
    except Exception as exc:
        _fail(exc)
        return
    console.print("[green]database schema created[/green]")


# ------------------------------------------------------------------ search
@search_app.command("init")
def search_init(config: str | None = None) -> None:
    """Create the versioned index and aliases idempotently."""
    try:
        container = _container(config)
        result = container.init_search()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, default=str))


@search_app.command("health")
def search_health(config: str | None = None) -> None:
    """Show index/alias health and document count."""
    try:
        container = _container(config)
        result = container.index_manager.health()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, default=str))


@search_app.command("reconcile")
def search_reconcile(config: str | None = None) -> None:
    """Reindex all active memories from the database into Elasticsearch."""
    try:
        container = _container(config)
        with container.services() as services:
            memories = services.memories.list_memories(
                workspace_id=_default_workspace(), status=MemoryStatus.ACTIVE, limit=10_000
            )
            indexed = 0
            failed = 0
            for memory in memories:
                if services.index_sync(memory):
                    indexed += 1
                else:
                    failed += 1
    except Exception as exc:
        _fail(exc)
        return
    console.print(
        f"indexed={indexed} failed={failed} pending_outbox={services.outbox.pending_count()}"
    )


# ------------------------------------------------------------------ memory
@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Natural-language or error query")],
    mode: Annotated[str, typer.Option("--mode", "-m")] = "hybrid_rrf",
    workspace: Annotated[str, typer.Option("--workspace", "-w")] = "demo-workspace",
    environment_fingerprint: Annotated[str, typer.Option("--fingerprint")] = "",
    task_family: Annotated[str, typer.Option("--task-family")] = "",
    error_code: Annotated[str, typer.Option("--error-code")] = "",
    final_k: Annotated[int, typer.Option("--final-k")] = 5,
    config: str | None = None,
) -> None:
    """Search memories with a selected retrieval mode."""
    try:
        container = _container(config)
        with container.services() as services:
            request = RetrievalRequest(
                context=RetrievalContext(
                    workspace_id=workspace,
                    query=query,
                    task_family=task_family,
                    environment_fingerprint=environment_fingerprint,
                    error_code=error_code,
                ),
                mode=RetrievalMode(mode),
                final_k=final_k,
            )
            decision = services.recall.retrieve(request)
            table = Table(title=f"Retrieval decision {decision.id} [{decision.status}]")
            for column in ("rank", "memory", "score", "status", "reason"):
                table.add_column(column)
            for candidate in decision.candidates:
                table.add_row(
                    str(candidate.fused_rank),
                    candidate.title or candidate.memory_id,
                    f"{candidate.final_score:.3f}",
                    "selected" if candidate.selected else "rejected",
                    ", ".join(candidate.rejection_reasons) or "ok",
                )
            console.print(table)
            console.print(f"decision={decision.id} reason={decision.reason}")
    except Exception as exc:
        _fail(exc)


@memory_app.command("show")
def memory_show(
    memory_id: str, workspace: Annotated[str, typer.Option("--workspace")] = "demo-workspace"
) -> None:
    """Show a memory with status, evidence, and compatibility."""
    try:
        container = _container()
        with container.services() as services:
            memory = services.memories.get_memory(memory_id, workspace)
            if memory is None:
                console.print(f"[red]memory {memory_id} not found[/red]")
                raise typer.Exit(code=1)
            console.print_json(memory.model_dump_json())
            for item in services.compatibility.list_for_memory(memory_id):
                console.print_json(item.model_dump_json())
    except Exception as exc:
        _fail(exc)


@memory_app.command("compatibility")
def memory_compatibility(
    memory_id: str,
    environment_id: Annotated[str, typer.Option("--environment")] = "mock-api-v2",
    workspace: Annotated[str, typer.Option("--workspace")] = "demo-workspace",
) -> None:
    """Evaluate compatibility of a memory against an environment version."""
    from recap.environments.mock_api import PaginatedApiEnvironment

    try:
        container = _container()
        version = environment_id.rsplit("-v", 1)[-1]
        environment = PaginatedApiEnvironment(version=version)
        fingerprint = environment.fingerprint()
        with container.services() as services:
            memory = services.memories.get_memory(memory_id, workspace)
            if memory is None:
                console.print(f"[red]memory {memory_id} not found[/red]")
                raise typer.Exit(code=1)
            result = services.memory_shift.evaluate_compatibility(
                memory,
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
                tool_schema_fingerprints=fingerprint.tool_schema_hashes,
                tool_versions=fingerprint.tool_versions,
            )
            console.print_json(result.model_dump_json())
    except Exception as exc:
        _fail(exc)


# ------------------------------------------------------------------ repair
@repair_app.command("compile")
def repair_compile(
    run_id: str,
    environment_version: Annotated[str, typer.Option("--environment-version")] = "1",
) -> None:
    """Compile repair candidates from a recorded failed run."""
    from recap.environments.mock_api import PaginatedApiEnvironment

    try:
        container = _container()
        environment = PaginatedApiEnvironment(version=environment_version).fingerprint()
        with container.services() as services:
            candidates = services.repair_compiler.compile(run_id, environment=environment)
            if not candidates:
                console.print("no_repair_candidate: insufficient evidence")
                return
            for candidate in candidates:
                console.print_json(candidate.model_dump_json())
    except Exception as exc:
        _fail(exc)


@repair_app.command("validate")
def repair_validate(
    candidate_id: str,
    environment_version: Annotated[str, typer.Option("--environment-version")] = "1",
    validator_name: Annotated[str, typer.Option("--validator")] = "mock_sandbox",
) -> None:
    """Validate a candidate in the safe mock sandbox."""
    from recap.environments.mock_api import PaginatedApiEnvironment

    try:
        container = _container()
        environment = PaginatedApiEnvironment(version=environment_version).fingerprint()
        with container.services() as services:
            result = services.repair_compiler.validate(
                candidate_id, environment=environment, validator_name=validator_name
            )
            console.print_json(result.model_dump_json())
            if result.status.value != "passed":
                raise typer.Exit(code=1)
    except Exception as exc:
        _fail(exc)


@repair_app.command("promote")
def repair_promote(candidate_id: str) -> None:
    """Promote a validated candidate through the policy gate."""
    try:
        container = _container()
        with container.services() as services:
            candidate = services.repairs.get(candidate_id)
            if candidate is None:
                console.print(f"[red]candidate {candidate_id} not found[/red]")
                raise typer.Exit(code=1)
            memory = services.promotion.promote(candidate)
            console.print_json(memory.model_dump_json())
    except Exception as exc:
        _fail(exc)


# -------------------------------------------------------------------- demo
@demo_app.command("seed")
def demo_seed() -> None:
    """Seed the deterministic demo: failed run, repair, validation, promotion."""
    from recap.demo import run_demo

    try:
        report = run_demo(_container(), write_report=True)
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(report, indent=2, default=str))


@demo_app.command("run")
def demo_run() -> None:
    """Run the full three-subsystem demo end to end."""
    from recap.demo import run_demo

    try:
        report = run_demo(_container(), write_report=True)
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(report["summary"], indent=2, default=str))
    console.print(f"report: {report.get('report_path', '')}")


# -------------------------------------------------------------------- eval
@eval_app.command("run")
def eval_run(
    split: Annotated[str, typer.Option("--split")] = "dev",
    output: Annotated[Path, typer.Option("--output")] = Path("benchmark/reports"),
) -> None:
    """Run the deterministic benchmark across retrieval baselines."""
    from recap.evals.runner import run_benchmark

    try:
        result = run_benchmark(_container(), split=split, output_dir=output)
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result["summary"], indent=2, default=str))
    console.print(f"report: {result['report_path']}")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("recap.api.app:app", host=host, port=port, reload=reload)


def _default_workspace() -> str:
    import os

    return os.environ.get("RECAP_WORKSPACE", "demo-workspace")
