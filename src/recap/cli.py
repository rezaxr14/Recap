"""RECAP command line interface.

Every command performs real work against the configured database and search
index. Commands never print success for an operation that did not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from recap.config import get_settings, load_settings
from recap.container import Container
from recap.domain.enums import MemoryStatus, RetrievalMode
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.errors import RecapError
from recap.infrastructure.outbox import OutboxWorker

app = typer.Typer(help="RECAP memory and recovery engine", no_args_is_help=True)
db_app = typer.Typer(help="Database operations")
search_app = typer.Typer(help="Elasticsearch operations")
memory_app = typer.Typer(help="Memory inspection")
repair_app = typer.Typer(help="Repair compilation and validation")
agent_app = typer.Typer(help="Tool-using agent runs")
chunk_app = typer.Typer(help="Deterministic chunk materialization")
demo_app = typer.Typer(help="Deterministic local demo")
eval_app = typer.Typer(help="Evaluation harness")
app.add_typer(db_app, name="db")
app.add_typer(search_app, name="search")
app.add_typer(memory_app, name="memory")
app.add_typer(repair_app, name="repair")
app.add_typer(agent_app, name="agent")
app.add_typer(chunk_app, name="chunk")
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


@app.command()
def providers(config: str | None = None) -> None:
    """Report configured model roles, models, and declared capabilities.

    No network request is made and no credential is printed. Use ``recap doctor``
    to probe reachability.
    """
    try:
        container = _container(config)
        report = container.provider_status()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(report, indent=2, default=str))


@app.command("embedding-cache")
def embedding_cache(
    config: str | None = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Delete all cached vectors (next rebuild re-embeds)"),
    ] = False,
) -> None:
    """Show or clear the content-addressed Postgres embedding cache."""
    from recap.infrastructure.embeddings.cache import SqlEmbeddingCache

    try:
        container = _container(config)
        cache = container.embedding_cache or SqlEmbeddingCache(container.database)
        result = cache.stats()
        result["enabled"] = container.settings.embeddings.cache_enabled
        if clear:
            result["deleted"] = cache.clear()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, indent=2, default=str))


@app.command()
def doctor(
    config: str | None = None,
    live: Annotated[
        bool, typer.Option("--live", help="Make a small real chat/embedding call")
    ] = False,
) -> None:
    """Check database, Elasticsearch, and configured provider endpoints."""
    try:
        container = _container(config)
        result = container.health()
        result["providers_health"] = container.provider_health(live=live)
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, indent=2, default=str))

    failures: list[str] = []
    if result.get("database") != "ok":
        failures.append("database")
    elasticsearch = result.get("elasticsearch")
    if isinstance(elasticsearch, dict) and elasticsearch.get("status") in {
        "unavailable",
        "error",
    }:
        failures.append("elasticsearch")
    provider_health = result.get("providers_health")
    if isinstance(provider_health, dict):
        for role, status in provider_health.items():
            if not isinstance(status, dict):
                continue
            if status.get("status") in {"error"}:
                failures.append(role)
            for key in ("live_chat", "live_embedding"):
                check = status.get(key)
                if isinstance(check, dict) and check.get("status") == "error":
                    failures.append(f"{role}:{key}")
    if failures:
        console.print(f"[red]doctor found failures: {', '.join(failures)}[/red]")
        raise typer.Exit(code=1)
    providers_info = result.get("providers")
    semantic = isinstance(providers_info, dict) and bool(providers_info.get("semantic_retrieval"))
    if not semantic:
        console.print(
            "[yellow]degraded mode: semantic retrieval is disabled "
            "(lexical/test embedding provider)[/yellow]"
        )
    console.print("[green]all checks passed[/green]")


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
    """Create the versioned memory and chunk indices plus aliases idempotently."""
    try:
        container = _container(config)
        result = container.init_search()
        result["chunks"] = container.init_search_chunks()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, default=str))


@search_app.command("health")
def search_health(config: str | None = None) -> None:
    """Show index/alias health and document count for memories and chunks."""
    try:
        container = _container(config)
        result = {
            "memories": container.index_manager.health(),
            "chunks": container.index_manager.chunks_health(),
        }
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(result, default=str))


@search_app.command("reconcile")
def search_reconcile(
    config: str | None = None,
    chunks: Annotated[
        bool, typer.Option("--chunks", help="Also rebuild chunk documents from Postgres")
    ] = False,
    prune_orphans: Annotated[
        bool,
        typer.Option(
            "--prune-orphans",
            help="Delete chunk documents whose chunk row or parent memory is gone",
        ),
    ] = False,
    workspace: Annotated[
        str,
        typer.Option("--workspace", "-w", help="Workspace to reconcile (default: demo-workspace)"),
    ] = "",
) -> None:
    """Reindex active memories (and optionally chunks) for one workspace."""
    from recap.maintenance import prune_orphan_chunks

    workspace_id = workspace or _default_workspace()
    indexed = 0
    failed = 0
    chunk_count = 0
    chunks_rebuilt = 0
    skipped_secret = 0
    failed_memories: list[dict[str, str]] = []
    orphan_result: dict[str, Any] | None = None
    try:
        container = _container(config)
        with container.services() as services:
            memories = services.memories.list_memories(
                workspace_id=workspace_id, status=MemoryStatus.ACTIVE, limit=10_000
            )
            for memory in memories:
                try:
                    if services.index_sync(memory):
                        indexed += 1
                    else:
                        failed += 1
                        failed_memories.append(
                            {
                                "memory_id": memory.id,
                                "stage": "memory_index",
                                "error": services.index_sync.last_error,
                            }
                        )
                    if chunks:
                        report = services.chunk_service.rebuild(memory)
                        chunk_count += report.count
                        if report.indexed:
                            chunks_rebuilt += 1
                        else:
                            failed += 1
                            failed_memories.append(
                                {
                                    "memory_id": memory.id,
                                    "stage": "chunk_index",
                                    "error": report.error or "chunk indexing failed",
                                }
                            )
                except ValueError as exc:
                    # The chunker fails closed on secrets. One bad memory must
                    # not abort the whole workspace reconcile.
                    failed += 1
                    message = str(exc)
                    if "secret" in message.lower():
                        skipped_secret += 1
                    failed_memories.append(
                        {"memory_id": memory.id, "stage": "chunk_build", "error": message}
                    )
                except Exception as exc:  # per-memory boundary
                    failed += 1
                    failed_memories.append(
                        {
                            "memory_id": memory.id,
                            "stage": "unknown",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            pending_outbox = services.outbox.pending_count()
        if prune_orphans:
            orphan_result = prune_orphan_chunks(container, workspace_id)
    except Exception as exc:
        _fail(exc)
        return
    summary: dict[str, Any] = {
        "workspace_id": workspace_id,
        "indexed": indexed,
        "failed": failed,
        "chunks_rebuilt": chunks_rebuilt,
        "chunks_total": chunk_count,
        "skipped_secret": skipped_secret,
        "pending_outbox": pending_outbox,
        "failed_memories": failed_memories,
    }
    if orphan_result is not None:
        summary["orphans"] = orphan_result
    console.print_json(json.dumps(summary, indent=2, default=str))


@search_app.command("drain")
def search_drain(
    config: str | None = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 20,
    max_batches: Annotated[int, typer.Option("--max-batches", min=1)] = 100,
) -> None:
    """Drain due index-outbox rows into Elasticsearch (retry with backoff)."""
    try:
        container = _container(config)
        counts = OutboxWorker(container, batch_size=batch_size, max_batches=max_batches).drain()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps(counts, indent=2))


@search_app.command("outbox")
def search_outbox(config: str | None = None) -> None:
    """Show index-outbox state counts."""
    try:
        container = _container(config)
        with container.services() as services:
            counts = services.outbox.counts_by_state()
            due = services.outbox.pending_count()
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(json.dumps({"states": counts, "due_pending": due}, indent=2))


# ------------------------------------------------------------------- chunk
@chunk_app.command("build")
def chunk_build(
    workspace: Annotated[str, typer.Option("--workspace", "-w")] = "demo-workspace",
    memory_id: Annotated[
        str, typer.Option("--memory-id", help="Build chunks for a single memory only")
    ] = "",
    status: Annotated[
        str, typer.Option("--status", help="Status filter for bulk build")
    ] = "active",
    limit: Annotated[int, typer.Option("--limit", min=1)] = 1000,
    config: str | None = None,
) -> None:
    """Build deterministic chunks from Postgres and index them."""
    try:
        container = _container(config)
        with container.services() as services:
            if memory_id:
                memory = services.memories.get_memory(memory_id, workspace_id=workspace)
                if memory is None:
                    console.print(f"[red]memory not found[/red]: {memory_id}")
                    raise typer.Exit(code=1)
                memories = [memory]
            else:
                memories = services.memories.list_memories(
                    workspace_id=workspace, status=MemoryStatus(status), limit=limit
                )
            total_chunks = 0
            failed = 0
            warnings = 0
            for memory in memories:
                report = services.chunk_service.rebuild(memory)
                total_chunks += report.count
                warnings += len(report.warnings)
                if not report.indexed:
                    failed += 1
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(
        json.dumps(
            {
                "memories": len(memories),
                "chunks": total_chunks,
                "index_failures": failed,
                "warnings": warnings,
                "pending_outbox": services.outbox.pending_count(),
            },
            indent=2,
        )
    )


@chunk_app.command("show")
def chunk_show(
    memory_id: Annotated[str, typer.Argument(help="Memory id")],
    version: Annotated[int, typer.Option("--version", min=1)] = 1,
    config: str | None = None,
) -> None:
    """List stored chunks for a memory version."""
    try:
        container = _container(config)
        with container.services() as services:
            chunks = services.chunk_store.list_for_memory(memory_id, version)
    except Exception as exc:
        _fail(exc)
        return
    table = Table(title=f"Chunks for {memory_id}:v{version}")
    for column in ("type", "seq", "tokens", "events", "chunk_id"):
        table.add_column(column)
    for chunk in chunks:
        table.add_row(
            str(chunk.chunk_type),
            f"{chunk.seq_start}-{chunk.seq_end}",
            str(chunk.token_estimate),
            str(len(chunk.event_ids)),
            chunk.chunk_id,
        )
    console.print(table)
    console.print(f"chunks={len(chunks)}")


@chunk_app.command("count")
def chunk_count(config: str | None = None) -> None:
    """Show the number of chunk documents in the chunk index."""
    try:
        container = _container(config)
        repository = container.chunk_search_repository
        if repository is None:
            console.print("[red]chunk index is not configured[/red]")
            raise typer.Exit(code=1)
        count = repository.count()
    except Exception as exc:
        _fail(exc)
        return
    console.print(f"chunk_documents={count}")


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


# -------------------------------------------------------------------- agent
@agent_app.command("run")
def agent_run(
    environment_version: Annotated[str, typer.Option("--environment-version")] = "1",
    workspace: Annotated[str, typer.Option("--workspace")] = "demo-workspace",
    max_tool_calls: Annotated[int, typer.Option("--max-tool-calls", min=1, max=32)] = 6,
    memory_id: Annotated[
        str | None, typer.Option("--memory-id", help="Inject one memory as advisory data")
    ] = None,
    expire_first_cursor: Annotated[
        bool,
        typer.Option(
            "--expire-first-cursor",
            help="Make the first v1 cursor invalid to exercise failure recording",
        ),
    ] = False,
    task_id: Annotated[str | None, typer.Option("--task-id")] = None,
) -> None:
    """Run the configured LLM agent against the mock API and record the run.

    The model chooses tool calls. RECAP validates and executes only the declared
    local tool. This command requires a configured ``llm`` role; it never falls
    back to the deterministic agent or a fake model.
    """
    from recap.environments.llm_agent import MemoryAdvisory
    from recap.environments.mock_api import PaginatedApiEnvironment
    from recap.environments.run_recording import record_llm_run
    from recap.errors import ConfigurationError, NotFoundError, ValidationError

    try:
        container = _container()
        if container.llm is None:
            raise ConfigurationError(
                "the agent llm role is not configured; set LLM_PROVIDER and LLM_BASE_URL "
                "and LLM_MODEL (run `recap providers` to inspect roles)"
            )
        if environment_version not in {"1", "2"}:
            raise ValidationError("--environment-version must be 1 or 2")
        if expire_first_cursor and environment_version != "1":
            raise ValidationError("--expire-first-cursor only applies to --environment-version 1")
        environment = PaginatedApiEnvironment(version=environment_version)
        if expire_first_cursor:
            environment.expire_cursor("cur_1_0001")
        advisory = None
        if memory_id is not None:
            with container.services() as services:
                memory = services.memories.get_memory(memory_id, workspace)
                if memory is None:
                    raise NotFoundError(f"memory {memory_id} not found in workspace {workspace}")
                advisory = MemoryAdvisory.from_memory(memory)
        run_id, result = record_llm_run(
            container,
            environment,
            provider=container.llm,
            workspace_id=workspace,
            task_id=task_id,
            memory=advisory,
            max_tool_calls=max_tool_calls,
        )
    except Exception as exc:
        _fail(exc)
        return
    console.print_json(
        json.dumps(
            {
                "run_id": run_id,
                "success": result.success,
                "agent": "llm-tool-agent-v1",
                "model": container.llm.model,
                "environment_version": environment_version,
                "tool_calls": result.tool_calls,
                "retries": result.retries,
                "model_turns": result.model_turns,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "duration_ms": result.duration_ms,
                "used_memory_id": result.used_memory_id,
                "final_error": result.final_error,
            },
            indent=2,
            default=str,
        )
    )


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
