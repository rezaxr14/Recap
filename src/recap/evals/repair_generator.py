"""Live acceptance for the LLM repair candidate generator.

This is not the deterministic benchmark. It records real failed runs, asks the
configured repair model for one typed candidate per run, then validates every
candidate through the normal static and mock-sandbox validators. Format errors,
bounded retries, and no-candidate outcomes are counted rather than hidden.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recap.config import Settings
from recap.container import Container
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.run_recording import record_reference_run
from recap.errors import ConfigurationError
from recap.repair_compiler.candidate_generation import LLMCandidateGenerator


@dataclass
class RepairRunResult:
    workspace_id: str
    environment_version: str
    run_id: str
    generation_error: str = ""
    candidates: int = 0
    static_passed: int = 0
    sandbox_passed: int = 0
    validation_checks: list[dict[str, Any]] = field(default_factory=list)
    generation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "environment_version": self.environment_version,
            "run_id": self.run_id,
            "generation_error": self.generation_error,
            "candidates": self.candidates,
            "static_passed": self.static_passed,
            "sandbox_passed": self.sandbox_passed,
            "validation_checks": self.validation_checks,
            "generation": self.generation,
        }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def run_repair_generator_acceptance(
    container: Container,
    *,
    runs: int = 4,
    output: Path | None = None,
) -> dict[str, Any]:
    settings: Settings = container.settings
    if settings.repair_compiler.generator == "deterministic":
        raise ConfigurationError(
            "repair generator acceptance requires REPAIR_GENERATOR=llm",
            details={"generator": settings.repair_compiler.generator},
        )
    if container.repair_llm is None:
        raise ConfigurationError("repair LLM provider is disabled; set REPAIR_LLM_* credentials")
    if runs < 1 or runs > 50:
        raise ConfigurationError("runs must be within [1, 50]")

    started = time.perf_counter()
    results: list[RepairRunResult] = []
    for index in range(runs):
        version = "1" if index % 2 == 0 else "2"
        workspace_id = f"repair-acceptance-{uuid.uuid4().hex[:10]}"
        environment = PaginatedApiEnvironment(version=version)
        fingerprint = environment.fingerprint()
        run_id, failed = record_reference_run(
            container,
            environment,
            workspace_id=workspace_id,
            task_id=f"repair-acceptance-{index + 1}",
        )
        if failed.success:
            raise ConfigurationError("acceptance setup did not produce a failed run")
        result = RepairRunResult(
            workspace_id=workspace_id,
            environment_version=version,
            run_id=run_id,
        )
        with container.services() as services:
            services.environments.upsert(workspace_id, fingerprint)
            generator = LLMCandidateGenerator(
                container.repair_llm,
                max_attempts=settings.repair_compiler.max_candidate_generation_attempts,
            )
            services.repair_compiler.generator = generator
            try:
                candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
            except Exception as exc:  # generation/format failure is measured, not hidden
                candidates = []
                result.generation_error = f"{type(exc).__name__}: {exc}"
                result.generation = generator.stats.as_dict()
                results.append(result)
                continue
            result.candidates = len(candidates)
            for candidate in candidates:
                checks: dict[str, Any] = {"candidate_id": candidate.id}
                for validator_name in ("static", "mock_sandbox"):
                    validation = services.repair_compiler.validate(
                        candidate.id,
                        environment=fingerprint,
                        validator_name=validator_name,
                    )
                    checks[validator_name] = {
                        "status": validation.status.value,
                        "checks": validation.checks,
                    }
                    if validation.status.value == "passed":
                        if validator_name == "static":
                            result.static_passed += 1
                        else:
                            result.sandbox_passed += 1
                result.validation_checks.append(checks)
            result.generation = generator.stats.as_dict()
        results.append(result)

    calls = len(results)
    generation_successes = sum(item.generation.get("successes", 0) for item in results)
    attempts = sum(item.generation.get("attempts", 0) for item in results)
    format_failures = sum(item.generation.get("format_failures", 0) for item in results)
    candidates_total = sum(item.candidates for item in results)
    static_passed = sum(item.static_passed for item in results)
    sandbox_passed = sum(item.sandbox_passed for item in results)
    error_breakdown = Counter(
        item.generation_error.split(":", 1)[0] for item in results if item.generation_error
    )
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "kind": "live_repair_generator_acceptance",
        "model": {
            "provider": settings.repair_llm.provider,
            "model_id": settings.repair_llm.model,
            "base_url": settings.repair_llm.base_url,
            "generator": settings.repair_compiler.generator,
            "max_attempts": settings.repair_compiler.max_candidate_generation_attempts,
        },
        "runs": calls,
        "generation": {
            "schema_successes": generation_successes,
            "attempts": attempts,
            "retries": max(0, attempts - calls),
            "format_failures": format_failures,
            "schema_success_rate": _rate(generation_successes, calls),
            "format_failure_rate": _rate(format_failures, attempts),
            "provider_errors": sum(error_breakdown.values()),
            "provider_error_breakdown": dict(sorted(error_breakdown.items())),
        },
        "candidates": {
            "count": candidates_total,
            "candidate_rate": _rate(candidates_total, calls),
            "static_passed": static_passed,
            "static_pass_rate": _rate(static_passed, candidates_total),
            "sandbox_passed": sandbox_passed,
            "sandbox_pass_rate": _rate(sandbox_passed, candidates_total),
            "end_to_end_pass_rate": _rate(sandbox_passed, calls),
        },
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "results": [item.as_dict() for item in results],
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report
