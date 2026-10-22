"""Deterministic task suite with train/dev/test splits and controlled failures.

Tasks are grouped by template so no scenario is duplicated across splits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_family: str
    task_version: str
    instruction: str
    environment_version: str
    split: str
    failure_mode: str
    expected_memory_signature: str = ""
    relevant_memory_tags: tuple[str, ...] = ()
    hard_negative_kinds: tuple[str, ...] = ()
    seed: int = 7
    difficulty: str = "easy"
    side_effect_risk: str = "read_only"
    # v3 fields. They are defaults so the legacy and v2 suites keep the same
    # public task shape while reports gain the labels v3 needs.
    suite: str = "v2"
    tier: str = "A"
    scenario_id: str = ""
    gold_memory_ids: tuple[str, ...] = ()
    should_select: bool = True
    retrieval_query: str = ""
    schema_variant: str = ""
    silent_drift: bool = False
    tenant: str = "acme"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "task_version": self.task_version,
            "instruction": self.instruction,
            "environment_version": self.environment_version,
            "split": self.split,
            "failure_mode": self.failure_mode,
            "expected_memory_signature": self.expected_memory_signature,
            "relevant_memory_tags": list(self.relevant_memory_tags),
            "hard_negative_kinds": list(self.hard_negative_kinds),
            "seed": self.seed,
            "difficulty": self.difficulty,
            "side_effect_risk": self.side_effect_risk,
            "suite": self.suite,
            "tier": self.tier,
            "scenario_id": self.scenario_id or self.task_id,
            "gold_memory_ids": list(self.gold_memory_ids),
            "should_select": self.should_select,
            "retrieval_query": self.retrieval_query,
            "schema_variant": self.schema_variant,
            "silent_drift": self.silent_drift,
            "tenant": self.tenant,
        }


def default_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="paginate-train-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Collect every record from the paginated API.",
            environment_version="1",
            split="train",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
        ),
        BenchmarkTask(
            task_id="paginate-train-02",
            task_family="api_pagination",
            task_version="1",
            instruction="Page through all tenant records and report the count.",
            environment_version="1",
            split="train",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=11,
        ),
        BenchmarkTask(
            task_id="paginate-dev-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Fetch all records for the tenant despite cursor expiry.",
            environment_version="1",
            split="dev",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=13,
        ),
        BenchmarkTask(
            task_id="paginate-dev-02",
            task_family="api_pagination",
            task_version="2",
            instruction="Collect all records after the API renamed the pagination parameter.",
            environment_version="2",
            split="dev",
            failure_mode="renamed_parameter",
            expected_memory_signature="ValidationError|UNKNOWN_PARAMETER|list_records",
            relevant_memory_tags=("pagination", "page_token"),
            seed=23,
            difficulty="medium",
        ),
        BenchmarkTask(
            task_id="paginate-test-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Collect all records from the paginated API.",
            environment_version="1",
            split="test",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=17,
        ),
        BenchmarkTask(
            task_id="paginate-test-v2-01",
            task_family="api_pagination",
            task_version="2",
            instruction="Collect all records from the upgraded paginated API.",
            environment_version="2",
            split="test",
            failure_mode="renamed_parameter",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "page_token"),
            seed=19,
            difficulty="medium",
        ),
    ]


FAMILY_SPLITS: dict[str, str] = {
    "invalid_cursor": "train",
    "renamed_parameter": "train",
    "rate_limit": "train",
    "timeout": "train",
    "auth_expiry": "train",
    "idempotency_key": "dev",
    "response_field": "dev",
    "required_parameter": "test",
    "enum_change": "test",
    "date_format": "test",
}

FAMILY_SIGNATURES: dict[str, str] = {
    "invalid_cursor": "ValidationError|INVALID_CURSOR|list_records",
    "renamed_parameter": "ValidationError|UNKNOWN_PARAMETER|list_records",
    "rate_limit": "TransientError|RATE_LIMITED|list_records",
    "timeout": "TransientError|TIMEOUT|list_records",
    "auth_expiry": "AuthError|TOKEN_EXPIRED|list_records",
    "idempotency_key": "ValidationError|MISSING_IDEMPOTENCY_KEY|list_records",
    "response_field": "ResponseShapeError|MISSING_RESPONSE_FIELD|list_records",
    "required_parameter": "ValidationError|MISSING_REQUIRED_PARAMETER|list_records",
    "enum_change": "ValidationError|INVALID_ENUM_VALUE|list_records",
    "date_format": "ValidationError|INVALID_DATE_FORMAT|list_records",
}

FAMILY_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "invalid_cursor": (
        "Collect every record from the paginated API.",
        "Fetch all tenant records after the cursor expires.",
        "Page through the full result set without reusing a stale cursor.",
        "Recover from an invalid cursor and report the record count.",
        "Walk every page of records for the tenant.",
        "Gather the complete tenant list from the cursor-based endpoint.",
    ),
    "renamed_parameter": (
        "Collect all records after the pagination parameter was renamed.",
        "Fetch the full tenant list from the upgraded pagination API.",
        "Use the current page token instead of the legacy cursor.",
        "Page through records after the cursor field changed name.",
        "Adapt the pagination call to the new request schema.",
        "Retrieve every record from the renamed pagination endpoint.",
    ),
    "rate_limit": (
        "Collect all records when the upstream API rate limits requests.",
        "Fetch the tenant list despite a temporary 429 response.",
        "Page through every record while respecting the retry delay.",
        "Recover from throttling and finish the collection.",
        "Handle the rate limit, then retrieve the complete result set.",
        "Gather all records after the API asks the client to slow down.",
    ),
    "timeout": (
        "Collect all records after a transient upstream timeout.",
        "Fetch the full tenant list despite an intermittent gateway timeout.",
        "Retry the timed-out read and page through every record.",
        "Finish the collection when the first request times out.",
        "Recover from a transient network timeout and report all records.",
        "Gather the complete tenant list after a slow upstream call.",
    ),
    "auth_expiry": (
        "Collect all records after the access token expires.",
        "Refresh the expired credential and fetch the full tenant list.",
        "Page through every record using a current access token.",
        "Recover from an authentication expiry and finish the task.",
        "Fetch all tenant records after renewing the bearer token.",
        "Handle the expired token, then collect the complete result set.",
    ),
    "idempotency_key": (
        "Collect all records when the API requires an idempotency key.",
        "Supply the request key and fetch the full tenant list.",
        "Page through every record with the required request identifier.",
        "Recover from a missing idempotency key and finish the collection.",
        "Fetch all tenant records after adding the request key.",
        "Handle the new idempotency requirement and report the count.",
    ),
    "response_field": (
        "Collect all records after the response field was renamed.",
        "Read the tenant records from the new payload field.",
        "Page through every record despite the changed response shape.",
        "Adapt to the renamed records field and report the total.",
        "Fetch the full tenant list from the updated JSON response.",
        "Recover from a missing items field and collect all records.",
    ),
    "required_parameter": (
        "Collect all records after the API added a required parameter.",
        "Supply the new account parameter and fetch the tenant list.",
        "Page through every record with the mandatory account value.",
        "Recover from the missing required field and finish the collection.",
        "Fetch all tenant records after adding the required parameter.",
        "Handle the new required field and report the record count.",
    ),
    "enum_change": (
        "Collect all records after the sort enum changed.",
        "Fetch the full tenant list using the current sort value.",
        "Page through every record with the updated enum.",
        "Recover from an invalid sort value and finish the collection.",
        "Fetch all tenant records after adapting the sort parameter.",
        "Handle the renamed enum value and report the total.",
    ),
    "date_format": (
        "Collect all records after the timestamp format changed.",
        "Fetch the full tenant list using the current date format.",
        "Page through every record with the normalized timestamp.",
        "Recover from an invalid date format and finish the collection.",
        "Fetch all tenant records after adapting the since value.",
        "Handle the new timestamp format and report the record count.",
    ),
}


def build_benchmark_v2_tasks() -> list[BenchmarkTask]:
    """Return 120 seeded tasks across ten failure families and two versions.

    Families are held out by split: train has five families, dev two, and test
    three. No family appears in more than one split, so a test score is not a
    paraphrased copy of a training scenario. The six instructions per family
    are deterministic paraphrases; they are not model-generated at runtime.
    """
    tasks: list[BenchmarkTask] = []
    for family, split in FAMILY_SPLITS.items():
        for version in ("1", "2"):
            for index, instruction in enumerate(FAMILY_INSTRUCTIONS[family], start=1):
                tasks.append(
                    BenchmarkTask(
                        task_id=f"{family}-v{version}-{index:02d}",
                        task_family=family,
                        task_version=version,
                        instruction=instruction,
                        environment_version=version,
                        split=split,
                        failure_mode=family,
                        expected_memory_signature=FAMILY_SIGNATURES[family],
                        relevant_memory_tags=("pagination", family),
                        hard_negative_kinds=(
                            "stale_version",
                            "wrong_family",
                            "bad_track_record",
                        ),
                        seed=100 * (list(FAMILY_SPLITS).index(family) + 1) + index,
                        difficulty="hard" if version == "2" else "medium",
                    )
                )
    return tasks


V3_TENANTS = ("acme", "northwind", "globex", "initech", "umbrella", "stark")

# Tier C instructions deliberately describe only the goal. They never mention a
# parameter, an error, a version change, or the recovery action.
V3_TIER_C_INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "silent_cursor": (
        "Collect the complete record set for tenant {tenant}. The result must not miss or repeat rows.",
        "Return every record belonging to tenant {tenant} and report the final count.",
        "Gather all rows for tenant {tenant}; the answer will be checked against the stored data.",
        "Produce the full tenant {tenant} record list and state how many rows it contains.",
        "Fetch the entire set of tenant {tenant} records, with no omissions or duplicates.",
        "Collect all tenant {tenant} rows from the records service and report the exact total.",
    ),
    "silent_field": (
        "Read every record for tenant {tenant} and summarize the count.",
        "Collect the complete tenant {tenant} record set from the service response.",
        "Return all rows for tenant {tenant} and report how many were found.",
        "Gather the tenant {tenant} records and state the exact total.",
        "Produce the full record list for tenant {tenant} from the service payload.",
        "Fetch every tenant {tenant} row and report the final count.",
    ),
    "silent_enum": (
        "Collect every tenant {tenant} record in the order the service defines, then report the total.",
        "Return all tenant {tenant} rows in canonical order and state the count.",
        "Gather tenant {tenant} records in the service's canonical order and report the total.",
        "Produce every tenant {tenant} row in canonical order; the exact sequence is checked.",
        "Fetch the full tenant {tenant} record list in service order and report the count.",
        "Collect all tenant {tenant} rows in the canonical sequence and state the total.",
    ),
    "silent_account": (
        "Collect all tenant {tenant} records and report the final count.",
        "Return the complete tenant {tenant} record set and state how many rows were found.",
        "Gather every tenant {tenant} row from the records service and report the total.",
        "Produce the full tenant {tenant} record list and give the exact count.",
        "Fetch all tenant {tenant} records and report how many were returned.",
        "Collect the entire tenant {tenant} record set and state the final total.",
    ),
    "novel_failure": (
        "Collect all tenant {tenant} records and report the total.",
        "Return every tenant {tenant} row and state the count.",
        "Gather the complete tenant {tenant} record list and report how many were found.",
        "Fetch all tenant {tenant} records and give the exact total.",
        "Produce the full tenant {tenant} row set and report its size.",
        "Collect every tenant {tenant} record and state the final count.",
    ),
}


def build_benchmark_v3_tasks() -> list[BenchmarkTask]:
    """Tiered, leak-free v3 suite.

    Tier A is a control where a plain loop succeeds. Tier B has an opaque error
    that can be resolved by probing. Tier C contains silent drift or a hidden
    convention: the request succeeds, but the answer is wrong without memory.
    ``should_select=False`` marks tasks where abstaining is the safe action.
    """
    tasks: list[BenchmarkTask] = []

    def add(
        *,
        family: str,
        tier: str,
        split: str,
        scenario: str,
        instruction: str,
        version: str = "2",
        should_select: bool = True,
        gold: bool = True,
        variant: str = "",
        silent: bool = False,
        tenant: str = "acme",
        index: int = 1,
        gold_scenario: str = "",
    ) -> None:
        memory_id = f"v3-gold-{gold_scenario or scenario}-v{version}" if gold else ""
        tasks.append(
            BenchmarkTask(
                task_id=f"v3-{scenario}-v{version}-{index:02d}",
                task_family=scenario,
                task_version=version,
                instruction=instruction,
                environment_version=version,
                split=split,
                failure_mode=family,
                relevant_memory_tags=("pagination", family),
                hard_negative_kinds=("stale_version", "wrong_family", "bad_track_record"),
                seed=5000 + len(tasks),
                difficulty="hard",
                suite="v3",
                tier=tier,
                scenario_id=scenario,
                gold_memory_ids=(memory_id,) if memory_id else (),
                should_select=should_select,
                retrieval_query=(
                    f"{instruction} Previous attempt collected an unexpected record set."
                ),
                schema_variant=variant,
                silent_drift=silent,
                tenant=tenant,
            )
        )

    # Tier A: self-explanatory/control. No memory is needed, and the deterministic
    # agent succeeds without one.
    for index, tenant in enumerate(V3_TENANTS, start=1):
        add(
            family="self_explanatory",
            tier="A",
            split="train",
            scenario="tier-a-control",
            instruction=(
                f"Return the full set of records for tenant {tenant} and report how many there are."
            ),
            should_select=False,
            gold=False,
            tenant=tenant,
            index=index,
        )

    # Tier B: opaque error, discoverable by probing.
    for index, tenant in enumerate(V3_TENANTS, start=1):
        add(
            family="opaque_required",
            tier="B",
            split="dev",
            scenario="tier-b-opaque",
            instruction=(
                f"Fetch every row belonging to tenant {tenant} from the records "
                "endpoint, then report the total."
            ),
            tenant=tenant,
            index=index,
        )

    # Tier C base scenarios.
    for family in ("silent_cursor", "silent_field", "silent_enum", "silent_account"):
        split = "dev" if family == "silent_cursor" else "test"
        for index, (tenant, template) in enumerate(
            zip(V3_TENANTS, V3_TIER_C_INSTRUCTIONS[family], strict=True), start=1
        ):
            add(
                family=family,
                tier="C",
                split=split,
                scenario=f"tier-c-{family}",
                instruction=template.format(tenant=tenant),
                silent=True,
                tenant=tenant,
                index=index,
            )

    # Partial drift: only the tenant description (an irrelevant schema path)
    # changes. A whole-fingerprint gate abstains; field-scoped compatibility does
    # not. The same task family with a relevant change is an abstention case.
    for index, tenant in enumerate(V3_TENANTS[:3], start=1):
        add(
            family="silent_cursor",
            tier="C",
            split="dev",
            scenario="tier-c-silent-cursor-irrelevant",
            instruction=(f"Collect every record for tenant {tenant} and report the exact count."),
            variant="irrelevant",
            silent=True,
            tenant=tenant,
            index=index,
            gold_scenario="tier-c-silent_cursor",
        )
    for index, tenant in enumerate(V3_TENANTS[3:], start=1):
        add(
            family="silent_cursor",
            tier="C",
            split="dev",
            scenario="tier-c-silent-cursor-relevant",
            instruction=(f"Collect all tenant {tenant} records and report the exact count."),
            variant="relevant",
            should_select=False,
            gold=False,
            silent=True,
            tenant=tenant,
            index=index,
        )

    # No relevant memory: the correct action is to abstain. The environment still
    # has a silent failure, so "always select something" is visibly unsafe.
    for index, (tenant, template) in enumerate(
        zip(V3_TENANTS, V3_TIER_C_INSTRUCTIONS["novel_failure"], strict=True), start=1
    ):
        add(
            family="novel_failure",
            tier="C",
            split="test",
            scenario="tier-c-novel-failure",
            instruction=template.format(tenant=tenant),
            should_select=False,
            gold=False,
            silent=True,
            tenant=tenant,
            index=index,
        )

    return tasks


def split_hash(tasks: list[BenchmarkTask]) -> str:
    payload = json.dumps(
        [task.to_dict() for task in sorted(tasks, key=lambda item: item.task_id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _normalize_instruction(text: str) -> str:
    """Lowercase and collapse punctuation/whitespace before comparing tasks."""
    return " ".join("".join(char if char.isalnum() else " " for char in text.lower()).split())


def _token_shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    tokens = _normalize_instruction(text).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def check_leakage(
    tasks: list[BenchmarkTask],
    *,
    near_duplicate_threshold: float = 0.8,
    require_family_holdout: bool = False,
) -> list[str]:
    """Flag task ids, instructions, and near-duplicate wording shared across splits.

    Exact normalized instructions are an obvious leak. Paraphrases are more
    subtle: two tasks from different splits can share almost all of their
    3-token shingles. Comparing shingles catches this without requiring an
    embedding provider or making the check nondeterministic.

    ``require_family_holdout`` additionally enforces that a failure family
    appears in exactly one split. The v2 suite is family-held-out by design;
    the small legacy suite intentionally reuses ``api_pagination`` across
    splits, so the check is opt-in there.
    """
    issues: list[str] = []
    task_ids: dict[str, set[str]] = {}
    normalized: dict[str, set[str]] = {}
    shingles: dict[str, set[tuple[str, ...]]] = {}
    families: dict[str, set[str]] = {}
    for task in tasks:
        task_ids.setdefault(task.task_id, set()).add(task.split)
        key = _normalize_instruction(task.instruction)
        normalized.setdefault(key, set()).add(task.split)
        shingles[task.task_id] = _token_shingles(task.instruction)
        families.setdefault(task.failure_mode, set()).add(task.split)

    for task_id, splits in task_ids.items():
        if len(splits) > 1:
            issues.append(f"duplicate task id across splits: {task_id} {sorted(splits)}")
    for key, splits in normalized.items():
        if len(splits) > 1:
            issues.append(f"identical instruction across splits: {sorted(splits)} ({key[:80]!r})")
    if require_family_holdout:
        for family, splits in families.items():
            if len(splits) > 1:
                issues.append(f"failure family appears across splits: {family} {sorted(splits)}")

    # Pairwise shingle comparison across splits. Task counts are small and the
    # check runs in tests/CI, not in the retrieval hot path.
    for index, left in enumerate(tasks):
        for right in tasks[index + 1 :]:
            if left.split == right.split:
                continue
            similarity = _jaccard(shingles[left.task_id], shingles[right.task_id])
            if similarity >= near_duplicate_threshold:
                issues.append(
                    "near-duplicate instruction across splits: "
                    f"{left.task_id} ({left.split}) vs {right.task_id} ({right.split}) "
                    f"jaccard={similarity:.3f}"
                )
    return issues
