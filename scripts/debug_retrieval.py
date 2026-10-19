from recap.config import get_settings
from recap.container import Container
from recap.domain.enums import RetrievalMode
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment

container = Container.build(get_settings())
env = PaginatedApiEnvironment(version="1")
fp = env.fingerprint()
with container.services() as s:
    mems = s.memories.list_memories("demo-workspace", limit=10)
    for m in mems:
        print("MEM", m.id, m.status, m.environment_fingerprint, m.content.tool_schema_fingerprints, m.content.failure_code)
    print("FP", fp.fingerprint, fp.tool_schema_hashes)
    req = RetrievalRequest(
        context=RetrievalContext(
            workspace_id="demo-workspace",
            query="API rejected the pagination cursor with INVALID_CURSOR",
            task_family="api_pagination",
            environment_id=fp.environment_id,
            environment_fingerprint=fp.fingerprint,
            environment_version=fp.environment_version,
            environment_type=fp.environment_type,
            tool_names=["list_records"],
            tool_versions=fp.tool_versions,
            tool_schema_fingerprints=fp.tool_schema_hashes,
            error_type="ValidationError",
            error_code="INVALID_CURSOR",
            error_message="cursor 'start' is invalid or expired",
            state={"records_collected": 0, "tool_schema_fingerprints": fp.tool_schema_hashes},
        ),
        mode=RetrievalMode.OUTCOME_AWARE,
        final_k=3,
    )
    d = s.recall.retrieve(req)
    print("STATUS", d.status, d.reason)
    print("WARN", d.warnings)
    for c in d.candidates:
        print("CAND", c.memory_id, c.fused_rank, c.final_score, c.rejection_reasons, c.applicability_reason, c.compatibility_status)
        print("  FEAT", c.features.model_dump())
