"""Unit tests for Elasticsearch mappings and query DSL.

These never touch a live cluster; they assert the request shape that the
integration suite then verifies against Elasticsearch 8.19.
"""

from __future__ import annotations

from typing import Any

from recap.infrastructure.elasticsearch.hybrid_queries import (
    build_es_rrf_query,
    build_lexical_query,
    build_vector_query,
)
from recap.infrastructure.elasticsearch.mappings import (
    IDENTIFIER_ANALYZER,
    IDENTIFIER_SUBFIELD,
    memory_index_body,
)


def _fields(query: dict[str, Any]) -> list[str]:
    should = query["query"]["bool"]["should"]
    multi = next(item["multi_match"] for item in should if "multi_match" in item)
    return list(multi["fields"])


def test_identifier_mapping_adds_analyzer_and_subfields() -> None:
    body = memory_index_body(dimension=256, identifier_analysis=True)
    properties = body["mappings"]["properties"]

    assert body["settings"]["analysis"]["analyzer"][IDENTIFIER_ANALYZER]["type"] == "custom"
    assert (
        body["settings"]["analysis"]["filter"]["recap_identifier_delimiters"]["type"]
        == "word_delimiter_graph"
    )
    for field in ("title", "summary", "failure_signature", "failure_code"):
        subfield = properties[field]["fields"][IDENTIFIER_SUBFIELD]
        assert subfield["type"] == "text"
        assert subfield["analyzer"] == IDENTIFIER_ANALYZER


def test_standard_mapping_has_no_identifier_subfields() -> None:
    body = memory_index_body(dimension=256)
    properties = body["mappings"]["properties"]

    assert "analysis" not in body["settings"]
    assert IDENTIFIER_SUBFIELD not in properties["failure_code"].get("fields", {})
    assert IDENTIFIER_SUBFIELD not in properties["title"].get("fields", {})


def test_int8_index_options_are_explicit() -> None:
    body = memory_index_body(dimension=768, dense_vector_type="int8_hnsw")
    embedding = body["mappings"]["properties"]["content_embedding"]

    assert embedding["dims"] == 768
    assert embedding["index_options"]["type"] == "int8_hnsw"
    assert body["mappings"]["_source"] == {"excludes": ["content_embedding"]}


def test_pure_lexical_query_drops_metadata_boosts() -> None:
    query = build_lexical_query(
        query="collect the full record set",
        workspace_id="ws",
        allowed_statuses=["active"],
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        task_family="api_pagination",
        tool_names=["list_records"],
        pure=True,
    )
    should = query["query"]["bool"]["should"]

    assert len(should) == 1
    assert "failure_signature" not in _fields(query)
    assert "failure_code" not in _fields(query)
    assert not any("match_phrase" in item for item in should)
    assert not any("term" in item for item in should)
    assert not any("terms" in item for item in should)


def test_identifier_analysis_adds_query_subfields() -> None:
    query = build_lexical_query(
        query="invalid cursor",
        workspace_id="ws",
        allowed_statuses=["active"],
        identifier_analysis=True,
    )
    fields = _fields(query)

    assert "failure_code.identifier^1.0" in fields
    assert "failure_signature.identifier^1.0" in fields
    assert "title.identifier^1.0" in fields


def test_vector_query_passes_prefilter_and_candidate_count() -> None:
    query = build_vector_query(
        embedding=[0.1, 0.2],
        workspace_id="ws",
        allowed_statuses=["active", "suspect"],
        size=10,
        num_candidates=500,
    )
    knn = query["knn"]

    assert knn["k"] == 10
    assert knn["num_candidates"] == 500
    filters = knn["filter"]["bool"]["filter"]
    assert {"term": {"workspace_id": "ws"}} in filters
    assert {"terms": {"status": ["active", "suspect"]}} in filters


def test_native_rrf_uses_knn_retriever_not_a_standard_query() -> None:
    query = build_es_rrf_query(
        query="invalid cursor",
        embedding=[0.1, 0.2],
        workspace_id="ws",
        allowed_statuses=["active"],
        failure_code="INVALID_CURSOR",
        size=10,
        rank_constant=60,
        num_candidates=200,
        pure=True,
    )
    retrievers = query["retriever"]["rrf"]["retrievers"]

    assert query["retriever"]["rrf"]["rank_constant"] == 60
    assert "standard" in retrievers[0]
    assert "knn" in retrievers[1]
    assert retrievers[1]["knn"]["num_candidates"] == 200
    # The pure lexical clause must not receive the error-code metadata boost.
    should = retrievers[0]["standard"]["query"]["bool"]["should"]
    assert not any("term" in item for item in should)
