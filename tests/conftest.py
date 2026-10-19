"""Shared fixtures.

Unit tests must never touch the network. Integration/e2e tests are opt-in via
``--run-integration`` (or ``RECAP_RUN_INTEGRATION=1``) and use an isolated
Postgres database and Elasticsearch index prefix so they can never corrupt
local development data.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from recap.config import Settings, load_settings

INTEGRATION_ENABLED = os.environ.get("RECAP_RUN_INTEGRATION") == "1"

TEST_DATABASE_URL = os.environ.get(
    "RECAP_TEST_DATABASE_URL",
    "postgresql+psycopg://recap:recap@localhost:5433/recap_test",
)
TEST_ES_URL = os.environ.get("RECAP_TEST_ELASTICSEARCH_URL", "http://localhost:9200")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that require live Postgres and Elasticsearch",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    enabled = config.getoption("--run-integration") or INTEGRATION_ENABLED
    if enabled:
        return
    skip = pytest.mark.skip(
        reason="needs live services; pass --run-integration or set RECAP_RUN_INTEGRATION=1"
    )
    for item in items:
        if {"integration", "e2e"} & set(item.keywords):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def live_settings(request: pytest.FixtureRequest) -> Settings:
    if not (
        INTEGRATION_ENABLED
        or request.config.getoption("--run-integration")
        or os.environ.get("PYTEST_RUN_INTEGRATION") == "1"
    ):
        # pytest_collection_modifyitems already skipped these tests; this guard
        # exists so an accidental direct call fails loudly instead of silently
        # hitting local development data.
        pytest.skip("integration tests are opt-in")
    base = load_settings()
    settings = base.model_copy(deep=True)
    settings.storage.postgres_url = TEST_DATABASE_URL
    settings.elasticsearch.url = TEST_ES_URL
    settings.elasticsearch.index_prefix = "recap-test"
    settings.elasticsearch.index_generation = 1
    settings.elasticsearch.memories_read_alias = "recap-test-memories-read"
    settings.elasticsearch.memories_write_alias = "recap-test-memories-write"
    settings.elasticsearch.chunks_read_alias = "recap-test-chunks-read"
    settings.elasticsearch.chunks_write_alias = "recap-test-chunks-write"
    settings.elasticsearch.chunks_index_generation = 1
    return settings


@pytest.fixture()
def workspace_id() -> str:
    return f"test-ws-{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def container(live_settings: Settings) -> Iterator[Container]:  # noqa: F821
    from recap.container import Container

    built = Container.build(live_settings)
    built.migrate()
    built.init_search()
    built.init_search_chunks()
    yield built
    built.es_client.close()
