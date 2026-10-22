from types import SimpleNamespace

from fastapi.testclient import TestClient

from recap.api import app as app_module


def _settings(api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            api_key=api_key,
            api_prefix="/api/v1",
            cors_origin_list=[],
            log_level="INFO",
        )
    )


def test_health_live_is_public_but_api_requires_key(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings("test-secret"))
    client = TestClient(app_module.create_app(container=object()))

    assert client.get("/health/live").status_code == 200
    assert client.get("/api/v1/diagnostics").status_code == 401
    assert client.get("/api/v1/diagnostics", headers={"X-API-Key": "wrong"}).status_code == 401


def test_empty_key_disables_auth_for_local_use(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(""))
    client = TestClient(app_module.create_app(container=object()), raise_server_exceptions=False)

    # No key configured: the middleware must not turn a missing key into 401.
    assert client.get("/api/v1/diagnostics").status_code != 401
