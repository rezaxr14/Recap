"""Engine/session management.

Supports PostgreSQL for durable use and SQLite for fast, dependency-free unit
tests of persistence logic. Production paths must use PostgreSQL.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from recap.config import StorageSettings
from recap.infrastructure.db.models import Base


def build_engine(url: str, *, echo: bool = False) -> Engine:
    kwargs: dict[str, object] = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_fk(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def build_engine_from_settings(settings: StorageSettings) -> Engine:
    return build_engine(settings.postgres_url)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def drop_all(engine: Engine) -> None:
    Base.metadata.drop_all(engine)


class Database:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        create_all(self.engine)
