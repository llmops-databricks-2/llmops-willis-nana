"""Conversation memory stored in Lakebase Postgres."""

from __future__ import annotations

import json
from typing import Any

import psycopg
from loguru import logger
from psycopg_pool import ConnectionPool

from arxiv_curator.config import ProjectConfig
from arxiv_curator.vector_search import LakebasePgVectorManager


class LakebaseMemory:
    """Session messages in Lakebase: one row per message (JSONB), loaded again on the next turn.

    Credentials are short-lived, so the pool is rebuilt after a connection error.
    Reads and writes degrade to a warning when the database is unreachable: the agent still answers,
    just without memory.
    """

    def __init__(self, config: ProjectConfig, table: str = "agent_messages") -> None:
        """Bind the memory to the Lakebase project of the config."""
        self.manager = LakebasePgVectorManager(config=config)
        self.schema = config.schema
        self.table = f'"{config.schema}"."{table}"'
        self._pool: ConnectionPool | None = None

    def _get_pool(self) -> ConnectionPool:
        if self._pool is None:
            self._pool = ConnectionPool(conninfo=self.manager.connection_string(), min_size=1, max_size=5, open=True)
        return self._pool

    def _reset_pool(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def ensure_schema(self) -> None:
        """Create the messages table and its index if they do not exist."""
        with self._get_pool().connection() as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS agent_messages_session_idx ON {self.table} (session_id, id)")
        logger.info(f"Memory table {self.table} ready")

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append messages ({role, content}) to the session."""
        try:
            with self._get_pool().connection() as conn:
                for message in messages:
                    conn.execute(
                        f"INSERT INTO {self.table} (session_id, message_data) VALUES (%s, %s)",
                        (session_id, json.dumps(message)),
                    )
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to save session messages: {e}")

    def load_messages(self, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return the last `limit` messages of the session, oldest first."""
        try:
            with self._get_pool().connection() as conn:
                rows = conn.execute(
                    f"SELECT message_data FROM {self.table} WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                ).fetchall()
            return [row[0] for row in reversed(rows)]
        except psycopg.OperationalError:
            self._reset_pool()
            raise
        except Exception as e:
            logger.warning(f"Failed to load session messages: {e}")
            return []

    def clear(self, session_id: str) -> None:
        """Delete the history of a session."""
        with self._get_pool().connection() as conn:
            conn.execute(f"DELETE FROM {self.table} WHERE session_id = %s", (session_id,))
