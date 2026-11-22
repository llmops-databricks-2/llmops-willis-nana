"""Conversation memory stored in Lakebase Postgres."""

from __future__ import annotations

from loguru import logger

from arxiv_curator.config import ProjectConfig
from arxiv_curator.vector_search import LakebasePgVectorManager


class LakebaseMemory:
    """One row per message, keyed by session: save each turn, reload the last N on the next one.

    Roles kept are user and assistant only: tool calls stay inside a turn, so the history
    stays small and always valid for the chat API.
    """

    def __init__(self, config: ProjectConfig, table: str = "agent_messages") -> None:
        """Bind the memory to the Lakebase project of the config."""
        self.manager = LakebasePgVectorManager(config=config)
        self.schema = config.schema
        self.table = f'"{config.schema}"."{table}"'

    def ensure_table(self) -> None:
        """Create the messages table if needed (idempotent)."""
        with self.manager.connection() as conn:
            conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS agent_messages_session_idx ON {self.table} (session_id, id)")
        logger.info(f"Memory table {self.table} ready")

    def save(self, session_id: str, messages: list[dict[str, str]]) -> None:
        """Append messages (role, content) to the session."""
        with self.manager.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (session_id, role, content) VALUES (%s, %s, %s)",
                [(session_id, m["role"], m["content"]) for m in messages],
            )

    def load(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        """Return the last `limit` messages of the session, oldest first."""
        with self.manager.connection() as conn:
            rows = conn.execute(
                f"SELECT role, content FROM {self.table} WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                (session_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def clear(self, session_id: str) -> None:
        """Delete the history of a session."""
        with self.manager.connection() as conn:
            conn.execute(f"DELETE FROM {self.table} WHERE session_id = %s", (session_id,))
