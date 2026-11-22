"""Custom-function tool: hybrid (vector + keyword) search over the call transcript chunks in Lakebase."""

from __future__ import annotations

import json

from databricks.sdk import WorkspaceClient

from arxiv_curator.config import ProjectConfig
from arxiv_curator.mcp import ToolInfo
from arxiv_curator.vector_search import LakebasePgVectorManager


class CallTranscriptSearch(LakebasePgVectorManager):
    """Search the speaker-turn windows; queries are embedded through the AI Gateway."""

    synced_table_name = "call_chunks_turns_synced"

    def embed_query(self, query: str) -> list[float]:
        """Embed with the same gateway model used for the chunks."""
        workspace = self.workspace or WorkspaceClient()
        response = workspace.api_client.do(
            "POST", "/ai-gateway/mlflow/v1/embeddings", body={"model": self.embedding_model, "input": query}
        )
        return response["data"][0]["embedding"]


def create_search_calls_tool(config: ProjectConfig, workspace: WorkspaceClient | None = None) -> ToolInfo:
    """Build the `search_calls` tool."""
    search = CallTranscriptSearch(config=config, workspace=workspace)

    def search_calls(query: str, num_results: int = 5, year: int | None = None) -> str:
        rows = search.hybrid_search(query=query, num_results=num_results, year=year)
        return json.dumps(
            [
                {
                    "call": row.get("title", "N/A"),
                    "date": f"{row.get('year')}-{row.get('month', 0):02d}-{row.get('day', 0):02d}",
                    "excerpt": row.get("text", "")[:800],
                    "score": round(float(row.get("score", 0.0)), 4),
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
        )

    return ToolInfo(
        name="search_calls",
        spec={
            "type": "function",
            "function": {
                "name": "search_calls",
                "description": (
                    "Hybrid semantic + keyword search in the transcripts of the sales calls. Use it for questions "
                    "about WHAT was said (needs, pains, objections, quotes). For counts and statistics use Genie."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look for, as a natural-language sentence"},
                        "num_results": {"type": "integer", "default": 5},
                        "year": {"type": "integer", "description": "Optional: only calls of this year"},
                    },
                    "required": ["query"],
                },
            },
        },
        exec_fn=search_calls,
    )
