"""Custom-function tool: semantic search over the call transcript chunks in Lakebase."""

from __future__ import annotations

from databricks.sdk import WorkspaceClient

from arxiv_curator.config import ProjectConfig
from arxiv_curator.vector_search import LakebasePgVectorManager

TOOL_NAME = "search_call_transcripts"
TOOL_DESCRIPTION = (
    "Semantic search in the transcripts of the sales calls. Use it for questions about WHAT was said "
    "(objections, needs, pains, quotes). Returns the best matching passages with call title and date. "
    "For counts and statistics use the Genie tool instead."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for, phrased as a natural-language sentence"},
        "num_results": {"type": "integer", "description": "Number of passages to return (default 5)"},
    },
    "required": ["query"],
}


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

    def as_tool(self, num_results: int = 5):  # noqa: ANN201
        """Return the function the agent calls."""

        def search_call_transcripts(query: str, num_results: int = num_results) -> str:
            rows = self.search(query, num_results=num_results)
            if not rows:
                return "No matching passage."
            return "\n\n".join(
                f"[{r['title']} | {r['year']}-{r['month']:02d}-{r['day']:02d} | score {r['score']:.2f}]\n{r['text']}"
                for r in rows
            )

        return search_call_transcripts


def register(toolbox, config: ProjectConfig) -> None:  # noqa: ANN001
    """Add the transcript search tool to a toolbox."""
    search = CallTranscriptSearch(config=config)
    toolbox.add_function(TOOL_NAME, TOOL_DESCRIPTION, TOOL_PARAMETERS, search.as_tool())

