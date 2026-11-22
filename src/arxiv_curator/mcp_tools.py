"""Expose MCP tools (Genie) and custom Python functions to an OpenAI-style chat model."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient


class McpToolbox:
    """List the tools of MCP servers as chat-completions tool specs and route the calls back."""

    def __init__(self, server_urls: list[str], workspace: WorkspaceClient | None = None) -> None:
        """Connect to each MCP server and index its tools by name."""
        self.workspace = workspace or WorkspaceClient()
        self._client_by_tool: dict[str, DatabricksMCPClient] = {}
        self._local_tools: dict[str, Callable[..., str]] = {}
        self.specs: list[dict[str, Any]] = []
        for url in server_urls:
            client = DatabricksMCPClient(server_url=url, workspace_client=self.workspace)
            for tool in client.list_tools():
                self._client_by_tool[tool.name] = client
                self.specs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                )

    @classmethod
    def genie(cls, genie_space_id: str, workspace: WorkspaceClient | None = None) -> McpToolbox:
        """Toolbox with the tools of one Genie space."""
        workspace = workspace or WorkspaceClient()
        return cls([f"{workspace.config.host}/api/2.0/mcp/genie/{genie_space_id}"], workspace)

    def add_function(self, name: str, description: str, parameters: dict[str, Any], fn: Callable[..., str]) -> None:
        """Register a custom Python function as a tool, next to the MCP ones."""
        self._local_tools[name] = fn
        self.specs.append(
            {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}
        )

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return its text output."""
        if name in self._local_tools:
            return self._local_tools[name](**arguments)
        result = self._client_by_tool[name].call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        return f"ERROR: {text}" if result.isError else text
