"""Tools for the agent: one `ToolInfo` interface for MCP tools (Genie) and custom Python functions."""

from __future__ import annotations

from collections.abc import Callable

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel, ConfigDict


class ToolInfo(BaseModel):
    """A tool the agent can call.

    Attributes:
        name: Tool name
        spec: JSON description of the tool (OpenAI chat-completions format)
        exec_fn: Function that implements the tool logic and returns text

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    spec: dict
    exec_fn: Callable[..., str]


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable[..., str]:
    """Create the function that runs one MCP tool (a client per call, so credentials never go stale)."""

    def exec_fn(**kwargs: object) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        text = "".join(c.text for c in response.content if getattr(c, "text", None))
        return f"ERROR: {text}" if response.isError else text

    return exec_fn


def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """List the tools of MCP servers and wrap them as ToolInfo."""
    tools = []
    for server_url in url_list:
        for mcp_tool in DatabricksMCPClient(server_url=server_url, workspace_client=w).list_tools():
            spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": mcp_tool.inputSchema or {"type": "object", "properties": {}},
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            tools.append(
                ToolInfo(name=mcp_tool.name, spec=spec, exec_fn=create_managed_exec_fn(server_url, mcp_tool.name, w))
            )
    return tools


def genie_tools(w: WorkspaceClient, genie_space_id: str) -> list[ToolInfo]:
    """Tools of one Genie space, through the Databricks managed MCP server."""
    return create_mcp_tools(w, [f"{w.config.host.rstrip('/')}/api/2.0/mcp/genie/{genie_space_id}"])
