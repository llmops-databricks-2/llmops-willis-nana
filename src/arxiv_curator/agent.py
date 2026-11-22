"""Call agent: a tool-calling loop on top of Lakebase memory and MCP tools.

Orchestration, per turn:
  code     1. load the session history from Lakebase
  model    2. the LLM answers, or asks for tools (it decides which, and when to stop)
  code     3. every requested tool is run, results are fed back (bounded by max_steps)
  code     4. the user question and the final answer are saved to Lakebase
"""

from __future__ import annotations

import json

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

from arxiv_curator.config import ProjectConfig
from arxiv_curator.mcp import ToolInfo
from arxiv_curator.memory import LakebaseMemory

SYSTEM_PROMPT = """You are an assistant for sales managers who analyse recorded sales calls.
Tools:
- Genie (query_space_*, then poll_response_* if the answer is not ready): counts, dates, durations, speakers.
  Ask it a QUESTION IN PLAIN LANGUAGE, never SQL, and ask each question once.
- search_calls: what was said in the calls (needs, pains, objections, quotes).
Never invent figures: if a tool returns nothing useful, say so.
Earlier messages of the conversation are provided: use them to resolve follow-up questions."""


class CallAgent:
    """Tool-calling agent with conversation memory."""

    def __init__(
        self,
        config: ProjectConfig,
        tools: list[ToolInfo],
        memory: LakebaseMemory,
        workspace: WorkspaceClient | None = None,
        max_steps: int = 8,
        history_limit: int = 20,
    ) -> None:
        """Wire the LLM (through the AI Gateway), the tools and the memory."""
        workspace = workspace or WorkspaceClient()
        self.model = config.llm_endpoint
        self.client = OpenAI(
            api_key=workspace.config.authenticate()["Authorization"].removeprefix("Bearer "),
            base_url=f"{workspace.config.host}/ai-gateway/mlflow/v1",
        )
        self.tools = {tool.name: tool for tool in tools}
        self.memory = memory
        self.max_steps = max_steps
        self.history_limit = history_limit

    @mlflow.trace(span_type="AGENT")
    def chat(self, session_id: str, question: str) -> str:
        """Answer one question of a session, with the history and the tools."""
        history = self.memory.load_messages(session_id, self.history_limit)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]

        answer = None
        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=[tool.spec for tool in self.tools.values()] or None
            )
            message = response.choices[0].message
            if not message.tool_calls:
                answer = message.content
                break
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                logger.info(f"tool call: {call.function.name}({call.function.arguments})")
                output = self._run_tool(call.function.name, call.function.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        if answer is None:
            answer = "I could not finish within the step limit. Please narrow the question."

        self.memory.save_messages(session_id, [{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        return answer

    @mlflow.trace(span_type="TOOL")
    def _run_tool(self, name: str, raw_arguments: str) -> str:
        try:
            return str(self.tools[name].exec_fn(**json.loads(raw_arguments or "{}")))
        except Exception as e:  # the model gets the error and can retry
            return f"ERROR: {e}"
