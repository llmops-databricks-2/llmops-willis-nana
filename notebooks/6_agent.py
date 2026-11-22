# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
# MAGIC %md
# MAGIC # 6. Call agent: tools and memory
# MAGIC
# MAGIC ```text
# MAGIC question ─► load history (Lakebase) ─► LLM ◄─► tools ─► save turn (Lakebase)
# MAGIC                                               ├── Genie (MCP): counts, dates, durations
# MAGIC                                               └── search_calls (custom function): what was said
# MAGIC ```
# MAGIC
# MAGIC Code decides the steps (load, loop bound, save). The model decides which tool to call and when to stop.
# MAGIC Design notes: `docs_agent_design.md`.

# COMMAND ----------

import uuid

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger

from arxiv_curator.agent import CallAgent
from arxiv_curator.call_search_tool import create_search_calls_tool
from arxiv_curator.config import load_config
from arxiv_curator.mcp import genie_tools
from arxiv_curator.memory import LakebaseMemory

# COMMAND ----------

dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("project_config.yml", env)
if not cfg.genie_space_id:
    raise ValueError("Set genie_space_id in project_config.yml (Genie space on the call tables)")

mlflow.set_experiment(cfg.experiment_path)
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tools: Genie (MCP) and a custom function

# COMMAND ----------

tools = [*genie_tools(w, cfg.genie_space_id), create_search_calls_tool(cfg, w)]
logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")

# Run the custom tool alone first
print(tools[-1].exec_fn(query="main problem of the prospect", num_results=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Memory: session messages in Lakebase

# COMMAND ----------

memory = LakebaseMemory(cfg)
memory.ensure_schema()
agent = CallAgent(cfg, tools, memory, workspace=w)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Two turns in one session
# MAGIC The follow-up ("it", "he") only makes sense if the first turn was loaded from Lakebase.

# COMMAND ----------

session_id = f"demo-{uuid.uuid4().hex[:8]}"
print(agent.chat(session_id, "How many calls do we have, and what was the main problem of the prospect in them?"))
print("-" * 60)
print(agent.chat(session_id, "Quote what he said about it."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Persistence proof
# MAGIC A brand-new agent and memory object (no process state) read the same session from Lakebase.

# COMMAND ----------

fresh_agent = CallAgent(cfg, tools, LakebaseMemory(cfg), workspace=w)
print(fresh_agent.chat(session_id, "Remind me what my first question was."))
print(memory.load_messages(session_id))
