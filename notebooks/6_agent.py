# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Call agent: Genie (MCP) + transcript search (custom function), conversation history in Lakebase, loaded again on the next turn."""

import uuid

import mlflow
from databricks.sdk.runtime import dbutils
from loguru import logger

from arxiv_curator import call_search_tool
from arxiv_curator.agent import CallAgent
from arxiv_curator.config import load_config
from arxiv_curator.mcp_tools import McpToolbox
from arxiv_curator.memory import LakebaseMemory

# COMMAND ----------

dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("project_config.yml", env)
if not cfg.genie_space_id:
    raise ValueError("Set genie_space_id in project_config.yml (Genie space on the call tables)")

mlflow.set_experiment(cfg.experiment_path)

memory = LakebaseMemory(cfg)
memory.ensure_table()
toolbox = McpToolbox.genie(cfg.genie_space_id)  # tool 1: Genie, through MCP
call_search_tool.register(toolbox, cfg)  # tool 2: custom function, vector search in Lakebase
logger.info(f"Tools: {[spec['function']['name'] for spec in toolbox.specs]}")
agent = CallAgent(cfg, toolbox, memory)

# COMMAND ----------

# Turn 1, then a follow-up that only makes sense with the memory ("those" = the calls of turn 1)
session_id = f"demo-{uuid.uuid4().hex[:8]}"
print(agent.chat(session_id, "How many calls do we have, and what was the main problem of the prospect in them?"))
print("-" * 60)
print(agent.chat(session_id, "Quote what he said about it."))

# COMMAND ----------

# Proof of persistence: a brand-new agent object (empty process state) reads the same session from Lakebase
print(CallAgent(cfg, toolbox, LakebaseMemory(cfg)).chat(session_id, "Remind me what my first question was."))
print(memory.load(session_id))
