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

# MAGIC %md
# MAGIC ## 0. Install the current package
# MAGIC The bundle deploy is blocked (jobs owned by someone else), so install the wheel built from `src/`
# MAGIC (`uv build`, uploaded to the workspace). Rebuild and re-upload it after each code change, with a new version.

# COMMAND ----------

# MAGIC %pip install --force-reinstall --no-deps /Workspace/Users/willis.nana@gmail.com/artifacts/arxiv_curator-0.0.2-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

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
# MAGIC ## 3. Quatre questions en français
# MAGIC Questions 1 à 3 : une même session (Genie, recherche dans les transcripts, puis une relance qui n'a de sens
# MAGIC que grâce à la mémoire). Question 4 : un nouvel agent, sans état local, relit la session depuis Lakebase.

# COMMAND ----------

session_id = f"demo-{uuid.uuid4().hex[:8]}"

# 1. Genie (MCP): chiffres et métadonnées
print(agent.chat(session_id, "Combien d'appels avons-nous, quelle est leur durée moyenne et qui y a participé ?"))

# COMMAND ----------

# 2. search_calls (fonction personnalisée): contenu des échanges
print(agent.chat(session_id, "Quel est le principal problème du prospect dans cet appel ?"))

# COMMAND ----------

# 3. Relance: « ses » et « à ce sujet » dépendent de la question précédente, chargée depuis Lakebase
print(agent.chat(session_id, "Cite ses propres mots à ce sujet."))

# COMMAND ----------

# 4. Persistance: nouvel agent et nouvelle mémoire, même session
fresh_agent = CallAgent(cfg, tools, LakebaseMemory(cfg), workspace=w)
print(fresh_agent.chat(session_id, "Rappelle-moi quelle était ma toute première question."))
print(memory.load_messages(session_id))
