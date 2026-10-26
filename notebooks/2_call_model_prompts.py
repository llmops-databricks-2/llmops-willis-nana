# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Analyse a call transcript with the course's chat model through the AI Gateway."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils, display
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("call_id", "7aedb6fc14518f47")
env = dbutils.widgets.get("env")
call_id = dbutils.widgets.get("call_id")
cfg = load_config("project_config.yml", env)

# COMMAND ----------

workspace_client = WorkspaceClient()
databricks_token = workspace_client.tokens.create(lifetime_seconds=1200).token_value

# COMMAND ----------

client = OpenAI(
    api_key=databricks_token,
    base_url=f"{workspace_client.config.host}/ai-gateway/mlflow/v1",
)

# COMMAND ----------

transcript_table = f"{cfg.full_schema_name}.call_analysis"
transcript_line = spark.table(transcript_table).filter(f"call_id = '{call_id}'").first()
if transcript_line is None:
    raise ValueError(f"No call {call_id} in {transcript_table}")

# COMMAND ----------

transcript = transcript_line["transcript"]

# COMMAND ----------

print(transcript)

# COMMAND ----------

prompt = f"""You are a sales call analyst. Here is a call transcript in French.

Summarize the call in 5 lines, then list the prospect's goal, budget and objections.
Only use what is in the transcript. If something is missing, say "not mentioned".

Transcript:
{transcript}
"""

# COMMAND ----------

response = client.responses.create(
    model=cfg.llm_endpoint,
    max_output_tokens=512,
    input=prompt,
)

display(response.output_text)

# COMMAND ----------

prompt = f"""You are a sales call analyst. Here is a call transcript in French.

The ICP is : 3 years of exp, alrealdy in the data market, strong in tech (SQL, Python, Spark, Databricks)

Is this lead quality? If not, why?
If yes, what is the next step?

Transcript:
{transcript}
"""

# COMMAND ----------

response = client.responses.create(
    model=cfg.llm_endpoint,
    max_output_tokens=512,
    input=prompt,
)

display(response.output_text)

# COMMAND ----------

print(response.usage)
print("input:", response.usage.input_tokens)
print("output:", response.usage.output_tokens)
print("total:", response.usage.total_tokens)

# COMMAND ----------
