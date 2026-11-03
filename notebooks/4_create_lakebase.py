# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Sync a Unity Catalog Delta table into a Lakebase Postgres table (Lakebase synced table)."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.runtime import dbutils
from databricks.sdk.service.postgres import (
    Project,
    ProjectSpec,
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
)
from loguru import logger

from arxiv_curator.config import load_config

# COMMAND ----------

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("source_table", "call_chunks")
dbutils.widgets.text("primary_key", "chunk_id")
env = dbutils.widgets.get("env")
source_table = dbutils.widgets.get("source_table")
primary_key = dbutils.widgets.get("primary_key")
cfg = load_config("project_config.yml", env)

PROJECT_ID = cfg.lakebase_instance
BRANCH = f"projects/{PROJECT_ID}/branches/production"
POSTGRES_DATABASE = "databricks_postgres"

SOURCE_TABLE = f"{cfg.full_schema_name}.{source_table}"
# The synced table gets a UC entry here and lands in Postgres as <schema>.<table>
SYNCED_TABLE = f"{cfg.full_schema_name}.{source_table}_synced"

w = WorkspaceClient()

# COMMAND ----------

# Reuse the environment's Lakebase project, or create it: this provisions a
# production branch and a primary read-write endpoint that scales to zero.
try:
    project = w.postgres.get_project(f"projects/{PROJECT_ID}")
    logger.info(f"Lakebase project {project.name} found")
except NotFound:
    logger.info(f"Creating Lakebase project {PROJECT_ID}...")
    project = w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name=PROJECT_ID)), project_id=PROJECT_ID
    ).wait()
    logger.info(f"Lakebase project {project.name} created")

# COMMAND ----------

# Create the synced table. TRIGGERED syncs only the changes on each run, which
# needs the Change Data Feed on the source (enabled in 3_parse_pdf for call_chunks).
try:
    synced = w.postgres.get_synced_table(f"synced_tables/{SYNCED_TABLE}")
    logger.info(f"Synced table {SYNCED_TABLE} already exists")
except NotFound:
    logger.info(f"Syncing {SOURCE_TABLE} into Lakebase...")
    synced = w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=SOURCE_TABLE,
                primary_key_columns=[primary_key],
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                branch=BRANCH,
                postgres_database=POSTGRES_DATABASE,
                create_database_objects_if_missing=True,
            )
        ),
        synced_table_id=SYNCED_TABLE,
    ).wait()
    logger.info(f"Synced table {SYNCED_TABLE} created")

# COMMAND ----------

# The first sync runs in the background: rerun this cell until the state is ONLINE
synced = w.postgres.get_synced_table(f"synced_tables/{SYNCED_TABLE}")
logger.info(f"State: {synced.status.detailed_state}")
logger.info(f"Message: {synced.status.message}")
logger.info(f"Pipeline: {synced.status.pipeline_id}")
logger.info(f"Postgres table: {POSTGRES_DATABASE} > {cfg.schema}.{source_table}_synced on {BRANCH}")
