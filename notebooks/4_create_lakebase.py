# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Sync the call chunk tables into Lakebase Postgres, with the embedding as a pgvector column."""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.runtime import dbutils
from databricks.sdk.service.postgres import (
    Project,
    ProjectSpec,
    SyncedTable,
    SyncedTableState,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("project_config.yml", env)

PROJECT_ID = cfg.lakebase_project_id
BRANCH = f"projects/{PROJECT_ID}/branches/production"
POSTGRES_DATABASE = cfg.lakebase_database

# One synced table per chunking strategy (see 3_parse_pdf): Delta source table -> Postgres table name
SYNCS = {
    f"{cfg.full_schema_name}.call_chunks": "call_chunks_synced",
    f"{cfg.full_schema_name}.call_chunks_turns": "call_chunks_turns_synced",
}
EMBEDDING_DIM = 1024  # dimension of the embedding endpoint used by CallProcessor

manager = LakebasePgVectorManager(config=cfg)

w = WorkspaceClient()

# COMMAND ----------

# Reuse the environment's Lakebase project, or create it (production branch + primary compute that scales to zero)
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

# The vector extension must exist before the synced table with a vector column is created:
# creating a synced table does not install extensions. manager.ensure_extension() installs lakebase_vector,
# which this project (Postgres 16) does not preload: pgvector's `vector` is available and is enough here.
with manager.connection() as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
logger.info("vector extension ready")

# COMMAND ----------

# Create the synced tables. TRIGGERED syncs the changes on each run (the chunk tables have the Change Data Feed on).
# The type override makes `embedding` (ARRAY<FLOAT> in Delta) a pgvector vector(1024) column in Postgres.
for source_table, synced_name in SYNCS.items():
    synced_table = f"{cfg.full_schema_name}.{synced_name}"
    try:
        w.postgres.get_synced_table(f"synced_tables/{synced_table}")
        logger.info(f"Synced table {synced_table} already exists")
    except NotFound:
        logger.info(f"Syncing {source_table} into Lakebase...")
        w.postgres.create_synced_table(
            synced_table=SyncedTable(
                spec=SyncedTableSyncedTableSpec(
                    source_table_full_name=source_table,
                    primary_key_columns=["id"],
                    scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                    branch=BRANCH,
                    postgres_database=POSTGRES_DATABASE,
                    create_database_objects_if_missing=True,
                    type_overrides=[
                        SyncedTableSyncedTableSpecTypeOverride(
                            column_name="embedding",
                            pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
                            size=EMBEDDING_DIM,
                        )
                    ],
                )
            ),
            synced_table_id=synced_table,
        ).wait()
        logger.info(f"Synced table {synced_table} created")

# COMMAND ----------

# The first sync runs in the background: wait until each table is online (up to 20 minutes)
ONLINE = {
    SyncedTableState.SYNCED_TABLE_ONLINE,
    SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE,
    SyncedTableState.SYNCED_TABLE_ONLINE_CONTINUOUS_UPDATE,
}
FAILED = {SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED, SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED}

for synced_name in SYNCS.values():
    synced_table = f"{cfg.full_schema_name}.{synced_name}"
    deadline = time.time() + 20 * 60
    while True:
        status = w.postgres.get_synced_table(f"synced_tables/{synced_table}").status
        logger.info(f"{synced_name} | {status.detailed_state} | {status.message}")
        if status.detailed_state in ONLINE:
            break
        if status.detailed_state in FAILED or time.time() > deadline:
            raise RuntimeError(f"Synced table {synced_table} is not online: {status.detailed_state}")
        time.sleep(30)

# COMMAND ----------

# Full-text index used by the hybrid search (the vector HNSW index is disabled in LakebasePgVectorManager:
# a sequential scan is exact and fast enough for a few hundred chunks), then check the Postgres side
for synced_name in SYNCS.values():
    manager.synced_table_name = synced_name
    manager.table_name = f'"{cfg.schema}"."{synced_name}"'
    # The Postgres table shows up a little after the synced table is online
    for _ in range(20):
        with manager.connection() as conn:
            if conn.execute("SELECT to_regclass(%s)", (manager.table_name,)).fetchone()[0]:
                break
        time.sleep(15)
    manager.ensure_indexes()
    with manager.connection() as conn:
        count = conn.execute(f"SELECT count(*) FROM {manager.table_name}").fetchone()[0]
        embedding_type = conn.execute(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = 'embedding'",
            (cfg.schema, synced_name),
        ).fetchone()[0]
    logger.info(f"{manager.table_name}: {count} rows, embedding is {embedding_type}")