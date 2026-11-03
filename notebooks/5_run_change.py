# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Run semantic, filtered and hybrid search on the call chunks and compare the returned evidence."""

import json
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.runtime import dbutils, display
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from arxiv_curator.config import load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("file_name", "L_SLIMANI_ Appel_Diagnostic_Gemini.pdf")
env = dbutils.widgets.get("env")
file_name = dbutils.widgets.get("file_name")
cfg = load_config("project_config.yml", env)

CHUNKS_TABLE = f"{cfg.full_schema_name}.call_chunks"
INDEX_NAME = f"{cfg.full_schema_name}.call_chunks_index"
VS_ENDPOINT = f"llmops-vs-{env}"
NUM_RESULTS = 5
COLUMNS = ["chunk_id", "chunk_position", "call_id", "chunk_to_retrieve"]

w = WorkspaceClient()

# COMMAND ----------

# Vector Search endpoint: reuse it if it exists (creating one is billed while it runs)
try:
    w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    logger.info(f"Endpoint {VS_ENDPOINT} found")
except NotFound:
    logger.info(f"Creating endpoint {VS_ENDPOINT}...")
    w.vector_search_endpoints.create_endpoint_and_wait(name=VS_ENDPOINT, endpoint_type=EndpointType.STANDARD)

# COMMAND ----------

# Delta Sync index on the chunks: embed chunk_to_embed (chunk text + context),
# return chunk_to_retrieve. The same index serves ANN and HYBRID queries.
try:
    w.vector_search_indexes.get_index(INDEX_NAME)
    logger.info(f"Index {INDEX_NAME} found")
except NotFound:
    logger.info(f"Creating index {INDEX_NAME}...")
    w.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=VS_ENDPOINT,
        primary_key="chunk_id",
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=CHUNKS_TABLE,
            pipeline_type=PipelineType.TRIGGERED,
            embedding_source_columns=[
                EmbeddingSourceColumn(name="chunk_to_embed", embedding_model_endpoint_name=cfg.embedding_endpoint)
            ],
        ),
    )

while not w.vector_search_indexes.get_index(INDEX_NAME).status.ready:
    logger.info("Waiting for the index to be ready...")
    time.sleep(30)
logger.info(f"Index {INDEX_NAME} ready")

# COMMAND ----------

# Same questions as the call analysis prompts of 2_call_model_prompts, phrased as searches.
# The last two hold exact terms (tools, next step), where keyword matching should help.
QUESTIONS = [
    "Quel est le budget du prospect ?",
    "Quelles sont les objections du prospect ?",
    "Quel est l'objectif du prospect ?",
    "Quelle expérience a le prospect avec Spark et Databricks ?",
    "Quelle est la prochaine étape ?",
]

call_id = spark.table(CHUNKS_TABLE).filter(F.col("file_name") == file_name).first()["call_id"]
num_chunks = spark.table(CHUNKS_TABLE).filter(F.col("call_id") == call_id).count()
second_half = num_chunks // 2

# Standard endpoint filters: a dict, one key per column ("col >=" for comparisons)
FILTERS = json.dumps({"call_id": call_id, "chunk_position >=": second_half})
logger.info(f"Call {call_id}: {num_chunks} chunks, filter keeps positions >= {second_half}")

MODES = {
    "semantic": {"query_type": "ANN"},
    "filtered": {"query_type": "ANN", "filters_json": FILTERS},
    "hybrid": {"query_type": "HYBRID"},
}


def search(question: str, **params: str) -> list[dict]:
    """Run one query and return the chunks with their rank and score (score is the last column)."""
    result = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME, columns=COLUMNS, query_text=question, num_results=NUM_RESULTS, **params
    )
    rows = result.result.data_array or []
    return [
        {**dict(zip(COLUMNS, row[:-1], strict=True)), "rank": rank, "score": float(row[-1])}
        for rank, row in enumerate(rows, start=1)
    ]


evidence = []
for question in QUESTIONS:
    for mode, params in MODES.items():
        for hit in search(question, **params):
            evidence.append(
                {
                    "question": question,
                    "mode": mode,
                    "rank": hit["rank"],
                    "chunk_position": int(hit["chunk_position"]),
                    "score": round(hit["score"], 4),
                    "chunk_id": hit["chunk_id"],
                    "evidence": hit["chunk_to_retrieve"],
                }
            )
logger.info(f"{len(evidence)} hits collected")

# COMMAND ----------

# Everything returned, by question and mode. Scores are not comparable across modes
# (cosine similarity for semantic and filtered, a fused score for hybrid): compare the ranks.
evidence_df = spark.createDataFrame(evidence)
display(evidence_df.orderBy("question", "mode", "rank"))

# COMMAND ----------

# What changes versus semantic search: which chunks appear, disappear or move.
summary = []
for question in QUESTIONS:
    by_mode = {
        mode: [h["chunk_id"] for h in sorted(evidence, key=lambda h: h["rank"]) if h["question"] == question and h["mode"] == mode]
        for mode in MODES
    }
    positions = {
        mode: [h["chunk_position"] for h in sorted(evidence, key=lambda h: h["rank"]) if h["question"] == question and h["mode"] == mode]
        for mode in MODES
    }
    base = by_mode["semantic"]
    for mode in ("filtered", "hybrid"):
        kept = [c for c in by_mode[mode] if c in base]
        summary.append(
            {
                "question": question,
                "mode": mode,
                "semantic_positions": str(positions["semantic"]),
                "mode_positions": str(positions[mode]),
                "shared_with_semantic": len(kept),
                "new_chunks": len(by_mode[mode]) - len(kept),
                "top1_changed": bool(by_mode[mode]) and bool(base) and by_mode[mode][0] != base[0],
            }
        )
display(spark.createDataFrame(summary))

# COMMAND ----------

# Read the top-1 evidence of each mode side by side for one question
QUESTION = QUESTIONS[0]
for mode in MODES:
    top = next((h for h in evidence if h["question"] == QUESTION and h["mode"] == mode and h["rank"] == 1), None)
    print(f"=== {mode} | {QUESTION}")
    print(f"position {top['chunk_position']} | score {top['score']}\n{top['evidence']}\n" if top else "no result\n")
