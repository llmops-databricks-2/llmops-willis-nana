# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Run semantic, filtered and hybrid search on the call chunks and capture what changes in the evidence."""

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils, display
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.vector_search import LakebasePgVectorManager

# COMMAND ----------

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("project_config.yml", env)

NUM_RESULTS = 5


class GatewayPgVectorManager(LakebasePgVectorManager):
    """Embed the queries through the AI Gateway: cfg.embedding_endpoint is a gateway service, not a serving endpoint."""

    def __init__(self, synced_table_name: str, **kwargs) -> None:  # noqa: ANN003
        """Search the Postgres synced table of the given name."""
        self.synced_table_name = synced_table_name
        super().__init__(**kwargs)

    def embed_query(self, query: str) -> list[float]:
        """Generate a query embedding with the same model used for the documents."""
        workspace = self.workspace or WorkspaceClient()
        response = workspace.api_client.do(
            "POST", "/ai-gateway/mlflow/v1/embeddings", body={"model": self.embedding_model, "input": query}
        )
        return response["data"][0]["embedding"]


# The chunks of the first strategy (one per parsed text element) and of the second (windows of speaker turns)
import yaml

with open("../project_config.yml") as f:
    lakebase_database = yaml.safe_load(f)[env].get("lakebase_database", "databricks_postgres")
manager = GatewayPgVectorManager("call_chunks_synced", config=cfg, database=lakebase_database)
turns_manager = GatewayPgVectorManager("call_chunks_turns_synced", config=cfg, database=lakebase_database)

# COMMAND ----------

# Which call and year does the synced table hold? The filtered search needs a value that exists.
with manager.connection() as conn:
    calls = conn.execute(
        f"""SELECT arxiv_id
        , title
        , authors
        , year
        , count(*)
        FROM {manager.table_name}
        GROUP BY 1, 2, 3, 4 ORDER BY 4, 1"""
    ).fetchall()
display(pd.DataFrame(calls, columns=["call_id", "title", "speakers", "year", "num_chunks"]))
YEAR = calls[0][3]

# COMMAND ----------

# The same questions are asked in every mode. The first five are phrased in natural language (semantic search
# should do well); the last two quote exact words of the call (full-text matching should help).
QUESTIONS = [
    "Quel est le budget du prospect ?",
    "Quelles sont les objections du prospect ?",
    "Quel est le parcours professionnel du prospect ?",
    "Quel accompagnement idéal le prospect attend-il ?",
    "Pourquoi le prospect veut-il se former ?",
    "prébourse",
    "Algérie",
]

# Filtered search keeps one call year: YEAR holds the data, YEAR - 1 shows a filter that excludes it
MODES = {
    "semantic": lambda q: manager.search(q, num_results=NUM_RESULTS),
    "filtered": lambda q: manager.search(q, num_results=NUM_RESULTS, year=YEAR),
    "filtered_other_year": lambda q: manager.search(q, num_results=NUM_RESULTS, year=YEAR - 1),
    "hybrid": lambda q: manager.hybrid_search(q, num_results=NUM_RESULTS),
}

rows = []
for question in QUESTIONS:
    for mode, run in MODES.items():
        for rank, hit in enumerate(run(question), start=1):
            rows.append(
                {
                    "question": question,
                    "mode": mode,
                    "rank": rank,
                    "id": hit["id"],
                    "year": hit["year"],
                    "score": round(float(hit["score"]), 4),
                    "evidence": hit["text"],
                }
            )
evidence = pd.DataFrame(rows)
logger.info(f"{len(evidence)} hits collected for {len(QUESTIONS)} questions and {len(MODES)} modes")

# COMMAND ----------

# Everything returned, by question and mode. Scores are not comparable across modes
# (cosine similarity for semantic and filtered, a weighted vector + full-text score for hybrid): compare the ranks.
display(evidence.sort_values(["question", "mode", "rank"]))

# COMMAND ----------

# What changes versus semantic search: chunks that appear, disappear or move, and whether the top-1 changes.
summary = []
for question in QUESTIONS:
    by_mode = {
        mode: group.sort_values("rank")["id"].tolist()
        for mode, group in evidence[evidence["question"] == question].groupby("mode")
    }
    base = by_mode.get("semantic", [])
    for mode in ("filtered", "filtered_other_year", "hybrid"):
        hits = by_mode.get(mode, [])
        summary.append(
            {
                "question": question,
                "mode": mode,
                "num_results": len(hits),
                "shared_with_semantic": len(set(hits) & set(base)),
                "new_vs_semantic": len(set(hits) - set(base)),
                "top1_changed": bool(hits) and bool(base) and hits[0] != base[0],
                "same_order": hits == base,
                "semantic_chunks": ", ".join(i.split("_")[-1] for i in base),
                "mode_chunks": ", ".join(i.split("_")[-1] for i in hits),
            }
        )
display(pd.DataFrame(summary))

# COMMAND ----------

# Read the evidence itself: the top-1 chunk of semantic and hybrid search where hybrid changed the top-1
changed = {s["question"] for s in summary if s["mode"] == "hybrid" and s["top1_changed"]}
for question in QUESTIONS:
    if question not in changed:
        continue
    for mode in ("semantic", "hybrid"):
        top = evidence[(evidence["question"] == question) & (evidence["mode"] == mode) & (evidence["rank"] == 1)]
        print(f"=== {mode} | {question}")
        print(f"{top.iloc[0]['id']} | score {top.iloc[0]['score']}\n{top.iloc[0]['evidence'][:600]}\n")
if not changed:
    print("Hybrid kept the same top-1 chunk as semantic search for every question.")

# COMMAND ----------

# The other chunking strategy: the same semantic search on the windows of speaker turns.
# A window holds a whole exchange (question and answer), so its top-1 is easier to read as evidence.
chunking = []
for question in QUESTIONS:
    for strategy, searcher in (("parsed_text_elements", manager), ("speaker_turn_windows", turns_manager)):
        top = searcher.search(question, num_results=1)[0]
        chunking.append(
            {
                "question": question,
                "strategy": strategy,
                "top1_chunk": top["id"].split("_")[-1],
                "score": round(float(top["score"]), 4),
                "chars": len(top["text"]),
                "evidence": top["text"][:300],
            }
        )
display(pd.DataFrame(chunking))
