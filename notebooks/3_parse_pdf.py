# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Parse a call transcript PDF of the volume with ai_parse_document and inspect the result."""

from databricks.sdk.runtime import dbutils, display
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from arxiv_curator.config import load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("file_name", "L_SLIMANI_ Appel_Diagnostic_Gemini.pdf")
env = dbutils.widgets.get("env")
file_name = dbutils.widgets.get("file_name")
cfg = load_config("project_config.yml", env)

PDF_PATH = f"/Volumes/{cfg.catalog}/{cfg.schema}/{cfg.volume}/{file_name}"
TABLE_NAME = f"{cfg.catalog}.{cfg.schema}.parsed_documents"

# COMMAND ----------

# Parse the PDF once and persist it: each ai_parse_document call is billed,
# so the inspection cells below read the Delta table instead of re-parsing.
# Transcripts have no figures, so AI descriptions are disabled.
parsed_df = (
    spark.read.format("binaryFile")
    .load(PDF_PATH)
    .select(
        "path",
        F.expr("ai_parse_document(content, map('version', '2.0', 'descriptionElementTypes', ''))").alias("parsed"),
    )
)
parsed_df.write.format("delta").mode("overwrite").saveAsTable(TABLE_NAME)
logger.info(f"Parsed {PDF_PATH} into {TABLE_NAME}")

# COMMAND ----------

parsed = spark.table(TABLE_NAME).filter(F.col("path").endswith(file_name))

# Overview: parse errors, page and element counts
display(
    parsed.selectExpr(
        "path",
        "parsed:error_status AS error_status",
        "size(variant_get(parsed, '$.document.pages', 'ARRAY<VARIANT>')) AS num_pages",
        "size(variant_get(parsed, '$.document.elements', 'ARRAY<VARIANT>')) AS num_elements",
        "parsed:metadata AS metadata",
    )
)

# COMMAND ----------

# One row per extracted element
elements = parsed.selectExpr(
    "explode(variant_get(parsed, '$.document.elements', 'ARRAY<VARIANT>')) AS e"
).selectExpr(
    "e:id::INT AS id",
    "e:type::STRING AS type",
    "e:bbox[0]:page_id::INT AS page_id",
    "e:confidence::DOUBLE AS confidence",
    "e:content::STRING AS content",
)

# Element types found in the document
display(elements.groupBy("type").count().orderBy(F.desc("count")))

# COMMAND ----------

display(elements.orderBy("id"))

# COMMAND ----------

# Full text, as the downstream steps would see it
text = "\n".join(row["content"] for row in elements.orderBy("id").collect() if row["content"])
print(text[:5000])

# COMMAND ----------

# Chunk the parsed documents for retrieval: ai_prep_search splits them at natural
# boundaries and adds context (title, headers, page) to the text to embed.
# call_id is computed as in 1_sales_data_ingestion, to join with call_analysis.
CHUNKS_TABLE = f"{cfg.catalog}.{cfg.schema}.call_chunks"

chunks_df = (
    spark.table(TABLE_NAME)
    .withColumn("file_name", F.element_at(F.split("path", "/"), -1))
    .withColumn("call_id", F.substring(F.sha2("file_name", 256), 1, 16))
    .withColumn("prep", F.expr("ai_prep_search(parsed)"))
    .withColumn("chunk", F.expr("explode(variant_get(prep, '$.document.contents', 'ARRAY<VARIANT>'))"))
    .selectExpr(
        "variant_get(chunk, '$.chunk_id', 'STRING') AS chunk_id",
        "variant_get(chunk, '$.chunk_position', 'INT') AS chunk_position",
        "variant_get(chunk, '$.chunk_to_retrieve', 'STRING') AS chunk_to_retrieve",
        "variant_get(chunk, '$.chunk_to_embed', 'STRING') AS chunk_to_embed",
        "call_id",
        "file_name",
        "path AS source_path",
    )
    .withColumn("prepped_at", F.current_timestamp())
)
chunks_df.write.format("delta").mode("overwrite").saveAsTable(CHUNKS_TABLE)

# Change Data Feed lets a Vector Search Delta Sync index pick up new chunks
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
logger.info(f"Chunks ready: {CHUNKS_TABLE}")

# COMMAND ----------

chunks = spark.table(CHUNKS_TABLE).filter(F.col("file_name") == file_name)

# Chunk count and size distribution (in characters)
display(
    chunks.select(
        F.count("*").alias("num_chunks"),
        F.min(F.length("chunk_to_retrieve")).alias("min_chars"),
        F.avg(F.length("chunk_to_retrieve")).cast("int").alias("avg_chars"),
        F.max(F.length("chunk_to_retrieve")).alias("max_chars"),
    )
)

# COMMAND ----------

display(chunks.orderBy("chunk_position"))

# COMMAND ----------

# Review chunk sizes: bucket the chunks by length (about 4 characters per token)
display(
    chunks.withColumn("chars", F.length("chunk_to_retrieve"))
    .withColumn(
        "bucket",
        F.when(F.col("chars") < 200, "1. < 200 (too short?)")
        .when(F.col("chars") < 1000, "2. 200-1000")
        .when(F.col("chars") < 2000, "3. 1000-2000")
        .otherwise("4. >= 2000 (too long?)"),
    )
    .groupBy("bucket")
    .agg(F.count("*").alias("num_chunks"), (F.avg("chars") / 4).cast("int").alias("avg_tokens"))
    .orderBy("bucket")
)

# COMMAND ----------

# The smallest chunks are the first suspects: a lone "Oui, d'accord." means nothing on its own
display(
    chunks.select("chunk_position", F.length("chunk_to_retrieve").alias("chars"), "chunk_to_retrieve")
    .orderBy("chars")
    .limit(10)
)

# COMMAND ----------

# Alternative strategy: group consecutive speaker turns of the transcript built in
# 1_sales_data_ingestion ("Speaker: text" lines) into windows of at most MAX_CHARS.
MAX_CHARS = 1500
OVERLAP_TURNS = 1


def chunk_by_turns(transcript: str) -> list[str]:
    """Group consecutive 'Speaker: text' lines into chunks of at most MAX_CHARS.

    The last OVERLAP_TURNS turns of a chunk are repeated at the start of the next one,
    so a question and its answer are not split apart. A single turn longer than
    MAX_CHARS stays whole.
    """
    turns = [line for line in transcript.split("\n") if line.strip()]
    chunks, current = [], []
    for turn in turns:
        if current and len("\n".join([*current, turn])) > MAX_CHARS:
            chunks.append("\n".join(current))
            overlap = current[-OVERLAP_TURNS:] if OVERLAP_TURNS else []
            # Keep the overlap only if it still fits with the new turn
            current = overlap if len("\n".join([*overlap, turn])) <= MAX_CHARS else []
        current.append(turn)
    if current:
        chunks.append("\n".join(current))
    return chunks


chunk_by_turns_udf = F.udf(chunk_by_turns, ArrayType(StringType()))

turn_chunks = (
    spark.table(f"{cfg.catalog}.{cfg.schema}.call_analysis")
    .filter(F.col("file_name") == file_name)
    .select("call_id", "file_name", F.posexplode(chunk_by_turns_udf("transcript")).alias("chunk_position", "chunk_text"))
)
display(turn_chunks)

# COMMAND ----------


def size_stats(df: DataFrame, text_col: str, strategy: str) -> DataFrame:
    """Chunk count and length distribution (in characters) of one strategy."""
    return df.select(
        F.lit(strategy).alias("strategy"),
        F.count("*").alias("num_chunks"),
        F.min(F.length(text_col)).alias("min_chars"),
        F.avg(F.length(text_col)).cast("int").alias("avg_chars"),
        F.max(F.length(text_col)).alias("max_chars"),
    )


# Compare both strategies side by side
display(
    size_stats(chunks, "chunk_to_retrieve", "ai_prep_search").unionByName(
        size_stats(turn_chunks, "chunk_text", f"speaker_turns_{MAX_CHARS}")
    )
)

# COMMAND ----------

# Qualitative check: which chunks would answer a question about the budget?
# A good chunk holds the whole exchange (question + answer), not half of it.
KEYWORD = "budget"
display(chunks.filter(F.lower("chunk_to_retrieve").contains(KEYWORD)).select("chunk_position", "chunk_to_retrieve"))
display(turn_chunks.filter(F.lower("chunk_text").contains(KEYWORD)).select("chunk_position", "chunk_text"))
