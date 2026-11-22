# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
# ]
# ///
"""Parse the call PDF, inspect the processed documents and chunk tables, and compare two chunking strategies."""

from databricks.sdk.runtime import dbutils, display
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from arxiv_curator.call_processor import CallProcessor
from arxiv_curator.config import load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("file_name", "L_SLIMANI_ Appel_Diagnostic_Gemini.pdf")
env = dbutils.widgets.get("env")
file_name = dbutils.widgets.get("file_name")
cfg = load_config("project_config.yml", env)

processor = CallProcessor(spark, cfg, file_name)

# COMMAND ----------

# Parse the PDF once (ai_parse_document is billed per call), then build both chunk tables with their embeddings
processor.parse_pdf()
processor.process_chunks(max_chars=1500, overlap_turns=1)

# COMMAND ----------

# 01 - Inspect the processed document: the raw ai_parse_document JSON, one row per PDF
parsed = spark.table(processor.parsed_table).filter(F.col("path").endswith(file_name))
display(parsed.select("path", "processed", F.length("parsed_content").alias("json_chars")))

# COMMAND ----------

# One row per element found by the parser, in reading order
elements = parsed.selectExpr(
    "explode(variant_get(parse_json(parsed_content), '$.document.elements', 'ARRAY<VARIANT>')) AS e"
).selectExpr(
    "e:id::INT AS id",
    "e:type::STRING AS type",
    "e:bbox[0]:page_id::INT AS page_id",
    "e:content::STRING AS content",
)

# Only the text elements become chunks; the section headers are the timestamps of the transcript
display(
    elements.groupBy("type")
    .agg(
        F.count("*").alias("num_elements"),
        F.sum(F.length("content")).alias("chars"),
        F.max(F.length("content")).alias("max_chars"),
    )
    .withColumn("kept_as_chunk", F.col("type") == "text")
    .orderBy(F.desc("num_elements"))
)

# COMMAND ----------

display(elements.orderBy("id").limit(12))

# COMMAND ----------

# 01 - Inspect the chunk tables: one row per chunk, with the call metadata and its embedding
for table in (processor.chunks_table, processor.turn_chunks_table):
    print(table)
    display(
        spark.table(table)
        .filter(F.col("arxiv_id") == processor.call_id)
        .groupBy("arxiv_id", "title", "authors", "year", "month", "day")
        .agg(
            F.count("*").alias("num_chunks"),
            F.sum(F.when(F.col("embedding").isNull(), 1).otherwise(0)).alias("missing_embeddings"),
            F.max(F.size("embedding")).alias("embedding_dim"),
        )
    )

# COMMAND ----------

# The chunks of the first strategy: a parsed text element is a page of the transcript, with both speakers mixed
chunks = spark.table(processor.chunks_table).filter(F.col("arxiv_id") == processor.call_id)
display(chunks.select("id", "chunk_id", F.length("text").alias("chars"), "text").orderBy(F.col("chunk_id").cast("int")))

# COMMAND ----------

# 02 - Review chunk sizes (about 4 characters per token): the same statistics for both strategies
turn_chunks = spark.table(processor.turn_chunks_table).filter(F.col("arxiv_id") == processor.call_id)


def size_stats(df: DataFrame, strategy: str) -> DataFrame:
    """Chunk count and length distribution (in characters) of one strategy."""
    return df.select(
        F.lit(strategy).alias("strategy"),
        F.count("*").alias("num_chunks"),
        F.min(F.length("text")).alias("min_chars"),
        F.expr("percentile(length(text), 0.5)").cast("int").alias("median_chars"),
        F.avg(F.length("text")).cast("int").alias("avg_chars"),
        F.max(F.length("text")).alias("max_chars"),
        F.sum(F.when(F.length("text") < 300, 1).otherwise(0)).alias("under_300_chars"),
        F.sum(F.when(F.length("text") > 1500, 1).otherwise(0)).alias("over_1500_chars"),
    )


display(size_stats(chunks, "parsed_text_elements").unionByName(size_stats(turn_chunks, "speaker_turn_windows_1500")))

# COMMAND ----------

# Length buckets of both strategies
for name, df in (("parsed_text_elements", chunks), ("speaker_turn_windows_1500", turn_chunks)):
    print(name)
    display(
        df.withColumn(
            "bucket",
            F.when(F.length("text") < 300, "1. < 300")
            .when(F.length("text") < 800, "2. 300-800")
            .when(F.length("text") < 1500, "3. 800-1500")
            .otherwise("4. >= 1500"),
        )
        .groupBy("bucket")
        .agg(F.count("*").alias("num_chunks"), (F.avg(F.length("text")) / 4).cast("int").alias("avg_tokens"))
        .orderBy("bucket")
    )

# COMMAND ----------

# 02 - A parsed text element is a page of the transcript, so it is cut at page breaks, even in the middle of a turn.
# The speaker-turn windows are cut between turns only, and repeat the last turn of a window at the start of the next.
display(
    turn_chunks.select("id", F.length("text").alias("chars"), "text").orderBy(F.col("chunk_id").cast("int")).limit(6)
)
