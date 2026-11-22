# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "..",
#   "pypdf==5.4.0",
# ]
# ///
# ruff: noqa: ERA001
"""Ingest the call transcript PDFs of the volume into the call_analysis Delta table."""

import hashlib
import os
import re
from collections import Counter
from datetime import datetime

from databricks.sdk.runtime import dbutils, display
from loguru import logger
from pypdf import PdfReader
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType

from arxiv_curator.config import load_config

# COMMAND ----------

# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config: the env widget is set by the job from ${bundle.target}, default dev
dbutils.widgets.text("env", "dev")
env = dbutils.widgets.get("env")
cfg = load_config("project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
VOLUME = cfg.volume
TABLE_NAME = "call_analysis"

# Call transcript PDFs are uploaded to the environment's volume
PDF_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

FR_MONTHS = {
    "janv": 1,
    "févr": 2,
    "fevr": 2,
    "mars": 3,
    "avr": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7,
    "août": 8,
    "aout": 8,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "déc": 12,
    "dec": 12,
}

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
logger.info(f"Env {env}: schema {CATALOG}.{SCHEMA} and volume {PDF_DIR} ready")

PDF_PATHS = sorted(os.path.join(PDF_DIR, name) for name in os.listdir(PDF_DIR) if name.lower().endswith(".pdf"))
if not PDF_PATHS:
    raise FileNotFoundError(f"No call transcript PDFs found in {PDF_DIR}")

# COMMAND ----------

# Parse call transcript PDFs (Google Meet / Gemini style: "Speaker: text" with timestamps)


def parse_call_date(text: str) -> str | None:
    """Parse a header like 'juil. 14, 2026' (French month abbreviation) into 'YYYY-MM-DD'."""
    m = re.search(r"^\s*([A-Za-zéûèÉ]+)\.?\s+(\d{1,2}),\s+(\d{4})", text, re.MULTILINE)
    if not m:
        return None
    month = FR_MONTHS.get(m.group(1).lower())
    return datetime(int(m.group(3)), month, int(m.group(2))).date().isoformat() if month else None


def parse_duration(text: str) -> int | None:
    """Parse 'Transcription terminée après 00:42:21' into seconds."""
    m = re.search(r"apr[èe]s\s+(\d{2}):(\d{2}):(\d{2})", text)
    if not m:
        return None
    h, mi, s = map(int, m.groups())
    return h * 3600 + mi * 60 + s


def flatten(text: str) -> str:
    """Collapse all whitespace into single spaces: the PDF yields one word per line."""
    return re.sub(r"\s+", " ", text).strip()


def parse_title(flat: str) -> str | None:
    """Header looks like: '<date> <title> - Transcription 00:00:00 ...'."""
    m = re.search(r"\d{4}\s+(.+?)\s+-\s*Transcription", flat)
    return m.group(1) if m else None


def detect_speakers(flat: str, min_count: int = 5) -> set[str]:
    """Find speaker labels: 'Name:' or 'First Last:' that repeat many times in the text."""
    single = Counter(re.findall(r"(?<![\w\-])([A-ZÀ-Ý][\w\-]*):\s", flat))
    double = Counter(re.findall(r"(?<![\w\-])([A-ZÀ-Ý][\w\-]*\s[A-ZÀ-Ý][\w\-]*):\s", flat))
    speakers = {name for name, count in single.items() if count >= min_count}
    for name, count in double.items():
        last = name.split()[-1]
        # Prefer 'Willis Nana' over 'Nana', but not 'Hm K-nigh-t' over 'K-nigh-t'
        if count >= min_count and last in speakers and count >= 0.8 * single[last]:
            speakers.discard(last)
            speakers.add(name)
    return speakers


def parse_turns(flat: str, speakers: set[str]) -> list[list[str]]:
    """Split the flattened transcript into [speaker, utterance] turns."""
    if not speakers:
        return []
    body = flat.split("Transcription terminée")[0]
    body = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "", body)  # drop timestamps
    body = re.sub(r"\s+", " ", body)
    names = "|".join(re.escape(s) for s in sorted(speakers, key=len, reverse=True))
    parts = re.split(rf"(?<![\w\-])({names}):\s", body)
    # parts = [header, speaker1, text1, speaker2, text2, ...]
    return [[parts[i], parts[i + 1].strip()] for i in range(1, len(parts) - 1, 2) if parts[i + 1].strip()]


def fetch_call_transcripts(pdf_paths: list[str]) -> list[dict]:
    """Read call transcript PDFs and extract their metadata and full text.

    Args:
        pdf_paths: List of PDF file paths (workspace or volume paths)

    Returns:
        List of call metadata dictionaries

    """
    calls = []
    for path in pdf_paths:
        reader = PdfReader(path)
        raw = "\n".join(page.extract_text() or "" for page in reader.pages)
        flat = flatten(raw)
        turns = parse_turns(flat, detect_speakers(flat))
        # One readable line per speaker turn instead of one word per line
        text = "\n".join(f"{speaker}: {utterance}" for speaker, utterance in turns) or flat

        call = {
            "call_id": hashlib.sha256(path.split("/")[-1].encode("utf-8")).hexdigest()[:16],
            "file_name": path.split("/")[-1],
            "file_path": path,
            "call_title": parse_title(flat),
            "call_date": parse_call_date(flat),
            "duration_seconds": parse_duration(flat),
            "speakers": sorted({t[0] for t in turns}),
            "num_turns": len(turns),
            "num_pages": len(reader.pages),
            "transcript": text,
            "ingestion_timestamp": datetime.now().isoformat(),
            "processed": None,  # Will be set by the analysis step
            "analysis": None,  # Will be set by the analysis step
        }
        calls.append(call)

    return calls


logger.info("Reading call transcripts...")
calls = fetch_call_transcripts(PDF_PATHS)
logger.info(f"Parsed {len(calls)} calls")
logger.info("Sample call:")
logger.info(f"Title: {calls[0]['call_title']}")
logger.info(f"Speakers: {calls[0]['speakers']}")
logger.info(f"Call ID: {calls[0]['call_id']}")
logger.info(f"Turns: {calls[0]['num_turns']}")

# COMMAND ----------

# Create Delta Table in Unity Catalog
# Store the call transcripts in a Delta table for downstream analysis.

# Define schema
schema = StructType(
    [
        StructField("call_id", StringType(), False),
        StructField("file_name", StringType(), False),
        StructField("file_path", StringType(), True),
        StructField("call_title", StringType(), True),
        StructField("call_date", StringType(), True),  # ISO date string (YYYY-MM-DD)
        StructField("duration_seconds", IntegerType(), True),
        StructField("speakers", ArrayType(StringType()), True),
        StructField("num_turns", IntegerType(), True),
        StructField("num_pages", IntegerType(), True),
        StructField("transcript", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
        StructField("processed", LongType(), True),  # Will be set by the analysis step
        StructField("analysis", StringType(), True),  # Will be set by the analysis step
    ]
)

# COMMAND ----------

# Create DataFrame
df = spark.createDataFrame(calls, schema=schema)

# Write to Delta table (overwrite, so re-running does not create duplicates)
table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

# COMMAND ----------

df.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(table_path)

logger.info(f"Delta table ready: {table_path}")
logger.info(f"Records in batch: {df.count()}")

# COMMAND ----------

display(spark.table(table_path))
