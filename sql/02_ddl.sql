-- DDL of the call tables written by CallProcessor (src/arxiv_curator/call_processor.py).
-- Run on Databricks after 01_cleanup.sql. The Lakebase synced tables are created by notebooks/4_create_lakebase.py.
-- call_chunks and call_chunks_turns have the layout of arxiv_chunks_table, so LakebasePgVectorManager searches them:
-- arxiv_id holds the call id, title the call title, authors the speakers and year/month/day the call date.

CREATE TABLE IF NOT EXISTS llmops_dev.willis_nana.call_parsed_docs_table (
    path STRING COMMENT 'Path of the PDF in the volume',
    parsed_content STRING COMMENT 'ai_parse_document output (version 2.0), as JSON',
    processed TIMESTAMP COMMENT 'When the PDF was parsed'
)
COMMENT 'Raw ai_parse_document output of the call transcript PDFs';

CREATE TABLE IF NOT EXISTS llmops_dev.willis_nana.call_chunks (
    arxiv_id STRING NOT NULL COMMENT 'Call id (call_analysis.call_id)',
    chunk_id STRING NOT NULL COMMENT 'Id of the parsed text element',
    text STRING COMMENT 'Cleaned text of the chunk',
    id STRING NOT NULL COMMENT 'Primary key: call id and chunk id',
    title STRING COMMENT 'Call title',
    summary STRING,
    authors STRING COMMENT 'Speakers of the call',
    year INT COMMENT 'Call date: year',
    month INT COMMENT 'Call date: month',
    day INT COMMENT 'Call date: day',
    embedding ARRAY<FLOAT> COMMENT 'Embedding of text (1024 dimensions)',
    CONSTRAINT call_chunks_pk PRIMARY KEY (id)
)
COMMENT 'Call chunks, strategy 1: one chunk per parsed text element'
TBLPROPERTIES (delta.enableChangeDataFeed = true);

CREATE TABLE IF NOT EXISTS llmops_dev.willis_nana.call_chunks_turns (
    arxiv_id STRING NOT NULL COMMENT 'Call id (call_analysis.call_id)',
    chunk_id STRING NOT NULL COMMENT 'Position of the window in the call',
    text STRING COMMENT 'Consecutive speaker turns, one per line',
    id STRING NOT NULL COMMENT 'Primary key: call id and chunk id',
    title STRING COMMENT 'Call title',
    summary STRING,
    authors STRING COMMENT 'Speakers of the call',
    year INT COMMENT 'Call date: year',
    month INT COMMENT 'Call date: month',
    day INT COMMENT 'Call date: day',
    embedding ARRAY<FLOAT> COMMENT 'Embedding of text (1024 dimensions)',
    CONSTRAINT call_chunks_turns_pk PRIMARY KEY (id)
)
COMMENT 'Call chunks, strategy 2: windows of at most 1500 characters between speaker turns'
TBLPROPERTIES (delta.enableChangeDataFeed = true);
