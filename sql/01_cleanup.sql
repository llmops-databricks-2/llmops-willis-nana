-- Cleanup of the call chunk objects, to keep one version only: call_chunks and call_chunks_turns (+ their synced tables).
-- Run on Databricks (SQL editor or notebook), in order. Replace llmops_dev if you clean another environment.

-- Step 1 - Lakebase synced tables. They are NOT dropped with SQL: delete them with the CLI (or Catalog > the table >
-- Delete), which also removes their sync pipeline. Run these in a terminal first:
--   databricks postgres delete-synced-table "synced_tables/llmops_dev.willis_nana.call_chunks_synced"
--   databricks postgres delete-synced-table "synced_tables/llmops_dev.willis_nana.call_chunks_turns_synced"
--   databricks postgres delete-synced-table "synced_tables/llmops_dev.willis_nana.call_chunks_v2_synced"
--   databricks postgres delete-synced-table "synced_tables/llmops_dev.willis_nana.call_chunks_turns_v2_synced"
-- (a "not found" on call_chunks_synced is fine: it is already gone)

-- Step 2 - Delta tables: the versions 2, the temporary tables and the old layout of call_chunks
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_chunks_v2;
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_chunks_turns_v2;
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_chunks_table;
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_chunks_turns_table;
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_chunks;       -- old layout (chunk_to_retrieve...): replaced by 02_ddl.sql
DROP TABLE IF EXISTS llmops_dev.willis_nana.call_parsed_docs_table;

-- Optional - the table written by an earlier version of 3_parse_pdf, not used anymore
-- DROP TABLE IF EXISTS llmops_dev.willis_nana.parsed_documents;
