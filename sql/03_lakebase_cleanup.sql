-- Cleanup on the Lakebase side. Run in the Lakebase SQL editor (project lb-llmops-willis-nana, branch production),
-- connected to the lb-llmops-willis-nana database (project_config.yml: lakebase_database), after deleting the synced
-- tables (01_cleanup.sql, step 1). Deleting a synced table leaves its Postgres table behind: drop it here.

DROP TABLE IF EXISTS willis_nana.call_chunks_synced;
DROP TABLE IF EXISTS willis_nana.call_chunks_turns_synced;
DROP TABLE IF EXISTS willis_nana.call_chunks_v2_synced;
DROP TABLE IF EXISTS willis_nana.call_chunks_turns_v2_synced;
