"""Parse a call transcript PDF and chunk it into the tables that the Lakebase search reads.

PDF in the volume
   ↓ (parse_pdf)
call_parsed_docs_table (ai_parse_document JSON)
   ↓ (process_chunks)
call_chunks (one chunk per parsed text element)
call_chunks_turns (windows of consecutive speaker turns)

Both chunk tables have the columns of arxiv_chunks_table, so LakebasePgVectorManager searches them unchanged:
`arxiv_id` holds the call id, `title` the call title, `authors` the speakers and year/month/day the call date.
"""

import hashlib
import re
from collections import Counter

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, IntegerType, StringType, StructField, StructType

from arxiv_curator.config import ProjectConfig
from arxiv_curator.data_processor import DataProcessor

CHUNK_SCHEMA = StructType(
    [
        StructField("arxiv_id", StringType(), False),
        StructField("chunk_id", StringType(), False),
        StructField("text", StringType(), True),
        StructField("id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("summary", StringType(), True),
        StructField("authors", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("day", IntegerType(), True),
        StructField("embedding", ArrayType(FloatType()), True),
    ]
)


class CallProcessor:
    """Parse one call transcript PDF and write its chunks, with embeddings, to Delta tables."""

    def __init__(
        self, spark: SparkSession, config: ProjectConfig, file_name: str, workspace: WorkspaceClient | None = None
    ) -> None:
        """Initialize the processor for one PDF of the config's volume.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with the catalog, schema, volume and embedding endpoint
            file_name: Name of the PDF in the volume
            workspace: WorkspaceClient used to embed the chunks, created if omitted

        """
        self.spark = spark
        self.workspace = workspace or WorkspaceClient()
        self.cfg = config
        self.file_name = file_name
        self.pdf_path = f"/Volumes/{config.catalog}/{config.schema}/{config.volume}/{file_name}"
        # Same id as 1_sales_data_ingestion, to join with call_analysis
        self.call_id = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]
        self.call_table = f"{config.full_schema_name}.call_analysis"
        self.parsed_table = f"{config.full_schema_name}.call_parsed_docs_table"
        self.chunks_table = f"{config.full_schema_name}.call_chunks"
        self.turn_chunks_table = f"{config.full_schema_name}.call_chunks_turns"

    def parse_pdf(self) -> None:
        """Parse the PDF with ai_parse_document and store the JSON in call_parsed_docs_table."""
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                path STRING,
                parsed_content STRING,
                processed TIMESTAMP
            )
        """)
        # Replace the previous parse of the same file: each ai_parse_document call is billed
        self.spark.sql(f"DELETE FROM {self.parsed_table} WHERE path LIKE '%{self._sql_escape(self.file_name)}'")
        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                path,
                CAST(ai_parse_document(content, map('version', '2.0', 'descriptionElementTypes', '')) AS STRING),
                current_timestamp()
            FROM READ_FILES('{self._sql_escape(self.pdf_path)}', format => 'binaryFile')
        """)
        logger.info(f"Parsed {self.pdf_path} into {self.parsed_table}")

    def process_chunks(self, max_chars: int = 1500, overlap_turns: int = 1) -> None:
        """Write both chunk tables: one chunk per text element, and windows of speaker turns.

        Args:
            max_chars: Maximum characters of a window of speaker turns
            overlap_turns: Turns repeated at the start of the next window

        """
        parsed_content = self._parsed_content()
        metadata = self._metadata()

        # Strategy 1: what DataProcessor does for papers, one cleaned chunk per parsed text element
        element_chunks = [
            (chunk_id, DataProcessor._clean_chunk(content))
            for chunk_id, content in DataProcessor._extract_chunks(parsed_content)
        ]
        self._save([(i, t) for i, t in element_chunks if t], self.chunks_table, metadata)

        # Strategy 2: the text of the whole call, cut between speaker turns
        full_text = " ".join(content for _, content in DataProcessor._extract_chunks(parsed_content))
        turns = self.split_turns(full_text, metadata["speakers"])
        windows = self.group_turns(turns, max_chars, overlap_turns)
        self._save([(str(i), window) for i, window in enumerate(windows)], self.turn_chunks_table, metadata)

    @staticmethod
    def detect_speakers(text: str, min_count: int = 5) -> list[str]:
        """Find the speaker labels: 'Name:' repeated many times in the text."""
        flat = re.sub(r"\s+", " ", text)
        counts = Counter(re.findall(r"(?<![\w\-])([A-ZÀ-Ý][\w\-]*(?: [A-ZÀ-Ý][\w\-]*)?):\s", flat))
        return [name for name, count in counts.items() if count >= min_count]

    @staticmethod
    def split_turns(text: str, speakers: list[str]) -> list[str]:
        """Split a transcript into one 'Speaker: utterance' string per turn."""
        if not speakers:
            return []
        flat = re.sub(r"\s+", " ", text).strip()
        names = "|".join(re.escape(name) for name in sorted(speakers, key=len, reverse=True))
        parts = re.split(rf"(?<![\w\-])({names}):\s", flat)
        return [f"{parts[i]}: {parts[i + 1].strip()}" for i in range(1, len(parts) - 1, 2) if parts[i + 1].strip()]

    @staticmethod
    def group_turns(turns: list[str], max_chars: int, overlap_turns: int) -> list[str]:
        """Group consecutive turns into windows of at most max_chars characters.

        The last overlap_turns turns of a window start the next one, if they still fit, so that a question and its
        answer are not split apart. A single turn longer than max_chars stays whole.
        """
        windows, current = [], []
        for turn in turns:
            if current and len("\n".join([*current, turn])) > max_chars:
                windows.append("\n".join(current))
                overlap = current[-overlap_turns:] if overlap_turns else []
                current = overlap if len("\n".join([*overlap, turn])) <= max_chars else []
            current.append(turn)
        if current:
            windows.append("\n".join(current))
        return windows

    @staticmethod
    def _sql_escape(value: str) -> str:
        return value.replace("'", "''")

    def _parsed_content(self) -> str:
        row = self.spark.table(self.parsed_table).filter(F.col("path").endswith(self.file_name)).first()
        if row is None:
            raise ValueError(f"{self.file_name} is not in {self.parsed_table}: run parse_pdf first")
        return row["parsed_content"]

    def _metadata(self) -> dict:
        """Title, date and speakers of the call, from call_analysis or else from the transcript itself."""
        row = None
        if self.spark.catalog.tableExists(self.call_table):
            row = self.spark.table(self.call_table).filter(F.col("call_id") == self.call_id).first()
        text = " ".join(content for _, content in DataProcessor._extract_chunks(self._parsed_content()))
        call_date = (row["call_date"] if row else None) or ""
        year, month, day = (int(part) for part in call_date.split("-")) if call_date else (None, None, None)
        speakers = list(row["speakers"]) if row and row["speakers"] else self.detect_speakers(text)
        return {
            "title": (row["call_title"] if row else None) or self.file_name.removesuffix(".pdf"),
            "authors": ", ".join(speakers),
            "speakers": speakers,
            "year": year,
            "month": month,
            "day": day,
        }

    def _embed(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Embed the texts through the AI Gateway: the embedding endpoint is a gateway service, not ai_query-able."""
        embeddings = []
        for start in range(0, len(texts), batch_size):
            response = self.workspace.api_client.do(
                "POST",
                "/ai-gateway/mlflow/v1/embeddings",
                body={"model": self.cfg.embedding_endpoint, "input": texts[start : start + batch_size]},
            )
            embeddings.extend(item["embedding"] for item in sorted(response["data"], key=lambda item: item["index"]))
        return embeddings

    def _save(self, chunks: list[tuple[str, str]], table: str, metadata: dict) -> None:
        """Embed the chunks and replace the rows of this call in the table."""
        embeddings = self._embed([text for _, text in chunks])
        rows = [
            (
                self.call_id,
                str(chunk_id),
                text,
                f"{self.call_id}_{chunk_id}",
                metadata["title"],
                None,
                metadata["authors"],
                metadata["year"],
                metadata["month"],
                metadata["day"],
                embedding,
            )
            for (chunk_id, text), embedding in zip(chunks, embeddings, strict=True)
        ]
        df = self.spark.createDataFrame(rows, schema=CHUNK_SCHEMA)
        if self.spark.catalog.tableExists(table) and "arxiv_id" in self.spark.table(table).columns:
            self.spark.sql(f"DELETE FROM {table} WHERE arxiv_id = '{self.call_id}'")
            df.write.mode("append").option("mergeSchema", "true").saveAsTable(table)
        else:
            # No table yet, or a table with another layout (an earlier layout of the table): replace it
            df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        # Change Data Feed lets the Lakebase synced table pick up the changes
        self.spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        logger.info(f"Saved {len(rows)} chunks to {table}")
