"""Gait records: one row per cow per daily run.

BigQueryStore is the real store. InMemoryStore backs tests and --dry-run.
Rows are appended with a load job rather than streaming inserts, so it also works on
BigQuery sandbox / free-tier projects, where streaming is not available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

STATUSES = (
    "scored",              # Gemini gave a trusted 1-5 score
    "rules_normal",        # rule screen found nothing; Gemini not called, no score
    "insufficient_data",   # pose/gait unusable (cow not walking, too few frames...)
    "flagged_no_frame",    # rules flagged but no clean frame to send
    "gemini_rejected",     # Gemini answered but failed the trust gate (quality / range / JSON)
    "gemini_error",        # call failed
    "gemini_skipped",      # rules flagged, Gemini disabled (dry run)
)


@dataclass
class GaitRecord:
    record_id: str
    cow_name: str
    recorded_at: datetime              # UTC
    status: str
    clip_path: str
    n_valid_frames: int
    rule_flag: bool
    rule_reasons: str = ""
    head_bob: Optional[float] = None
    hind_asymmetry: Optional[float] = None
    back_arch: Optional[float] = None
    frame_time_s: Optional[float] = None
    gemini_called: bool = False
    gait_score: Optional[int] = None   # only ever set for status == "scored"
    gemini_confidence: Optional[float] = None
    gemini_rationale: str = ""

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row["recorded_at"] = self.recorded_at.isoformat()
        return row


class Store(Protocol):
    def add(self, record: GaitRecord) -> None: ...
    def history(self, cow_name: str, limit: int = 30) -> List[Dict[str, Any]]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.rows: List[GaitRecord] = []

    def add(self, record: GaitRecord) -> None:
        self.rows.append(record)

    def history(self, cow_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        mine = [r for r in self.rows if r.cow_name == cow_name]
        mine.sort(key=lambda r: r.recorded_at, reverse=True)
        return [r.to_row() for r in mine[:limit]]


SCHEMA = [
    ("record_id", "STRING", "REQUIRED"),
    ("cow_name", "STRING", "REQUIRED"),
    ("recorded_at", "TIMESTAMP", "REQUIRED"),
    ("status", "STRING", "REQUIRED"),
    ("clip_path", "STRING", "NULLABLE"),
    ("n_valid_frames", "INTEGER", "NULLABLE"),
    ("rule_flag", "BOOLEAN", "NULLABLE"),
    ("rule_reasons", "STRING", "NULLABLE"),
    ("head_bob", "FLOAT", "NULLABLE"),
    ("hind_asymmetry", "FLOAT", "NULLABLE"),
    ("back_arch", "FLOAT", "NULLABLE"),
    ("frame_time_s", "FLOAT", "NULLABLE"),
    ("gemini_called", "BOOLEAN", "NULLABLE"),
    ("gait_score", "INTEGER", "NULLABLE"),
    ("gemini_confidence", "FLOAT", "NULLABLE"),
    ("gemini_rationale", "STRING", "NULLABLE"),
]


class BigQueryStore:
    def __init__(
        self,
        project: str,
        dataset: str = "cattle_monitor",
        table: str = "gait_records",
        location: str = "US",
        client: Any = None,
    ) -> None:
        from google.cloud import bigquery  # lazy

        self._bq = bigquery
        self.client = client or bigquery.Client(project=project, location=location)
        self.table_id = f"{project}.{dataset}.{table}"
        self.dataset_id = f"{project}.{dataset}"
        self.location = location

    def ensure_table(self) -> None:
        """Create the dataset and table if missing. Call once; it is a cloud write."""
        ds = self._bq.Dataset(self.dataset_id)
        ds.location = self.location
        self.client.create_dataset(ds, exists_ok=True)
        table = self._bq.Table(
            self.table_id, schema=[self._bq.SchemaField(n, t, m) for n, t, m in SCHEMA]
        )
        table.time_partitioning = self._bq.TimePartitioning(field="recorded_at")
        table.clustering_fields = ["cow_name"]
        self.client.create_table(table, exists_ok=True)

    def add(self, record: GaitRecord) -> None:
        job = self.client.load_table_from_json(
            [record.to_row()],
            self.table_id,
            job_config=self._bq.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema=[self._bq.SchemaField(n, t, m) for n, t, m in SCHEMA],
            ),
        )
        job.result()

    def history(self, cow_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        q = f"""SELECT * FROM `{self.table_id}`
                WHERE cow_name = @cow ORDER BY recorded_at DESC LIMIT @n"""
        cfg = self._bq.QueryJobConfig(
            query_parameters=[
                self._bq.ScalarQueryParameter("cow", "STRING", cow_name),
                self._bq.ScalarQueryParameter("n", "INT64", limit),
            ]
        )
        return [dict(r) for r in self.client.query(q, job_config=cfg).result()]
