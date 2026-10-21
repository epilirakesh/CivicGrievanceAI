"""
CivicGrievanceAI - ASTRA SQLite Database Service
Manages persistence of civic grievances, multimodal analysis outputs,
and administrative case management workflows.
"""

import csv
import io
import json
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "civic_complaints.db"

VALID_STATUSES = ["New", "Under Review", "In Progress", "Resolved"]
VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]


def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def migrate_schema(conn: sqlite3.Connection):
    """
    Safely adds missing columns to existing tables in an idempotent,
    backward-compatible manner without dropping tables or losing records.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(complaints)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    # Mapping of expected columns to their safe ALTER TABLE statements
    expected_columns = {
        "admin_notes": "ALTER TABLE complaints ADD COLUMN admin_notes TEXT DEFAULT ''",
        "audio_path": "ALTER TABLE complaints ADD COLUMN audio_path TEXT DEFAULT ''",
        "audio_language": "ALTER TABLE complaints ADD COLUMN audio_language TEXT DEFAULT ''",
        "intake_mode": "ALTER TABLE complaints ADD COLUMN intake_mode TEXT DEFAULT 'Text'",
        "assigned_to": "ALTER TABLE complaints ADD COLUMN assigned_to TEXT DEFAULT ''",
        "assigned_department": "ALTER TABLE complaints ADD COLUMN assigned_department TEXT DEFAULT ''",
        "ward": "ALTER TABLE complaints ADD COLUMN ward TEXT DEFAULT ''",
        "landmark": "ALTER TABLE complaints ADD COLUMN landmark TEXT DEFAULT ''",
        "address": "ALTER TABLE complaints ADD COLUMN address TEXT DEFAULT ''",
        "latitude": "ALTER TABLE complaints ADD COLUMN latitude REAL DEFAULT NULL",
        "longitude": "ALTER TABLE complaints ADD COLUMN longitude REAL DEFAULT NULL",
        "location_accuracy": "ALTER TABLE complaints ADD COLUMN location_accuracy REAL DEFAULT NULL",
        "location_timestamp": "ALTER TABLE complaints ADD COLUMN location_timestamp TEXT DEFAULT ''",
        "location_source": "ALTER TABLE complaints ADD COLUMN location_source TEXT DEFAULT ''",
    }

    for col_name, alter_stmt in expected_columns.items():
        if col_name not in existing_cols:
            cursor.execute(alter_stmt)
    conn.commit()


def init_db(db_path: Optional[Union[str, Path]] = None):
    """Initializes the complaints table, indexes, and applies schema migrations."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS complaints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                complaint_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                citizen_name TEXT,
                citizen_contact TEXT,
                location TEXT,
                complaint_text TEXT,
                text_category TEXT,
                text_confidence REAL,
                text_margin REAL,
                text_all_scores_json TEXT,
                text_needs_review INTEGER DEFAULT 0,
                image_path TEXT,
                annotated_image_path TEXT,
                detected_classes_json TEXT,
                detected_objects_count INTEGER DEFAULT 0,
                priority TEXT DEFAULT 'Medium',
                status TEXT DEFAULT 'New',
                admin_notes TEXT DEFAULT '',
                audio_path TEXT DEFAULT '',
                audio_language TEXT DEFAULT '',
                intake_mode TEXT DEFAULT 'Text',
                assigned_to TEXT DEFAULT '',
                assigned_department TEXT DEFAULT '',
                ward TEXT DEFAULT '',
                landmark TEXT DEFAULT '',
                address TEXT DEFAULT '',
                latitude REAL DEFAULT NULL,
                longitude REAL DEFAULT NULL,
                location_accuracy REAL DEFAULT NULL,
                location_timestamp TEXT DEFAULT '',
                location_source TEXT DEFAULT ''
            )
        """)
        migrate_schema(conn)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON complaints(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON complaints(text_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON complaints(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_needs_review ON complaints(text_needs_review)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_priority ON complaints(priority)")
        conn.commit()


def generate_grievance_id(db_path: Optional[Union[str, Path]] = None) -> str:
    """
    Generates a unique, collision-safe grievance ID in the format:
    AST-YYYYMMDD-XXXX (e.g., AST-20261017-4B9E)
    """
    date_str = datetime.now().strftime("%Y%m%d")
    for _ in range(10):
        suffix = secrets.token_hex(2).upper()
        candidate = f"AST-{date_str}-{suffix}"
        # Check collision
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM complaints WHERE complaint_id = ?", (candidate,))
            if not cursor.fetchone():
                return candidate
    # Fallback with timestamp microsecond
    return f"AST-{date_str}-{datetime.now().strftime('%f')[:4].upper()}"


def insert_complaint(data: Dict[str, Any], db_path: Optional[Union[str, Path]] = None) -> str:
    """Inserts a new grievance record into SQLite and returns the complaint_id."""
    init_db(db_path)
    cid = data.get("complaint_id") or generate_grievance_id(db_path)

    # Initial case status workflow: 'Under Review' if flagged by AI, else 'New'
    initial_status = data.get("status")
    if not initial_status:
        initial_status = "Under Review" if data.get("text_needs_review") else "New"

    lat = data.get("latitude")
    lon = data.get("longitude")
    try:
        lat = float(lat) if lat is not None and str(lat).strip() != "" else None
    except (ValueError, TypeError):
        lat = None
    try:
        lon = float(lon) if lon is not None and str(lon).strip() != "" else None
    except (ValueError, TypeError):
        lon = None

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO complaints (
                complaint_id,
                citizen_name,
                citizen_contact,
                location,
                complaint_text,
                text_category,
                text_confidence,
                text_margin,
                text_all_scores_json,
                text_needs_review,
                image_path,
                annotated_image_path,
                detected_classes_json,
                detected_objects_count,
                priority,
                status,
                admin_notes,
                audio_path,
                audio_language,
                intake_mode,
                assigned_to,
                assigned_department,
                ward,
                landmark,
                address,
                latitude,
                longitude,
                location_accuracy,
                location_timestamp,
                location_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cid,
            data.get("citizen_name", "Anonymous"),
            data.get("citizen_contact", ""),
            data.get("location", "Unspecified"),
            data.get("complaint_text", ""),
            data.get("text_category"),
            data.get("text_confidence"),
            data.get("text_margin"),
            json.dumps(data.get("text_all_scores", {})),
            1 if data.get("text_needs_review") else 0,
            data.get("image_path"),
            data.get("annotated_image_path"),
            json.dumps(data.get("detected_classes", [])),
            data.get("detected_objects_count", 0),
            data.get("priority", "Medium"),
            initial_status,
            data.get("admin_notes", ""),
            data.get("audio_path", ""),
            data.get("audio_language", ""),
            data.get("intake_mode", "Text"),
            data.get("assigned_to", ""),
            data.get("assigned_department", data.get("text_category") or ""),
            data.get("ward", ""),
            data.get("landmark", ""),
            data.get("address", data.get("location", "")),
            lat,
            lon,
            data.get("location_accuracy"),
            data.get("location_timestamp", ""),
            data.get("location_source", "Manual")
        ))
        conn.commit()
    return cid


def _parse_row(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    d["text_all_scores"] = json.loads(d.get("text_all_scores_json") or "{}")
    d["detected_classes"] = json.loads(d.get("detected_classes_json") or "[]")
    d["text_needs_review"] = bool(d.get("text_needs_review", 0))
    d["audio_path"] = d.get("audio_path") or ""
    d["audio_language"] = d.get("audio_language") or ""
    d["intake_mode"] = d.get("intake_mode") or "Text"
    d["assigned_to"] = d.get("assigned_to") or ""
    d["assigned_department"] = d.get("assigned_department") or (d.get("text_category") or "")
    d["ward"] = d.get("ward") or ""
    d["landmark"] = d.get("landmark") or ""
    d["address"] = d.get("address") or d.get("location") or ""
    d["latitude"] = d.get("latitude")
    d["longitude"] = d.get("longitude")
    d["location_accuracy"] = d.get("location_accuracy")
    d["location_timestamp"] = d.get("location_timestamp") or ""
    d["location_source"] = d.get("location_source") or ""
    return d


def get_all_complaints(limit: int = 500, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Retrieves all complaints ordered by most recent first."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM complaints ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [_parse_row(r) for r in rows]


def get_complaint_by_id(complaint_id: str, db_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single complaint by its unique complaint ID."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,))
        row = cursor.fetchone()
        return _parse_row(row) if row else None


def update_complaint_status(
    complaint_id: str,
    new_status: str,
    admin_notes: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> bool:
    """Updates case status and optional administrative dispatch notes."""
    init_db(db_path)
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Allowed: {VALID_STATUSES}")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if admin_notes is not None:
            cursor.execute(
                "UPDATE complaints SET status = ?, admin_notes = ? WHERE complaint_id = ?",
                (new_status, admin_notes, complaint_id)
            )
        else:
            cursor.execute(
                "UPDATE complaints SET status = ? WHERE complaint_id = ?",
                (new_status, complaint_id)
            )
        conn.commit()
        return cursor.rowcount > 0


def update_complaint_assignment(
    complaint_id: str,
    assigned_to: str,
    assigned_department: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> bool:
    """Updates officer or department assignment for a complaint."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if assigned_department:
            cursor.execute(
                "UPDATE complaints SET assigned_to = ?, assigned_department = ? WHERE complaint_id = ?",
                (assigned_to, assigned_department, complaint_id)
            )
        else:
            cursor.execute(
                "UPDATE complaints SET assigned_to = ? WHERE complaint_id = ?",
                (assigned_to, complaint_id)
            )
        conn.commit()
        return cursor.rowcount > 0


def search_and_filter_complaints(
    search_query: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    needs_review: Optional[bool] = None,
    limit: int = 500,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Performs parameterized search across grievance ID and complaint text with filters."""
    init_db(db_path)
    query_parts = ["SELECT * FROM complaints WHERE 1=1"]
    params: List[Any] = []

    if search_query and search_query.strip():
        term = f"%{search_query.strip()}%"
        query_parts.append("AND (complaint_id LIKE ? OR complaint_text LIKE ? OR location LIKE ?)")
        params.extend([term, term, term])

    if status and status != "All":
        query_parts.append("AND status = ?")
        params.append(status)

    if category and category != "All":
        query_parts.append("AND text_category = ?")
        params.append(category)

    if needs_review is not None:
        query_parts.append("AND text_needs_review = ?")
        params.append(1 if needs_review else 0)

    query_parts.append("ORDER BY id DESC LIMIT ?")
    params.append(limit)

    full_sql = " ".join(query_parts)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(full_sql, params)
        rows = cursor.fetchall()
        return [_parse_row(r) for r in rows]


def export_complaints_csv(complaints: List[Dict[str, Any]]) -> str:
    """Generates an RFC-4180 compliant CSV string of grievance records."""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Grievance ID",
        "Created At",
        "Citizen Name",
        "Location",
        "Intake Mode",
        "Audio Language",
        "Complaint Text",
        "Text Category",
        "Text Confidence",
        "Score Margin",
        "Needs Review",
        "Detected Classes",
        "Object Count",
        "Priority",
        "Status",
        "Admin Notes"
    ]
    writer.writerow(headers)

    for c in complaints:
        writer.writerow([
            c.get("complaint_id", ""),
            c.get("created_at", ""),
            c.get("citizen_name", ""),
            c.get("location", ""),
            c.get("intake_mode", "Text"),
            c.get("audio_language", ""),
            c.get("complaint_text", ""),
            c.get("text_category", ""),
            f"{c.get('text_confidence', 0.0):.4f}" if c.get("text_confidence") is not None else "",
            f"{c.get('text_margin', 0.0):.4f}" if c.get("text_margin") is not None else "",
            "YES" if c.get("text_needs_review") else "NO",
            ", ".join(c.get("detected_classes", [])),
            c.get("detected_objects_count", 0),
            c.get("priority", "Medium"),
            c.get("status", "New"),
            c.get("admin_notes", "")
        ])

    return output.getvalue()
