"""
Tests for services/database.py
Validates SQLite initialization, collision-safe ID generation, CRUD,
parameterized queries, and CSV export using a temporary test database.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.database import (
    init_db,
    generate_grievance_id,
    insert_complaint,
    get_all_complaints,
    get_complaint_by_id,
    update_complaint_status,
    search_and_filter_complaints,
    export_complaints_csv,
    VALID_STATUSES
)


def run_database_tests():
    print("=" * 60)
    print("🧪 Running Database Integration Tests (Isolated Temp DB)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_complaints.db"

        # 1. Initialization
        init_db(test_db)
        assert test_db.is_file(), "Test database file was not created."
        print("✅ 1. Database initialization and table creation passed.")

        # 2. Collision-safe ID generation
        id1 = generate_grievance_id(test_db)
        id2 = generate_grievance_id(test_db)
        assert id1.startswith("AST-"), f"Invalid ID format: {id1}"
        assert id1 != id2, "Generated IDs collided!"
        print(f"✅ 2. Unique ID generation passed ({id1}, {id2}).")

        # 3. Insert and Retrieve
        payload = {
            "citizen_name": "Ranjeet Test",
            "citizen_contact": "9999999999",
            "location": "Ward 5, Sector B",
            "complaint_text": "Water pipeline leakage near park.",
            "text_category": "WATER_SUPPLY",
            "text_confidence": 0.285,
            "text_margin": 0.082,
            "text_all_scores": {"WATER_SUPPLY": 0.285, "ROADS_INFRASTRUCTURE": 0.203},
            "text_needs_review": False,
            "detected_classes": ["Water"],
            "detected_objects_count": 1,
            "priority": "Medium"
        }
        cid = insert_complaint(payload, test_db)
        assert cid.startswith("AST-"), "Returned CID did not match format."

        rec = get_complaint_by_id(cid, test_db)
        assert rec is not None, f"Could not retrieve complaint with ID {cid}"
        assert rec["citizen_name"] == "Ranjeet Test"
        assert rec["text_category"] == "WATER_SUPPLY"
        assert rec["status"] == "New"
        print(f"✅ 3. Record insertion and retrieval by ID passed ({cid}).")

        # 4. Status Workflow Transition
        for next_status in ["Under Review", "In Progress", "Resolved"]:
            ok = update_complaint_status(cid, next_status, admin_notes=f"Moved to {next_status}", db_path=test_db)
            assert ok, f"Failed updating status to {next_status}"
            updated = get_complaint_by_id(cid, test_db)
            assert updated["status"] == next_status
            assert f"Moved to {next_status}" in updated["admin_notes"]
        print("✅ 4. Status workflow transitions (New -> Under Review -> In Progress -> Resolved) passed.")

        # 5. Search and Parameterized Filtering
        # Insert a second record flagged for review
        payload2 = {
            "citizen_name": "Review Case",
            "location": "MG Road",
            "complaint_text": "Pole wire sparking dangerously.",
            "text_category": "ELECTRICAL_SAFETY",
            "text_needs_review": True
        }
        cid2 = insert_complaint(payload2, test_db)

        # Search by text
        search_res = search_and_filter_complaints(search_query="sparking", db_path=test_db)
        assert len(search_res) == 1 and search_res[0]["complaint_id"] == cid2
        print("✅ 5a. Parameterized text search passed.")

        # Filter by needs_review flag
        review_res = search_and_filter_complaints(needs_review=True, db_path=test_db)
        assert len(review_res) == 1 and review_res[0]["complaint_id"] == cid2
        print("✅ 5b. Filtering by review flag passed.")

        # 6. CSV Export
        all_recs = get_all_complaints(db_path=test_db)
        csv_data = export_complaints_csv(all_recs)
        assert "Grievance ID,Created At,Citizen Name" in csv_data
        assert cid in csv_data
        assert cid2 in csv_data
        print("✅ 6. CSV export formatting passed.")

        # 7. Persistence Across Connection Close
        # Re-opening new connection to test_db
        reopened_recs = get_all_complaints(db_path=test_db)
        assert len(reopened_recs) == 2, "Records did not persist across connections!"
        print("✅ 7. SQLite persistence across connection close passed.")

    print("\n🎉 ALL DATABASE INTEGRATION TESTS PASSED!\n")


if __name__ == "__main__":
    run_database_tests()
