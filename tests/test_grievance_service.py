"""
Tests for services/grievance_service.py
Validates the GrievanceAnalyzer class across text analysis, image detection,
error handling, and multimodal processing.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.grievance_service import GrievanceAnalyzer
from services.database import get_complaint_by_id


def run_service_tests():
    print("=" * 60)
    print("🧪 Running Grievance Service Integration Tests")
    print("=" * 60)

    analyzer = GrievanceAnalyzer()

    # 1. Text Analysis - English
    res_en = analyzer.analyze_text("Garbage has not been collected from our street for three days.")
    assert res_en["success"] is True, "English text analysis failed."
    assert res_en["category"] == "GARBAGE_SANITATION", f"Unexpected category: {res_en['category']}"
    assert "all_scores" in res_en, "Missing all_scores in response."
    print("✅ 1. English text complaint analysis passed (GARBAGE_SANITATION).")

    # 2. Text Analysis - Hindi
    res_hi = analyzer.analyze_text("हमारी सड़क पर बहुत बड़े गड्ढे हैं।")
    assert res_hi["success"] is True, "Hindi text analysis failed."
    assert res_hi["category"] == "ROADS_INFRASTRUCTURE", f"Unexpected category: {res_hi['category']}"
    print("✅ 2. Hindi text complaint analysis passed (ROADS_INFRASTRUCTURE).")

    # 3. Text Analysis - Error handling on empty text
    res_empty = analyzer.analyze_text("   ")
    assert res_empty["success"] is False, "Empty text should fail gracefully."
    assert "error" in res_empty, "Missing error message on empty text."
    print("✅ 3. Empty text handling passed (graceful validation failure).")

    # 4. Image Analysis - Valid Image
    test_img = PROJECT_ROOT / "sample_images" / "potholes.webp"
    if test_img.is_file():
        res_img = analyzer.analyze_image(test_img)
        assert res_img["success"] is True, "Image analysis failed."
        assert res_img["has_detections"] is True, "Expected detections on potholes.webp"
        assert "Road" in res_img["detected_classes"], "Expected 'Road' class in detected classes."
        print(f"✅ 4. Image detection passed ({res_img['detection_count']} detections, classes: {res_img['detected_classes']}).")
    else:
        print("⚠️ 4. Image test skipped (sample_images/potholes.webp not found).")

    # 5. Image Analysis - Missing File
    res_missing_img = analyzer.analyze_image("non_existent_image_12345.jpg")
    assert res_missing_img["success"] is False, "Missing image should return success=False."
    assert "not found" in res_missing_img["error"], "Expected 'not found' in error message."
    print("✅ 5. Missing image error handling passed.")

    # 6. Multimodal Intake & Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_intake.db"
        rec = analyzer.process_grievance(
            complaint_text="Paani ki pipeline phat gayi hai sadak par.",
            image_path=str(test_img) if test_img.is_file() else None,
            citizen_name="Asha Devi",
            location="Ward 14",
            save_to_db=True,
            db_path=test_db
        )
        assert rec["complaint_id"].startswith("AST-"), f"Unexpected ID format: {rec['complaint_id']}"
        saved = get_complaint_by_id(rec["complaint_id"], db_path=test_db)
        assert saved is not None, "Failed to retrieve persisted multimodal complaint."
        assert saved["citizen_name"] == "Asha Devi"
        print(f"✅ 6. Full multimodal intake & persistence passed ({rec['complaint_id']}).")

    print("\n🎉 ALL GRIEVANCE SERVICE TESTS PASSED!\n")


if __name__ == "__main__":
    run_service_tests()
