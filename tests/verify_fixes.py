"""
Comprehensive Verification Script for ASTRA Fixes
Verifies:
1. Image inference on real pothole image produces detections from that image.
2. A second different image does NOT reuse the first image's annotated result.
3. Original and annotated images remain correctly associated.
4. Real complaint AST-20261017-6449 is preserved in SQLite.
5. Demo records are removed safely.
6. Text-only registration and status updates work.
7. Audio model audit.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import (
    default_analyzer,
    get_all_complaints,
    get_complaint_by_id,
    update_complaint_status,
    insert_complaint
)


def run_verification():
    print("=" * 70)
    print("🚀 RUNNING ASTRA VERIFICATION CHECKS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Check 1: Real Pothole Image (Parel Webp from AST-20261017-6449)
    # -------------------------------------------------------------
    parel_img = PROJECT_ROOT / "test_outputs" / "dashboard_uploads" / "upload_20261017023054_parel.webp"
    assert parel_img.is_file(), f"Target image not found: {parel_img}"

    res1 = default_analyzer.analyze_image(parel_img)
    assert res1["success"] is True, "Inference on parel.webp failed."
    assert res1["detection_count"] == 6, f"Expected 6 detections on parel.webp, got {res1['detection_count']}"
    assert "annotated_upload_20261017023054_parel.jpg" in res1["annotated_image_path"]
    assert Path(res1["annotated_image_path"]).is_file(), "Annotated image not found on disk."
    print("✅ Check 1 PASSED: Real uploaded pothole image produced 6 detections on parel.webp, bound to its own annotated file.")

    # -------------------------------------------------------------
    # Check 2: Second Different Image (wire.avif) Does NOT Reuse First Image
    # -------------------------------------------------------------
    wire_img = PROJECT_ROOT / "sample_images" / "wire.avif"
    assert wire_img.is_file(), f"Wire image not found: {wire_img}"

    res2 = default_analyzer.analyze_image(wire_img)
    assert res2["success"] is True, "Inference on wire.avif failed."
    assert res2["detection_count"] == 1, f"Expected 1 detection on wire.avif, got {res2['detection_count']}"
    assert res2["detected_classes"] == ["Electricity"], f"Expected Electricity, got {res2['detected_classes']}"
    assert "annotated_wire.jpg" in res2["annotated_image_path"]
    assert res2["annotated_image_path"] != res1["annotated_image_path"], "Result paths collided!"
    print("✅ Check 2 PASSED: Second image (wire.avif) produced 1 Electricity detection, completely distinct from first image.")

    # -------------------------------------------------------------
    # Check 3 & 4: AST-20261017-6449 Preserved & Correctly Associated
    # -------------------------------------------------------------
    case_6449 = get_complaint_by_id("AST-20261017-6449")
    assert case_6449 is not None, "Real complaint AST-20261017-6449 missing from SQLite!"
    assert case_6449["citizen_name"] == "Shubham Sharma"
    assert "Parel" in case_6449["location"]
    assert Path(case_6449["image_path"]).is_file(), "Original uploaded image missing on disk."
    assert Path(case_6449["annotated_image_path"]).is_file(), "Annotated image missing on disk."
    assert "parel" in case_6449["annotated_image_path"].lower(), f"Annotated image is not parel: {case_6449['annotated_image_path']}"
    print(f"✅ Check 3 & 4 PASSED: Real complaint AST-20261017-6449 is preserved with valid image references on disk.")

    # -------------------------------------------------------------
    # Check 5: Demo / Synthetic Test Records Removed
    # -------------------------------------------------------------
    all_recs = get_all_complaints()
    cids = [c["complaint_id"] for c in all_recs]
    print(f"Current live records in DB: {cids}")
    assert "AST-20261017-6449" in cids
    # Verify synthetic script records are gone
    assert "CGC-20261017015826" not in cids
    assert "CGC-20261017015930" not in cids
    assert "CGC-20261017015933" not in cids
    assert "AST-20261017-C422" not in cids
    assert "AST-20261017-8DBF" not in cids
    print("✅ Check 5 PASSED: Synthetic script test records removed; real grievance preserved.")

    # -------------------------------------------------------------
    # Check 6: Text-Only Registration & Status Updates
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_flow.db"
        text_case = default_analyzer.process_grievance(
            complaint_text="Garbage pile overflowing near market square.",
            citizen_name="Test User",
            location="Market Square",
            save_to_db=True,
            db_path=test_db
        )
        assert text_case["complaint_id"].startswith("AST-")
        assert text_case["text_category"] == "GARBAGE_SANITATION"

        # Update status
        ok = update_complaint_status(text_case["complaint_id"], "In Progress", admin_notes="Assigned to Sanitation Unit 4", db_path=test_db)
        assert ok is True
        fetched = get_complaint_by_id(text_case["complaint_id"], db_path=test_db)
        assert fetched["status"] == "In Progress"
        assert fetched["admin_notes"] == "Assigned to Sanitation Unit 4"
        print("✅ Check 6 PASSED: Text-only registration and status update workflow verified.")

    # -------------------------------------------------------------
    # Check 7: Audio Model Audit
    # -------------------------------------------------------------
    missing_audio_deps = []
    for pkg in ["soundfile", "transformers", "torchaudio", "onnxruntime"]:
        try:
            __import__(pkg)
        except ImportError:
            missing_audio_deps.append(pkg)

    print(f"Audio dependencies status: Missing={missing_audio_deps}")
    assert len(missing_audio_deps) == 0, f"Expected all audio dependencies to be installed, missing: {missing_audio_deps}"
    print("✅ Check 7 PASSED: Audio dependencies (soundfile, transformers, torchaudio, onnxruntime) confirmed installed and available.")

    print("\n🎉 ALL 7 VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    run_verification()
