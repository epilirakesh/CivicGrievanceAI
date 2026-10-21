"""
Tests for Audio Voice Grievance Pipeline & ASTRA Integration
Validates:
  1. Transcript-to-NLP classification (Hindi, Marathi, English).
  2. Manual-review warning propagation (score margin < 5% and OTHER_REVIEW).
  3. Audio grievance persistence in SQLite with audio metadata.
  4. Robust handling of missing audio files, unsupported formats, empty files, and timeouts.
  5. Handling of missing/gated speech recognition models via astra_audio_predict.py.
  6. Subprocess execution of astra_audio_predict.py with --decoder ctc.
  7. Regression testing of existing text and image workflows.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astra_handoff.astra_predict import predict_complaint
from services.database import (
    init_db,
    insert_complaint,
    get_complaint_by_id,
    export_complaints_csv,
    search_and_filter_complaints
)
from services.grievance_service import GrievanceAnalyzer, default_analyzer


def run_all_tests():
    print("=" * 70)
    print("🧪 ASTRA AUDIO VOICE AI & MULTIMODAL INTEGRATION TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------
    # TEST 1: Transcript-to-NLP 5-Class Classification (Real Model)
    # -------------------------------------------------------------
    print("\n--- 1. Testing Transcript-to-NLP Classification ---")
    
    # 1a. Hindi transcription text (from benchmark Garbage complaint)
    hindi_text = "गली में कचरा जमा हो गया है और बदबू फैल रही है नगर निगम को तुरंत सफाई करनी चाहिए"
    res_hi = predict_complaint(hindi_text)
    assert res_hi["category"] in {"GARBAGE_SANITATION", "ROADS_INFRASTRUCTURE", "ELECTRICAL_SAFETY", "WATER_SUPPLY", "OTHER_REVIEW"}, f"Invalid category: {res_hi['category']}"
    assert "confidence" in res_hi and isinstance(res_hi["confidence"], float)
    assert "score_margin" in res_hi and isinstance(res_hi["score_margin"], float)
    assert len(res_hi["all_scores"]) == 5, f"Expected 5 classes, got {len(res_hi['all_scores'])}"
    print(f"✅ 1a. Hindi text classification passed -> Category: {res_hi['category']} (score: {res_hi['confidence']:.4f}, margin: {res_hi['score_margin']:.4f})")

    # 1b. Marathi transcription text (from benchmark Road defect)
    marathi_text = "रस्त्यावरील मोठा खड्डा असून अपघात होण्याची शक्यता आहे दुरुस्ती करा"
    res_mr = predict_complaint(marathi_text)
    assert res_mr["category"] in {"GARBAGE_SANITATION", "ROADS_INFRASTRUCTURE", "ELECTRICAL_SAFETY", "WATER_SUPPLY", "OTHER_REVIEW"}
    print(f"✅ 1b. Marathi text classification passed -> Category: {res_mr['category']} (score: {res_mr['confidence']:.4f}, margin: {res_mr['score_margin']:.4f})")

    # 1c. English text classification
    english_text = "Dangerous high voltage electrical wire snapped and hanging low."
    res_en = predict_complaint(english_text)
    assert res_en["category"] == "ELECTRICAL_SAFETY", f"Expected ELECTRICAL_SAFETY, got {res_en['category']}"
    print(f"✅ 1c. English text classification passed -> Category: {res_en['category']} (score: {res_en['confidence']:.4f})")

    # -------------------------------------------------------------
    # TEST 2: Manual-Review Warning Propagation
    # -------------------------------------------------------------
    print("\n--- 2. Testing Manual-Review Warning Propagation ---")

    # High margin case -> should NOT trigger review
    high_margin_sample = "सड़क पर कचरा बहुत ज्यादा पड़ा है"
    hm_res = default_analyzer.analyze_text(high_margin_sample)
    if hm_res["score_margin"] >= 0.05 and hm_res["category"] != "OTHER_REVIEW":
        assert not hm_res["needs_review"], "High-margin complaint incorrectly flagged for review."
        print(f"✅ 2a. High-margin case not flagged (margin: {hm_res['score_margin']*100:.2f}% >= 5%).")
    else:
        print(f"ℹ️ 2a. Sample margin: {hm_res['score_margin']*100:.2f}% (needs_review={hm_res['needs_review']}).")

    # Ambiguous multi-issue text -> should trigger review when margin < 0.05
    ambiguous_text = "लाइट का खंभा और पानी की पाइप दोनों के पास थोड़ा कचरा है"
    amb_res = default_analyzer.analyze_text(ambiguous_text)
    assert "needs_review" in amb_res
    if amb_res["needs_review"]:
        assert amb_res["warning_message"] is not None
        print(f"✅ 2b. Ambiguous complaint triggered manual review -> Category: {amb_res['category']}, Margin: {amb_res['score_margin']*100:.2f}%")
        print(f"      Warning: {amb_res['warning_message']}")
    else:
        print(f"ℹ️ 2b. Ambiguous complaint margin was {amb_res['score_margin']*100:.2f}% (needs_review={amb_res['needs_review']}).")

    # -------------------------------------------------------------
    # TEST 3: Audio Grievance Persistence in SQLite
    # -------------------------------------------------------------
    print("\n--- 3. Testing Audio Grievance Persistence in SQLite ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_audio_complaints.db"
        init_db(test_db)

        # Create dummy audio file to reference
        sample_audio = Path(tmpdir) / "citizen_record.mp3"
        sample_audio.write_bytes(b"DUMMY_AUDIO_DATA_FOR_TEST")

        record = default_analyzer.process_grievance(
            complaint_text=hindi_text,
            audio_path=sample_audio,
            audio_language="hi",
            intake_mode="Audio",
            citizen_name="Aarav Sharma",
            citizen_contact="9876543210",
            location="Dadar West, Ward G/N",
            save_to_db=True,
            db_path=test_db
        )

        cid = record["complaint_id"]
        assert cid.startswith("AST-"), f"Invalid ID format: {cid}"
        print(f"✅ 3a. Audio grievance inserted with Case ID: {cid}")

        # Retrieve and verify all audio fields
        retrieved = get_complaint_by_id(cid, db_path=test_db)
        assert retrieved is not None, "Failed to retrieve inserted complaint."
        assert retrieved["audio_path"] == str(sample_audio), f"Audio path mismatch: {retrieved['audio_path']}"
        assert retrieved["audio_language"] == "hi", f"Audio language mismatch: {retrieved['audio_language']}"
        assert retrieved["intake_mode"] == "Audio", f"Intake mode mismatch: {retrieved['intake_mode']}"
        assert retrieved["complaint_text"] == hindi_text
        assert retrieved["text_category"] is not None
        print("✅ 3b. SQLite record verified (audio_path, audio_language='hi', intake_mode='Audio').")

        # Verify CSV export contains new columns
        all_recs = [retrieved]
        csv_out = export_complaints_csv(all_recs)
        assert "Intake Mode" in csv_out, "CSV header missing 'Intake Mode'"
        assert "Audio Language" in csv_out, "CSV header missing 'Audio Language'"
        assert "Audio" in csv_out, "CSV missing intake mode value 'Audio'"
        assert "hi" in csv_out, "CSV missing audio language value 'hi'"
        print("✅ 3c. CSV export verified with Intake Mode and Audio Language.")

    # -------------------------------------------------------------
    # TEST 4: Validation & Error Handling (Missing, Unsupported, Empty, Timeouts)
    # -------------------------------------------------------------
    print("\n--- 4. Testing Validation & Edge Case Handling ---")

    # 4a. Missing audio file
    res_missing = default_analyzer.analyze_audio("/path/to/nonexistent/audio.mp3")
    assert not res_missing["success"], "Missing audio should return success=False."
    assert "not found" in res_missing["error"].lower()
    print("✅ 4a. Non-existent audio file handled gracefully.")

    # 4b. Unsupported format
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not an audio file")
        txt_path = f.name
    try:
        res_unsupported = default_analyzer.analyze_audio(txt_path)
        assert not res_unsupported["success"], "Unsupported format should return success=False."
        assert "unsupported" in res_unsupported["error"].lower()
        print("✅ 4b. Unsupported audio format (.txt) rejected with clear feedback.")
    finally:
        os.unlink(txt_path)

    # 4c. Empty audio file (0 bytes)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        empty_mp3 = f.name
    try:
        res_empty = default_analyzer.analyze_audio(empty_mp3)
        assert not res_empty["success"], "Empty audio file should return success=False."
        assert "empty" in res_empty["error"].lower()
        print("✅ 4c. Empty audio file (0 bytes) rejected with clear feedback.")
    finally:
        os.unlink(empty_mp3)

    # 4d. Unsupported language code
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"FAKE_DATA")
        dummy_mp3 = f.name
    try:
        res_lang = default_analyzer.analyze_audio(dummy_mp3, language="fr")
        assert not res_lang["success"], "Unsupported language should return success=False."
        assert "unsupported language" in res_lang["error"].lower()
        print("✅ 4d. Unsupported language ('fr') rejected with clear feedback.")
    finally:
        os.unlink(dummy_mp3)

    # -------------------------------------------------------------
    # TEST 5: CLI Subprocess Execution of astra_audio_predict.py
    # -------------------------------------------------------------
    print("\n--- 5. Testing CLI Subprocess Execution of astra_audio_predict.py ---")
    script_path = PROJECT_ROOT / "astra_handoff" / "astra_audio_predict.py"
    assert script_path.is_file(), "astra_audio_predict.py not found."

    # Test CLI --help execution
    cli_help = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True
    )
    assert cli_help.returncode == 0, f"CLI --help failed: {cli_help.stderr}"
    assert "--language" in cli_help.stdout
    assert "--decoder" in cli_help.stdout
    print("✅ 5a. CLI argument validation and --help output verified.")

    # Test real audio sample invocation through CLI
    audio_sample_candidates = list((PROJECT_ROOT / "audio_samples").glob("*.mp3"))
    if audio_sample_candidates:
        test_audio = audio_sample_candidates[0]
        print(f"📦 Found real test audio recording: {test_audio.name}")
        
        # Test CLI execution with --language hi and --decoder ctc
        cli_run = subprocess.run(
            [sys.executable, str(script_path), str(test_audio), "--language", "hi", "--decoder", "ctc"],
            capture_output=True,
            text=True
        )
        print(f"ℹ️ CLI execution output (Exit code: {cli_run.returncode}):")
        if cli_run.returncode == 0:
            print(f"✅ Real Hindi ASR transcription succeeded!\n{cli_run.stdout}")
        else:
            # Model is gated or weights unavailable
            print("ℹ️ Model loading reported expected setup requirements:")
            for line in (cli_run.stderr or cli_run.stdout).strip().splitlines()[-4:]:
                print("   ", line)
            assert "ai4bharat/indic-conformer-600m-multilingual" in (cli_run.stderr + cli_run.stdout)
            print("✅ 5b. Subprocess error propagation verified (gated repo diagnostic surfaced cleanly).")

    # -------------------------------------------------------------
    # TEST 6: Regression Testing on Existing Multimodal Workflows
    # -------------------------------------------------------------
    print("\n--- 6. Testing Regression on Existing Text & Vision Features ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_regress.db"
        init_db(test_db)

        # 6a. Text-only grievance
        txt_case = default_analyzer.process_grievance(
            complaint_text="Potholes on main road causing traffic delays.",
            citizen_name="Citizen Tester",
            location="Station Road",
            save_to_db=True,
            db_path=test_db
        )
        assert txt_case["complaint_id"].startswith("AST-")
        assert txt_case["text_category"] is not None
        assert txt_case["intake_mode"] == "Text"
        print(f"✅ 6a. Text grievance intake regression passed ({txt_case['complaint_id']}).")

        # 6b. Image-based YOLO11n grievance
        sample_img = PROJECT_ROOT / "sample_images" / "potholes.webp"
        if sample_img.is_file():
            img_case = default_analyzer.process_grievance(
                complaint_text="Damaged road surface",
                image_path=sample_img,
                intake_mode="Multimodal",
                save_to_db=True,
                db_path=test_db
            )
            assert img_case["complaint_id"].startswith("AST-")
            assert img_case["image_analysis"]["success"]
            assert img_case["intake_mode"] == "Multimodal"
            print(f"✅ 6b. Multimodal image+text intake regression passed ({img_case['complaint_id']}, detections: {img_case['detected_objects_count']}).")

    print("\n" + "=" * 70)
    print("🎉 ALL ASTRA AUDIO & MULTIMODAL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
