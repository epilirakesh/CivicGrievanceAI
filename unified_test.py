"""
CivicGrievanceAI - Unified Testing Script
Evaluates both core analytical modalities independently:
  1. Image-based issue detection (Prithvi's YOLO11n model)
  2. Text-based complaint classification (Shubham's ASTRA NLP model)

Requirements:
- Independent reporting of image and text results (no category merging).
- Explicit manual review alerts for low-margin text predictions.
- No new model weights or package downloads.
"""

import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
ASTRA_DIR = PROJECT_ROOT / "astra_handoff"
ASTRA_MODEL_PATH = ASTRA_DIR / "astra_baseline_model_122.joblib"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "sample_images"
OUTPUT_DIR = PROJECT_ROOT / "test_outputs" / "unified_test"


def test_image_detection(image_path: Path = None):
    """Runs YOLO11n object detection on a local test image."""
    print("\n" + "=" * 70)
    print("📷 MODULE 1: IMAGE-BASED ISSUE DETECTION (YOLO11n)")
    print("=" * 70)

    # 1. Verify YOLO model weights
    if not YOLO_MODEL_PATH.is_file():
        print(f"❌ Error: YOLO model weights not found at '{YOLO_MODEL_PATH}'.")
        print("   Please ensure 'best.pt' is placed in 'models/'.")
        return False

    # 2. Locate a local test image
    target_image = None
    if image_path and Path(image_path).is_file():
        target_image = Path(image_path)
    elif SAMPLE_IMAGES_DIR.is_dir():
        supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}
        candidates = [f for f in SAMPLE_IMAGES_DIR.iterdir() if f.suffix.lower() in supported_exts]
        if candidates:
            # Prefer potholes.webp if available, else first image
            target_image = next((f for f in candidates if "pothole" in f.name.lower()), candidates[0])

    if not target_image:
        print("⚠️  No local test image available in 'sample_images/'.")
        print("   Skipping image detection. (To test, place an image in 'sample_images/'.)")
        return False

    print(f"📦 Model File   : {YOLO_MODEL_PATH.resolve()}")
    print(f"🖼️  Test Image   : {target_image.resolve()}")
    print(f"📐 Input Size   : 768x768")
    print("-" * 70)

    try:
        from ultralytics import YOLO
        model = YOLO(str(YOLO_MODEL_PATH))

        results = model.predict(
            source=str(target_image),
            imgsz=768,
            conf=0.25,
            save=True,
            project=str(PROJECT_ROOT / "test_outputs"),
            name="unified_test",
            exist_ok=True,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        print(f"🏷️  Configured Classes : {model.names}")
        if len(boxes) == 0:
            print("⚠️  Bounding Box Detected : NO (0 objects passed standard 0.25 confidence threshold)")
        else:
            print(f"✅ Bounding Box Detected : YES ({len(boxes)} detection(s) found)\n")
            print(f"{'#':<4} {'Detected Class':<18} {'Confidence':<14} {'Bounding Box (xyxy)':<30}")
            print("-" * 70)
            for i, box in enumerate(boxes, 1):
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"Class {cls_id}")
                score = float(box.conf[0].item())
                xyxy = [round(c, 1) for c in box.xyxy[0].tolist()]
                print(f"{i:<4} {cls_name:<18} {score*100:>6.2f}%        {str(xyxy):<30}")

        saved_path = Path(result.save_dir)
        annotated_files = [f for f in saved_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if annotated_files:
            print(f"\n💾 Annotated Image Saved : {annotated_files[0].resolve()}")
            print(f"👉 To view on Mac, run  : open '{annotated_files[0].resolve()}'")

        return True

    except Exception as e:
        print(f"❌ Error during image inference: {e}")
        return False


def test_text_classification():
    """Runs ASTRA NLP complaint classification on standard multilingual test cases."""
    print("\n" + "=" * 70)
    print("📝 MODULE 2: TEXT-BASED COMPLAINT CLASSIFICATION (ASTRA NLP)")
    print("=" * 70)

    # 1. Verify ASTRA model and module existence
    if not ASTRA_MODEL_PATH.is_file():
        print(f"❌ Error: ASTRA model weights not found at '{ASTRA_MODEL_PATH}'.")
        return False

    predict_script = ASTRA_DIR / "astra_predict.py"
    if not predict_script.is_file():
        print(f"❌ Error: Classifier script not found at '{predict_script}'.")
        return False

    try:
        # Import existing prediction function
        from astra_handoff.astra_predict import predict_complaint
    except ImportError as e:
        print(f"❌ Import Error: Could not import 'predict_complaint' from 'astra_handoff.astra_predict'.")
        print(f"   Details: {e}")
        return False

    # 2. Benchmark test complaints
    test_cases = [
        ("English", "Garbage has not been collected from our street for three days."),
        ("Hinglish", "Hamare area mein paani nahi aa raha hai."),
        ("Hindi", "हमारी सड़क पर बहुत बड़े गड्ढे हैं।")
    ]

    print(f"📦 Model File   : {ASTRA_MODEL_PATH.resolve()}")
    print("ℹ️  Note: Scores represent uncalibrated model probabilities, not guaranteed accuracy.")
    print("-" * 70)

    all_passed = True
    for idx, (lang, text) in enumerate(test_cases, 1):
        print(f"\n--- [Case {idx} - {lang}] ---")
        print(f"💬 Complaint Text : \"{text}\"")

        try:
            res = predict_complaint(text)
            category = res.get("category", "UNKNOWN")
            confidence = res.get("confidence", 0.0)
            margin = res.get("score_margin", 0.0)
            needs_review = res.get("needs_review", False)
            all_scores = res.get("all_scores", {})

            print(f"🏷️  Predicted Category : {category}")
            print(f"📊 Top Score          : {confidence*100:.2f}% (Margin over 2nd: {margin*100:.2f}%)")

            # Display all candidate scores
            scores_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in all_scores.items()])
            print(f"📈 All Class Scores   : {scores_str}")

            if needs_review:
                print("⚠️  MANUAL REVIEW REQUIRED : Score margin between top candidates is below 0.05.")
                print("    This prediction is flagged for human dispatcher verification.")
            else:
                print("✅ Confidence Margin   : Sufficient (margin >= 0.05, no manual review flag)")

        except Exception as e:
            print(f"❌ Error predicting complaint: {e}")
            all_passed = False

    return all_passed


def main():
    print("=" * 70)
    print("🚀 CivicGrievanceAI - Unified Multi-Modal Integration Test")
    print("=" * 70)

    # Optional image path from command-line argument
    cli_image = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Run tests independently
    img_success = test_image_detection(cli_image)
    txt_success = test_text_classification()

    print("\n" + "=" * 70)
    print("🏁 SUMMARY OF UNIFIED TEST RUN")
    print("=" * 70)
    print(f"📷 Image Detection Module : {'PASSED' if img_success else 'FAILED or SKIPPED'}")
    print(f"📝 Text Classifier Module : {'PASSED' if txt_success else 'FAILED'}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
