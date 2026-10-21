"""
CivicGrievanceAI - Model Verification Script
Tests Prithvi's YOLO11n civic complaint object-detection model.

Task: Detect Road, Electricity, and Water issues.
Trained input size: 768x768
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def run_verification(model_path: str = "models/best.pt", image_path: str = None, imgsz: int = 768, conf: float = 0.25):
    model_file = Path(model_path)
    if not model_file.is_file():
        print(f"❌ Error: Model weights not found at '{model_path}'.")
        sys.exit(1)

    # If image_path not explicitly provided, find the first image in sample_images/
    if image_path is None:
        sample_dir = Path("sample_images")
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
        images = [f for f in sample_dir.iterdir() if f.suffix.lower() in image_extensions]
        if not images:
            print(f"❌ Error: No images found in '{sample_dir}/'. Please place an image there.")
            sys.exit(1)
        image_file = images[0]
    else:
        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"❌ Error: Image file not found at '{image_path}'.")
            sys.exit(1)

    print("=" * 60)
    print("🚀 CivicGrievanceAI - YOLO11n Model Verification")
    print("=" * 60)
    print(f"📦 Loading weights from: {model_file.resolve()}")
    model = YOLO(str(model_file))

    print(f"🏷️  Classes configured in model: {model.names}")
    print(f"🖼️  Running inference on: {image_file.name} (imgsz={imgsz}, conf={conf})")
    print("-" * 60)

    # Run inference with save=True to save the annotated result into test_outputs/
    results = model.predict(
        source=str(image_file),
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=str(Path("test_outputs").resolve()),
        name=image_file.stem,
        exist_ok=False,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    if len(boxes) == 0:
        print("❌ Bounding Box Detected: NO (0 objects found above confidence threshold)")
        print("   Tip: Try testing another image or adjusting confidence threshold.")
    else:
        print(f"✅ Bounding Box Detected: YES ({len(boxes)} detection(s) found)\n")
        print(f"{'#':<4} {'Class Name':<15} {'Confidence':<12} {'Bounding Box (xyxy)':<30}")
        print("-" * 65)
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, f"Class {cls_id}")
            score = float(box.conf[0].item())
            xyxy = [round(coord, 1) for coord in box.xyxy[0].tolist()]
            print(f"{i:<4} {cls_name:<15} {score*100:>6.2f}%     {str(xyxy):<30}")

    saved_dir = Path(result.save_dir)
    annotated_files = [f for f in saved_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    print("\n" + "=" * 60)
    print(f"💾 Annotated directory: {saved_dir.resolve()}")
    if annotated_files:
        saved_img = annotated_files[0]
        print(f"🖼️  Annotated image saved: {saved_img.name}")
        print(f"👉 To view on your Mac, run: open '{saved_img.resolve()}'")
    print("=" * 60)

if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_verification(image_path=img_arg)
