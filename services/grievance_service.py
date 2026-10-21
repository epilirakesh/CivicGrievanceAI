"""
CivicGrievanceAI - Core Reusable Integration Service
Unifies multimodal issue analysis:
  - Prithvi's YOLO11n object detector (models/best.pt)
  - Shubham's ASTRA NLP text classifier (astra_handoff/astra_predict.py)
  - SQLite grievance persistence (services/database.py)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.database import insert_complaint


class GrievanceAnalyzer:
    """Reusable service for civic complaint analysis and routing."""

    def __init__(self, yolo_weights: Optional[Path] = None):
        self.yolo_path = Path(yolo_weights) if yolo_weights else PROJECT_ROOT / "models" / "best.pt"
        self._yolo_model = None
        self._asr_model = None
        self._asr_model_target = None

    def get_asr_model(
        self,
        model_target: str = "ai4bharat/indic-conformer-600m-multilingual",
        hf_token: Optional[str] = None
    ) -> Any:
        """Lazy load and cache the ASR IndicConformer model in memory across complaints."""
        if self._asr_model is not None and self._asr_model_target == model_target:
            return self._asr_model

        from astra_handoff.astra_audio_predict import get_asr_model as load_asr
        self._asr_model = load_asr(model_target=model_target, hf_token=hf_token)
        self._asr_model_target = model_target
        return self._asr_model

    @property
    def yolo_model(self):
        """Lazy load YOLO model to optimize startup time and memory."""
        if self._yolo_model is None:
            if not self.yolo_path.is_file():
                raise FileNotFoundError(f"YOLO model weights not found at '{self.yolo_path}'.")
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(self.yolo_path))
        return self._yolo_model

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyzes text complaint using ASTRA NLP model."""
        if not text or not str(text).strip():
            return {
                "success": False,
                "error": "Complaint text cannot be empty.",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        try:
            from astra_handoff.astra_predict import predict_complaint
            res = predict_complaint(text.strip())

            category = res.get("category", "OTHER_REVIEW")
            confidence = res.get("confidence", 0.0)
            margin = res.get("score_margin", 0.0)
            needs_review = res.get("needs_review", False)
            all_scores = res.get("all_scores", {})

            warning = None
            if needs_review:
                warning = (
                    f"Confidence margin ({margin*100:.2f}%) between top categories is below 5%. "
                    "Manual dispatcher verification required."
                )

            return {
                "success": True,
                "category": category,
                "confidence": round(float(confidence), 4),
                "score_margin": round(float(margin), 4),
                "all_scores": all_scores,
                "needs_review": needs_review,
                "warning_message": warning
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "category": "OTHER_REVIEW",
                "confidence": 0.0,
                "needs_review": True,
                "warning_message": f"Text classification error: {e}"
            }

    def analyze_image(
        self,
        image_input: Union[str, Path],
        conf_threshold: float = 0.25,
        save_annotated: bool = True,
        output_name: str = "service_detect"
    ) -> Dict[str, Any]:
        """Analyzes complaint image using YOLO11n object detector."""
        img_path = Path(image_input)
        if not img_path.is_file():
            return {
                "success": False,
                "error": f"Image file not found at '{img_path}'.",
                "detections": [],
                "detected_classes": [],
                "detection_count": 0,
                "annotated_image_path": None
            }

        try:
            results = self.yolo_model.predict(
                source=str(img_path),
                imgsz=768,
                conf=conf_threshold,
                save=False,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes
            detections = []
            detected_classes = set()

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.yolo_model.names.get(cls_id, f"Class {cls_id}")
                score = float(box.conf[0].item())
                xyxy = [round(c, 1) for c in box.xyxy[0].tolist()]

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(score, 4),
                    "box_xyxy": xyxy
                })
                detected_classes.add(cls_name)

            annotated_file = None
            if save_annotated:
                annotated_dir = PROJECT_ROOT / "test_outputs" / "annotated"
                annotated_dir.mkdir(parents=True, exist_ok=True)
                annotated_filename = annotated_dir / f"annotated_{img_path.stem}.jpg"
                result.save(filename=str(annotated_filename))
                annotated_file = str(annotated_filename.resolve())

            return {
                "success": True,
                "detections": detections,
                "detected_classes": sorted(list(detected_classes)),
                "detection_count": len(detections),
                "has_detections": len(detections) > 0,
                "annotated_image_path": annotated_file
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "detected_classes": [],
                "detection_count": 0,
                "annotated_image_path": None
            }

    def analyze_audio(
        self,
        audio_path: Union[str, Path],
        language: str = "hi",
        decoder: str = "ctc",
        model_name_or_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        timeout: int = 600,
        use_subprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Shubham's existing IndicConformer + ASTRA NLP speech pipeline.
        Reuses the loaded model in-memory across calls to eliminate 25-30s reload times.
        Supports fallback to subprocess if requested.
        """
        audio_file = Path(audio_path)
        if not audio_file.is_file():
            return {
                "success": False,
                "error": f"Audio file not found at '{audio_file}'.",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        # Check supported audio extensions
        valid_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".webm", ".caf", ".amr", ".mp4", ".wma"}
        if audio_file.suffix.lower() not in valid_exts:
            return {
                "success": False,
                "error": f"Unsupported audio format '{audio_file.suffix}'. Supported formats: {', '.join(sorted(valid_exts))}.",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        # Validate non-empty file
        if audio_file.stat().st_size == 0:
            return {
                "success": False,
                "error": "Audio file is empty (0 bytes).",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        if language not in ("hi", "mr"):
            return {
                "success": False,
                "error": f"Unsupported language '{language}'. Supported: 'hi' (Hindi), 'mr' (Marathi).",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        target_model = model_name_or_path or "ai4bharat/indic-conformer-600m-multilingual"

        # Fast path: in-process execution reusing cached model in memory
        if not use_subprocess:
            try:
                from astra_handoff.astra_audio_predict import predict_audio
                active_model = self.get_asr_model(model_target=target_model, hf_token=hf_token)
                result = predict_audio(
                    audio_path=audio_file,
                    language=language,
                    decoder=decoder,
                    model=active_model,
                    hf_token=hf_token
                )
                return result
            except Exception as e:
                err_str = str(e)
                # If in-process model load fails, return clear error without crashing
                return {
                    "success": False,
                    "error": err_str,
                    "transcript": "",
                    "category": None,
                    "confidence": 0.0,
                    "needs_review": False
                }

        # Subprocess execution path (with generous timeout)
        script_path = PROJECT_ROOT / "astra_handoff" / "astra_audio_predict.py"
        if not script_path.is_file():
            return {
                "success": False,
                "error": f"Audio prediction script not found at '{script_path}'.",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

        import json
        import os
        import subprocess

        python_bin = sys.executable
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_python.is_file():
            python_bin = str(venv_python)

        cmd = [
            python_bin,
            str(script_path),
            str(audio_file),
            "--language", language,
            "--decoder", decoder,
            "--json"
        ]

        if model_name_or_path:
            cmd.extend(["--model-name-or-path", str(model_name_or_path)])
        if hf_token:
            cmd.extend(["--hf-token", str(hf_token)])

        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = str(hf_token)

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                env=env
            )

            if res.returncode != 0:
                err_detail = (res.stderr or res.stdout or f"Subprocess exited with code {res.returncode}").strip()
                return {
                    "success": False,
                    "error": err_detail,
                    "transcript": "",
                    "category": None,
                    "confidence": 0.0,
                    "needs_review": False,
                    "raw_stderr": res.stderr,
                    "raw_stdout": res.stdout
                }

            # Parse JSON line from output
            output_lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            parsed_json = None
            for line in reversed(output_lines):
                if line.startswith("{") and line.endswith("}"):
                    try:
                        parsed_json = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue

            if not parsed_json or not parsed_json.get("success", False):
                err = parsed_json.get("error") if parsed_json else f"Unparseable audio output: {res.stdout}"
                return {
                    "success": False,
                    "error": err or "Transcription failed without specific error.",
                    "transcript": "",
                    "category": None,
                    "confidence": 0.0,
                    "needs_review": False
                }

            transcript = parsed_json.get("transcript", "").strip()
            category = parsed_json.get("category")
            confidence = parsed_json.get("confidence", 0.0)
            margin = parsed_json.get("score_margin", 0.0)
            needs_review = parsed_json.get("needs_review", False)
            all_scores = parsed_json.get("all_scores", {})

            warning = None
            if needs_review:
                warning = (
                    f"Confidence margin ({margin*100:.2f}%) between top categories is below 5%. "
                    "Manual dispatcher verification required."
                )

            return {
                "success": True,
                "transcript": transcript,
                "language": language,
                "decoder": decoder,
                "category": category,
                "confidence": round(float(confidence), 4),
                "score_margin": round(float(margin), 4),
                "all_scores": all_scores,
                "needs_review": needs_review,
                "warning_message": warning,
                "raw_stdout": res.stdout
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Audio transcription timed out after {timeout} seconds.",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Subprocess execution error: {e}",
                "transcript": "",
                "category": None,
                "confidence": 0.0,
                "needs_review": False
            }

    def process_grievance(
        self,
        complaint_text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
        audio_language: Optional[str] = None,
        intake_mode: str = "Text",
        citizen_name: str = "Anonymous",
        citizen_contact: str = "",
        location: str = "Unspecified",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_accuracy: Optional[float] = None,
        location_timestamp: Optional[str] = None,
        ward: Optional[str] = None,
        landmark: Optional[str] = None,
        address: Optional[str] = None,
        assigned_to: str = "",
        save_to_db: bool = True,
        db_path: Optional[Union[str, Path]] = None,
        precomputed_nlp: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Performs full end-to-end multimodal grievance intake, evaluation, and persistence."""
        # Use precomputed NLP results if available (e.g. from reviewed audio pipeline), else analyze
        if precomputed_nlp and precomputed_nlp.get("success"):
            text_result = precomputed_nlp
        else:
            text_result = self.analyze_text(complaint_text) if complaint_text else None

        image_result = self.analyze_image(image_path) if image_path else None

        # Determine automated priority assessment
        priority = "Medium"
        if image_result and "Electricity" in image_result.get("detected_classes", []):
            priority = "High"
        elif text_result and text_result.get("category") == "ELECTRICAL_SAFETY":
            priority = "High"
        elif image_result and image_result.get("detection_count", 0) >= 5:
            priority = "High"

        # Check if manual review is required
        needs_review = False
        review_reasons = []

        if text_result and text_result.get("needs_review"):
            needs_review = True
            review_reasons.append("Text classifier confidence margin < 5%")

        if text_result and text_result.get("category") == "OTHER_REVIEW":
            needs_review = True
            review_reasons.append("Category classified as OTHER_REVIEW")

        resolved_loc = address or location or "Unspecified"

        record = {
            "citizen_name": citizen_name,
            "citizen_contact": citizen_contact,
            "location": resolved_loc,
            "complaint_text": complaint_text or "",
            "text_category": text_result.get("category") if text_result else None,
            "text_confidence": text_result.get("confidence") if text_result else None,
            "text_margin": text_result.get("score_margin") if text_result else None,
            "text_all_scores": text_result.get("all_scores", {}) if text_result else {},
            "text_needs_review": needs_review,
            "image_path": str(image_path) if image_path else None,
            "annotated_image_path": image_result.get("annotated_image_path") if image_result else None,
            "detected_classes": image_result.get("detected_classes", []) if image_result else [],
            "detected_objects_count": image_result.get("detection_count", 0) if image_result else 0,
            "priority": priority,
            "status": "Under Review" if needs_review else "New",
            "review_reasons": review_reasons,
            "text_analysis": text_result,
            "image_analysis": image_result,
            "audio_path": str(audio_path) if audio_path else "",
            "audio_language": audio_language or "",
            "intake_mode": intake_mode,
            "latitude": latitude,
            "longitude": longitude,
            "location_accuracy": location_accuracy,
            "location_timestamp": location_timestamp or "",
            "location_source": "Browser GPS" if latitude and longitude else "Manual",
            "ward": ward or "",
            "landmark": landmark or "",
            "address": resolved_loc,
            "assigned_to": assigned_to or ""
        }

        if save_to_db:
            cid = insert_complaint(record, db_path=db_path)
            record["complaint_id"] = cid

        return record


# Global singleton instance for easy import across modules
default_analyzer = GrievanceAnalyzer()

