r"""ASTRA audio grievance demo: IndicConformer ASR + existing five-class classifier.
Run from project root: python astra_handoff/astra_audio_predict.py audio_samples/Audio_1.mp3 --language hi --decoder ctc
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

# In-memory model cache to avoid reloading 2.5GB model on every complaint
_CACHED_ASR_MODEL = None
_CACHED_MODEL_TARGET = None


def resolve_classifier_python(project_root: Path, explicit_path: Optional[Path] = None) -> Path:
    """Resolves the Python interpreter for running astra_predict across platforms."""
    if explicit_path and Path(explicit_path).is_file():
        return Path(explicit_path).resolve()

    env_python = os.getenv("ASTRA_CLASSIFIER_PYTHON")
    if env_python and Path(env_python).is_file():
        return Path(env_python).resolve()

    candidates = [
        project_root / ".venv" / "bin" / "python",                # macOS/Linux standard ASTRA venv
        project_root / "indic_venv" / "bin" / "python",           # macOS/Linux indic_venv
        project_root / "indic_venv" / "Scripts" / "python.exe",   # Windows indic_venv
        project_root / ".venv" / "Scripts" / "python.exe",        # Windows .venv
        Path(sys.executable),                                     # Active interpreter
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    return Path(sys.executable).resolve()


def ensure_local_model_directory(
    local_dir: Path,
    repo_id: str = "ai4bharat/indic-conformer-600m-multilingual",
    token: Optional[str] = None
) -> Path:
    """
    Ensures a dedicated local directory exists where all ONNX models and weights
    are real regular files (not symlinks to blobs), preventing ONNXRuntimeError:
    'External data path escapes model directory'.
    """
    import shutil
    local_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = local_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # 1. First check if Hugging Face hub cache has the snapshot downloaded
    cache_hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    repo_slug = f"models--{repo_id.replace('/', '--')}"
    snaps_dir = cache_hub / repo_slug / "snapshots"
    if snaps_dir.is_dir():
        for snap in snaps_dir.iterdir():
            snap_assets = snap / "assets"
            if snap.is_dir() and snap_assets.is_dir() and (snap_assets / "encoder.onnx").exists():
                # Link root files
                for f in snap.iterdir():
                    if f.is_file():
                        dst = local_dir / f.name
                        if not dst.exists():
                            real_f = f.resolve()
                            try:
                                os.link(real_f, dst)
                            except OSError:
                                shutil.copyfile(real_f, dst)
                # Link asset files as real files
                for f in snap_assets.iterdir():
                    dst = assets_dir / f.name
                    if not dst.exists():
                        real_f = f.resolve()
                        try:
                            os.link(real_f, dst)
                        except OSError:
                            shutil.copyfile(real_f, dst)
                break

    # 2. If assets are still missing, download snapshot directly without symlinks
    if not (assets_dir / "encoder.onnx").is_file():
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
            local_dir_use_symlinks=False
        )

    return local_dir


def load_indic_conformer_local(local_dir: Path) -> Any:
    """Loads IndicASRModel directly from dedicated local directory without symlink escapes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_onnx", str(local_dir / "model_onnx.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_onnx"] = mod
    spec.loader.exec_module(mod)
    config = mod.IndicASRConfig(ts_folder=str(local_dir))
    model = mod.IndicASRModel(config)
    model.eval()
    return model


def get_asr_model(
    model_target: str = "ai4bharat/indic-conformer-600m-multilingual",
    hf_token: Optional[str] = None
) -> Any:
    """
    Lazy loads and caches the IndicConformer ASR model in memory.
    Resolves ONNX models and weights inside a dedicated local directory to prevent
    symlink escapes in onnxruntime, and reuses the loaded model across requests.
    """
    global _CACHED_ASR_MODEL, _CACHED_MODEL_TARGET

    target = model_target or os.getenv("ASTRA_ASR_MODEL") or "ai4bharat/indic-conformer-600m-multilingual"
    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if _CACHED_ASR_MODEL is not None and _CACHED_MODEL_TARGET == target:
        return _CACHED_ASR_MODEL

    # Persist token locally if provided so future calls and background workers are authenticated
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass
        os.environ["HF_TOKEN"] = token

    project_root = Path(__file__).resolve().parent.parent
    local_model_dir = project_root / "models" / "indic_conformer_600m"

    # Priority 1: Check dedicated local directory (resolves all external data without symlinks)
    if (local_model_dir / "assets" / "encoder.onnx").is_file():
        print(f"Loading IndicConformer model from dedicated local directory: {local_model_dir}...", flush=True)
        try:
            model = load_indic_conformer_local(local_model_dir)
            _CACHED_ASR_MODEL = model
            _CACHED_MODEL_TARGET = target
            return _CACHED_ASR_MODEL
        except Exception as e:
            print(f"Warning: Local load failed ({e}), falling back to preparation...", flush=True)

    # Priority 2: Prepare local model directory from cache or download
    try:
        ensure_local_model_directory(local_model_dir, repo_id=target, token=token)
        if (local_model_dir / "assets" / "encoder.onnx").is_file():
            model = load_indic_conformer_local(local_model_dir)
            _CACHED_ASR_MODEL = model
            _CACHED_MODEL_TARGET = target
            return _CACHED_ASR_MODEL
    except Exception as prep_err:
        err_msg = str(prep_err)
        if "401" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            raise RuntimeError(
                f"Model '{target}' is a gated Hugging Face repository (~2.55 GB). "
                "Please accept conditions at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual "
                "and provide your Hugging Face access token."
            ) from prep_err

    # Priority 3: AutoModel fallback
    try:
        from transformers import AutoModel
        load_kwargs = {"trust_remote_code": True}
        if token:
            load_kwargs["token"] = token
        print(f"Loading IndicConformer model from '{target}' via transformers...", flush=True)
        model = AutoModel.from_pretrained(target, **load_kwargs)
        model.eval()
        _CACHED_ASR_MODEL = model
        _CACHED_MODEL_TARGET = target
        return _CACHED_ASR_MODEL
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "restricted" in err_msg.lower():
            raise RuntimeError(
                f"Model '{target}' is a gated Hugging Face repository (~2.55 GB). "
                "Please accept conditions at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual "
                "and provide your Hugging Face access token."
            ) from e
        raise RuntimeError(f"Failed to load speech recognition model '{target}': {err_msg}") from e


def predict_audio(
    audio_path: Union[str, Path],
    language: str = "hi",
    decoder: str = "ctc",
    model: Optional[Any] = None,
    model_target: Optional[str] = None,
    hf_token: Optional[str] = None,
    classifier_python: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Transcribes Hindi/Marathi audio using IndicConformer and classifies the grievance into 5 departments.
    Reuses provided/cached model in-memory to prevent recurring 25-30s reload latencies.
    """
    audio = Path(audio_path).resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    if audio.stat().st_size == 0:
        raise ValueError("Audio file is empty (0 bytes).")
    if language not in ("hi", "mr"):
        raise ValueError(f"Unsupported language '{language}'. Supported: 'hi' (Hindi), 'mr' (Marathi).")

    try:
        import soundfile as sf
        import torch
    except ImportError as e:
        raise RuntimeError(f"Missing audio dependency: {e}. Please ensure soundfile and torch are installed.") from e

    # Obtain or reuse cached model
    active_model = model or get_asr_model(
        model_target=model_target or "ai4bharat/indic-conformer-600m-multilingual",
        hf_token=hf_token
    )

    # Convert audio to 16kHz mono WAV using FFmpeg
    with tempfile.TemporaryDirectory(prefix="astra_audio_") as tmp:
        wav_path = Path(tmp) / "audio_16k.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(audio),
                 "-ar", "16000", "-ac", "1", "-y", str(wav_path)],
                check=True,
                capture_output=True,
                text=True
            )
        except FileNotFoundError:
            raise RuntimeError("FFmpeg executable not found in PATH. Please install ffmpeg.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg audio conversion failed: {e.stderr or e}")

        samples, rate = sf.read(wav_path, dtype="float32")
        if rate != 16000:
            raise RuntimeError(f"Expected 16000 Hz sample rate, got {rate} Hz.")
        if getattr(samples, "ndim", 1) != 1:
            raise RuntimeError("Expected mono waveform after FFmpeg conversion.")
        waveform = torch.from_numpy(samples).unsqueeze(0)

    # Speech recognition inference
    with torch.inference_mode():
        transcript = str(active_model(waveform, language, decoder)).strip()

    if not transcript:
        return {
            "success": False,
            "error": "No transcription generated from audio.",
            "transcript": "",
            "category": None,
            "confidence": 0.0,
            "needs_review": False
        }

    # Department classification: try direct in-process first for speed
    prediction = None
    try:
        from astra_handoff.astra_predict import predict_complaint
        prediction = predict_complaint(transcript)
    except ImportError:
        try:
            from astra_predict import predict_complaint
            prediction = predict_complaint(transcript)
        except ImportError:
            pass

    # Subprocess fallback if running in decoupled environments
    if prediction is None:
        project = Path(__file__).resolve().parent.parent
        resolved_py = classifier_python or resolve_classifier_python(project)
        predict_path = Path(__file__).resolve().parent / "astra_predict.py"
        classifier_code = (
            "import json,sys; "
            "from pathlib import Path; "
            "sys.path.insert(0,str(Path(sys.argv[1]).resolve().parent)); "
            "from astra_predict import predict_complaint; "
            "text=Path(sys.argv[2]).read_text(encoding='utf-8'); "
            "r=predict_complaint(text); print(json.dumps(r,ensure_ascii=False))"
        )
        with tempfile.TemporaryDirectory(prefix="astra_text_") as tmp:
            transcript_path = Path(tmp) / "transcript.txt"
            transcript_path.write_text(transcript, encoding="utf-8")
            result = subprocess.run(
                [str(resolved_py), "-X", "utf8", "-c", classifier_code,
                 str(predict_path), str(transcript_path)],
                capture_output=True, text=True, encoding="utf-8", check=True,
            )
        prediction = json.loads(result.stdout.strip())

    margin = float(prediction.get("score_margin", 0.0))
    needs_review = bool(prediction.get("needs_review", False))
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
        "category": prediction.get("category"),
        "confidence": float(prediction.get("confidence", 0.0)),
        "score_margin": margin,
        "all_scores": prediction.get("all_scores", {}),
        "needs_review": needs_review,
        "warning_message": warning
    }


def main():
    parser = argparse.ArgumentParser(description="Transcribe a Hindi/Marathi civic complaint and classify it with ASTRA.")
    parser.add_argument("audio", type=Path, help="MP3/WAV/M4A audio path")
    parser.add_argument("--language", required=True, choices=("hi", "mr"))
    parser.add_argument("--decoder", default="ctc", choices=("ctc", "rnnt"))
    parser.add_argument("--classifier-python", type=Path, default=None, help="Explicit path to classifier Python interpreter")
    parser.add_argument("--model-name-or-path", type=str, default=None, help="HF model name or local path (defaults to ai4bharat/indic-conformer-600m-multilingual)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for gated models")
    parser.add_argument("--json", action="store_true", help="Output final result as JSON to stdout")
    args = parser.parse_args()

    try:
        res = predict_audio(
            audio_path=args.audio,
            language=args.language,
            decoder=args.decoder,
            model_target=args.model_name_or_path,
            hf_token=args.hf_token,
            classifier_python=args.classifier_python
        )
    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e), "transcript": ""}, ensure_ascii=False))
        else:
            sys.stderr.write(f"ERROR: {e}\n")
        sys.exit(1)

    if args.json:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print("\nTRANSCRIPT:", res["transcript"], flush=True)
        print("\nASTRA CATEGORY:", res["category"])
        print("MODEL SCORE:", res["confidence"])
        print("ALL SCORES:", res["all_scores"])
        print("SCORE MARGIN:", res["score_margin"])
        if res["needs_review"]:
            print("MANUAL REVIEW REQUIRED: Category scores are too close.")
        else:
            print("Review flag: Not triggered")
        print("Note: scores are not calibrated confidence or production validation.")


if __name__ == "__main__":
    main()
