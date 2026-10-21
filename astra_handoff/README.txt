ASTRA — NLP AND SPEECH MODULE HANDOFF

FILES
1. astra_predict.py
   Five-class text complaint classifier.

2. astra_baseline_model_122.joblib
   Trained TF-IDF + Logistic Regression model.

3. astra_audio_predict.py
   Hindi/Marathi audio transcription and classification.

FIVE NLP CATEGORIES
ROADS_INFRASTRUCTURE
GARBAGE_SANITATION
ELECTRICAL_SAFETY
WATER_SUPPLY
OTHER_REVIEW

ENVIRONMENT REQUIREMENTS
- Windows
- Python with scikit-learn and joblib for NLP
- Separate speech Python environment with PyTorch,
  Transformers and SoundFile
- FFmpeg installed and available in PATH
- Internet/model cache access for IndicConformer

CURRENT PROJECT LAYOUT
Project root/
    venv/
    indic_venv/
    astra_handoff/
        astra_predict.py
        astra_baseline_model_122.joblib
        astra_audio_predict.py

TEXT CLASSIFICATION
Import predict_complaint from astra_predict.py.

Returns:
- category
- confidence (uncalibrated model score)
- all_scores
- score_margin
- needs_review

AUDIO CLASSIFICATION
Run from project root:

.\venv\Scripts\python.exe .\astra_handoff\astra_audio_predict.py .\test1.mp3 --language hi --decoder ctc

Use --language mr for Marathi.

The audio script invokes indic_venv for text classification.
Both environments must exist at the expected locations.

MODEL RESULTS
Preliminary NLP test accuracy: 80.65%
Macro F1-score: approximately 0.80
Training examples: 91
Test examples: 31

DATASET
125 complaints across five categories.
122 provisional training candidates.
Mostly team-written examples.
Human review and independent evaluation remain pending.

IMPORTANT
Model scores are not calibrated probabilities.
The manual-review threshold is experimental.
The audio warning code has passed syntax checks,
but its updated end-to-end display has not been retested.

INTEGRATION STATUS
Text classifier: tested.
Hindi audio pipeline: tested.
Marathi transcription: tested.
Marathi classification: one known incorrect prediction.
Full Streamlit integration: pending.